"""Human-in-the-loop: detección de tareas críticas, pausa obligatoria y aprobación externa.

Flujo (después de que el validador cierra la tarea):

    validator ─▶ risk_gate ─┬─▶ END                      (tarea no crítica)
                            └─▶ human_approval ──interrupt()──  ⏸  espera POST /tasks/{id}/approve
                                     │ aprobado + publicar ─▶ publish_report ─▶ END
                                     └ aprobado sin publicar / rechazado ─▶ END

`risk_gate` decide (de forma determinista, sin LLM) si la tarea es CRÍTICA:
  1. Herramienta con efectos secundarios: la petición pide publicar/enviar el informe
     (`publish_report` escribe un fichero fuera del sistema → no se ejecuta sin humano).
  2. Costo muy alto: los tokens consumidos superan `HITL_TOKEN_THRESHOLD`.
  3. Calidad: el validador cerró por límite de revisiones sin aprobar.
  4. El cliente lo pidió explícitamente (`require_approval: true`).

El estado pausado vive en los checkpoints de Redis (RedisSaver), así que la aprobación
puede llegar horas después o a otro worker distinto.
"""

from __future__ import annotations

import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.config import get_config
from langgraph.graph import END
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from app.observability import tokens_used
from app.store import JobStatus

HITL_TOKEN_THRESHOLD = int(os.getenv("HITL_TOKEN_THRESHOLD", "60000"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "reports"))

_PUBLISH_RE = re.compile(r"\b(publica\w*|envia\w*|enviar|comparte|compartir|distribu\w+|publish|send)\b")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def wants_publication(question: str) -> bool:
    """El sistema identifica peticiones con efecto secundario (publicar/enviar el informe)."""
    return bool(_PUBLISH_RE.search(_normalize(question)))


# --------------------------------------------------------------------------- #
# Herramienta con efectos secundarios
# --------------------------------------------------------------------------- #
@tool
def publish_report(job_id: str, title: str, content: str) -> str:
    """Publica el informe final escribiéndolo en el directorio de informes compartido.
    Tiene efectos secundarios: solo se ejecuta tras aprobación humana."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{job_id}.md"
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.write_text(f"# {title}\n\n_Publicado {stamp} · job {job_id}_\n\n{content}\n", encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------- #
# Nodos del grafo
# --------------------------------------------------------------------------- #
def _thread_id() -> str:
    return get_config()["configurable"]["thread_id"]


def risk_gate(state: dict) -> dict:
    """Clasifica la tarea. Escribe las razones en el estado (checkpointed) para que la
    decisión no cambie si el nodo HITL se re-ejecuta al reanudar."""
    tokens = tokens_used(_thread_id())
    reasons: list[str] = []
    if state.get("publish_requested"):
        reasons.append("side_effect: la tarea ejecuta publish_report (escribe/distribuye el informe)")
    if tokens > HITL_TOKEN_THRESHOLD:
        reasons.append(f"high_cost: {tokens} tokens consumidos (> umbral {HITL_TOKEN_THRESHOLD})")
    if state.get("validation_feedback"):
        reasons.append("low_quality: el validador cerró por límite de revisiones sin aprobar")
    if state.get("force_approval"):
        reasons.append("requested: el cliente pidió aprobación humana")
    return {"critical_reasons": reasons, "tokens_at_gate": tokens}


def route_after_gate(state: dict) -> Literal["human_approval", "__end__"]:
    return "human_approval" if state.get("critical_reasons") else END


def human_approval(state: dict) -> Command[Literal["publish_report", "__end__"]]:
    """Pausa obligatoria: `interrupt` guarda el estado en Redis y devuelve el control.
    Al reanudar con Command(resume=decision) la ejecución continúa desde aquí."""
    action = "publish_report" if state.get("publish_requested") else "release_answer"
    decision = interrupt({
        "type": "approval_required",
        "action": action,
        "reasons": state.get("critical_reasons", []),
        "tokens_used": state.get("tokens_at_gate", 0),
        "draft": state.get("final_answer", ""),
    })
    decision = decision if isinstance(decision, dict) else {"approved": bool(decision)}
    approval = {
        "approved": bool(decision.get("approved")),
        "reviewer": decision.get("reviewer") or "anónimo",
        "comment": decision.get("comment") or "",
        "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    note = f"[human_approval] {'APROBADO' if approval['approved'] else 'RECHAZADO'} por {approval['reviewer']}"
    update = {"approval": approval, "messages": [AIMessage(content=note, name="human")]}
    if not approval["approved"]:
        update["final_answer"] = (
            f"⛔ Respuesta retenida: la revisión humana rechazó la acción '{action}'. "
            f"Comentario: {approval['comment'] or '(sin comentario)'}"
        )
        return Command(goto=END, update=update)
    return Command(goto="publish_report" if action == "publish_report" else END, update=update)


def publish_report_node(state: dict) -> dict:
    path = publish_report.invoke({
        "job_id": _thread_id(),
        "title": "Informe del orquestador multi-agente",
        "content": state.get("final_answer", ""),
    })
    return {"published_to": path, "messages": [AIMessage(content=f"[publish_report] {path}", name="publisher")]}


# --------------------------------------------------------------------------- #
# Endpoint de aprobación
# --------------------------------------------------------------------------- #
class ApprovalRequest(BaseModel):
    approved: bool = Field(description="true = continuar; false = rechazar la acción crítica")
    reviewer: str | None = Field(default=None, max_length=100)
    comment: str | None = Field(default=None, max_length=1000)


router = APIRouter(tags=["human-in-the-loop"])


@router.post("/tasks/{job_id}/approve", status_code=202)
async def approve_task(job_id: str, body: ApprovalRequest, request: Request) -> dict:
    """Entrega la decisión humana. No ejecuta el grafo: encola un mensaje `resume`
    que un worker procesa con Command(resume=...)."""
    store = request.app.state.store
    # Transición atómica AWAITING_APPROVAL → PENDING: evita dobles aprobaciones concurrentes.
    claimed = await store.claim_for_resume(job_id)
    if claimed is None:
        raise HTTPException(404, "job no encontrado")
    if claimed is False:
        job = await store.get(job_id)
        raise HTTPException(409, f"el job está en estado {job['status']}, no espera aprobación")
    await store.enqueue({"job_id": job_id, "kind": "resume", "decision": body.model_dump()})
    return {"job_id": job_id, "status": JobStatus.PENDING, "decision": body.model_dump()}
