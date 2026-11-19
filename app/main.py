"""API REST del orquestador multi-agente.

    POST /tasks                 encola la tarea y devuelve job_id (202, no espera al grafo)
    GET  /tasks/{job_id}        estado del job (polling)
    GET  /tasks                 últimos jobs
    POST /tasks/{job_id}/approve  decisión humana para un job en AWAITING_APPROVAL (app/hitl.py)
    GET  /health                ping a Redis

Los endpoints solo hablan con Redis mediante `redis.asyncio`; el grafo corre en el
worker (`python -m app.worker`) o, con WORKER_MODE=inline, en una tarea asyncio de
este mismo proceso.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

load_dotenv()

from app.hitl import router as hitl_router  # noqa: E402
from app.store import JobStatus, JobStore  # noqa: E402

log = logging.getLogger("api")
WORKER_MODE = os.getenv("WORKER_MODE", "external")  # external | inline


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    app.state.store = JobStore.from_url()
    async with AsyncExitStack() as stack:
        worker_task = None
        if WORKER_MODE == "inline":
            from app.graph import build_production_graph, redis_checkpointer
            from app.observability import flush, init_observability
            from app.worker import Worker

            init_observability()
            saver = await stack.enter_async_context(redis_checkpointer())
            worker = Worker(app.state.store, build_production_graph(saver))
            worker_task = asyncio.create_task(worker.run())
            stack.callback(flush)
        try:
            yield
        finally:
            if worker_task:
                worker.stop()
                await asyncio.wait_for(worker_task, timeout=30)
            await app.state.store.close()


app = FastAPI(
    title="Orquestador multi-agente · API",
    version="1.0.0",
    description="Supervisor + investigador + analista + validador (LangGraph) con Redis, "
                "observabilidad y aprobación humana.",
    lifespan=lifespan,
)
app.include_router(hitl_router)


class TaskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000, examples=[
        "¿Cuánto crecieron los usuarios activos de NovaPay entre Q1 y Q4 de 2024?"])
    publish: bool | None = Field(
        default=None,
        description="Publicar el informe (herramienta con efectos secundarios → requiere aprobación). "
                    "Si es null, el sistema lo detecta en la pregunta ('publica', 'envía'…).")
    require_approval: bool = Field(default=False, description="Forzar la pausa de aprobación humana.")


@app.post("/tasks", status_code=202)
async def create_task(body: TaskRequest, request: Request, response: Response) -> dict:
    job = await request.app.state.store.create(
        body.question, publish=body.publish, require_approval=body.require_approval)
    response.headers["Location"] = f"/tasks/{job['job_id']}"
    return {"job_id": job["job_id"], "status": job["status"], "status_url": f"/tasks/{job['job_id']}"}


@app.get("/tasks/{job_id}")
async def get_task(job_id: str, request: Request) -> dict:
    job = await request.app.state.store.get(job_id)
    if job is None:
        raise HTTPException(404, "job no encontrado")
    if job["status"] == JobStatus.AWAITING_APPROVAL:
        job["approve_url"] = f"/tasks/{job_id}/approve"
    job.pop("traceback", None)
    return job


@app.get("/tasks")
async def list_tasks(request: Request, limit: int = Query(20, ge=1, le=200)) -> list[dict]:
    jobs = await request.app.state.store.list(limit)
    return [{k: j.get(k) for k in ("job_id", "status", "question", "created_at", "duration_s", "tokens_total")}
            for j in jobs]


@app.get("/health")
async def health(request: Request) -> dict:
    try:
        await request.app.state.store.redis.ping()
    except Exception as exc:
        raise HTTPException(503, f"Redis no disponible: {exc}")
    return {"status": "ok", "worker_mode": WORKER_MODE}
