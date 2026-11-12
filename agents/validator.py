"""Nodo de Validación: aplica una rúbrica al borrador antes de dar END.

Si falla y queda presupuesto de revisiones, devuelve el control al Supervisor con
feedback dirigido al agente responsable. Si no, cierra igualmente (parada garantizada).
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from agents._specialist import format_contributions, original_question
from agents.requirements import describe_missing, missing_analyses
from config import MAX_REVISIONS
from state import OrchestratorState


class ValidationResult(BaseModel):
    answers_all_parts: bool = Field(description="¿Responde a todas las partes de la pregunta?")
    figures_traceable: bool = Field(description="¿Cada cifra aparece en las aportaciones del researcher o del analyst?")
    sources_cited: bool = Field(description="¿Cita las fuentes [kb-...] de los datos?")
    no_contradictions: bool = Field(description="¿Está libre de contradicciones con los datos fuente?")
    proper_methods: bool = Field(
        description="¿Cada análisis pedido se hizo con su herramienta del analyst? (sentimiento = sentiment_analysis "
        "sobre textos de reseñas; las estrellas no cuentan)."
    )
    feedback: str = Field(description="Qué hay que corregir exactamente (vacío si todo está bien).")
    agent_to_refine: Literal["researcher", "analyst", "synthesizer", "none"] = Field(
        description="Qué agente debe corregir el problema principal ('none' si aprueba)."
    )

    @property
    def passed(self) -> bool:
        return all([self.answers_all_parts, self.figures_traceable, self.sources_cited,
                    self.no_contradictions, self.proper_methods])


VALIDATOR_PROMPT = """Eres el VALIDADOR (control de calidad). Evalúa el borrador con esta rúbrica:
1. answers_all_parts: cubre todas las partes de la pregunta.
2. figures_traceable: cada número del borrador aparece en las aportaciones (no inventado).
3. sources_cited: los datos fuente llevan su cita [kb-...].
4. no_contradictions: no contradice los datos del investigador ni los cálculos del analista.
5. proper_methods: cada análisis pedido se hizo con su herramienta (mira "herramientas:" en cada
   aportación). Si se pide sentimiento y el analyst no ejecutó sentiment_analysis, es un fallo
   aunque el borrador hable de valoraciones en estrellas -> agent_to_refine = analyst.
Sé estricto pero justo: las limitaciones declaradas explícitamente no son un fallo."""


def make_validator_node(llm):
    grader = llm.with_structured_output(ValidationResult, method="json_schema")

    def validator_node(state: OrchestratorState) -> dict:
        result: ValidationResult = grader.invoke([
            ("system", VALIDATOR_PROMPT),
            ("human",
             f"Pregunta:\n{original_question(state)}\n\n"
             f"Aportaciones:\n{format_contributions(state)}\n\n"
             f"Borrador a validar:\n{state.get('draft_answer', '')}"),
        ])
        # Comprobación determinista: el LLM no puede aprobar si falta un análisis exigido.
        missing = missing_analyses(original_question(state), state)
        if missing:
            result.proper_methods = False
            result.agent_to_refine = "analyst"
            result.feedback = (
                f"Falta ejecutar con la herramienta del analista: {describe_missing(missing)}. "
                + (result.feedback or "")
            ).strip()
        revisions = state.get("revision_count", 0)

        if result.passed or revisions >= MAX_REVISIONS:
            final = state.get("draft_answer", "")
            if not result.passed:
                final += f"\n\n> ⚠️ Validación no superada tras {MAX_REVISIONS} revisiones: {result.feedback}"
            verdict = "APROBADO" if result.passed else "CERRADO POR LÍMITE DE REVISIONES"
            return {
                "task_completed": True,
                "final_answer": final,
                "validation_feedback": "" if result.passed else result.feedback,
                "messages": [AIMessage(content=f"[validator] {verdict}", name="validator")],
            }

        feedback = f"(para {result.agent_to_refine}) {result.feedback}"
        return {
            "task_completed": False,
            "revision_count": revisions + 1,
            "validation_feedback": feedback,
            "messages": [AIMessage(content=f"[validator] RECHAZADO {feedback}", name="validator")],
        }

    return validator_node
