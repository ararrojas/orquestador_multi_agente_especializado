"""Supervisor: router inteligente y controlador de flujo.

Decide con un LLM (salida estructurada) quién interviene ahora, pero sus decisiones
pasan por guardas deterministas que garantizan la terminación:
  1. presupuesto global de pasos (MAX_STEPS),
  2. tope de llamadas por especialista (MAX_CALLS_PER_AGENT),
  3. criterio de suficiencia: no se sintetiza sin aporte de ambos dominios.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from agents._specialist import format_contributions, original_question
from agents.requirements import describe_missing, missing_analyses
from config import MAX_CALLS_PER_AGENT, MAX_STEPS
from state import OrchestratorState


class SupervisorDecision(BaseModel):
    reasoning: str = Field(description="Por qué eliges este siguiente paso (1-3 frases).")
    next_agent: Literal["researcher", "analyst", "synthesizer"] = Field(
        description="Nodo que debe intervenir ahora. 'synthesizer' = la información es suficiente."
    )
    instruction: str = Field(
        description="Instrucción concreta y autocontenida para ese agente (qué buscar o calcular)."
    )


SUPERVISOR_PROMPT = """Eres el SUPERVISOR de un equipo multi-agente de análisis e investigación.
Dada la conversación actual, decide quién debe intervenir ahora o si es momento de finalizar.

Equipo:
- researcher: busca datos en fuentes (base de conocimiento interna y, si existe, web). No calcula.
- analyst: hace cálculos, crecimientos, estadísticos y análisis de sentimiento SOBRE los datos ya recopilados. No busca.
- synthesizer: redacta la respuesta final. Elígelo solo si se cumple la rúbrica de suficiencia.

Rúbrica de suficiencia (TODAS deben cumplirse para elegir synthesizer):
1. Cada parte de la pregunta tiene datos del researcher con fuente citada.
2. Toda cifra derivada (porcentajes, medias, sentimiento) la calculó el analyst con herramientas.
3. Cada análisis que pide la pregunta se hizo con SU herramienta: si se pide sentimiento, el analyst
   debe ejecutar sentiment_analysis sobre los TEXTOS de las reseñas. Una valoración en estrellas u
   otra métrica parecida NO sustituye al análisis de sentimiento. "Análisis pendientes" debe ser "ninguno".
4. No hay contradicciones sin resolver entre researcher y analyst.
5. Si hay feedback del validador, el agente responsable ya lo atendió.

Reglas de flujo:
- El orden natural es researcher -> analyst -> synthesizer.
- Al enviar al analyst, nombra en la instrucción cada herramienta que debe usar y pásale los datos
  necesarios (para sentimiento: pide que use la lista literal de reseñas aportada por el researcher).
- Si faltan los textos de reseñas, pide al researcher que los recupere literalmente antes.
- Si el analyst dice que faltan datos, vuelve al researcher con una instrucción precisa de qué falta.
- Si hay conflicto entre agentes, los DATOS FUENTE (researcher) mandan sobre la interpretación; pide al analyst recalcular con ellos.
- Las instrucciones deben ser específicas y autocontenidas: el especialista NO ve la conversación.
- Presupuesto: te quedan {remaining} pasos. No repitas trabajo ya hecho."""


def make_supervisor_node(llm):
    router = llm.with_structured_output(SupervisorDecision, method="json_schema")

    def supervisor_node(state: OrchestratorState) -> dict:
        step = state.get("step_count", 0) + 1
        calls = state.get("agent_calls", {}) or {}
        contributions = state.get("contributions", []) or []
        has = {c["agent"] for c in contributions}

        # Guarda 1: presupuesto agotado -> sintetizar con lo que haya.
        if step > MAX_STEPS:
            decision = SupervisorDecision(
                reasoning=f"Presupuesto de {MAX_STEPS} pasos agotado: se sintetiza con la información disponible.",
                next_agent="synthesizer",
                instruction="Sintetiza con lo disponible y declara explícitamente las lagunas.",
            )
        else:
            context = (
                f"Pregunta del usuario:\n{original_question(state)}\n\n"
                f"Aportaciones hasta ahora:\n{format_contributions(state)}\n\n"
                f"Llamadas por agente: {calls or 'ninguna'}\n"
                f"Análisis pendientes: {describe_missing(missing_analyses(original_question(state), state))}\n"
                f"Feedback del validador: {state.get('validation_feedback') or 'ninguno'}"
            )
            decision = router.invoke([
                ("system", SUPERVISOR_PROMPT.format(remaining=MAX_STEPS - step + 1)),
                ("human", context),
            ])

            # Guarda 2: criterio de suficiencia mínimo (ambos dominios deben haber aportado).
            if decision.next_agent == "synthesizer":
                for needed in ("researcher", "analyst"):
                    if needed not in has and calls.get(needed, 0) < MAX_CALLS_PER_AGENT:
                        decision = SupervisorDecision(
                            reasoning=f"Override: falta la aportación de '{needed}' antes de sintetizar.",
                            next_agent=needed,
                            instruction=decision.instruction if needed == "researcher" else
                            "Procesa los datos del investigador: calcula las métricas y sentimiento que pide la pregunta.",
                        )
                        break

            # Guarda 3: tope por agente (evita bucles de correcciones eternas).
            if decision.next_agent != "synthesizer" and calls.get(decision.next_agent, 0) >= MAX_CALLS_PER_AGENT:
                decision = SupervisorDecision(
                    reasoning=f"Override: '{decision.next_agent}' alcanzó {MAX_CALLS_PER_AGENT} llamadas; se sintetiza.",
                    next_agent="synthesizer",
                    instruction="Sintetiza con lo disponible y declara las lagunas.",
                )

        note = f"[supervisor · paso {step}] → {decision.next_agent}: {decision.reasoning}"
        return {
            "next_agent": decision.next_agent,
            "current_instruction": decision.instruction,
            "supervisor_reasoning": decision.reasoning,
            "step_count": step,
            "messages": [AIMessage(content=note, name="supervisor")],
        }

    return supervisor_node
