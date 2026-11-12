"""Fase de síntesis final: integra las aportaciones de los especialistas."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from agents._specialist import format_contributions, original_question
from llm import message_text
from state import OrchestratorState

SYNTH_PROMPT = """Eres el SINTETIZADOR. Redacta la respuesta final para el usuario usando
EXCLUSIVAMENTE las aportaciones de los especialistas.

- Responde a todas las partes de la pregunta, en español, de forma clara y estructurada.
- Cita las fuentes del investigador ([kb-00X]) y atribuye las cifras calculadas al analista.
- No añadas datos ni cálculos nuevos. Si algo falta, dilo en una sección "Limitaciones".
- Si hay feedback del validador, corrígelo en esta versión."""


def make_synthesizer_node(llm):
    def synthesizer_node(state: OrchestratorState) -> dict:
        context = (
            f"Pregunta del usuario:\n{original_question(state)}\n\n"
            f"Aportaciones:\n{format_contributions(state)}\n\n"
            f"Feedback del validador sobre el borrador anterior: {state.get('validation_feedback') or 'ninguno'}"
        )
        draft = message_text(llm.invoke([("system", SYNTH_PROMPT), ("human", context)]))
        return {"draft_answer": draft, "messages": [AIMessage(content=draft, name="synthesizer")]}

    return synthesizer_node
