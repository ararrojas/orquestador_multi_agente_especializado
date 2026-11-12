"""Agente de Búsqueda/Investigación: recupera hechos de fuentes, no calcula."""

from __future__ import annotations

from agents._specialist import build_specialist_node, original_question
from state import OrchestratorState
from tools.search_tools import research_tools

RESEARCH_PROMPT = """Eres el AGENTE INVESTIGADOR de un equipo multi-agente.
Tu único trabajo es encontrar datos en las fuentes disponibles con tus herramientas.

Reglas:
- Usa SIEMPRE las herramientas; no respondas de memoria.
- Para datos internos (NovaPay: informes, reseñas, benchmark) usa search_knowledge_base;
  usa web_search solo si existe y se necesita contexto externo.
- Haz varias búsquedas cortas si la instrucción cubre varios temas.
- Devuelve los datos tal cual aparecen (cifras exactas, textos literales de reseñas)
  y cita el ID de la fuente entre corchetes, p. ej. [kb-004].
- NO hagas cálculos ni análisis de sentimiento: eso lo hace el analista.
- Si algo no aparece en las fuentes, dilo explícitamente ("no encontrado").

Formato de salida:
## Hallazgos
- <dato> [fuente]
## Datos brutos para el analista
<series numéricas en orden cronológico y/o lista de textos literales>
## Lagunas
<lo que no se encontró>"""


def make_research_node(llm):
    def build_context(state: OrchestratorState) -> str:
        # Contexto mínimo: pregunta original + instrucción del supervisor (+ feedback si es revisión).
        ctx = (
            f"Pregunta del usuario:\n{original_question(state)}\n\n"
            f"Instrucción del supervisor:\n{state.get('current_instruction', '')}"
        )
        if state.get("validation_feedback"):
            ctx += f"\n\nFeedback del validador a corregir:\n{state['validation_feedback']}"
        return ctx

    return build_specialist_node("researcher", llm, research_tools(), RESEARCH_PROMPT, build_context)
