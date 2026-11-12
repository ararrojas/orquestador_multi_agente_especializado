"""Agente de Análisis/Cómputo: procesa los datos del investigador con herramientas
deterministas (cálculo, crecimiento, estadísticos, sentimiento)."""

from __future__ import annotations

from agents._specialist import build_specialist_node, format_contributions
from state import OrchestratorState
from tools.analysis_tools import analysis_tools

ANALYST_PROMPT = """Eres el AGENTE ANALISTA de un equipo multi-agente.
Recibes datos ya recopilados por el investigador y los procesas.

Reglas:
- Todo número que reportes debe salir de una herramienta (calculator, growth_rate,
  descriptive_stats, sentiment_analysis). No hagas aritmética mental.
- Trabaja SOLO con los datos que te pasan. No inventes datos. Si faltan, dilo.
- Para sentimiento, pasa cada reseña como un elemento separado de la lista.
- Sé conciso: resultados, método e interpretación breve.

Formato de salida:
## Resultados
- <métrica>: <valor> (herramienta usada)
## Interpretación
<2-4 frases>
## Limitaciones
<tamaño de muestra, datos faltantes, etc.>"""


def make_analyst_node(llm):
    def build_context(state: OrchestratorState) -> str:
        # El analista NO ve la conversación: solo su instrucción y los hallazgos del investigador.
        ctx = (
            f"Instrucción del supervisor:\n{state.get('current_instruction', '')}\n\n"
            f"Datos aportados por el investigador:\n{format_contributions(state, 'researcher')}"
        )
        if state.get("validation_feedback"):
            ctx += f"\n\nFeedback del validador a corregir:\n{state['validation_feedback']}"
        return ctx

    return build_specialist_node("analyst", llm, analysis_tools(), ANALYST_PROMPT, build_context)
