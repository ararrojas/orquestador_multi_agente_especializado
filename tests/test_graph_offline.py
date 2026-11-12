"""Recorre el grafo completo con un LLM guionizado:
supervisor→researcher→supervisor→analyst→supervisor→synthesizer→validator(rechaza)
→supervisor→analyst→supervisor→synthesizer→validator(aprueba)→END."""

import warnings

from langchain_core.messages import AIMessage

from graph import build_graph, initial_state
from tests.fake_llm import ScriptedChatModel

warnings.filterwarnings("ignore")


def _tool_call(name, args, i):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{i}"}])


def sup(next_agent, instruction="haz tu parte"):
    return {"reasoning": f"toca {next_agent}", "next_agent": next_agent, "instruction": instruction}


def verdict(ok, agent="none", feedback=""):
    return {"answers_all_parts": ok, "figures_traceable": ok, "sources_cited": True,
            "no_contradictions": True, "proper_methods": True, "feedback": feedback, "agent_to_refine": agent}


def make_llm():
    chat = [
        _tool_call("search_knowledge_base", {"query": "usuarios activos Q1 Q4 2024"}, 1),
        "## Hallazgos\n- MAU Q1 1.200.000 [kb-001], Q4 1.720.000 [kb-004]",
        _tool_call("growth_rate", {"values": [1200000, 1720000], "labels": ["Q1", "Q4"]}, 2),
        "## Resultados\n- Crecimiento MAU: +43.33% (growth_rate)",
        "Borrador v1 sin sentimiento",
        _tool_call("sentiment_analysis", {"texts": ["excelente", "terrible caída"]}, 3),
        "## Resultados\n- Sentimiento medio calculado (sentiment_analysis)",
        "Borrador v2: MAU +43.33% [kb-001][kb-004]; sentimiento calculado por el analista.",
    ]
    structured = [
        sup("researcher"), sup("analyst"), sup("synthesizer"),
        verdict(False, "analyst", "Falta el análisis de sentimiento."),
        sup("analyst", "calcula el sentimiento"), sup("synthesizer"),
        verdict(True),
    ]
    return ScriptedChatModel(chat, structured)


def test_full_delegation_flow_with_refinement():
    llm = make_llm()
    app = build_graph(llm)
    visited = []
    final = {}
    for update in app.stream(initial_state("pregunta de prueba"), {"recursion_limit": 60}, stream_mode="updates"):
        for node, delta in update.items():
            visited.append(node)
            final.update(delta or {})

    assert visited == [
        "supervisor", "researcher", "supervisor", "analyst", "supervisor", "synthesizer", "validator",
        "supervisor", "analyst", "supervisor", "synthesizer", "validator",
    ]
    assert final["task_completed"] is True
    assert final["final_answer"].startswith("Borrador v2")
    assert llm.exhausted



def test_supervisor_cannot_skip_analyst():
    """Si el LLM quiere sintetizar sin análisis, la guarda de suficiencia lo impide."""
    chat = [
        _tool_call("search_knowledge_base", {"query": "ingresos"}, 1), "datos [kb-001]",
        "## Resultados ok",
        "Borrador final",
    ]
    structured = [sup("researcher"), sup("synthesizer"), sup("synthesizer"), verdict(True)]
    app = build_graph(ScriptedChatModel(chat, structured))
    out = app.invoke(initial_state("q"), {"recursion_limit": 60})
    agents = [c["agent"] for c in out["contributions"]]
    assert agents == ["researcher", "analyst"]
    assert out["task_completed"]


def test_step_budget_forces_termination(monkeypatch):
    """Un supervisor que siempre pide al investigador termina igualmente."""
    import agents.supervisor as s

    monkeypatch.setattr(s, "MAX_STEPS", 3)
    monkeypatch.setattr(s, "MAX_CALLS_PER_AGENT", 99)
    chat = ["datos 1", "datos 2", "datos 3", "Borrador"]
    structured = [sup("researcher"), sup("researcher"), sup("analyst"), verdict(True)]
    out = build_graph(ScriptedChatModel(chat, structured)).invoke(initial_state("q"), {"recursion_limit": 60})
    assert out["step_count"] == 4 and out["next_agent"] == "synthesizer"
    assert out["task_completed"]


def test_validator_rejects_sentiment_without_tool():
    """Aunque el LLM validador apruebe, si se pidió sentimiento y el analista no usó
    sentiment_analysis, la comprobación determinista rechaza y devuelve el trabajo al analista."""
    chat = [
        _tool_call("search_knowledge_base", {"query": "reseñas Q3 Q4"}, 1), "reseñas y estrellas [kb-008]",
        _tool_call("calculator", {"expression": "4.4 - 3.7"}, 2), "diferencia de estrellas 0.7",
        "Borrador v1 con estrellas",
        _tool_call("sentiment_analysis", {"texts": ["excelente", "terrible caída"]}, 3), "sentimiento calculado",
        "Borrador v2 con sentimiento",
    ]
    structured = [
        sup("researcher"), sup("analyst"), sup("synthesizer"),
        verdict(True),                      # el LLM aprobaría (error que debe corregirse)
        sup("analyst", "ejecuta sentiment_analysis"), sup("synthesizer"),
        verdict(True),
    ]
    out = build_graph(ScriptedChatModel(chat, structured)).invoke(
        initial_state("Compara el sentimiento de las reseñas de Q3 y Q4"), {"recursion_limit": 60}
    )
    assert out["revision_count"] == 1
    assert out["final_answer"] == "Borrador v2 con sentimiento"
    rejected = [m.content for m in out["messages"] if getattr(m, "name", "") == "validator"]
    assert "RECHAZADO (para analyst)" in rejected[0] and "sentiment_analysis" in rejected[0]
