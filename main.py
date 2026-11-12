"""CLI: ejecuta el orquestador y muestra el flujo de delegación paso a paso.

    python main.py                      # pregunta de demo
    python main.py "tu pregunta"        # pregunta propia
    python main.py --diagram            # regenera docs/graph.mmd (y PNG si hay red)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*create_react_agent.*")

DEMO_QUESTION = (
    "Investiga la evolución de NovaPay durante 2024: ¿cuánto crecieron los usuarios activos "
    "mensuales y los ingresos entre Q1 y Q4 (porcentaje total y por trimestre)? Además, compara "
    "el sentimiento de las reseñas de usuarios de Q3 frente a Q4 y explica qué pudo causar la diferencia."
)


def export_diagram(app, out_dir: Path = Path("docs")) -> None:
    out_dir.mkdir(exist_ok=True)
    (out_dir / "graph.mmd").write_text(app.get_graph().draw_mermaid(), encoding="utf-8")
    print(f"Mermaid guardado en {out_dir / 'graph.mmd'}")
    try:
        (out_dir / "graph.png").write_bytes(app.get_graph().draw_mermaid_png())
        print(f"PNG guardado en {out_dir / 'graph.png'}")
    except Exception as exc:  # draw_mermaid_png usa el servicio mermaid.ink
        print(f"(PNG no generado: {exc})")


def run(question: str, app=None) -> dict:
    from graph import build_graph, initial_state

    import agents._specialist as specialist

    specialist.progress_hook = lambda text: print(text, flush=True)
    app = app or build_graph()
    final_state: dict = {}
    print(f"\n👤 Usuario: {question}\n")
    print("(cada llamada al LLM puede tardar 10-25 s; la ejecución completa, varios minutos)\n")
    stream = app.stream(initial_state(question), {"recursion_limit": 60}, stream_mode=["updates", "values"])
    for mode, chunk in stream:
        if mode == "values":
            final_state = chunk  # estado completo tras cada super-paso
            continue
        for node, delta in chunk.items():
            if not delta:
                continue
            if node == "supervisor":
                print(f"🧭 supervisor (paso {delta['step_count']}) → {delta['next_agent']}")
                print(f"   motivo: {delta['supervisor_reasoning']}")
                print(f"   instrucción: {delta['current_instruction']}\n")
            elif node in ("researcher", "analyst"):
                c = delta["contributions"][0]
                icon = "🔎" if node == "researcher" else "📊"
                print(f"{icon} {node} usó {c['tools_used'] or 'ninguna herramienta'}")
                print("   " + c["output"][:600].replace("\n", "\n   ") + ("…" if len(c["output"]) > 600 else "") + "\n")
            elif node == "synthesizer":
                print("📝 synthesizer: borrador generado\n")
            elif node == "validator":
                print(f"✅ {delta['messages'][-1].content}\n")
    print("=" * 80)
    print(f"Pasos: {final_state.get('step_count')} | revisiones: {final_state.get('revision_count')} "
          f"| llamadas por agente: {final_state.get('agent_calls')}")
    print("=" * 80)
    print(final_state.get("final_answer", "(sin respuesta final)"))
    return final_state


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--diagram":
        from graph import build_graph

        export_diagram(build_graph())
    else:
        try:
            run(" ".join(args) or DEMO_QUESTION)
        except Exception as exc:  # errores del proveedor (503 saturado, 504 timeout, 429 cuota…)
            print(f"\n❌ Error del proveedor LLM: {type(exc).__name__}: {str(exc)[:300]}")
            print("   Si es 503/504 el modelo está saturado: reintenta o cambia LLM_MODEL en .env.")
            sys.exit(1)
