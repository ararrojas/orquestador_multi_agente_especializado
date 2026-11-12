"""Qué herramientas del analista exige cada tipo de pregunta.

Se usa en dos sitios:
- el Supervisor lo ve en su contexto ("análisis pendientes"),
- el Validador lo comprueba de forma determinista antes de aprobar.
Así una cifra "parecida" (p. ej. estrellas en tiendas) no sustituye al análisis pedido.
"""

from __future__ import annotations

import re
import unicodedata

from state import OrchestratorState

# (etiqueta, regex sobre la pregunta normalizada, herramientas que lo satisfacen)
RULES: list[tuple[str, str, set[str]]] = [
    ("análisis de sentimiento de reseñas", r"sentim|opinion|resena|review", {"sentiment_analysis"}),
    ("cálculo de crecimientos/porcentajes", r"crec|porcentaj|variacion|%|growth", {"growth_rate", "calculator"}),
]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def missing_analyses(question: str, state: OrchestratorState) -> list[tuple[str, set[str]]]:
    """Análisis que la pregunta exige y que el analista aún no ejecutó con su herramienta."""
    used = {t for c in state.get("contributions", []) or [] if c["agent"] == "analyst" for t in c["tools_used"]}
    q = _normalize(question)
    return [(label, tools) for label, pattern, tools in RULES if re.search(pattern, q) and not (tools & used)]


def describe_missing(missing: list[tuple[str, set[str]]]) -> str:
    if not missing:
        return "ninguno"
    return "; ".join(f"{label} (herramienta: {' o '.join(sorted(tools))})" for label, tools in missing)
