"""Herramientas del Agente de Análisis/Cómputo.

Todas son deterministas: el LLM decide *qué* calcular, pero los números los
produce código, no el modelo (evita aritmética alucinada).
"""

from __future__ import annotations

import ast
import math
import operator
import re
import statistics
import unicodedata

from langchain_core.tools import tool

# --------------------------------------------------------------------------- #
# Calculadora segura (sin eval)
# --------------------------------------------------------------------------- #
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos, ast.FloorDiv: operator.floordiv,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponente demasiado grande")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Expresión no permitida: {ast.dump(node)[:60]}")


@tool
def calculator(expression: str) -> str:
    """Evalúa una expresión aritmética (+, -, *, /, **, %, paréntesis).
    Usa punto como separador decimal y sin separador de miles. Ej: '(7.4 - 4.8) / 4.8 * 100'."""
    try:
        result = _eval_node(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        return f"ERROR: {exc}"
    return f"{expression} = {round(result, 6)}"


@tool
def growth_rate(values: list[float], labels: list[str] | None = None) -> str:
    """Calcula el crecimiento porcentual entre periodos consecutivos, el total
    (primero→último) y la tasa compuesta por periodo (CAGR). `values` en orden
    cronológico; `labels` opcional con el nombre de cada periodo."""
    if len(values) < 2:
        return "ERROR: se necesitan al menos 2 valores."
    labels = labels if labels and len(labels) == len(values) else [f"P{i + 1}" for i in range(len(values))]
    lines = []
    for i in range(1, len(values)):
        prev, cur = values[i - 1], values[i]
        pct = (cur - prev) / prev * 100 if prev else float("nan")
        lines.append(f"{labels[i - 1]}→{labels[i]}: {prev} → {cur} ({pct:+.2f}%)")
    total = (values[-1] - values[0]) / values[0] * 100 if values[0] else float("nan")
    periods = len(values) - 1
    cagr = ((values[-1] / values[0]) ** (1 / periods) - 1) * 100 if values[0] > 0 else float("nan")
    lines.append(f"Total {labels[0]}→{labels[-1]}: {total:+.2f}%")
    lines.append(f"Crecimiento compuesto por periodo: {cagr:+.2f}%")
    return "\n".join(lines)


@tool
def descriptive_stats(values: list[float]) -> str:
    """Estadísticos descriptivos de una serie numérica: n, media, mediana, desviación
    típica, mínimo y máximo."""
    if not values:
        return "ERROR: lista vacía."
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return (
        f"n={len(values)} media={statistics.mean(values):.4f} mediana={statistics.median(values):.4f} "
        f"desv_tipica={sd:.4f} min={min(values)} max={max(values)}"
    )


# --------------------------------------------------------------------------- #
# Análisis de sentimiento por léxico (ES/EN)
# --------------------------------------------------------------------------- #
_POSITIVE = {
    "excelente", "rapida", "rapido", "facil", "encanta", "util", "utiles", "buena", "bueno",
    "increible", "recomiendo", "bien", "genial", "perfecta", "perfecto", "mejor", "satisfecho",
    "great", "excellent", "love", "easy", "fast", "good", "recommend",
}
_NEGATIVE = {
    "lento", "lenta", "frustrante", "caras", "caro", "cara", "terrible", "errores", "error",
    "malo", "mala", "pesima", "pesimo", "decepcionado", "decepcionante", "caida", "problema",
    "problemas", "peor", "horrible", "bad", "slow", "terrible", "awful", "expensive",
}
_NEGATORS = {"no", "nunca", "ni", "not", "never"}
_INTENSIFIERS = {"muy": 1.5, "super": 1.5, "really": 1.5, "very": 1.5}


def _norm_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"[a-z]+", text)


def _score(text: str) -> float:
    tokens = _norm_tokens(text)
    score = 0.0
    for i, tok in enumerate(tokens):
        polarity = 1.0 if tok in _POSITIVE else -1.0 if tok in _NEGATIVE else 0.0
        if not polarity:
            continue
        window = tokens[max(0, i - 3):i]
        if any(w in _NEGATORS for w in window):
            polarity *= -1
        if i > 0 and tokens[i - 1] in _INTENSIFIERS:
            polarity *= _INTENSIFIERS[tokens[i - 1]]
        score += polarity
    n = len(tokens) or 1
    return max(-1.0, min(1.0, score / math.sqrt(n))) if score else 0.0


@tool
def sentiment_analysis(texts: list[str]) -> str:
    """Analiza el sentimiento de una lista de textos (reseñas, opiniones). Devuelve
    una puntuación en [-1, 1] y etiqueta por texto, más el agregado (media y % de
    positivos/negativos/neutros). Pasa cada reseña como un elemento separado."""
    if not texts:
        return "ERROR: lista vacía."
    rows, scores = [], []
    for i, t in enumerate(texts, 1):
        s = _score(t)
        scores.append(s)
        label = "positivo" if s > 0.05 else "negativo" if s < -0.05 else "neutro"
        rows.append(f"{i}. {label:8s} ({s:+.2f}) {t[:80]}")
    n = len(scores)
    pos = sum(s > 0.05 for s in scores) / n * 100
    neg = sum(s < -0.05 for s in scores) / n * 100
    rows.append(
        f"AGREGADO: media={sum(scores) / n:+.3f} | positivos={pos:.0f}% "
        f"negativos={neg:.0f}% neutros={100 - pos - neg:.0f}% (n={n})"
    )
    return "\n".join(rows)


def analysis_tools() -> list:
    """Herramientas acotadas del analista (solo cómputo, sin acceso a fuentes)."""
    return [calculator, growth_rate, descriptive_stats, sentiment_analysis]
