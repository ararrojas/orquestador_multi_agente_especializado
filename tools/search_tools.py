"""Herramientas del Agente de Investigación.

- `search_knowledge_base`: búsqueda vectorial simulada (TF-IDF + similitud coseno)
  sobre la base de conocimiento local `data/knowledge_base.json`, que hace de
  "Vector DB" de las pre-entregas anteriores. Sin dependencias externas.
- `web_search`: búsqueda real con Tavily, solo si hay `TAVILY_API_KEY`.
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"

_STOPWORDS = {
    "a", "al", "con", "de", "del", "el", "en", "es", "la", "las", "lo", "los", "por",
    "para", "que", "se", "su", "un", "una", "y", "o", "the", "of", "and", "to", "in",
    "como", "mas", "muy", "sobre", "entre", "tras", "fue",
}


def _tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in _STOPWORDS and len(t) > 1]


class _TfidfIndex:
    """Índice TF-IDF mínimo que simula una Vector DB (embeddings dispersos)."""

    def __init__(self, docs: list[dict]):
        self.docs = docs
        tokenized = [_tokenize(d["title"] + " " + d["content"]) for d in docs]
        n = len(docs)
        df = Counter(t for toks in tokenized for t in set(toks))
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self.vectors = [self._vectorize(toks) for toks in tokenized]

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        vec = {t: c * self.idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, k: int = 4) -> list[tuple[float, dict]]:
        q = self._vectorize(_tokenize(query))
        scored = [
            (sum(q.get(t, 0.0) * w for t, w in vec.items()), doc)
            for vec, doc in zip(self.vectors, self.docs)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(s, d) for s, d in scored[:k] if s > 0]


@lru_cache(maxsize=1)
def _index() -> _TfidfIndex:
    return _TfidfIndex(json.loads(KB_PATH.read_text(encoding="utf-8")))


@tool
def search_knowledge_base(query: str, k: int = 4) -> str:
    """Busca en la base de conocimiento interna (informes trimestrales, reseñas,
    benchmarks, postmortems). Devuelve los fragmentos más relevantes con su ID y
    fuente para poder citarlos. Usa consultas cortas y específicas; puedes llamarla
    varias veces con consultas distintas."""
    hits = _index().search(query, k=max(1, min(k, 8)))
    if not hits:
        return "Sin resultados en la base de conocimiento para esa consulta."
    return "\n\n".join(
        f"[{d['id']}] {d['title']} (fuente: {d['source']}, score={s:.2f})\n{d['content']}"
        for s, d in hits
    )


def _build_web_search():
    if not os.getenv("TAVILY_API_KEY"):
        return None
    from langchain_tavily import TavilySearch

    return TavilySearch(max_results=3, name="web_search")


def research_tools() -> list:
    """Herramientas acotadas del investigador (solo lectura de fuentes)."""
    tools = [search_knowledge_base]
    web = _build_web_search()
    if web is not None:
        tools.append(web)
    return tools
