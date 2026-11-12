"""Modelo de chat guionizado para probar el grafo sin red ni API key.

Las herramientas SÍ se ejecutan de verdad; solo las decisiones del LLM están guionizadas.
"""

from __future__ import annotations

from collections import deque

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import PrivateAttr


class ScriptedChatModel(BaseChatModel):
    _chat: deque = PrivateAttr()
    _structured: deque = PrivateAttr()

    def __init__(self, chat_responses, structured_responses, **kw):
        super().__init__(**kw)
        self._chat = deque(chat_responses)
        self._structured = deque(structured_responses)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self._chat.popleft()
        if isinstance(msg, str):
            msg = AIMessage(content=msg)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def _next(_input):
            data = self._structured.popleft()
            return schema(**data) if isinstance(data, dict) else data

        return RunnableLambda(_next)

    @property
    def exhausted(self) -> bool:
        return not self._chat and not self._structured
