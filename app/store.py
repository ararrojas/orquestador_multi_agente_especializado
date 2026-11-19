"""Estado de los trabajos en Redis (independiente de los checkpoints de LangGraph).

    job:{id}      HASH  status, question, timestamps, result, error, interrupción HITL, tokens…
    jobs:index    ZSET  ids por fecha de creación (para listar)
    jobs:queue    LIST  cola de mensajes para los workers ({"job_id", "kind": start|resume, ...})

Todo usa `redis.asyncio`: la API nunca bloquea el event loop esperando a Redis.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from enum import StrEnum
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import WatchError

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_KEY = "jobs:queue"
INDEX_KEY = "jobs:index"
JOB_TTL_S = int(os.getenv("JOB_TTL_S", str(7 * 24 * 3600)))


class JobStatus(StrEnum):
    PENDING = "PENDING"                    # encolado, ningún worker lo tomó aún
    RUNNING = "RUNNING"                    # un worker está ejecutando el grafo
    AWAITING_APPROVAL = "AWAITING_APPROVAL"  # pausado en el nodo HITL (interrupt)
    DONE = "DONE"
    FAILED = "FAILED"


TERMINAL = {JobStatus.DONE, JobStatus.FAILED}
# Campos que se guardan serializados en JSON dentro del hash.
_JSON_FIELDS = {"interrupt", "approval", "critical_reasons", "agent_calls", "result"}


def _now() -> float:
    return round(time.time(), 3)


class JobStore:
    def __init__(self, redis: Redis):
        self.redis = redis

    @classmethod
    def from_url(cls, url: str = REDIS_URL) -> "JobStore":
        return cls(Redis.from_url(url, decode_responses=True))

    async def close(self) -> None:
        await self.redis.aclose()

    @staticmethod
    def key(job_id: str) -> str:
        return f"job:{job_id}"

    async def create(self, question: str, **options: Any) -> dict:
        job_id = uuid.uuid4().hex
        now = _now()
        job = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "question": question,
            "options": json.dumps(options),
            "created_at": now,
            "updated_at": now,
            "tokens_total": 0,
        }
        async with self.redis.pipeline(transaction=True) as p:
            p.hset(self.key(job_id), mapping=job)
            p.expire(self.key(job_id), JOB_TTL_S)
            p.zadd(INDEX_KEY, {job_id: now})
            p.rpush(QUEUE_KEY, json.dumps({"job_id": job_id, "kind": "start"}))
            await p.execute()
        return await self.get(job_id)

    async def get(self, job_id: str) -> dict | None:
        raw = await self.redis.hgetall(self.key(job_id))
        if not raw:
            return None
        job: dict[str, Any] = dict(raw)
        for f in _JSON_FIELDS | {"options"}:
            if f in job:
                job[f] = json.loads(job[f])
        for f in ("created_at", "updated_at", "started_at", "finished_at", "duration_s"):
            if f in job:
                job[f] = float(job[f])
        job["tokens_total"] = int(job.get("tokens_total", 0))
        return job

    async def update(self, job_id: str, **fields: Any) -> None:
        mapping = {}
        for k, v in fields.items():
            if v is None:
                continue
            mapping[k] = json.dumps(v, ensure_ascii=False) if k in _JSON_FIELDS else v
        mapping["updated_at"] = _now()
        await self.redis.hset(self.key(job_id), mapping=mapping)

    async def claim_for_resume(self, job_id: str) -> bool | None:
        """AWAITING_APPROVAL → PENDING de forma atómica (WATCH/MULTI).
        None = no existe · False = no estaba esperando aprobación · True = reclamado."""
        key = self.key(job_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    await pipe.watch(key)
                    status = await pipe.hget(key, "status")
                    if status is None:
                        return None
                    if status != JobStatus.AWAITING_APPROVAL:
                        return False
                    pipe.multi()
                    pipe.hset(key, mapping={"status": JobStatus.PENDING, "updated_at": _now()})
                    await pipe.execute()
                    return True
                except WatchError:
                    continue

    async def add_tokens(self, job_id: str, tokens: int) -> None:
        if tokens:
            await self.redis.hincrby(self.key(job_id), "tokens_total", tokens)

    async def enqueue(self, message: dict) -> None:
        await self.redis.rpush(QUEUE_KEY, json.dumps(message))

    async def dequeue(self, timeout: float = 1.0) -> dict | None:
        item = await self.redis.blpop([QUEUE_KEY], timeout=timeout)
        return json.loads(item[1]) if item else None

    async def list(self, limit: int = 20) -> list[dict]:
        ids = await self.redis.zrevrange(INDEX_KEY, 0, limit - 1)
        jobs = [await self.get(i) for i in ids]
        return [j for j in jobs if j]
