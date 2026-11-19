"""Worker: consume la cola de Redis y ejecuta el grafo en segundo plano.

Se puede ejecutar como proceso separado (recomendado, escala horizontalmente):

    python -m app.worker

o dentro del proceso de la API (`WORKER_MODE=inline`, cómodo en desarrollo).

Cada mensaje de la cola es `{"job_id", "kind": "start" | "resume", "decision"?}`.
Transiciones de estado en Redis:

    PENDING → RUNNING → DONE
                      → AWAITING_APPROVAL → (POST /approve) → PENDING → RUNNING → DONE
                      → FAILED   (cualquier excepción o timeout: el cliente nunca se queda
                                  esperando en un polling infinito)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
import traceback

from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv()

from app.graph import build_production_graph, production_input, redis_checkpointer  # noqa: E402
from app.observability import flush, init_observability, job_context, release_usage, traced, usage_handler  # noqa: E402
from app.store import JobStatus, JobStore  # noqa: E402

log = logging.getLogger("worker")

WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "5"))
JOB_TIMEOUT_S = float(os.getenv("JOB_TIMEOUT_S", "900"))
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "80"))


class Worker:
    def __init__(self, store: JobStore, graph, *, concurrency: int = WORKER_CONCURRENCY,
                 timeout_s: float = JOB_TIMEOUT_S):
        self.store = store
        self.graph = graph
        self.timeout_s = timeout_s
        self._sem = asyncio.Semaphore(concurrency)
        self._tasks: set[asyncio.Task] = set()
        self._stopping = asyncio.Event()

    # ------------------------------------------------------------------ loop
    async def run(self) -> None:
        log.info("worker listo (concurrencia=%s, timeout=%ss)", self._sem._value, self.timeout_s)
        while not self._stopping.is_set():
            await self._sem.acquire()
            try:
                message = await self.store.dequeue(timeout=1)
            except Exception:
                self._sem.release()
                log.exception("error leyendo la cola; reintento en 1 s")
                await asyncio.sleep(1)
                continue
            if message is None:
                self._sem.release()
                continue
            task = asyncio.create_task(self.handle(message))
            self._tasks.add(task)
            task.add_done_callback(self._done)
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    def _done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        self._sem.release()

    def stop(self) -> None:
        self._stopping.set()

    # ------------------------------------------------------------------ job
    async def handle(self, message: dict) -> None:
        """Nunca deja escapar una excepción: cualquier fallo termina en FAILED."""
        job_id = message.get("job_id", "?")
        job = await self.store.get(job_id)
        if job is None:
            log.warning("mensaje para job inexistente %s", job_id)
            return
        started = time.time()
        await self.store.update(job_id, status=JobStatus.RUNNING,
                                started_at=job.get("started_at", started), attempt_kind=message["kind"])
        try:
            await asyncio.wait_for(self.execute(job_id=job_id, job=job, message=message), self.timeout_s)
        except asyncio.TimeoutError:
            await self._fail(job_id, f"Timeout: la ejecución superó {self.timeout_s:.0f} s")
        except Exception as exc:  # error del LLM, de una herramienta, de Redis…
            log.exception("job %s falló", job_id)
            await self._fail(job_id, f"{type(exc).__name__}: {exc}", traceback.format_exc(limit=5))
        finally:
            await self.store.add_tokens(job_id, release_usage(job_id))

    async def _fail(self, job_id: str, error: str, tb: str | None = None) -> None:
        try:
            await self.store.update(job_id, status=JobStatus.FAILED, error=error[:2000],
                                    traceback=tb, finished_at=time.time())
        except Exception:
            log.exception("no se pudo marcar FAILED el job %s", job_id)

    @traced("job.execute")
    async def execute(self, *, job_id: str, job: dict, message: dict) -> dict:
        options = job.get("options", {})
        config = {
            "configurable": {"thread_id": job_id},
            "callbacks": [usage_handler(job_id)],
            "recursion_limit": RECURSION_LIMIT,
            "run_name": f"job:{message['kind']}",
            "metadata": {"job_id": job_id, "kind": message["kind"]},
            "tags": ["pre7-api"],
        }
        if message["kind"] == "start":
            graph_input = production_input(job["question"], publish=options.get("publish"),
                                           require_approval=options.get("require_approval", False))
        elif message["kind"] == "resume":
            graph_input = Command(resume=message["decision"])
        else:
            raise ValueError(f"tipo de mensaje desconocido: {message['kind']}")

        with job_context(job_id, kind=message["kind"]):
            # Los nodos del grafo son síncronos: LangGraph los ejecuta en un thread pool
            # bajo ainvoke, así el event loop del worker atiende otros jobs en paralelo.
            await self.graph.ainvoke(graph_input, config)

        snapshot = await self.graph.aget_state({"configurable": {"thread_id": job_id}})
        values = snapshot.values
        common = {
            "critical_reasons": values.get("critical_reasons") or [],
            "agent_calls": values.get("agent_calls") or {},
            "step_count": values.get("step_count", 0),
            "revision_count": values.get("revision_count", 0),
        }
        if snapshot.interrupts:
            payload = snapshot.interrupts[0].value
            await self.store.update(job_id, status=JobStatus.AWAITING_APPROVAL, interrupt=payload, **common)
            return {"status": JobStatus.AWAITING_APPROVAL, "reasons": payload.get("reasons")}

        finished = time.time()
        await self.store.update(
            job_id,
            status=JobStatus.DONE,
            result=values.get("final_answer", ""),
            approval=values.get("approval"),
            published_to=values.get("published_to"),
            finished_at=finished,
            duration_s=round(finished - float(job.get("started_at") or job["created_at"]), 3),
            **common,
        )
        return {"status": JobStatus.DONE, "published_to": values.get("published_to")}


async def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_observability()
    store = JobStore.from_url()
    async with redis_checkpointer() as saver:
        worker = Worker(store, build_production_graph(saver))
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, worker.stop)
        try:
            await worker.run()
        finally:
            flush()
            await store.close()


if __name__ == "__main__":
    asyncio.run(main())
