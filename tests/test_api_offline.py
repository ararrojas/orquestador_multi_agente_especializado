"""API + worker + HITL sin red: Redis simulado (fakeredis), checkpointer en memoria y
LLM guionizado. Los tests de integración con Redis real están en scripts/ (ver README)."""

import asyncio
import warnings

import fakeredis
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app import hitl
from app.graph import build_production_graph
from app.main import app
from app.store import JobStatus, JobStore
from app.worker import Worker
from tests.fake_llm import ScriptedChatModel
from tests.test_graph_offline import _tool_call, sup, verdict

warnings.filterwarnings("ignore")


def happy_llm():
    chat = [
        _tool_call("search_knowledge_base", {"query": "usuarios Q1 Q4"}, 1), "MAU Q1 1.2M [kb-001]",
        _tool_call("growth_rate", {"values": [1200000, 1720000]}, 2), "MAU +43.33% (growth_rate)",
        "Respuesta final: MAU +43.33% [kb-001]",
    ]
    return ScriptedChatModel(chat, [sup("researcher"), sup("analyst"), sup("synthesizer"), verdict(True)])


class BrokenLLM(ScriptedChatModel):
    def with_structured_output(self, schema, **kwargs):
        raise_ = lambda _: (_ for _ in ()).throw(RuntimeError("503 modelo saturado"))  # noqa: E731
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(raise_)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(hitl, "REPORTS_DIR", tmp_path / "reports")
    store = JobStore(fakeredis.FakeAsyncRedis(decode_responses=True))
    app.state.store = store
    client = TestClient(app)  # sin `with`: no ejecuta el lifespan (no hay Redis real)
    return store, client, tmp_path


def run_queue(store, llm, n_messages=1):
    async def _go():
        worker = Worker(store, build_production_graph(InMemorySaver(), llm), concurrency=5, timeout_s=30)
        for _ in range(n_messages):
            await worker.handle(await store.dequeue(timeout=1))
    asyncio.run(_go())


def test_post_returns_job_id_immediately(env):
    store, client, _ = env
    r = client.post("/tasks", json={"question": "¿Cuánto crecieron los usuarios?"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert client.get(f"/tasks/{job_id}").json()["status"] == JobStatus.PENDING
    assert client.get("/tasks/nope").status_code == 404


def test_non_critical_task_runs_to_done(env):
    store, client, _ = env
    job_id = client.post("/tasks", json={"question": "¿Cuánto crecieron los usuarios?"}).json()["job_id"]
    run_queue(store, happy_llm())
    job = client.get(f"/tasks/{job_id}").json()
    assert job["status"] == JobStatus.DONE
    assert job["result"].startswith("Respuesta final")
    assert job["critical_reasons"] == []


def test_exception_in_background_marks_failed(env):
    store, client, _ = env
    job_id = client.post("/tasks", json={"question": "q de prueba"}).json()["job_id"]
    run_queue(store, BrokenLLM([], []))
    job = client.get(f"/tasks/{job_id}").json()
    assert job["status"] == JobStatus.FAILED
    assert "503 modelo saturado" in job["error"]


def test_publish_requires_human_approval_then_publishes(env):
    store, client, tmp = env
    llm = happy_llm()
    saver = InMemorySaver()
    job_id = client.post("/tasks", json={"question": "Calcula el crecimiento y publica el informe"}).json()["job_id"]

    async def first_run():
        worker = Worker(store, build_production_graph(saver, llm), timeout_s=30)
        await worker.handle(await store.dequeue(timeout=1))
    asyncio.run(first_run())

    job = client.get(f"/tasks/{job_id}").json()
    assert job["status"] == JobStatus.AWAITING_APPROVAL
    assert job["interrupt"]["action"] == "publish_report"
    assert any(r.startswith("side_effect") for r in job["interrupt"]["reasons"])
    assert not (tmp / "reports").exists()  # la herramienta NO se ejecutó sin aprobación

    r = client.post(f"/tasks/{job_id}/approve", json={"approved": True, "reviewer": "ara"})
    assert r.status_code == 202
    assert client.post(f"/tasks/{job_id}/approve", json={"approved": True}).status_code == 409

    async def resume():
        worker = Worker(store, build_production_graph(saver, llm), timeout_s=30)
        await worker.handle(await store.dequeue(timeout=1))
    asyncio.run(resume())

    job = client.get(f"/tasks/{job_id}").json()
    assert job["status"] == JobStatus.DONE
    assert job["approval"]["approved"] is True and job["approval"]["reviewer"] == "ara"
    assert (tmp / "reports" / f"{job_id}.md").read_text().count("MAU +43.33%") == 1


def test_rejection_withholds_answer(env):
    store, client, tmp = env
    llm, saver = happy_llm(), InMemorySaver()
    job_id = client.post("/tasks", json={"question": "¿Cuánto crecieron?", "require_approval": True}).json()["job_id"]

    async def go():
        w = Worker(store, build_production_graph(saver, llm), timeout_s=30)
        await w.handle(await store.dequeue(timeout=1))
        assert (await store.get(job_id))["status"] == JobStatus.AWAITING_APPROVAL
        await store.claim_for_resume(job_id)
        await store.enqueue({"job_id": job_id, "kind": "resume",
                             "decision": {"approved": False, "comment": "no"}})
        await w.handle(await store.dequeue(timeout=1))
    asyncio.run(go())

    job = client.get(f"/tasks/{job_id}").json()
    assert job["status"] == JobStatus.DONE
    assert job["result"].startswith("⛔ Respuesta retenida")
    assert not (tmp / "reports").exists()


def test_detects_side_effect_requests():
    assert hitl.wants_publication("Analiza y envía el informe al equipo")
    assert hitl.wants_publication("Publícalo")
    assert not hitl.wants_publication("¿Cuánto crecieron los ingresos?")
