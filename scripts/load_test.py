"""Prueba de carga: 5 peticiones concurrentes contra la API + polling hasta terminar.

    python scripts/load_test.py                       # 5 jobs, aprueba automáticamente los HITL
    python scripts/load_test.py --n 5 --no-approve    # deja los HITL esperando (aprobar a mano)
    python scripts/load_test.py --url http://localhost:8000

Imprime por job: estado, latencia extremo a extremo, tokens y si pasó por HITL, y al final
p50/p95 del lado cliente. El costo por ejecución y la p95 que se entregan se leen del
dashboard de Phoenix/LangSmith (ver README); esto es solo una referencia cruzada.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx

QUESTIONS = [
    "¿Cuánto crecieron los usuarios activos mensuales de NovaPay entre Q1 y Q4 de 2024 (total y por trimestre)?",
    "¿Cómo evolucionaron los ingresos de NovaPay en 2024? Calcula el crecimiento total y trimestral.",
    "Compara el sentimiento de las reseñas de usuarios de Q3 frente a Q4 de NovaPay y explica la diferencia.",
    "Compara el churn y el CAC de NovaPay en Q4 2024 con el benchmark de neobancos europeos: calcula la diferencia porcentual.",
    "Analiza el crecimiento de ingresos de NovaPay en 2024 y publica el informe para el equipo directivo.",
]
TERMINAL = {"DONE", "FAILED"}


async def run_job(client: httpx.AsyncClient, i: int, question: str, approve: bool, poll_s: float) -> dict:
    t0 = time.perf_counter()
    r = await client.post("/tasks", json={"question": question})
    r.raise_for_status()
    job_id = r.json()["job_id"]
    enqueue_ms = (time.perf_counter() - t0) * 1000
    print(f"[{i}] encolado {job_id} en {enqueue_ms:.0f} ms")
    hitl = False
    while True:
        await asyncio.sleep(poll_s)
        job = (await client.get(f"/tasks/{job_id}")).json()
        if job["status"] == "AWAITING_APPROVAL" and not hitl:
            hitl = True
            print(f"[{i}] ⏸  HITL: {job['interrupt']['reasons']}")
            if not approve:
                return {"i": i, "job_id": job_id, "status": "AWAITING_APPROVAL",
                        "latency_s": time.perf_counter() - t0, "tokens": job["tokens_total"], "hitl": True}
            await client.post(f"/tasks/{job_id}/approve",
                              json={"approved": True, "reviewer": "load_test", "comment": "auto-aprobado"})
        if job["status"] in TERMINAL:
            latency = time.perf_counter() - t0
            print(f"[{i}] {job['status']} en {latency:.1f} s · tokens={job['tokens_total']}"
                  + (f" · error={job.get('error')}" if job["status"] == "FAILED" else ""))
            return {"i": i, "job_id": job_id, "status": job["status"], "latency_s": latency,
                    "tokens": job["tokens_total"], "hitl": hitl, "enqueue_ms": enqueue_ms}


def pct(values: list[float], p: float) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(p) - 1]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--no-approve", action="store_true")
    ap.add_argument("--poll", type=float, default=2.0)
    args = ap.parse_args()

    async with httpx.AsyncClient(base_url=args.url, timeout=30) as client:
        (await client.get("/health")).raise_for_status()
        t0 = time.perf_counter()
        results = await asyncio.gather(*[
            run_job(client, i, QUESTIONS[i % len(QUESTIONS)], not args.no_approve, args.poll)
            for i in range(args.n)
        ])
        wall = time.perf_counter() - t0

    print("\n job_id                            estado              latencia   tokens  HITL")
    for r in results:
        print(f" {r['job_id']}  {r['status']:18s} {r['latency_s']:7.1f} s {r['tokens']:8d}  {'sí' if r['hitl'] else 'no'}")
    lat = sorted(r["latency_s"] for r in results if r["status"] == "DONE")
    if lat:
        print(f"\n completados={len(lat)}/{len(results)} · p50={pct(lat, 50):.1f} s · p95={pct(lat, 95):.1f} s "
              f"· pared total={wall:.1f} s (latencia cliente, incluye la aprobación HITL)")
    enq = [r["enqueue_ms"] for r in results if "enqueue_ms" in r]
    if enq:
        print(f" POST /tasks: máx {max(enq):.0f} ms (la API responde sin esperar al grafo)")


if __name__ == "__main__":
    asyncio.run(main())
