from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import psycopg
from confluent_kafka import Consumer, Producer
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

import infersched_pb2


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int((len(ordered) - 1) * quantile), len(ordered) - 1)
    return ordered[index]


class SubmitRequest(BaseModel):
    prompt_tokens: int = Field(ge=1, le=8192)
    max_output_tokens: int = Field(default=16, ge=1, le=2048)
    model_id: str = "benchmark-model"
    priority: int = Field(default=1, ge=0, le=100)
    seed: int = Field(default=1, ge=0)
    prompt_hash: str | None = None


@dataclass
class GatewayState:
    capacity: int
    brokers: str
    database: str
    producer: Producer = field(init=False)
    consumer: Consumer = field(init=False)
    pending: dict[str, float] = field(default_factory=dict)
    results: dict[str, dict] = field(default_factory=dict)
    latencies_ms: list[float] = field(default_factory=list)
    submitted: int = 0
    completed: int = 0
    rejected: int = 0
    stop: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    thread: threading.Thread | None = None

    def __post_init__(self) -> None:
        self.producer = Producer(
            {"bootstrap.servers": self.brokers, "enable.idempotence": True}
        )
        self.consumer = Consumer(
            {
                "bootstrap.servers": self.brokers,
                "group.id": f"infersched-gateway-{uuid.uuid4()}",
                # Unique gateway group plus request-id filtering avoids the
                # initial-assignment race where a fast result beats `latest`.
                "auto.offset.reset": "earliest",
                "enable.auto.commit": True,
            }
        )

    def start(self) -> None:
        self.consumer.subscribe(["inference.results"])
        self.thread = threading.Thread(target=self._consume, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stop = True
        if self.thread:
            self.thread.join(timeout=3)
        self.consumer.close()
        self.producer.flush(3)

    def submit(self, body: SubmitRequest) -> tuple[str, str]:
        with self.lock:
            if len(self.pending) >= self.capacity:
                self.rejected += 1
                raise HTTPException(status_code=429, detail="gateway queue full")
            request_id = str(uuid.uuid4())
            trace_id = uuid.uuid4().hex
            span_id = uuid.uuid4().hex[:16]
            self.pending[request_id] = time.perf_counter()
            self.submitted += 1
        request = infersched_pb2.InferenceRequest(
            request_id=request_id,
            model_id=body.model_id,
            model_revision=f"{body.model_id}-v1",
            tokenizer_revision="tokenizer-v1",
            prompt_tokens=body.prompt_tokens,
            prompt_hash=body.prompt_hash or f"prompt-{request_id}",
            priority=body.priority,
            sampling=infersched_pb2.SamplingParams(
                max_output_tokens=body.max_output_tokens,
                temperature=0.0,
                top_p=1.0,
                seed=body.seed,
            ),
            trace_id=trace_id,
            span_id=span_id,
        )
        try:
            self.producer.produce(
                "inference.requests", key=request_id, value=request.SerializeToString()
            )
            self.producer.poll(0)
        except BufferError as error:
            with self.lock:
                self.pending.pop(request_id, None)
                self.rejected += 1
            raise HTTPException(status_code=429, detail="Kafka producer queue full") from error
        return request_id, trace_id

    def _consume(self) -> None:
        while not self.stop:
            message = self.consumer.poll(0.1)
            if message is None or message.error():
                continue
            try:
                result = json.loads(message.value())
            except (json.JSONDecodeError, TypeError):
                continue
            request_id = result.get("request_id")
            if not request_id:
                continue
            with self.lock:
                started = self.pending.pop(request_id, None)
                self.results[request_id] = result
                if started is not None:
                    self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
                    self.completed += 1

    def metrics(self) -> dict:
        with self.lock:
            samples = list(self.latencies_ms)
            return {
                "submitted": self.submitted,
                "completed": self.completed,
                "rejected": self.rejected,
                "queue_depth": len(self.pending),
                "e2e_p50_ms": percentile(samples, 0.50),
                "e2e_p90_ms": percentile(samples, 0.90),
                "e2e_p99_ms": percentile(samples, 0.99),
            }


STATE = GatewayState(
    capacity=int(os.getenv("INFERSCHED_GATEWAY_CAPACITY", "1024")),
    brokers=os.getenv("INFERSCHED_BROKERS", "localhost:9092"),
    database=os.getenv(
        "INFERSCHED_DATABASE",
        "host=localhost port=55432 dbname=infersched user=infersched password=infersched",
    ),
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE.start()
    yield
    STATE.close()


app = FastAPI(title="InferSched Control Plane", version="0.7.0", lifespan=lifespan)


@app.post("/v1/inference", status_code=202)
def submit(body: SubmitRequest) -> dict:
    request_id, trace_id = STATE.submit(body)
    return {"request_id": request_id, "trace_id": trace_id, "state": "QUEUED"}


@app.get("/v1/inference/{request_id}")
def result(request_id: str) -> dict:
    with STATE.lock:
        cached = STATE.results.get(request_id)
        pending = request_id in STATE.pending
    if cached:
        return {"state": "COMPLETED", **cached}
    if pending:
        return {"request_id": request_id, "state": "PENDING"}
    with psycopg.connect(STATE.database) as connection:
        row = connection.execute(
            "SELECT state, terminal_result FROM inference_requests WHERE request_id = %s",
            (request_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="request not found")
    payload = json.loads(row[1]) if row[1] else {}
    return {"request_id": request_id, "state": row[0], **payload}


@app.get("/metrics")
def metrics() -> dict:
    return STATE.metrics()


@app.get("/healthz")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def dashboard() -> Response:
    return Response(
        """<!doctype html><title>InferSched</title>
<style>body{font:16px system-ui;max-width:850px;margin:40px auto;background:#10141c;color:#e8eef8}
pre{padding:24px;background:#18202d;border-radius:12px;color:#75e6a4}</style>
<h1>InferSched control plane</h1><p>Live gateway latency and overload metrics.</p><pre id=m>loading</pre>
<script>setInterval(async()=>m.textContent=JSON.stringify(await(await fetch('/metrics')).json(),null,2),1000)</script>""",
        media_type="text/html",
    )
