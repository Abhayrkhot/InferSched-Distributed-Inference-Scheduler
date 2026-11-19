"""InferSched gateway: admission, durable submission, result correlation, metrics.

Request path: validate -> bounded admission -> produce to `inference.requests`
and wait for the broker acknowledgement -> 202. A consumer thread correlates
`inference.results` back to this gateway's pending requests and wakes
long-polling clients. Pending requests that never resolve expire after a
timeout so a lost result cannot permanently consume admission capacity.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

import psycopg
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field

import infersched_pb2

# Prometheus histogram buckets for end-to-end latency, in seconds.
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
TERMINAL_STATES = {"COMPLETED", "FAILED", "FAILED_TERMINAL", "CANCELLED", "TIMED_OUT"}


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
    # Shared prompt prefix (e.g. a system prompt): requests with the same
    # prefix_id share their first `prefix_tokens` tokens of KV cache.
    prefix_id: str | None = Field(default=None, max_length=128)
    prefix_tokens: int = Field(default=0, ge=0, le=8192)


@dataclass
class Pending:
    started: float
    started_unix: float
    trace_id: str
    span_id: str
    acked_after_s: float | None = None  # Broker acknowledgement latency.
    waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = field(
        default_factory=list
    )


class Histogram:
    def __init__(self, buckets: tuple[float, ...]) -> None:
        self.buckets = buckets
        self.counts = [0] * len(buckets)
        self.total = 0
        self.sum = 0.0

    def observe(self, value: float) -> None:
        self.total += 1
        self.sum += value
        for index, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[index] += 1


def _default_producer(brokers: str) -> Any:
    from confluent_kafka import Producer

    return Producer(
        {
            "bootstrap.servers": brokers,
            "enable.idempotence": True,
            "linger.ms": 1,
        }
    )


def _default_consumer(brokers: str) -> Any:
    from confluent_kafka import Consumer

    return Consumer(
        {
            "bootstrap.servers": brokers,
            # Each gateway reads every result partition and filters to its own
            # request ids. Start offsets are pinned to the high watermarks at
            # assignment (see GatewayState._on_assign), so a restart neither
            # replays the topic's history nor races `latest` resolution.
            "group.id": f"infersched-gateway-{uuid.uuid4()}",
            "enable.auto.commit": False,
        }
    )


class GatewayState:
    def __init__(
        self,
        capacity: int,
        brokers: str,
        database: str,
        *,
        producer: Any | None = None,
        consumer_factory: Callable[[], Any] | None = None,
        produce_timeout_s: float = 5.0,
        result_timeout_s: float = 120.0,
        result_cache_size: int = 10_000,
        latency_window: int = 10_000,
        traces: bool = True,
    ) -> None:
        self.capacity = capacity
        self.brokers = brokers
        self.database = database
        self.produce_timeout_s = produce_timeout_s
        self.result_timeout_s = result_timeout_s
        self.result_cache_size = result_cache_size
        self.traces = traces
        self.producer = producer if producer is not None else _default_producer(brokers)
        self._consumer_factory = consumer_factory or (lambda: _default_consumer(brokers))
        self.consumer: Any | None = None
        self.lock = threading.Lock()
        self.pending: dict[str, Pending] = {}
        self.results: OrderedDict[str, dict] = OrderedDict()
        # Requests this gateway gave up waiting for; a late result is still kept.
        self.expired: OrderedDict[str, None] = OrderedDict()
        self.latencies_ms: deque[float] = deque(maxlen=latency_window)
        # Event-loop scheduling lag samples: a blocked loop delays every request.
        self.loop_lag_ms: deque[float] = deque(maxlen=2000)
        self.histogram = Histogram(LATENCY_BUCKETS)
        self.submitted = 0
        self.completed = 0
        self.failed = 0
        self.rejected = 0
        self.timed_out = 0
        self.produce_failures = 0
        self.stop = False
        self.threads: list[threading.Thread] = []
        # Set once the results consumer is positioned; /readyz gates on it so
        # no request is accepted before its result could be observed.
        self.ready = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def _on_assign(self, consumer: Any, partitions: list) -> None:
        for partition in partitions:
            _, high = consumer.get_watermark_offsets(partition, timeout=5, cached=False)
            partition.offset = high
        consumer.assign(partitions)
        self.ready.set()

    def start(self) -> None:
        self.consumer = self._consumer_factory()
        self.consumer.subscribe(["inference.results"], on_assign=self._on_assign)
        for target in (self._consume, self._poll_producer):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self.threads.append(thread)

    def close(self) -> None:
        self.stop = True
        for thread in self.threads:
            thread.join(timeout=3)
        if self.consumer is not None:
            self.consumer.close()
        self.producer.flush(3)

    # -- submission --------------------------------------------------------

    def _reserve(self, body: SubmitRequest) -> tuple[str, Pending]:
        if not self.ready.is_set():
            raise HTTPException(status_code=503, detail="gateway starting")
        with self.lock:
            if len(self.pending) >= self.capacity:
                self.rejected += 1
                raise HTTPException(status_code=429, detail="gateway queue full")
            request_id = str(uuid.uuid4())
            entry = Pending(
                started=time.perf_counter(),
                started_unix=time.time(),
                trace_id=uuid.uuid4().hex,
                span_id=uuid.uuid4().hex[:16],
            )
            self.pending[request_id] = entry
            self.submitted += 1
        return request_id, entry

    def _release(self, request_id: str, *, produce_failed: bool) -> None:
        with self.lock:
            self.pending.pop(request_id, None)
            self.submitted -= 1
            self.rejected += 1
            if produce_failed:
                self.produce_failures += 1

    def _build(self, body: SubmitRequest, request_id: str, entry: Pending) -> bytes:
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
            trace_id=entry.trace_id,
            span_id=entry.span_id,
            prefix_id=body.prefix_id or "",
            prefix_tokens=min(body.prefix_tokens, body.prompt_tokens) if body.prefix_id else 0,
        )
        return request.SerializeToString()

    async def submit(self, body: SubmitRequest) -> tuple[str, str]:
        """Returns only after the broker acknowledged the request (or raises)."""
        request_id, entry = self._reserve(body)
        loop = asyncio.get_running_loop()
        delivered: asyncio.Future[str | None] = loop.create_future()

        def on_delivery(error: Any, _message: Any) -> None:
            def resolve() -> None:
                if not delivered.done():
                    delivered.set_result(None if error is None else str(error))

            loop.call_soon_threadsafe(resolve)

        try:
            self.producer.produce(
                "inference.requests",
                key=request_id,
                value=self._build(body, request_id, entry),
                on_delivery=on_delivery,
            )
        except BufferError as error:
            self._release(request_id, produce_failed=True)
            raise HTTPException(status_code=429, detail="Kafka producer queue full") from error
        try:
            failure = await asyncio.wait_for(delivered, timeout=self.produce_timeout_s)
        except asyncio.TimeoutError:
            failure = "delivery timed out"
        if failure is not None:
            # Not acknowledged: the client must not believe the request is queued.
            self._release(request_id, produce_failed=True)
            raise HTTPException(status_code=503, detail=f"request not persisted: {failure}")
        entry.acked_after_s = time.perf_counter() - entry.started
        return request_id, entry.trace_id

    # -- result correlation ------------------------------------------------

    def _poll_producer(self) -> None:
        while not self.stop:
            self.producer.poll(0.05)

    def handle_result(self, payload: bytes | str, created_ms: int | None = None) -> bool:
        """Correlates one `inference.results` message; True if it was ours."""
        try:
            result = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return False
        request_id = result.get("request_id") if isinstance(result, dict) else None
        if not request_id:
            return False
        status = result.get("status", "COMPLETED")
        result["state"] = "COMPLETED" if status == "COMPLETED" else "FAILED"
        with self.lock:
            entry = self.pending.pop(request_id, None)
            if entry is None:
                if request_id not in self.expired:
                    return False  # Another gateway's request.
                self.expired.pop(request_id, None)
            self._remember(request_id, result)
            if entry is None:
                return True
            latency_s = time.perf_counter() - entry.started
            self.latencies_ms.append(latency_s * 1000.0)
            self.histogram.observe(latency_s)
            if result["state"] == "COMPLETED":
                self.completed += 1
            else:
                self.failed += 1
            waiters = list(entry.waiters)
        for loop, event in waiters:
            loop.call_soon_threadsafe(event.set)
        transit_ms = None if created_ms is None else time.time() * 1000 - created_ms
        self._emit_span(request_id, entry, result["state"], transit_ms)
        return True

    def _remember(self, request_id: str, result: dict) -> None:
        self.results[request_id] = result
        self.results.move_to_end(request_id)
        while len(self.results) > self.result_cache_size:
            self.results.popitem(last=False)

    def expire_pending(self, now: float | None = None) -> int:
        """Drops pending requests older than the result timeout."""
        now = time.perf_counter() if now is None else now
        expired: list[Pending] = []
        with self.lock:
            for request_id, entry in list(self.pending.items()):
                if now - entry.started >= self.result_timeout_s:
                    del self.pending[request_id]
                    self.expired[request_id] = None
                    expired.append(entry)
                    self.timed_out += 1
            while len(self.expired) > self.result_cache_size:
                self.expired.popitem(last=False)
        for entry in expired:
            for loop, event in entry.waiters:
                loop.call_soon_threadsafe(event.set)
        return len(expired)

    def _consume(self) -> None:
        last_expiry = time.perf_counter()
        while not self.stop:
            message = self.consumer.poll(0.1)
            if message is not None and not message.error():
                kind, created_ms = message.timestamp()
                self.handle_result(message.value(), created_ms if kind else None)
            if time.perf_counter() - last_expiry >= 1.0:
                self.expire_pending()
                last_expiry = time.perf_counter()

    def _emit_span(
        self, request_id: str, entry: Pending, state: str, transit_ms: float | None = None
    ) -> None:
        if not self.traces:
            return
        span = {
            "trace_id": entry.trace_id,
            "span_id": entry.span_id,
            "parent_span_id": "",
            "name": "gateway.request",
            "service": "gateway",
            "start_unix_us": int(entry.started_unix * 1_000_000),
            "duration_us": int((time.perf_counter() - entry.started) * 1_000_000),
            "attributes": {
                "request_id": request_id,
                "state": state,
                # Result outbox publish (CreateTime) -> received by this gateway.
                **({"result_transit_ms": f"{transit_ms:.1f}"} if transit_ms is not None else {}),
                **(
                    {"submit_ack_ms": f"{entry.acked_after_s * 1000:.1f}"}
                    if entry.acked_after_s is not None
                    else {}
                ),
            },
        }
        try:
            self.producer.produce("inference.traces", key=entry.trace_id, value=json.dumps(span))
        except BufferError:
            pass  # Tracing is best effort.

    async def wait_for(self, request_id: str, timeout_s: float) -> None:
        loop = asyncio.get_running_loop()
        event = asyncio.Event()
        with self.lock:
            entry = self.pending.get(request_id)
            if entry is None:
                return
            entry.waiters.append((loop, event))
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            pass
        finally:
            with self.lock:
                entry = self.pending.get(request_id)
                if entry is not None and (loop, event) in entry.waiters:
                    entry.waiters.remove((loop, event))

    # -- observability -----------------------------------------------------

    async def monitor_loop_lag(self, interval_s: float = 0.01) -> None:
        loop = asyncio.get_running_loop()
        while not self.stop:
            started = loop.time()
            await asyncio.sleep(interval_s)
            self.loop_lag_ms.append(max(0.0, (loop.time() - started - interval_s) * 1000))

    def stats(self) -> dict:
        with self.lock:
            samples = list(self.latencies_ms)
            return {
                "submitted": self.submitted,
                "completed": self.completed,
                "failed": self.failed,
                "rejected": self.rejected,
                "timed_out": self.timed_out,
                "produce_failures": self.produce_failures,
                "queue_depth": len(self.pending),
                "e2e_p50_ms": percentile(samples, 0.50),
                "e2e_p90_ms": percentile(samples, 0.90),
                "e2e_p99_ms": percentile(samples, 0.99),
                "event_loop_lag_p99_ms": percentile(list(self.loop_lag_ms), 0.99),
                "event_loop_lag_max_ms": max(self.loop_lag_ms, default=0.0),
            }

    def prometheus(self) -> str:
        with self.lock:
            counters = {
                "submitted": self.submitted,
                "completed": self.completed,
                "failed": self.failed,
                "rejected": self.rejected,
                "timed_out": self.timed_out,
                "produce_failures": self.produce_failures,
            }
            depth = len(self.pending)
            buckets = list(self.histogram.counts)
            total = self.histogram.total
            latency_sum = self.histogram.sum
        lines = []
        for name, value in counters.items():
            metric = f"infersched_gateway_requests_{name}_total"
            lines += [f"# TYPE {metric} counter", f"{metric} {value}"]
        lag = list(self.loop_lag_ms)
        lines += [
            "# TYPE infersched_gateway_event_loop_lag_p99_seconds gauge",
            f"infersched_gateway_event_loop_lag_p99_seconds {percentile(lag, 0.99) / 1000}",
            "# TYPE infersched_gateway_queue_depth gauge",
            f"infersched_gateway_queue_depth {depth}",
            "# TYPE infersched_gateway_queue_capacity gauge",
            f"infersched_gateway_queue_capacity {self.capacity}",
            "# TYPE infersched_gateway_e2e_latency_seconds histogram",
        ]
        for bound, count in zip(self.histogram.buckets, buckets):
            lines.append(f'infersched_gateway_e2e_latency_seconds_bucket{{le="{bound}"}} {count}')
        lines += [
            f'infersched_gateway_e2e_latency_seconds_bucket{{le="+Inf"}} {total}',
            f"infersched_gateway_e2e_latency_seconds_sum {latency_sum}",
            f"infersched_gateway_e2e_latency_seconds_count {total}",
        ]
        return "\n".join(lines) + "\n"


def _state_from_env() -> GatewayState:
    return GatewayState(
        capacity=int(os.getenv("INFERSCHED_GATEWAY_CAPACITY", "1024")),
        brokers=os.getenv("INFERSCHED_BROKERS", "localhost:9092"),
        database=os.getenv(
            "INFERSCHED_DATABASE",
            "host=localhost port=55432 dbname=infersched user=infersched password=infersched",
        ),
        produce_timeout_s=float(os.getenv("INFERSCHED_PRODUCE_TIMEOUT_S", "5")),
        result_timeout_s=float(os.getenv("INFERSCHED_RESULT_TIMEOUT_S", "120")),
        result_cache_size=int(os.getenv("INFERSCHED_RESULT_CACHE", "10000")),
        traces=os.getenv("INFERSCHED_TRACES", "true") == "true",
    )


STATE = _state_from_env()


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE.start()
    monitor = asyncio.create_task(STATE.monitor_loop_lag())
    yield
    monitor.cancel()
    STATE.close()


app = FastAPI(title="InferSched Control Plane", version="0.8.0", lifespan=lifespan)


@app.post("/v1/inference", status_code=202)
async def submit(body: SubmitRequest) -> dict:
    request_id, trace_id = await STATE.submit(body)
    return {"request_id": request_id, "trace_id": trace_id, "state": "QUEUED"}


def _lookup_database(request_id: str) -> tuple[str, str | None] | None:
    with psycopg.connect(STATE.database, connect_timeout=2) as connection:
        return connection.execute(
            "SELECT state, terminal_result FROM inference_requests WHERE request_id = %s",
            (request_id,),
        ).fetchone()


@app.get("/v1/inference/{request_id}")
async def result(
    request_id: str,
    wait_ms: int = Query(default=0, ge=0, le=30_000),
) -> dict:
    """Result lookup; `wait_ms` long-polls instead of client-side busy polling."""
    if wait_ms > 0:
        await STATE.wait_for(request_id, wait_ms / 1000.0)
    with STATE.lock:
        cached = STATE.results.get(request_id)
        pending = request_id in STATE.pending
    if cached:
        return cached
    if pending:
        return {"request_id": request_id, "state": "PENDING"}
    row = await asyncio.to_thread(_lookup_database, request_id)
    if row is None:
        raise HTTPException(status_code=404, detail="request not found")
    payload = json.loads(row[1]) if row[1] else {}
    state = "FAILED" if row[0] == "FAILED_TERMINAL" else row[0]
    return {"request_id": request_id, **payload, "state": state}


@app.get("/metrics")
def metrics() -> Response:
    return Response(STATE.prometheus(), media_type="text/plain; version=0.0.4")


@app.get("/v1/stats")
def stats() -> dict:
    return STATE.stats()


@app.get("/healthz")
def health() -> dict:
    """Liveness: the process is serving HTTP."""
    return {"status": "ok"}


@app.get("/readyz")
def ready() -> Response:
    """Readiness: the results consumer is positioned and requests can resolve."""
    if STATE.ready.is_set():
        return Response('{"status":"ready"}', media_type="application/json")
    return Response('{"status":"starting"}', status_code=503, media_type="application/json")


@app.get("/")
def dashboard() -> Response:
    return Response(
        """<!doctype html><title>InferSched</title>
<style>body{font:16px system-ui;max-width:850px;margin:40px auto;background:#10141c;color:#e8eef8}
pre{padding:24px;background:#18202d;border-radius:12px;color:#75e6a4}</style>
<h1>InferSched control plane</h1><p>Live gateway latency and overload metrics.</p><pre id=m>loading</pre>
<script>setInterval(async()=>m.textContent=JSON.stringify(await(await fetch('/v1/stats')).json(),null,2),1000)</script>""",
        media_type="text/html",
    )
