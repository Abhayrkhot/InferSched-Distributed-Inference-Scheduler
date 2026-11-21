import asyncio
import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import infersched_pb2
from infersched_api.app import GatewayState, SubmitRequest, app


class FakeProducer:
    """Records produced messages; delivers (or fails) on the next poll()."""

    def __init__(self, error=None):
        self.error = error
        self.messages = []
        self._callbacks = []

    def produce(self, topic, key=None, value=None, on_delivery=None):
        self.messages.append((topic, key, value))
        if on_delivery is not None:
            self._callbacks.append(on_delivery)

    def poll(self, _timeout=0):
        callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            callback(self.error, None)
        return len(callbacks)

    def flush(self, _timeout=0):
        return 0


class SilentProducer(FakeProducer):
    """Never acknowledges anything (broker unreachable)."""

    def poll(self, _timeout=0):
        return 0


def make_state(producer=None, ready=True, **kwargs) -> GatewayState:
    state = GatewayState(
        capacity=kwargs.pop("capacity", 8),
        brokers="unused",
        database="",
        producer=producer or FakeProducer(),
        traces=kwargs.pop("traces", False),
        **kwargs,
    )
    if ready:
        state.ready.set()
    return state


class FakePartition:
    def __init__(self, partition):
        self.partition = partition
        self.offset = -1


class FakeConsumer:
    def __init__(self):
        self.assigned = None

    def get_watermark_offsets(self, partition, timeout=None, cached=False):
        return 0, 100 + partition.partition

    def assign(self, partitions):
        self.assigned = [(p.partition, p.offset) for p in partitions]


async def submit_with_delivery(state: GatewayState, body: SubmitRequest):
    task = asyncio.create_task(state.submit(body))
    await asyncio.sleep(0)
    state.producer.poll()
    return await task


def result_payload(request_id: str, status: str = "COMPLETED", **extra) -> bytes:
    return json.dumps({"request_id": request_id, "status": status, **extra}).encode()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_dashboard(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/").status_code == 200
    assert client.get("/v1/stats").json()["queue_depth"] == 0


def test_not_ready_until_results_consumer_is_positioned() -> None:
    async def scenario():
        state = make_state(ready=False)
        with pytest.raises(HTTPException) as error:
            await state.submit(SubmitRequest(prompt_tokens=16))
        assert error.value.status_code == 503
        consumer = FakeConsumer()
        state._on_assign(consumer, [FakePartition(0), FakePartition(1)])
        # Positioned at the high watermarks: no replay of older results.
        assert consumer.assigned == [(0, 100), (1, 101)]
        assert state.ready.is_set()
        await submit_with_delivery(state, SubmitRequest(prompt_tokens=16))

    asyncio.run(scenario())


def test_request_validation(client: TestClient) -> None:
    assert client.post("/v1/inference", json={"prompt_tokens": 0}).status_code == 422
    too_long_prefix = {"prompt_tokens": 16, "prefix_id": "x" * 129}
    assert client.post("/v1/inference", json=too_long_prefix).status_code == 422


def test_prometheus_exposition(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "# TYPE infersched_gateway_requests_submitted_total counter" in body
    assert 'infersched_gateway_e2e_latency_seconds_bucket{le="+Inf"}' in body
    assert "infersched_gateway_queue_depth" in body


def test_submit_returns_only_after_broker_ack_and_carries_prefix() -> None:
    async def scenario():
        state = make_state()
        body = SubmitRequest(prompt_tokens=300, prefix_id="system-a", prefix_tokens=256)
        request_id, trace_id = await submit_with_delivery(state, body)
        topic, key, value = state.producer.messages[0]
        assert topic == "inference.requests" and key == request_id
        request = infersched_pb2.InferenceRequest.FromString(value)
        assert request.trace_id == trace_id
        assert (request.prefix_id, request.prefix_tokens) == ("system-a", 256)
        assert state.stats()["queue_depth"] == 1

    asyncio.run(scenario())


def test_unacknowledged_produce_is_rejected_and_releases_capacity() -> None:
    async def scenario():
        failing = make_state(producer=FakeProducer(error="broker down"), capacity=1)
        with pytest.raises(HTTPException) as error:
            await submit_with_delivery(failing, SubmitRequest(prompt_tokens=16))
        assert error.value.status_code == 503
        assert failing.stats()["queue_depth"] == 0
        assert failing.stats()["produce_failures"] == 1

        silent = make_state(producer=SilentProducer(), produce_timeout_s=0.05)
        with pytest.raises(HTTPException) as error:
            await silent.submit(SubmitRequest(prompt_tokens=16))
        assert error.value.status_code == 503
        assert silent.stats()["queue_depth"] == 0

    asyncio.run(scenario())


def test_bounded_gateway_rejects_with_429() -> None:
    async def scenario():
        state = make_state(capacity=1)
        await submit_with_delivery(state, SubmitRequest(prompt_tokens=16))
        with pytest.raises(HTTPException) as error:
            await state.submit(SubmitRequest(prompt_tokens=16))
        assert error.value.status_code == 429

    asyncio.run(scenario())


def test_results_wake_long_poll_and_failures_are_terminal() -> None:
    async def scenario():
        state = make_state()
        ok_id, _ = await submit_with_delivery(state, SubmitRequest(prompt_tokens=16))
        bad_id, _ = await submit_with_delivery(state, SubmitRequest(prompt_tokens=16))
        waiter = asyncio.create_task(state.wait_for(ok_id, timeout_s=5))
        await asyncio.sleep(0)
        assert state.handle_result(result_payload(ok_id, engine_id="engine-a"))
        await asyncio.wait_for(waiter, timeout=1)
        assert state.results[ok_id]["state"] == "COMPLETED"
        assert state.handle_result(result_payload(bad_id, status="FAILED", error="x"))
        assert state.results[bad_id]["state"] == "FAILED"
        stats = state.stats()
        assert (stats["completed"], stats["failed"], stats["queue_depth"]) == (1, 1, 0)
        assert "infersched_gateway_requests_failed_total 1" in state.prometheus()

    asyncio.run(scenario())


def test_foreign_results_are_ignored_and_cache_is_bounded() -> None:
    async def scenario():
        state = make_state(result_cache_size=2)
        assert not state.handle_result(result_payload("someone-elses-request"))
        assert not state.handle_result(b"not json")
        ids = [
            (await submit_with_delivery(state, SubmitRequest(prompt_tokens=16)))[0]
            for _ in range(3)
        ]
        for request_id in ids:
            assert state.handle_result(result_payload(request_id))
        assert list(state.results) == ids[1:]

    asyncio.run(scenario())


def test_lost_results_expire_and_free_capacity() -> None:
    async def scenario():
        state = make_state(capacity=1, result_timeout_s=10)
        request_id, _ = await submit_with_delivery(state, SubmitRequest(prompt_tokens=16))
        started = state.pending[request_id].started
        assert state.expire_pending(now=started + 9) == 0
        assert state.expire_pending(now=started + 10) == 1
        assert state.stats()["timed_out"] == 1
        # Capacity is free again, and a late result is still retained.
        await submit_with_delivery(state, SubmitRequest(prompt_tokens=16))
        assert state.handle_result(result_payload(request_id))
        assert state.results[request_id]["state"] == "COMPLETED"

    asyncio.run(scenario())


def test_request_expires_exactly_at_its_deadline() -> None:
    # With this start time (seen in CI), (started + 10) - started is
    # 9.99999999999997, so an elapsed-time comparison missed the deadline.
    async def scenario():
        state = make_state(capacity=1, result_timeout_s=10)
        request_id, _ = await submit_with_delivery(state, SubmitRequest(prompt_tokens=16))
        started = 252.15820775
        state.pending[request_id].started = started
        assert state.expire_pending(now=started + 10) == 1

    asyncio.run(scenario())
