import pytest
from fastapi.testclient import TestClient

from fastapi import HTTPException

from infersched_api.app import GatewayState, SubmitRequest, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_dashboard(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/").status_code == 200


def test_request_validation(client: TestClient) -> None:
    response = client.post("/v1/inference", json={"prompt_tokens": 0})
    assert response.status_code == 422


def test_bounded_gateway_rejects_with_429() -> None:
    state = GatewayState(capacity=1, brokers="localhost:9092", database="")
    state.submit(SubmitRequest(prompt_tokens=16))
    with pytest.raises(HTTPException) as error:
        state.submit(SubmitRequest(prompt_tokens=16))
    assert error.value.status_code == 429
    state.close()
