"""
Integration tests for the NVE HTTP server.

Uses aiohttp's built-in test client — no real model is loaded.
The ModelStore is monkey-patched to return a fake echo backend.
"""

from __future__ import annotations

import json
import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from nve.serve.config import ServerConfig
from nve.serve.metrics import MetricsCollector
from nve.serve.model_store import ModelStore, ModelRecord, _Backend
from nve.serve.batch_scheduler import BatchScheduler
from nve.serve.server import NVEServer, create_app


# ── Fake backend ──────────────────────────────────────────────────────────────

class EchoBackend(_Backend):
    def generate(self, prompt, max_new_tokens, temperature, top_p):
        return {
            "text": f"[echo] {prompt}",
            "prompt_tokens": len(prompt.split()),
            "generated_tokens": 5,
            "time_s": 0.01,
            "tokens_per_sec": 500.0,
            "prefill_time_ms": 1.0,
            "decode_time_ms": 9.0,
            "backend": "echo",
        }

    def generate_stream(self, prompt, max_new_tokens, temperature, top_p):
        words = f"[echo] {prompt}".split()
        for i, w in enumerate(words):
            yield (w + " ") if i < len(words) - 1 else w

    @property
    def backend_name(self):
        return "echo"


@pytest.fixture
def server_with_model():
    cfg = ServerConfig(
        host="127.0.0.1",
        port=0,  # random port
        max_batch_size=4,
        batch_timeout_ms=20,
    )
    srv = NVEServer(cfg)
    # Inject fake model.
    record = ModelRecord(name="test-model", model_path="/fake/path", backend=EchoBackend())
    srv.model_store._models["test-model"] = record
    return srv


@pytest.fixture
def app(server_with_model):
    return create_app(server_with_model)


@pytest.fixture
def cli(loop, app):
    """aiohttp test client fixture."""
    return loop.run_until_complete(
        TestClient(TestServer(app)).__aenter__()
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/health")
    assert resp.status == 200
    body = await resp.json()
    assert body["status"] == "ok"
    assert "uptime_s" in body


@pytest.mark.asyncio
async def test_list_models_empty(aiohttp_client):
    cfg = ServerConfig()
    srv = NVEServer(cfg)
    a = create_app(srv)
    client = await aiohttp_client(a)
    resp = await client.get("/v1/models")
    assert resp.status == 200
    body = await resp.json()
    assert body["models"] == []


@pytest.mark.asyncio
async def test_list_models_with_model(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/v1/models")
    assert resp.status == 200
    body = await resp.json()
    assert len(body["models"]) == 1
    assert body["models"][0]["name"] == "test-model"
    assert body["models"][0]["backend"] == "echo"


@pytest.mark.asyncio
async def test_generate(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.post(
        "/v1/generate",
        json={"prompt": "Hello world", "model": "test-model", "max_new_tokens": 10},
    )
    assert resp.status == 200
    body = await resp.json()
    assert "text" in body
    assert "[echo]" in body["text"]
    assert body["model"] == "test-model"


@pytest.mark.asyncio
async def test_generate_no_model(aiohttp_client):
    cfg = ServerConfig()
    srv = NVEServer(cfg)
    a = create_app(srv)
    client = await aiohttp_client(a)
    resp = await client.post("/v1/generate", json={"prompt": "hi"})
    assert resp.status == 503


@pytest.mark.asyncio
async def test_generate_missing_prompt(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.post("/v1/generate", json={"model": "test-model"})
    assert resp.status == 400
    body = await resp.json()
    assert "error" in body


@pytest.mark.asyncio
async def test_generate_stream(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.post(
        "/v1/generate/stream",
        json={"prompt": "Hello", "model": "test-model"},
    )
    assert resp.status == 200
    assert "text/event-stream" in resp.content_type
    body = await resp.text()
    assert "data:" in body
    assert "done" in body


@pytest.mark.asyncio
async def test_batch_generate(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.post(
        "/v1/batch",
        json={
            "prompts": ["Hello", "World", "Test"],
            "model": "test-model",
            "max_new_tokens": 5,
        },
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["count"] == 3
    assert len(body["results"]) == 3
    for r in body["results"]:
        assert "text" in r


@pytest.mark.asyncio
async def test_unload_model(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.delete("/v1/models/test-model")
    assert resp.status == 200
    body = await resp.json()
    assert body["status"] == "unloaded"


@pytest.mark.asyncio
async def test_unload_nonexistent(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.delete("/v1/models/does-not-exist")
    assert resp.status == 404


@pytest.mark.asyncio
async def test_metrics_prometheus(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/metrics")
    assert resp.status == 200
    text = await resp.text()
    assert "nve_uptime_seconds" in text


@pytest.mark.asyncio
async def test_metrics_json(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/v1/metrics")
    assert resp.status == 200
    body = await resp.json()
    assert "uptime_s" in body
    assert "counters" in body
