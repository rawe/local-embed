"""Tests for the embedding service."""

from __future__ import annotations

import math
import os

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SKIP_MODEL = os.getenv("SKIP_MODEL_TESTS", "0").strip().lower() in ("1", "true", "yes")
skip_model = pytest.mark.skipif(SKIP_MODEL, reason="SKIP_MODEL_TESTS is set")

EXPECTED_DIM = 768


@pytest.fixture(scope="module")
def model():
    """Instantiate the EmbeddingModel once for the entire module."""
    from embed_provider.model import EmbeddingModel

    return EmbeddingModel()


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI TestClient (triggers lifespan -> model load)."""
    from embed_provider.api import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Model wrapper tests
# ---------------------------------------------------------------------------


@skip_model
class TestModelWrapper:
    """Tests that exercise the SentenceTransformer model directly."""

    def test_embed_one_returns_list_of_floats(self, model):
        result = model.embed_one("Hello world")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_embed_one_dimension(self, model):
        result = model.embed_one("Hello world")
        assert len(result) == EXPECTED_DIM

    def test_embed_one_no_nans(self, model):
        result = model.embed_one("Hello world")
        assert not any(math.isnan(v) for v in result)

    def test_embed_many_returns_correct_count(self, model):
        texts = ["first sentence", "second sentence"]
        results = model.embed_many(texts)
        assert len(results) == 2

    def test_embed_many_each_correct_dimension(self, model):
        texts = ["first sentence", "second sentence"]
        results = model.embed_many(texts)
        for emb in results:
            assert len(emb) == EXPECTED_DIM
            assert isinstance(emb, list)
            assert all(isinstance(v, float) for v in emb)


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@skip_model
class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "model" in body
        assert "device" in body


@skip_model
class TestEmbeddingsEndpoint:
    def test_single_string_input(self, client):
        resp = client.post("/v1/embeddings", json={"input": "Hello world"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert len(body["data"]) == 1
        assert body["data"][0]["object"] == "embedding"
        assert body["data"][0]["index"] == 0
        assert len(body["data"][0]["embedding"]) == EXPECTED_DIM

    def test_list_input(self, client):
        resp = client.post(
            "/v1/embeddings", json={"input": ["Hello", "World"]}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]) == 2
        for i, item in enumerate(body["data"]):
            assert item["index"] == i
            assert len(item["embedding"]) == EXPECTED_DIM

    def test_model_field_in_response(self, client):
        resp = client.post("/v1/embeddings", json={"input": "test"})
        body = resp.json()
        assert "model" in body
        assert isinstance(body["model"], str)

    def test_empty_list_returns_400(self, client):
        resp = client.post("/v1/embeddings", json={"input": []})
        assert resp.status_code == 400

    def test_empty_string_returns_400(self, client):
        resp = client.post("/v1/embeddings", json={"input": ""})
        assert resp.status_code == 400

    def test_list_with_empty_string_returns_400(self, client):
        resp = client.post(
            "/v1/embeddings", json={"input": ["hello", ""]}
        )
        assert resp.status_code == 400
