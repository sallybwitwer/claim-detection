"""Tests for the /detect-claims endpoint in src/api.py.

`predict` is replaced with a fake in most tests: the real one loads a ~570 MB
checkpoint, which would make the suite slow and would require weights that are
not in the repository. The fake records what the endpoint passed it, which is
how the model-name handling is checked. One optional test at the bottom runs
the real model end to end.
"""

import os

import pytest
from fastapi.testclient import TestClient

import src.api as api
from enums import Model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def calls(monkeypatch):
    """Replace api.predict with a fake; yields the list of (model, claims) it saw."""
    recorded = []

    def fake_predict(model, claims):
        Model[model]  # the same lookup the real predict does, so bad names still fail
        recorded.append((model, claims))
        return [
            {"text": c, "label": "claim", "prob_claim": 0.99} for c in claims
        ]

    monkeypatch.setattr(api, "predict", fake_predict)
    return recorded


@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture
def lenient_client():
    """Returns 500 instead of re-raising, so error handling can be asserted."""
    return TestClient(api.app, raise_server_exceptions=False)


def post(client, **body):
    return client.post("/detect-claims", json=body)


# --- happy path -----------------------------------------------------------


def test_returns_one_prediction_per_claim(client, calls):
    resp = post(client, model="BERT", claims=["Crime rose 12%.", "We must do better."])

    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 2


def test_prediction_keeps_text_label_and_probability(client, calls):
    resp = post(client, model="BERT", claims=["Crime rose 12%."])

    assert resp.json()["predictions"] == [
        {"text": "Crime rose 12%.", "label": "claim", "prob_claim": 0.99}
    ]


def test_claims_reach_predict_unchanged(client, calls):
    post(client, model="BERT", claims=["  padded  ", "unicode: café"])

    assert calls[0][1] == ["  padded  ", "unicode: café"]


def test_empty_claims_list_returns_no_predictions(client, calls):
    resp = post(client, model="BERT", claims=[])

    assert resp.status_code == 200
    assert resp.json()["predictions"] == []


# --- model name handling --------------------------------------------------


def test_bert_reaches_predict_as_a_known_checkpoint(client, calls):
    post(client, model="BERT", claims=["Crime rose 12%."])

    assert calls[0][0] in Model.__members__


def test_modernbert_reaches_predict_as_a_known_checkpoint(client, calls):
    post(client, model="ModernBERT", claims=["Crime rose 12%."])

    assert calls[0][0] in Model.__members__


@pytest.mark.parametrize("name", ["bert", "BERT", "modernbert", "ModernBERT", "MODERNBERT"])
def test_model_name_is_case_insensitive(client, calls, name):
    post(client, model=name, claims=["Crime rose 12%."])

    assert calls[0][0] in Model.__members__


def test_unknown_model_is_a_client_error_not_a_server_error(client, calls):
    resp = post(client, model="RoBERTa", claims=["Crime rose 12%."])

    assert resp.status_code == 400
    assert "RoBERTa" in resp.json()["detail"]


# --- request validation ---------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"claims": ["a claim"]},           # model missing
        {"model": "BERT"},                 # claims missing
        {"model": "BERT", "claims": "a"},  # claims not a list
        {"model": 42, "claims": ["a"]},    # model not a string
    ],
)
def test_malformed_requests_are_rejected(client, calls, body):
    assert post(client, **body).status_code == 422


# --- optional end-to-end check -------------------------------------------

_bert_ckpt = os.path.join(REPO_ROOT, Model.BERT.value)


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1" or not os.path.isdir(_bert_ckpt),
    reason="set RUN_INTEGRATION=1 and train BERT first; loads a ~440 MB checkpoint",
)
def test_real_model_separates_a_claim_from_an_opinion(client):
    resp = post(
        client,
        model="BERT",
        claims=[
            "The unemployment rate fell to 3.4% in January 2023.",
            "I think we should all try to be kinder to one another.",
        ],
    )

    labels = [p["label"] for p in resp.json()["predictions"]]
    assert labels == ["claim", "not_claim"]
