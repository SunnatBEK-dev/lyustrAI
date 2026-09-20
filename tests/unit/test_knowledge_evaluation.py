import json

import pytest

from ai_sdk.embeddings.base import BaseEmbeddingClient
from app.evaluate_knowledge import evaluate, load_dataset


class ConstantEmbeddingClient(BaseEmbeddingClient):
    def embed(self, texts):
        return [[1.0, float(len(text) % 7)] for text in texts]


def test_knowledge_evaluation_runs_from_stable_source_labels(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "architecture.md").write_text(
        "Provider-neutral architecture uses explicit contracts.",
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "architecture",
                    "query": "How are provider contracts designed?",
                    "expected_sources": ["architecture.md"],
                }
            ]
        ),
        encoding="utf-8",
    )

    report, passed = evaluate(
        corpus,
        dataset,
        embedding_client=ConstantEmbeddingClient(),
    )

    assert passed is True
    assert "Dataset cases: 1" in report
    assert "Observed failure categories" in report
    assert "Status: **PASS**" in report


@pytest.mark.parametrize(
    "payload",
    [
        [],
        [{"id": "", "query": "Question", "expected_sources": ["a.md"]}],
        [
            {"id": "duplicate", "query": "One", "expected_sources": ["a.md"]},
            {"id": "duplicate", "query": "Two", "expected_sources": ["a.md"]},
        ],
    ],
)
def test_knowledge_dataset_rejects_invalid_cases(tmp_path, payload):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset|fields"):
        load_dataset(dataset)


def test_knowledge_evaluation_rejects_unknown_source(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "known.md").write_text("Known facts", encoding="utf-8")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "unknown",
                    "query": "Question",
                    "expected_sources": ["missing.md"],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown sources"):
        evaluate(
            corpus,
            dataset,
            embedding_client=ConstantEmbeddingClient(),
        )
