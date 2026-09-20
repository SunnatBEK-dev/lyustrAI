from ai_sdk.embeddings.base import BaseEmbeddingClient
from ai_sdk.retrieval.retriever import SemanticRetriever
from ai_sdk.retrieval.search import (
    SearchResult,
    fuse_ranked_results,
)
from ai_sdk.retrieval.vector_store import BaseVectorStore


class HybridRetriever(SemanticRetriever):
    """Combine embedding similarity and BM25 lexical retrieval."""

    def __init__(
        self,
        embedding_client: BaseEmbeddingClient,
        vector_store: BaseVectorStore,
        *,
        semantic_weight: float = 0.7,
        candidate_multiplier: int = 3,
        rank_constant: int = 60,
    ) -> None:
        super().__init__(
            embedding_client=embedding_client,
            vector_store=vector_store,
        )

        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("Semantic weight must be between zero and one.")

        if candidate_multiplier <= 0:
            raise ValueError("Candidate multiplier must be greater than zero.")

        if rank_constant < 0:
            raise ValueError("Rank constant cannot be negative.")

        self.semantic_weight = semantic_weight
        self.candidate_multiplier = candidate_multiplier
        self.rank_constant = rank_constant

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Retrieval query cannot be empty.")

        if k <= 0:
            raise ValueError("Top-k value must be greater than zero.")

        candidate_k = k * self.candidate_multiplier
        semantic_results = []
        lexical_results = []

        if self.semantic_weight > 0.0:
            query_vector = self.embedding_client.embed_one(query)
            semantic_results = self.vector_store.search(
                query_vector=query_vector,
                k=candidate_k,
            )

        if self.semantic_weight < 1.0:
            lexical_results = self.vector_store.lexical_search(
                query=query,
                k=candidate_k,
            )

        return fuse_ranked_results(
            semantic_results,
            lexical_results,
            semantic_weight=self.semantic_weight,
            k=k,
            rank_constant=self.rank_constant,
        )
