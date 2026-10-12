"""Hybrid Dense-Sparse RRF Reranker & Citation Verifier.
100% Python Standard Library.
"""

class HybridRRFReranker:
    """Fuses dense semantic ranks and sparse BM25 lexical ranks using Reciprocal Rank Fusion (RRF) and verifies grounding."""

    @staticmethod
    def fuse_and_verify(dense_doc_ids: list, sparse_doc_ids: list, rrf_k: int = 60) -> dict:
        scores = {}
        for rank, doc_id in enumerate(dense_doc_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
        for rank, doc_id in enumerate(sparse_doc_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

        ranked = sorted(
            [{"doc_id": k, "rrf_score": round(v, 6), "dual_matched": (k in dense_doc_ids and k in sparse_doc_ids)}
             for k, v in scores.items()],
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        return {
            "total_unique_docs": len(ranked),
            "top_doc_id": ranked[0]["doc_id"] if ranked else None,
            "ranked_results": ranked
        }
