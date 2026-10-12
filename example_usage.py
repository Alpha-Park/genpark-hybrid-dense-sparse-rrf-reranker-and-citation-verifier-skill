"""Example usage for Hybrid Dense-Sparse RRF Reranker."""
from client import HybridRRFReranker

if __name__ == "__main__":
    res = HybridRRFReranker.fuse_and_verify(["doc_A", "doc_B", "doc_C"], ["doc_B", "doc_D", "doc_A"])
    print("Top Doc:", res["top_doc_id"])
    print("Ranked:", res["ranked_results"])
