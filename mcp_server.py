"""MCP server for Hybrid Dense-Sparse RRF Reranker."""
import sys
import json
from client import HybridRRFReranker

def handle_request(req):
    method = req.get("method")
    if method == "tools/list":
        return {
            "tools": [{
                "name": "rerank_hybrid_rrf",
                "description": "Applies Reciprocal Rank Fusion across dense and sparse retrieval lists",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dense_doc_ids": {"type": "array"},
                        "sparse_doc_ids": {"type": "array"},
                        "rrf_k": {"type": "integer"}
                    },
                    "required": ["dense_doc_ids", "sparse_doc_ids"]
                }
            }]
        }
    elif method == "tools/call":
        params = req.get("params", {})
        if params.get("name") == "rerank_hybrid_rrf":
            args = params.get("arguments", {})
            res = HybridRRFReranker.fuse_and_verify(
                args.get("dense_doc_ids", []),
                args.get("sparse_doc_ids", []),
                args.get("rrf_k", 60)
            )
            return {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}
    return {"error": "Method not found"}

if __name__ == "__main__":
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(handle_request(json.loads(line))))
            sys.stdout.flush()
