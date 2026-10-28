from support_agent.kb import retrieve
from support_agent.semantic_kb import retrieve_semantic

RRF_K = 60


def retrieve_hybrid(query: str, top_k: int = 3) -> list[dict]:
    """Combine keyword and semantic rankings using RRF."""
    if not query.strip() or top_k <= 0:
        return []

    keyword_results = retrieve(query, top_k=3)
    semantic_results = retrieve_semantic(query, top_k=3)

    fused = {}

    for results, source in [
        (keyword_results, "keyword"),
        (semantic_results, "semantic"),
    ]:
        for rank, article in enumerate(results, start=1):
            article_id = article["id"]

            if article_id not in fused:
                fused[article_id] = {
                    key: value for key, value in article.items() if key != "score"
                }
                fused[article_id]["rrf_score"] = 0.0
                fused[article_id]["keyword_rank"] = None
                fused[article_id]["semantic_rank"] = None

            fused[article_id]["rrf_score"] += 1 / (RRF_K + rank)
            fused[article_id][f"{source}_rank"] = rank

    ranked = sorted(
        fused.values(),
        key=lambda article: article["rrf_score"],
        reverse=True,
    )

    return [
        {
            **article,
            "score": article["rrf_score"],
        }
        for article in ranked[:top_k]
    ]
