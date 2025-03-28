def rerank_with_rrf(top_entries, original_scores, k=60):
    """
    Applies Reciprocal Rank Fusion (RRF) to rerank the top entries.

    Args:
        top_entries (List[dict]): List of candidate entries.
        original_scores (List[float]): Original similarity scores (same order).
        k (int): RRF constant (higher = less impact of lower ranks).

    Returns:
        List[dict]: Reranked entries.
    """
    # Assign initial rank positions
    ranks = {i: i + 1 for i in range(len(top_entries))}

    # Compute RRF score for each entry
    rrf_scores = {i: 1 / (k + ranks[i]) for i in ranks}

    # Combine with original scores if needed, or use only RRF
    reranked = sorted(range(len(top_entries)), key=lambda i: -rrf_scores[i])
    return [top_entries[i] for i in reranked]


def rerank_with_mmr(query_embedding, candidate_embeddings, top_k=5, lambda_param=0.5):
    """
    Maximal Marginal Relevance reranking.

    Args:
        query_embedding (Tensor): Normalized query embedding (1 x D).
        candidate_embeddings (Tensor): Normalized candidate embeddings (N x D).
        top_k (int): Number of items to rerank/select.
        lambda_param (float): Trade-off between relevance and diversity.

    Returns:
        List[int]: Indices of reranked items.
    """
    import torch

    selected = []
    candidates = list(range(candidate_embeddings.size(0)))

    query_embedding = query_embedding.squeeze(0)  # Shape: (D,)
    sim_to_query = torch.matmul(candidate_embeddings, query_embedding)  # (N,)

    while len(selected) < top_k and candidates:
        mmr_scores = []
        for idx in candidates:
            if not selected:
                diversity_penalty = 0
            else:
                selected_embs = candidate_embeddings[selected]
                sim_to_selected = torch.matmul(selected_embs, candidate_embeddings[idx])
                diversity_penalty = sim_to_selected.max().item()

            mmr_score = lambda_param * sim_to_query[idx].item() - (1 - lambda_param) * diversity_penalty
            mmr_scores.append((idx, mmr_score))

        best_idx, _ = max(mmr_scores, key=lambda x: x[1])
        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected
