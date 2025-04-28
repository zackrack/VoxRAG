from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load a reranker model (e.g., MiniLM-based)
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L6-v2")
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L6-v2")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def rerank_segments(query, docs):
    # Inside rerank_segments:
    inputs = tokenizer(
        [f"{query} [SEP] {doc}" for doc in docs],
        return_tensors='pt', padding=True, truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1)
    sorted_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
    return sorted_docs[:10]

from collections import defaultdict

def reciprocal_rank_fusion(ranked_lists, k=60, top_n=10, id_key="segment_id"):
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).
    
    Args:
        ranked_lists (list of list): Each list is a ranked list of dicts (e.g. FAISS results).
        k (int): RRF damping factor.
        top_n (int): Number of results to return.
        id_key (str): Key in each item that uniquely identifies a document.
    
    Returns:
        List of top_n items fused by RRF score.
    """
    scores = defaultdict(float)
    doc_lookup = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            doc_id = doc[id_key]
            scores[doc_id] += 1 / (k + rank + 1)
            doc_lookup[doc_id] = doc  # Preserve the full entry

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_lookup[doc_id] for doc_id, _ in sorted_docs[:top_n]]

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def maximal_marginal_relevance(query_embedding, doc_embeddings, lambda_param=0.5, top_n=10):
    """
    Select top_n documents using Maximal Marginal Relevance (MMR).
    
    Args:
        query_embedding (np.array): Query vector (shape: [1, D])
        doc_embeddings (np.array): Document vectors (shape: [N, D])
        lambda_param (float): Trade-off between relevance and diversity.
        top_n (int): Number of documents to select.
    
    Returns:
        List of indices of selected documents.
    """
    selected = []
    candidate_indices = list(range(len(doc_embeddings)))
    doc_embeddings = np.array(doc_embeddings)

    sim_to_query = cosine_similarity(doc_embeddings, query_embedding.reshape(1, -1)).flatten()
    sim_to_docs = cosine_similarity(doc_embeddings)

    for _ in range(top_n):
        mmr_score = []
        for idx in candidate_indices:
            diversity = max([sim_to_docs[idx][j] for j in selected], default=0)
            score = lambda_param * sim_to_query[idx] - (1 - lambda_param) * diversity
            mmr_score.append((score, idx))
        
        _, best_idx = max(mmr_score)
        selected.append(best_idx)
        candidate_indices.remove(best_idx)

    return selected
