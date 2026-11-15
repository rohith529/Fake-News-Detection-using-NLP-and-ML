"""Similarity Engine: compute Agreement_Scores between a claim and retrieved articles."""
import logging
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import clean

_MODE = os.environ.get("SIMILARITY_MODE", "lightweight").lower()
_log  = logging.getLogger(__name__)
_log.info("Similarity mode: %s", _MODE)

_semantic_model = None
if _MODE == "semantic":
    try:
        from sentence_transformers import SentenceTransformer
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        _log.info("Loaded sentence-transformers model: all-MiniLM-L6-v2")
    except Exception as e:
        import sys
        print(f"WARNING: sentence-transformers not available; falling back to lightweight. Reason: {e}",
              file=sys.stderr)
        _semantic_model = None


def compute_scores(claim_text: str, articles: list[dict]) -> list[float]:
    """Return Agreement_Score for each article, in input order.

    Each score is in [0.0, 1.0]. Returns [] when articles is empty.

    Lightweight mode (default):
        TF-IDF cosine similarity between the cleaned claim and each
        article title.

    Semantic mode:
        Falls through to lightweight for now — task 2.2 will add the
        real sentence-embedding logic.
    """
    if not articles:
        return []

    # ── Semantic path ─────────────────────────────────────────────────────────
    if _MODE == "semantic" and _semantic_model is not None:
        from numpy.linalg import norm
        claim_emb  = _semantic_model.encode([claim_text])[0]
        title_embs = _semantic_model.encode([a.get("title", "") for a in articles])
        c_norm = claim_emb / (norm(claim_emb) + 1e-9)
        scores_arr = []
        for emb in title_embs:
            a_norm = emb / (norm(emb) + 1e-9)
            s = float(np.dot(c_norm, a_norm))
            scores_arr.append(max(0.0, min(1.0, s)))
        return scores_arr

    # ── Lightweight path ─────────────────────────────────────────────────────
    cleaned_claim  = clean(claim_text)
    cleaned_titles = [clean(a.get("title", "")) for a in articles]

    corpus = [cleaned_claim] + cleaned_titles

    vectorizer = TfidfVectorizer(min_df=1, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # claim vector is row 0; article vectors are rows 1…n
    claim_vec    = tfidf_matrix[0]
    article_vecs = tfidf_matrix[1:]

    raw_scores = cosine_similarity(claim_vec, article_vecs).flatten()

    return [float(max(0.0, min(1.0, s))) for s in raw_scores]


def compute_source_signal(articles: list[dict], scores: list[float]) -> tuple[float, bool]:
    """Return (source_signal, confidence_flag).

    source_signal: reliability-weighted mean of the top-3 Agreement_Scores.
                   Returns 0.5 (neutral) when articles is empty.
    confidence_flag: True iff len(articles) >= 2.
    """
    if not articles:
        return 0.5, False

    confidence_flag = len(articles) >= 2

    # Pair scores with reliability weights; sort descending by score; take top 3
    pairs = sorted(zip(scores, [a["reliability"] for a in articles]),
                   key=lambda x: x[0], reverse=True)
    top = pairs[:3]

    total_weight = sum(r for _, r in top)
    if total_weight == 0:
        source_signal = 0.5
    else:
        source_signal = sum(s * r for s, r in top) / total_weight

    return float(max(0.0, min(1.0, source_signal))), confidence_flag
