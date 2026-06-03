import os
import re
from collections import Counter
from typing import Dict, List, Optional, Protocol, Sequence


MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_BACKEND = "tfidf"


class SearchBackend(Protocol):
    """
    Minimal interface used by semantic_search.
    """

    name: str

    def rank(self, query: str, texts: Sequence[str]) -> List[float]:
        ...


class KeywordSearchBackend:
    """
    Pure-Python fallback that keeps the app usable without ML dependencies.
    """

    name = "keyword-overlap"

    def rank(self, query: str, texts: Sequence[str]) -> List[float]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return [0.0 for _ in texts]

        query_counts = Counter(query_tokens)
        query_terms = set(query_tokens)
        scores = []

        for text in texts:
            text_tokens = _tokenize(text)
            text_counts = Counter(text_tokens)
            text_terms = set(text_tokens)
            overlap = query_terms.intersection(text_terms)

            if not text_terms:
                scores.append(0.0)
                continue

            weighted_overlap = sum(
                min(query_counts[token], text_counts[token]) for token in overlap
            )
            precision = weighted_overlap / max(len(text_tokens), 1)
            recall = weighted_overlap / max(len(query_tokens), 1)
            jaccard = len(overlap) / len(query_terms.union(text_terms))
            scores.append(float((0.35 * precision) + (0.45 * recall) + (0.20 * jaccard)))

        return scores


class TfidfSearchBackend:
    """
    Local semantic-ish retrieval using TF-IDF when scikit-learn is available.
    """

    name = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self._cosine_similarity = cosine_similarity
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )

    def rank(self, query: str, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []

        try:
            matrix = self._vectorizer.fit_transform([query, *texts])
            scores = self._cosine_similarity(matrix[0:1], matrix[1:]).ravel()
            return [float(score) for score in scores]
        except ValueError:
            return KeywordSearchBackend().rank(query, texts)


class SentenceTransformerSearchBackend:
    """
    Optional SBERT backend for stronger semantic retrieval.
    """

    name = "sentence-transformers"

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        self._cosine_similarity = cosine_similarity
        self._model = SentenceTransformer(model_name)

    def rank(self, query: str, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []

        query_embedding = self._model.encode([query])
        text_embeddings = self._model.encode(list(texts))
        scores = self._cosine_similarity(query_embedding, text_embeddings)[0]
        return [float(score) for score in scores]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def load_embedding_model(backend: Optional[str] = None) -> SearchBackend:
    """
    Load a retrieval backend.

    The default backend is local TF-IDF for predictable offline tests and demos.
    Set MEETING_INTEL_SEARCH_BACKEND=sentence-transformers to use SBERT.
    """
    selected_backend = (backend or os.getenv("MEETING_INTEL_SEARCH_BACKEND", DEFAULT_BACKEND)).lower()

    if selected_backend in {"sentence-transformers", "sentence_transformers", "sbert"}:
        try:
            return SentenceTransformerSearchBackend()
        except Exception:
            return _local_search_backend()

    return _local_search_backend()


def _local_search_backend() -> SearchBackend:
    try:
        return TfidfSearchBackend()
    except Exception:
        return KeywordSearchBackend()


def detect_query_intent(query: str) -> str:
    """
    Detect the likely intent of the query.
    """
    lowered_query = query.lower()

    decision_keywords = ["decide", "decided", "decision", "agreed", "finalized", "approved"]
    action_keywords = ["action", "task", "owner", "who will", "who is", "follow up", "send", "schedule"]

    if any(keyword in lowered_query for keyword in decision_keywords):
        return "decision"

    if any(keyword in lowered_query for keyword in action_keywords):
        return "action"

    return "general"


def semantic_search(
    query: str,
    records: List[Dict[str, str]],
    model: SearchBackend,
    top_k: int = 3,
) -> List[Dict[str, object]]:
    """
    Perform semantic search over transcript records.

    Returns the top_k most relevant records with similarity scores.
    Applies a simple intent-aware score boost.
    """
    if not records:
        return []

    texts = [record["text"] for record in records]
    similarity_scores = model.rank(query, texts)
    query_intent = detect_query_intent(query)

    scored_results = []
    for record, score in zip(records, similarity_scores):
        boosted_score = float(score)
        lowered_text = record["text"].lower()

        if query_intent == "decision":
            decision_patterns = ["decided", "agreed", "finalized", "approved", "confirmed"]
            if any(pattern in lowered_text for pattern in decision_patterns):
                boosted_score += 0.15

        elif query_intent == "action":
            action_patterns = ["i will", "we should", "let's", "lets", "need to", "plan to", "going to"]
            if any(pattern in lowered_text for pattern in action_patterns):
                boosted_score += 0.15

        scored_results.append({
            "speaker": record["speaker"],
            "text": record["text"],
            "score": boosted_score,
        })

    scored_results.sort(key=lambda item: item["score"], reverse=True)

    return scored_results[:top_k]
