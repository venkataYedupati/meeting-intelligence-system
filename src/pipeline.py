from functools import lru_cache
from typing import Dict, List, Optional

from src.output_formatter import build_meeting_output
from src.preprocess import (
    clean_text,
    extract_action_items,
    extract_decisions,
    parse_transcript_lines,
    split_transcript_lines,
)
from src.search import SearchBackend, load_embedding_model, semantic_search
from src.summary import generate_meeting_summary
from src.topics import segment_topics


@lru_cache(maxsize=1)
def get_search_backend() -> SearchBackend:
    """
    Cache the retrieval backend so API and Streamlit requests do not reload it.
    """
    return load_embedding_model()


def prepare_records(transcript: str) -> Dict[str, object]:
    """
    Clean and parse a transcript into speaker-text records.
    """
    cleaned_transcript = clean_text(transcript)
    lines = split_transcript_lines(cleaned_transcript)
    records = parse_transcript_lines(lines)

    return {
        "cleaned_transcript": cleaned_transcript,
        "records": records,
    }


def analyze_transcript(
    transcript: str,
    query: str,
    top_k: int = 3,
    search_backend: Optional[SearchBackend] = None,
) -> Dict[str, object]:
    """
    Run the standard meeting intelligence pipeline and keep intermediate artifacts.
    """
    prepared = prepare_records(transcript)
    records: List[Dict[str, str]] = prepared["records"]  # type: ignore[assignment]

    action_items = extract_action_items(records)
    decisions = extract_decisions(records)
    topics = segment_topics(records)
    backend = search_backend or get_search_backend()
    search_results = semantic_search(query, records, backend, top_k=top_k)

    summary = generate_meeting_summary(
        action_items=action_items,
        decisions=decisions,
        topics=topics,
    )

    output = build_meeting_output(
        summary=summary,
        action_items=action_items,
        decisions=decisions,
        topics=topics,
        search_results=search_results,
    )

    return {
        "cleaned_transcript": prepared["cleaned_transcript"],
        "records": records,
        "retrieval_backend": backend.name,
        "output": output,
    }


def run_pipeline(transcript: str, query: str, top_k: int = 3) -> Dict[str, object]:
    """
    Public helper for callers that only need the final standard output.
    """
    return analyze_transcript(transcript=transcript, query=query, top_k=top_k)["output"]
