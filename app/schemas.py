from typing import List

from pydantic import BaseModel


class MeetingRequest(BaseModel):
    transcript: str
    query: str = "What was decided about the demo?"
    top_k: int = 3


class AgenticMeetingRequest(MeetingRequest):
    top_k: int = 5


class TranscriptRecord(BaseModel):
    speaker: str
    text: str


class SearchResult(BaseModel):
    speaker: str
    text: str
    score: float


class TopicGroup(BaseModel):
    topic: str
    items: List[TranscriptRecord]


class MeetingSummary(BaseModel):
    overview: str
    key_decisions: List[str]
    key_action_items: List[str]


class MeetingOutput(BaseModel):
    summary: MeetingSummary
    action_items: List[TranscriptRecord]
    decisions: List[TranscriptRecord]
    topics: List[TopicGroup]
    search_results: List[SearchResult]
