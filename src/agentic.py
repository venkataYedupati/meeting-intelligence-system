import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

from src.pipeline import analyze_transcript
from src.search import detect_query_intent


Record = Dict[str, str]

STOP_WORDS = {
    "about",
    "after",
    "again",
    "also",
    "that",
    "this",
    "there",
    "their",
    "them",
    "then",
    "they",
    "with",
    "will",
    "would",
    "should",
    "could",
    "need",
    "needs",
    "going",
    "take",
    "make",
    "send",
    "what",
    "when",
    "where",
    "which",
    "who",
    "was",
    "were",
    "decide",
    "decided",
    "decision",
    "actions",
    "action",
    "open",
}


@dataclass
class AgentContext:
    transcript: str
    query: str
    records: List[Record]
    baseline: Dict[str, Any]
    retrieval_backend: str


@dataclass
class AgentResult:
    key: str
    agent: str
    role: str
    payload: Any
    confidence: float
    evidence_count: int
    status: str = "completed"


class MeetingAgent:
    key = "agent"
    name = "Meeting Agent"
    role = "Analyze meeting context"

    def run(self, context: AgentContext, shared: Dict[str, Any]) -> AgentResult:
        raise NotImplementedError

    def result(self, payload: Any, confidence: float, evidence_count: int) -> AgentResult:
        return AgentResult(
            key=self.key,
            agent=self.name,
            role=self.role,
            payload=payload,
            confidence=round(confidence, 2),
            evidence_count=evidence_count,
        )


class ParticipantProfilerAgent(MeetingAgent):
    key = "participants"
    name = "Participant Profiler Agent"
    role = "Identify speakers, contribution levels, and ownership signals"

    def run(self, context: AgentContext, shared: Dict[str, Any]) -> AgentResult:
        speaker_counts = Counter(record["speaker"] for record in context.records)
        action_counts = Counter(
            item["speaker"] for item in context.baseline.get("action_items", [])
        )
        decision_counts = Counter(
            item["speaker"] for item in context.baseline.get("decisions", [])
        )
        total_turns = max(len(context.records), 1)

        participants = []
        for speaker, turns in speaker_counts.most_common():
            participants.append(
                {
                    "speaker": speaker,
                    "turns": turns,
                    "participation_share": round(turns / total_turns, 2),
                    "action_items_owned": action_counts.get(speaker, 0),
                    "decisions_contributed": decision_counts.get(speaker, 0),
                }
            )

        return self.result(
            payload=participants,
            confidence=0.95 if participants else 0.0,
            evidence_count=len(context.records),
        )


class ActionPlanningAgent(MeetingAgent):
    key = "execution_plan"
    name = "Action Planning Agent"
    role = "Turn raw action-item mentions into owner, deadline, priority, and evidence"

    def run(self, context: AgentContext, shared: Dict[str, Any]) -> AgentResult:
        action_items = context.baseline.get("action_items", [])
        plan = []

        for index, item in enumerate(action_items, start=1):
            text = item["text"]
            owner = _extract_owner(item)
            due_date = _extract_due_date(text)
            priority = _infer_priority(text, due_date)
            confidence = 0.9 if owner and due_date else 0.75 if owner else 0.55

            plan.append(
                {
                    "id": f"A{index}",
                    "task": _clean_task_text(text),
                    "owner": owner,
                    "due_date": due_date,
                    "priority": priority,
                    "status": "open",
                    "confidence": confidence,
                    "evidence": {
                        "speaker": item["speaker"],
                        "text": text,
                    },
                }
            )

        average_confidence = _average([item["confidence"] for item in plan], default=0.0)
        return self.result(
            payload=plan,
            confidence=average_confidence,
            evidence_count=len(action_items),
        )


class DecisionRegisterAgent(MeetingAgent):
    key = "decision_register"
    name = "Decision Register Agent"
    role = "Normalize decisions with topics, status, and downstream work"

    def run(self, context: AgentContext, shared: Dict[str, Any]) -> AgentResult:
        decisions = context.baseline.get("decisions", [])
        action_items = context.baseline.get("action_items", [])
        topics = context.baseline.get("topics", [])
        decision_register = []

        for index, item in enumerate(decisions, start=1):
            text = item["text"]
            decision_register.append(
                {
                    "id": f"D{index}",
                    "decision": text,
                    "made_by": item["speaker"],
                    "status": _decision_status(text),
                    "topic": _topic_for_text(topics, text),
                    "downstream_actions": _related_texts(text, action_items),
                    "confidence": 0.88,
                    "evidence": {
                        "speaker": item["speaker"],
                        "text": text,
                    },
                }
            )

        return self.result(
            payload=decision_register,
            confidence=0.88 if decision_register else 0.0,
            evidence_count=len(decisions),
        )


class RiskDependencyAgent(MeetingAgent):
    key = "risks"
    name = "Risk and Dependency Agent"
    role = "Find blockers, risks, unresolved dependencies, and open concerns"

    def run(self, context: AgentContext, shared: Dict[str, Any]) -> AgentResult:
        risks = []

        for record in context.records:
            text = record["text"]
            if not _has_risk_signal(text):
                continue

            risks.append(
                {
                    "id": f"R{len(risks) + 1}",
                    "type": _risk_type(text),
                    "severity": _risk_severity(text),
                    "owner_signal": record["speaker"],
                    "description": text,
                    "mitigation_needed": True,
                    "evidence": {
                        "speaker": record["speaker"],
                        "text": text,
                    },
                }
            )

        return self.result(
            payload=risks,
            confidence=0.82 if risks else 0.4,
            evidence_count=len(risks),
        )


class QueryAnswerAgent(MeetingAgent):
    key = "answer"
    name = "Query Answer Agent"
    role = "Synthesize an answer from retrieval results and structured meeting evidence"

    def run(self, context: AgentContext, shared: Dict[str, Any]) -> AgentResult:
        query = context.query.strip()
        intent = detect_query_intent(query)
        citations = _select_answer_citations(context.baseline, intent, query)

        if not query:
            answer = "No query was provided."
            confidence = 0.0
        elif citations:
            evidence_text = "; ".join(
                f"{item['speaker']}: {item['text']}" for item in citations[:3]
            )
            answer = f"Based on the strongest meeting evidence, {evidence_text}"
            confidence = _average(
                [float(item.get("score", 0.75)) for item in citations],
                default=0.75,
            )
        else:
            answer = "No direct answer was found in the meeting transcript."
            confidence = 0.25

        return self.result(
            payload={
                "query": query,
                "intent": intent,
                "answer": answer,
                "citations": citations,
            },
            confidence=min(confidence, 0.95),
            evidence_count=len(citations),
        )


class FollowUpQuestionAgent(MeetingAgent):
    key = "follow_up_questions"
    name = "Follow-up Question Agent"
    role = "Generate next questions from missing owners, deadlines, and unresolved risks"

    def run(self, context: AgentContext, shared: Dict[str, Any]) -> AgentResult:
        questions = []
        execution_plan = shared.get("execution_plan", [])
        risks = shared.get("risks", [])
        decisions = shared.get("decision_register", [])

        for item in execution_plan:
            if not item.get("owner") or item.get("owner") == "Team":
                questions.append(
                    {
                        "question": f"Who directly owns action item {item['id']}?",
                        "reason": item["task"],
                        "priority": "high",
                        "related_id": item["id"],
                    }
                )

            if not item.get("due_date"):
                questions.append(
                    {
                        "question": f"What is the deadline for action item {item['id']}?",
                        "reason": item["task"],
                        "priority": "medium",
                        "related_id": item["id"],
                    }
                )

        for risk in risks:
            questions.append(
                {
                    "question": f"What is the mitigation plan for risk {risk['id']}?",
                    "reason": risk["description"],
                    "priority": "high" if risk["severity"] == "high" else "medium",
                    "related_id": risk["id"],
                }
            )

        if not decisions:
            questions.append(
                {
                    "question": "Which decisions need to be explicitly confirmed before the next meeting?",
                    "reason": "No clear decision statements were detected.",
                    "priority": "medium",
                    "related_id": None,
                }
            )

        if not execution_plan:
            questions.append(
                {
                    "question": "Which follow-up tasks should be assigned to owners?",
                    "reason": "No clear action items were detected.",
                    "priority": "medium",
                    "related_id": None,
                }
            )

        return self.result(
            payload=questions[:8],
            confidence=0.78 if questions else 0.9,
            evidence_count=len(questions),
        )


class QualityGateAgent(MeetingAgent):
    key = "quality"
    name = "Quality Gate Agent"
    role = "Score output completeness and list analysis gaps"

    def run(self, context: AgentContext, shared: Dict[str, Any]) -> AgentResult:
        execution_plan = shared.get("execution_plan", [])
        decisions = shared.get("decision_register", [])
        risks = shared.get("risks", [])
        follow_ups = shared.get("follow_up_questions", [])
        gaps = []

        if not context.records:
            gaps.append("Transcript has no parseable speaker turns.")
        if not execution_plan:
            gaps.append("No explicit action items were detected.")
        if not decisions:
            gaps.append("No explicit decisions were detected.")

        missing_deadlines = [
            item["id"] for item in execution_plan if not item.get("due_date")
        ]
        missing_owners = [
            item["id"]
            for item in execution_plan
            if not item.get("owner") or item.get("owner") == "Team"
        ]

        if missing_deadlines:
            gaps.append(
                "Action items missing deadlines: " + ", ".join(missing_deadlines)
            )
        if missing_owners:
            gaps.append("Action items missing direct owners: " + ", ".join(missing_owners))

        score = 1.0
        score -= 0.20 if not execution_plan else 0.0
        score -= 0.20 if not decisions else 0.0
        score -= min(len(missing_deadlines) * 0.05, 0.20)
        score -= min(len(missing_owners) * 0.05, 0.20)
        score -= 0.05 if risks and not follow_ups else 0.0
        score = round(max(score, 0.0), 2)

        payload = {
            "quality_score": score,
            "gaps": gaps,
            "coverage": {
                "speaker_turns": len(context.records),
                "action_items": len(execution_plan),
                "decisions": len(decisions),
                "risks_or_dependencies": len(risks),
                "follow_up_questions": len(follow_ups),
            },
        }

        return self.result(
            payload=payload,
            confidence=0.9,
            evidence_count=len(context.records),
        )


class AgenticMeetingOrchestrator:
    """
    Coordinates specialist agents over the standard meeting-intelligence output.
    """

    def __init__(self, agents: Optional[Sequence[MeetingAgent]] = None) -> None:
        self.agents = list(
            agents
            or [
                ParticipantProfilerAgent(),
                ActionPlanningAgent(),
                DecisionRegisterAgent(),
                RiskDependencyAgent(),
                QueryAnswerAgent(),
                FollowUpQuestionAgent(),
                QualityGateAgent(),
            ]
        )

    def analyze(self, transcript: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        artifacts = analyze_transcript(transcript=transcript, query=query, top_k=top_k)
        context = AgentContext(
            transcript=transcript,
            query=query,
            records=artifacts["records"],  # type: ignore[arg-type]
            baseline=artifacts["output"],  # type: ignore[arg-type]
            retrieval_backend=str(artifacts["retrieval_backend"]),
        )

        shared: Dict[str, Any] = {}
        trace = []

        for agent in self.agents:
            try:
                result = agent.run(context, shared)
            except Exception as error:
                result = AgentResult(
                    key=agent.key,
                    agent=agent.name,
                    role=agent.role,
                    payload={"error": str(error)},
                    confidence=0.0,
                    evidence_count=0,
                    status="failed",
                )

            shared[result.key] = result.payload
            trace.append(
                {
                    "agent": result.agent,
                    "key": result.key,
                    "role": result.role,
                    "status": result.status,
                    "confidence": result.confidence,
                    "evidence_count": result.evidence_count,
                }
            )

        output = dict(context.baseline)
        output["agentic"] = {
            "mode": "deterministic_multi_agent",
            "retrieval_backend": context.retrieval_backend,
            "participants": shared.get("participants", []),
            "answer": shared.get("answer", {}),
            "execution_plan": shared.get("execution_plan", []),
            "decision_register": shared.get("decision_register", []),
            "risks": shared.get("risks", []),
            "follow_up_questions": shared.get("follow_up_questions", []),
            "quality": shared.get("quality", {}),
            "agent_trace": trace,
        }

        return output


def run_agentic_analysis(transcript: str, query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Public entry point for agentic meeting analysis.
    """
    return AgenticMeetingOrchestrator().analyze(
        transcript=transcript,
        query=query,
        top_k=top_k,
    )


def _extract_owner(item: Record) -> Optional[str]:
    speaker = item["speaker"]
    text = item["text"].strip()
    lowered = text.lower()

    if re.search(r"\bi will\b|\bi'll\b", lowered):
        return speaker

    if re.search(
        r"\bwe should\b|\bwe need to\b|\bwe plan to\b|"
        r"\bwe are going to\b|\bwe're going to\b|\blet's\b|\blets\b",
        lowered,
    ):
        return "Team"

    match = re.search(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+"
        r"(?:will|should|needs to|need to|is going to|plans to|owns)\b",
        text,
    )
    if match:
        return match.group(1)

    return speaker or None


def _extract_due_date(text: str) -> Optional[str]:
    day = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    patterns = [
        rf"\bby\s+({day}|today|tomorrow|eod|end of day|next week|next {day})\b",
        rf"\bbefore\s+({day}|today|tomorrow|eod|end of day|next week|next {day})\b",
        rf"\bon\s+({day}|[A-Z][a-z]+\s+\d{{1,2}})\b",
        rf"\bfor\s+({day}|today|tomorrow|eod|end of day|next week|next {day})\b",
        rf"\b(today|tomorrow|eod|end of day|next week|next {day})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def _infer_priority(text: str, due_date: Optional[str]) -> str:
    lowered = text.lower()
    if re.search(r"\burgent\b|\bcritical\b|\basap\b|\bblocked\b|\bmust\b", lowered):
        return "high"
    if due_date:
        return "medium"
    return "normal"


def _clean_task_text(text: str) -> str:
    cleaned = re.sub(
        r"^\s*(i will|i'll|we should|we need to|we plan to|we are going to|"
        r"we're going to|let's|lets|need to|plan to|going to)\s+",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    return cleaned[:1].upper() + cleaned[1:] if cleaned else text


def _decision_status(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\bapproved\b|\bfinalized\b|\bconfirmed\b|\bagreed\b|\bdecided\b", lowered):
        return "confirmed"
    return "proposed"


def _topic_for_text(topics: Sequence[Dict[str, Any]], text: str) -> str:
    for topic in topics:
        for item in topic.get("items", []):
            if item.get("text") == text:
                return str(topic.get("topic", "general"))
    return "general"


def _related_texts(text: str, records: Sequence[Record]) -> List[str]:
    source_terms = _significant_terms(text)
    related = []

    for record in records:
        item_terms = _significant_terms(record["text"])
        if len(source_terms.intersection(item_terms)) >= 2:
            related.append(record["text"])

    return related[:3]


def _significant_terms(text: str) -> Set[str]:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return {token for token in tokens if len(token) > 3 and token not in STOP_WORDS}


def _has_risk_signal(text: str) -> bool:
    lowered = text.lower()
    patterns = [
        r"\brisk\b",
        r"\bconcern\b",
        r"\bunclear\b",
        r"\bissue\b",
        r"\bproblem\b",
        r"\bdelay\b",
        r"\bblocked\b",
        r"\bblocker\b",
        r"\bcannot\b",
        r"\bcan't\b",
        r"\bdependency\b",
        r"\bdepends on\b",
        r"\bwaiting on\b",
        r"\bnot ready\b",
        r"\bmissing\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def _risk_type(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\bblocked\b|\bblocker\b|\bcannot\b|\bcan't\b", lowered):
        return "blocker"
    if re.search(r"\bdependency\b|\bdepends on\b|\bwaiting on\b", lowered):
        return "dependency"
    if re.search(r"\bunclear\b|\bquestion\b", lowered):
        return "open_question"
    return "risk"


def _risk_severity(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\bcritical\b|\bblocked\b|\bblocker\b|\bcannot\b|\bcan't\b|\bfailed\b", lowered):
        return "high"
    if re.search(r"\bdelay\b|\bdependency\b|\bwaiting on\b|\bconcern\b", lowered):
        return "medium"
    return "low"


def _select_answer_citations(
    baseline: Dict[str, Any],
    intent: str,
    query: str,
) -> List[Dict[str, Any]]:
    if intent == "decision" and baseline.get("decisions"):
        return _rank_structured_citations(
            items=baseline["decisions"],
            query=query,
            source="decision",
            base_score=0.86,
        )

    if intent == "action" and baseline.get("action_items"):
        return _rank_structured_citations(
            items=baseline["action_items"],
            query=query,
            source="action_item",
            base_score=0.82,
        )

    return [
        {
            "speaker": item["speaker"],
            "text": item["text"],
            "score": item["score"],
            "source": "search_result",
        }
        for item in baseline.get("search_results", [])[:3]
    ]


def _rank_structured_citations(
    items: Sequence[Record],
    query: str,
    source: str,
    base_score: float,
) -> List[Dict[str, Any]]:
    query_terms = _significant_terms(query)
    scored_items = []

    for item in items:
        item_terms = _significant_terms(item["text"])
        overlap = len(query_terms.intersection(item_terms))
        score = base_score + min(overlap * 0.04, 0.12)

        scored_items.append(
            {
                "speaker": item["speaker"],
                "text": item["text"],
                "score": round(score, 2),
                "source": source,
                "overlap": overlap,
            }
        )

    if query_terms and any(item["overlap"] > 0 for item in scored_items):
        scored_items = [item for item in scored_items if item["overlap"] > 0]

    scored_items.sort(key=lambda item: item["score"], reverse=True)

    for item in scored_items:
        item.pop("overlap")

    return scored_items[:3]


def _average(values: Sequence[float], default: float) -> float:
    if not values:
        return default
    return round(sum(values) / len(values), 2)
