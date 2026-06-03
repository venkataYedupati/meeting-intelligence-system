from src.agentic import run_agentic_analysis


def test_agentic_analysis_returns_specialist_outputs() -> None:
    transcript = (
        "John: We should confirm pricing with finance.\n"
        "Sarah: I will send the launch deck by Friday.\n"
        "Mike: We decided to move the product demo to Thursday.\n"
        "Priya: The API dependency is blocked until the vendor replies."
    )

    output = run_agentic_analysis(
        transcript=transcript,
        query="What was decided about the demo?",
        top_k=3,
    )

    assert "agentic" in output
    agentic = output["agentic"]

    assert agentic["execution_plan"]
    assert agentic["decision_register"]
    assert agentic["risks"]
    assert agentic["follow_up_questions"]
    assert agentic["quality"]["coverage"]["speaker_turns"] == 4
    assert len(agentic["agent_trace"]) == 7

    owned_action = next(
        item for item in agentic["execution_plan"] if item["owner"] == "Sarah"
    )
    assert owned_action["due_date"].lower() == "friday"

    assert agentic["decision_register"][0]["status"] == "confirmed"
    assert agentic["risks"][0]["severity"] == "high"
