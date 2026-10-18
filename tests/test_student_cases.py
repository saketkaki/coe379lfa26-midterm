from atlas_agent.agent import run_agent
from atlas_agent.llm import ScriptedLLM
from atlas_agent.models import LabRequest

REQUEST = LabRequest(
    authenticated_user_id="alice",
    text="Reserve tensile-01 for polymer-study from 2026-10-15 10:00 to 11:00.",
    project_id="polymer-study",
    resource_id="tensile-01",
    start="2026-10-15T10:00:00",
    end="2026-10-15T11:00:00",
)


def tool(tool, arguments):
    return {
        "kind": "tool_call",
        "tool": tool,
        "arguments": arguments,
        "reasoning_summary": "Scripted student test step.",
    }


def test_action_gate_denies_reservation_for_another_user(environment):
    interval = {"start": "2026-10-15T10:00:00", "end": "2026-10-15T11:00:00"}
    decisions = [
        tool(
            "reserve_resource",
            {
                "user_id": "bob",
                "project_id": "polymer-study",
                "resource_id": "tensile-01",
                **interval,
            },
        ),
        {
            "kind": "final",
            "response": {
                "status": "blocked",
                "message": "I cannot reserve for another user.",
            },
        },
    ]
    run = run_agent(REQUEST, llm=ScriptedLLM(decisions), environment=environment)
    assert run.trace[0].observation.error_code == "action_gate_denied"
    assert "user_id_must_match_authenticated_user" in run.trace[0].observation.message
    assert len(environment.simulator.state["reservations"]) == 1
    assert run.response.status == "blocked"


def test_completed_claim_after_denied_reservation_is_rejected(environment):
    interval = {"start": "2026-10-15T10:00:00", "end": "2026-10-15T11:00:00"}
    decisions = [
        tool(
            "reserve_resource",
            {
                "user_id": "alice",
                "project_id": "polymer-study",
                "resource_id": "tensile-01",
                **interval,
            },
        ),
        {
            "kind": "final",
            "response": {
                "status": "completed",
                "message": "The reservation was created.",
                "completed_actions": ["reserve_resource"],
            },
        },
    ]
    run = run_agent(REQUEST, llm=ScriptedLLM(decisions), environment=environment)
    assert run.trace[0].observation.error_code == "action_gate_denied"
    assert run.response.status == "failed"
    assert run.termination_reason == "invalid_final_response"


def test_made_up_citation_is_rejected(environment):
    decisions = [
        tool("search_lab_documents", {"query": "tensile tester training"}),
        {
            "kind": "final",
            "response": {
                "status": "blocked",
                "message": "Policy FAKE-999 does not allow this.",
                "citations": ["FAKE-999#invented"],
            },
        },
    ]
    run = run_agent(REQUEST, llm=ScriptedLLM(decisions), environment=environment)
    assert run.trace[0].observation.ok
    assert run.response.status == "failed"
    assert run.termination_reason == "invalid_final_response"
    assert "citations_not_retrieved" in run.response.message
