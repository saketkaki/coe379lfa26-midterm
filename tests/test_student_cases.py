from atlas_agent.agent import run_agent
from atlas_agent.agent_models import ToolCallDecision
from atlas_agent.llm import ScriptedLLM
from atlas_agent.models import AgentResponse, LabRequest, ToolObservation, TraceEvent
from atlas_agent.validation import validate_final_response, validate_request_binding


def test_action_gate_blocks_reservation_for_another_user(environment):
    request = LabRequest(
        authenticated_user_id="alice",
        text="Reserve tensile-01 for polymer-study from 2026-10-15 10:00 to 11:00.",
        project_id="polymer-study",
        resource_id="tensile-01",
        start="2026-10-15T10:00:00",
        end="2026-10-15T11:00:00",
    )
    decisions = [
        {
            "kind": "tool_call",
            "tool": "reserve_resource",
            "arguments": {
                "user_id": "bob",
                "project_id": "polymer-study",
                "resource_id": "tensile-01",
                "start": "2026-10-15T10:00:00",
                "end": "2026-10-15T11:00:00",
            },
            "reasoning_summary": "Reserve for a different user.",
        },
        {
            "kind": "final",
            "response": {
                "status": "blocked",
                "message": "I cannot reserve for another user.",
            },
        },
    ]
    run = run_agent(request, llm=ScriptedLLM(decisions), environment=environment)
    assert run.response.status == "blocked"
    assert run.trace[0].observation.error_code == "action_gate_denied"
    assert len(environment.simulator.state["reservations"]) == 1


def test_blocked_response_cannot_hide_successful_reservation():
    trace = [
        TraceEvent(
            step=1,
            decision={"kind": "tool_call", "tool": "reserve_resource", "arguments": {}},
            observation=ToolObservation(
                ok=True,
                tool="reserve_resource",
                data={"status": "confirmed"},
            ),
            mutating=True,
        )
    ]
    response = AgentResponse(
        status="blocked",
        message="The reservation could not be made.",
    )
    errors = validate_final_response(response, trace)
    assert errors


def test_request_binding_rejects_looking_up_another_users_permissions():
    request = LabRequest(
        authenticated_user_id="alice",
        text="What am I allowed to reserve?",
    )
    decision = ToolCallDecision(
        kind="tool_call",
        tool="get_user_authorizations",
        arguments={"user_id": "bob"},
        reasoning_summary="Check another user's permissions.",
    )
    errors = validate_request_binding(decision, request=request)
    assert "user_id_must_match_authenticated_user" in errors
