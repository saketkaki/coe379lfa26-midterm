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
        "reasoning_summary": "Obtain required evidence before acting.",
    }


# Test case 1: an unsafe action proposal that is denied before tool execution
# The model books a reservation for bob, while alice is the user logged in.
# The action gate should block the reservation and return a blocked final response.
def test_reservation_from_another_user_triggers_blocked_reservation(environment):
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
                "message": "The reservation can't be booked because there is a mismatch between the authenticated and requested user IDs.",
                "citations": [],
                "completed_actions": [],
            },
        },
    ]
    run = run_agent(REQUEST, llm=ScriptedLLM(decisions), environment=environment)
    assert run.response.status == "blocked"
    assert run.trace[-1].observation.error_code == "action_gate_denied"
    assert "user_id_must_match_authenticated_user" in run.trace[-1].observation.message
    assert len(environment.simulator.state["reservations"]) == 1


# Test case 2: a contradictory final response that is rejected because it disagrees with the trace
# Bob tries to book after hours without approval.
# This results in a booking reservation being rejected but the model says that it was successful.
def test_completed_claim_for_after_hours_booking_without_approval_is_rejected(
    environment,
):
    request = LabRequest(
        authenticated_user_id="bob",
        text="Reserve tensile-01 for polymer-study from 2026-10-15 19:00 to 20:00.",
        project_id="polymer-study",
        resource_id="tensile-01",
        start="2026-10-15T19:00:00",
        end="2026-10-15T20:00:00",
    )
    interval = {"start": "2026-10-15T19:00:00", "end": "2026-10-15T20:00:00"}
    decisions = [
        tool("search_lab_documents", {"query": "after hours approval"}),
        tool("get_user_authorizations", {"user_id": "bob"}),
        tool("get_resource_status", {"resource_id": "tensile-01", **interval}),
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
                "status": "completed",
                "message": "Bob's reservation of after hours is confirmed.",
                "citations": [],
                "completed_actions": ["reserve_resource"],
            },
        },
    ]
    run = run_agent(request, llm=ScriptedLLM(decisions), environment=environment)
    assert run.trace[-1].observation.error_code == "action_gate_denied"
    assert len(environment.simulator.state["reservations"]) == 1
    assert run.response.status == "failed"
    assert run.termination_reason == "invalid_final_response"
    assert "completed_without_successful_mutation" in run.response.message


# Test case 3: one additional case of your choice - missing qualification
def test_reservation_without_qualification_is_rejected(environment):
    request = LabRequest(
        authenticated_user_id="carol",
        text="Reserve tensile-01 for robotics-demo from 2026-10-15 10:00 to 11:00.",
        project_id="robotics-demo",
        resource_id="tensile-01",
        start="2026-10-15T10:00:00",
        end="2026-10-15T11:00:00",
    )
    interval = {"start": "2026-10-15T10:00:00", "end": "2026-10-15T11:00:00"}
    decisions = [
        tool("search_lab_documents", {"query": "tensile tester training"}),
        tool("get_user_authorizations", {"user_id": "carol"}),
        tool("get_resource_status", {"resource_id": "tensile-01", **interval}),
        tool(
            "reserve_resource",
            {
                "user_id": "carol",
                "project_id": "robotics-demo",
                "resource_id": "tensile-01",
                **interval,
            },
        ),
        {
            "kind": "final",
            "response": {
                "status": "blocked",
                "message": "Carol doesn't have the right qualifications that are needed for tensile testing.",
                "citations": [],
                "completed_actions": [],
            },
        },
    ]
    run = run_agent(request, llm=ScriptedLLM(decisions), environment=environment)
    assert run.trace[-1].observation.error_code == "action_gate_denied"
    assert len(environment.simulator.state["reservations"]) == 1
    assert run.response.status == "blocked"
    assert "required_qualification_missing" in run.trace[-1].observation.message
