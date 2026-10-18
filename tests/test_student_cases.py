"""Student-authored scenarios for the Atlas agent.

Every test uses ScriptedLLM, so the model's decisions are fixed in advance and
each test checks how the application handles them.
"""

from atlas_agent.agent import run_agent
from atlas_agent.llm import ScriptedLLM
from atlas_agent.models import LabRequest

MORNING = {"start": "2026-10-15T10:00:00", "end": "2026-10-15T11:00:00"}


def tool(name, arguments):
    return {
        "kind": "tool_call",
        "tool": name,
        "arguments": arguments,
        "reasoning_summary": "Scripted step.",
    }


def final(status, message, citations=None, completed_actions=None):
    return {
        "kind": "final",
        "response": {
            "status": status,
            "message": message,
            "citations": citations or [],
            "completed_actions": completed_actions or [],
        },
    }


def reservation_count(environment):
    return len(environment.simulator.state["reservations"])


# Required case 1: an unsafe action proposal is denied before tool execution.
def test_gate_denies_reservation_for_unqualified_user_without_touching_lab(environment):
    # Bob has lab_safety but not laser_safety, so he may not book the laser cutter,
    # even after every lookup has been done.
    request = LabRequest(
        authenticated_user_id="bob",
        text="Reserve laser-01 for polymer-study from 10:00 to 11:00.",
        project_id="polymer-study",
        resource_id="laser-01",
        **MORNING,
    )
    llm = ScriptedLLM(
        [
            tool("search_lab_documents", {"query": "laser cutter training requirement"}),
            tool("get_user_authorizations", {"user_id": "bob"}),
            tool("get_resource_status", {"resource_id": "laser-01", **MORNING}),
            tool(
                "reserve_resource",
                {
                    "user_id": "bob",
                    "project_id": "polymer-study",
                    "resource_id": "laser-01",
                    **MORNING,
                },
            ),
            final("blocked", "Bob is missing the laser_safety qualification."),
        ]
    )
    before = reservation_count(environment)

    run = run_agent(request, llm=llm, environment=environment)

    denied = run.trace[-1]
    assert denied.decision["tool"] == "reserve_resource"
    assert denied.mutating is True
    assert denied.observation.ok is False
    assert denied.observation.error_code == "action_gate_denied"
    # The simulator never saw the reservation.
    assert reservation_count(environment) == before
    assert run.response.status == "blocked"
    assert run.termination_reason == "model_finished"


# Required case 2: a contradictory final response is rejected because it
# disagrees with the trace.
def test_final_response_claiming_denied_reservation_succeeded_is_rejected(environment):
    # The model skips every check, gets denied, then claims it booked anyway.
    request = LabRequest(
        authenticated_user_id="alice",
        text="Reserve tensile-01 for polymer-study from 10:00 to 11:00.",
        project_id="polymer-study",
        resource_id="tensile-01",
        **MORNING,
    )
    llm = ScriptedLLM(
        [
            tool(
                "reserve_resource",
                {
                    "user_id": "alice",
                    "project_id": "polymer-study",
                    "resource_id": "tensile-01",
                    **MORNING,
                },
            ),
            final(
                "completed",
                "Your reservation is confirmed.",
                completed_actions=["reserve_resource"],
            ),
        ]
    )
    before = reservation_count(environment)

    run = run_agent(request, llm=llm, environment=environment)

    assert run.trace[0].observation.error_code == "action_gate_denied"
    assert run.response.status == "failed"
    assert run.termination_reason == "invalid_final_response"
    assert "completed_without_successful_mutation" in run.response.message
    assert "completed_action_not_in_successful_trace" in run.response.message
    assert reservation_count(environment) == before


# Additional case 3: the model tries to book for a different user.
def test_gate_denies_identity_substitution_even_after_checks(environment):
    # Alice is logged in, but the model books for Bob instead.
    request = LabRequest(
        authenticated_user_id="alice",
        text="Reserve tensile-01 for polymer-study from 10:00 to 11:00.",
        project_id="polymer-study",
        resource_id="tensile-01",
        **MORNING,
    )
    llm = ScriptedLLM(
        [
            tool("search_lab_documents", {"query": "tensile tester reservation policy"}),
            tool("get_user_authorizations", {"user_id": "alice"}),
            tool("get_resource_status", {"resource_id": "tensile-01", **MORNING}),
            tool(
                "reserve_resource",
                {
                    "user_id": "bob",
                    "project_id": "polymer-study",
                    "resource_id": "tensile-01",
                    **MORNING,
                },
            ),
            final("blocked", "I could not make that reservation."),
        ]
    )
    before = reservation_count(environment)

    run = run_agent(request, llm=llm, environment=environment)

    denied = run.trace[-1]
    assert denied.observation.error_code == "action_gate_denied"
    assert "user_id_must_match_authenticated_user" in denied.observation.message
    assert reservation_count(environment) == before
    assert run.response.status == "blocked"


# Additional case 4: a citation that was never retrieved is rejected.
def test_final_response_with_made_up_citation_is_rejected(environment):
    request = LabRequest(
        authenticated_user_id="alice",
        text="What training do I need for the tensile tester?",
    )
    llm = ScriptedLLM(
        [
            tool("search_lab_documents", {"query": "tensile tester training"}),
            final(
                "blocked",
                "Policy FAKE-999 says you need special training.",
                citations=["FAKE-999#invented"],
            ),
        ]
    )

    run = run_agent(request, llm=llm, environment=environment)

    assert run.trace[0].observation.ok is True
    assert run.response.status == "failed"
    assert run.termination_reason == "invalid_final_response"
    assert "citations_not_retrieved" in run.response.message


# Additional case 5: a reservation that really happened cannot be hidden.
def test_final_response_hiding_a_successful_reservation_is_rejected(environment):
    request = LabRequest(
        authenticated_user_id="alice",
        text="Reserve tensile-01 for polymer-study from 10:00 to 11:00.",
        project_id="polymer-study",
        resource_id="tensile-01",
        **MORNING,
    )
    llm = ScriptedLLM(
        [
            tool(
                "search_lab_documents",
                {"query": "tensile tester lab_safety authorization calibration"},
            ),
            tool("get_user_authorizations", {"user_id": "alice"}),
            tool("get_resource_status", {"resource_id": "tensile-01", **MORNING}),
            tool(
                "reserve_resource",
                {
                    "user_id": "alice",
                    "project_id": "polymer-study",
                    "resource_id": "tensile-01",
                    **MORNING,
                },
            ),
            final("blocked", "Sorry, I could not reserve the tensile tester."),
        ]
    )
    before = reservation_count(environment)

    run = run_agent(request, llm=llm, environment=environment)

    # The reservation really went through...
    assert run.trace[-1].observation.ok is True
    assert reservation_count(environment) == before + 1
    # ...so a "blocked" answer that hides it must be rejected.
    assert run.response.status == "failed"
    assert run.termination_reason == "invalid_final_response"
