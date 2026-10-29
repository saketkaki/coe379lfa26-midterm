"""Milestone 3: implement three selected validation responsibilities."""

from __future__ import annotations
from xml.parsers.expat import errors

from .agent_models import ToolCallDecision
from .models import AgentResponse, LabRequest, TraceEvent
from .policy_context import ReservationContext, build_reservation_context


def validate_request_binding(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
) -> list[str]:
    """TODO 3.1: Prevent identity and resolved-field substitution."""
    # This is done while looking at the test cases in test_validation file
    errors: list[str] = []

    # Test case 1: test_request_binding_rejects_identity_and_project_substitution
    # Found the user_id example code from the course notes and based the rest off of that
    if "user_id" in decision.arguments and decision.arguments["user_id"] != request.authenticated_user_id:
        errors.append("user_id_must_match_authenticated_user")
    
    if "project_id" in decision.arguments and decision.arguments["project_id"] != request.project_id:
        errors.append("project_id_does_not_match_request")

    if "resource_id" in decision.arguments and decision.arguments["resource_id"] != request.resource_id:
        errors.append("resource_id_does_not_match_request")

    # Test case 2: test_request_binding_rejects_reservation_when_request_is_unresolved
    if "user_id" in decision.arguments and request.authenticated_user_id is None:
        errors.append("user_id_unresolved_in_request")

    if "project_id" in decision.arguments and request.project_id is None:
        errors.append("project_id_unresolved_in_request")

    if "resource_id" in decision.arguments and request.resource_id is None:
        errors.append("resource_id_unresolved_in_request")

    if "start" in decision.arguments and request.start is None:
        errors.append("start_unresolved_in_request")

    if "end" in decision.arguments and request.end is None:
        errors.append("end_unresolved_in_request")
    
    return errors


def validate_reservation_preconditions(context: ReservationContext) -> list[str]:
    """TODO 3.2: Return an error for every required context fact that is false."""
    errors: list[str] = []

    if not context.arguments_complete:
        errors.append("reservation_arguments_missing")

    if not context.policy_retrieved:
        errors.append("reservation_policy_not_retrieved")

    if not context.authorizations_checked:
        errors.append("reservation_authorizations_check_missing")

    if not context.resource_status_checked:
        errors.append("reservation_resource_status_check_missing")

    if not context.project_authorized:
        errors.append("reservation_project_not_authorized")

    if not context.qualification_current:
        errors.append("required_qualification_missing")

    if not context.resource_available:
        errors.append("resource_not_available")

    if not context.calibration_current:
        errors.append("calibration_not_current")

    if not context.maintenance_clear:
        errors.append("maintenance_not_clear")

    if not context.after_hours_approval_satisfied:
        errors.append("after_hours_approval_missing")

    return errors


def validate_proposed_tool_call(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
    trace: list[TraceEvent],
) -> list[str]:
    """Supplied wrapper. Do not modify this function."""
    errors = validate_request_binding(decision, request=request)
    if decision.tool == "reserve_resource":
        context = build_reservation_context(decision, trace)
        errors.extend(validate_reservation_preconditions(context))
    return errors


def retrieved_passage_ids(trace: list[TraceEvent]) -> set[str]:
    """Supplied helper. Do not modify this function."""
    identifiers: set[str] = set()
    for event in trace:
        if event.observation.tool != "search_lab_documents" or not event.observation.ok:
            continue
        identifiers.update(
            passage["passage_id"]
            for passage in event.observation.data.get("passages", [])
        )
    return identifiers


def validate_final_response(
    response: AgentResponse, trace: list[TraceEvent]
) -> list[str]:
    """TODO 3.3: Validate citations, claimed actions, and final status."""
    # This is done while looking at the test cases in test_validation file
    errors: list[str] = []

    # Part 3a: Validate citations
    for citation in response.citations:
        if citation not in retrieved_passage_ids(trace):
            # Using the error message from the test case in test_validation.py
            errors.append("citations_not_retrieved")

    # Part 3b: Validate completed actions
    successful_mutations: list[str] = []
    for event in trace:
        if event.mutating and event.observation.ok:
            successful_mutations.append(event.observation.tool)

    for entry in response.completed_actions:
        if entry not in successful_mutations:
            # Using the error message from the test case in test_validation.py
            errors.append("completed_action_not_in_successful_trace")

    # Part 3c: Validate final status
    if response.status == "completed":
        if not successful_mutations:
            errors.append("completed_without_successful_mutation")

    # Part 3d: Validate incomplete status
    if response.status != "completed":
        if response.completed_actions:
            errors.append("incomplete_with_completed_actions")
        if successful_mutations:
            errors.append("incomplete_with_successful_mutation")

    return errors