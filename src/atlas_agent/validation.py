"""Milestone 3: implement three selected validation responsibilities."""

from __future__ import annotations

from .agent_models import ToolCallDecision
from .models import AgentResponse, LabRequest, TraceEvent
from .policy_context import ReservationContext, build_reservation_context
from .tools import MUTATING_TOOLS

# Request fields the model must copy exactly from the trusted request.
BOUND_FIELDS = ("project_id", "resource_id", "start", "end")

# Each ReservationContext fact, paired with the error to report when it is False.
# Listed in the same order as the ReservationContext fields.
PRECONDITION_ERRORS = (
    ("arguments_complete", "reservation_arguments_incomplete"),
    ("policy_retrieved", "policy_not_retrieved"),
    ("authorizations_checked", "authorizations_not_checked"),
    ("resource_status_checked", "resource_status_not_checked"),
    ("project_authorized", "project_not_authorized"),
    ("qualification_current", "required_qualification_missing"),
    ("resource_available", "resource_not_available"),
    ("calibration_current", "calibration_not_current"),
    ("maintenance_clear", "maintenance_not_clear"),
    ("after_hours_approval_satisfied", "after_hours_approval_missing"),
)


def validate_request_binding(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
) -> list[str]:
    """TODO 3.1: Prevent identity and resolved-field substitution."""
    args = decision.arguments
    errors: list[str] = []

    # The model may never act for anyone except the logged-in user.
    if "user_id" in args and args["user_id"] != request.authenticated_user_id:
        errors.append("user_id_must_match_authenticated_user")

    for field in BOUND_FIELDS:
        if field not in args:
            continue
        trusted_value = getattr(request, field)
        if trusted_value is None:
            # The request never resolved this field, so the model guessed it.
            # Looking things up with a guess is fine; changing state is not.
            if decision.tool in MUTATING_TOOLS:
                errors.append(f"{field}_unresolved_in_request")
        elif args[field] != trusted_value:
            errors.append(f"{field}_does_not_match_request")

    return errors


def validate_reservation_preconditions(context: ReservationContext) -> list[str]:
    """TODO 3.2: Return an error for every required context fact that is false."""
    return [
        error
        for fact, error in PRECONDITION_ERRORS
        if not getattr(context, fact)
    ]


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
    errors: list[str] = []

    # 3a. Every citation must come from a successful search in this run.
    retrieved = retrieved_passage_ids(trace)
    unretrieved = [c for c in response.citations if c not in retrieved]
    if unretrieved:
        errors.append("citations_not_retrieved:" + ",".join(unretrieved))

    # 3b. Every claimed action must match a mutation that actually succeeded.
    successful_mutations = [
        event.observation.tool
        for event in trace
        if event.mutating and event.observation.ok
    ]
    if any(
        action not in successful_mutations for action in response.completed_actions
    ):
        errors.append("completed_action_not_in_successful_trace")

    if response.status == "completed":
        # 3c. "completed" needs a real success and must say what was done.
        if not successful_mutations:
            errors.append("completed_without_successful_mutation")
        if not response.completed_actions:
            errors.append("completed_without_claimed_action")
    else:
        # 3d. Any other status claims nothing and hides no successful change.
        if response.completed_actions:
            errors.append("completed_actions_with_non_completed_status")
        if successful_mutations:
            errors.append("successful_mutation_not_reported")

    return errors
