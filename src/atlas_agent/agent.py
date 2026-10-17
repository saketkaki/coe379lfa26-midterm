"""Milestone 2: complete the bounded decision–action–observation loop."""

from __future__ import annotations

import json

from pydantic import ValidationError

from .agent_models import (
    ClarificationDecision,
    DecisionAdapter,
    FinalDecision,
    ToolCallDecision,
)
from .llm import StructuredLLM
from .models import AgentResponse, AgentRun, LabRequest, ToolObservation, TraceEvent
from .prompts import initialize_messages
from .tools import MUTATING_TOOLS, ToolEnvironment
from .validation import validate_final_response, validate_proposed_tool_call


def _failed(message: str, trace: list[TraceEvent], reason: str) -> AgentRun:
    return AgentRun(
        response=AgentResponse(status="failed", message=message),
        trace=trace,
        termination_reason=reason,
    )


def run_agent(
    request: LabRequest,
    *,
    llm: StructuredLLM,
    environment: ToolEnvironment,
    max_steps: int = 10,
) -> AgentRun:
    messages = initialize_messages(request)
    trace: list[TraceEvent] = []

    for step in range(1, max_steps + 1):
        try:
            # TODO 2.1: Ask the model what it wants to do next.
            decision = llm.generate(messages=messages, response_model=DecisionAdapter)
        except (ValidationError, ValueError, TypeError, RuntimeError) as exc:
            return _failed(
                f"The model did not produce a valid decision: {exc}",
                trace,
                "invalid_model_decision",
            )

        if isinstance(decision, ClarificationDecision):
            # TODO 2.2: The model needs more info, so stop and ask the user.
            # No tool gets called here.
            question_for_user = AgentResponse(
                status="clarification_required",
                message=decision.question,
            )
            return AgentRun(
                response=question_for_user,
                trace=trace,
                termination_reason="clarification_required",
            )

        if isinstance(decision, FinalDecision):
            # TODO 2.3: The model says it's done. Before trusting its answer,
            # fact-check it against what actually happened in the trace.
            problems = validate_final_response(decision.response, trace)
            if problems:
                return _failed(
                    "The final answer didn't match what actually happened: "
                    + "; ".join(problems),
                    trace,
                    "invalid_final_response",
                )

            # Everything checks out, so hand the model's answer to the user.
            return AgentRun(
                response=decision.response,
                trace=trace,
                termination_reason="model_finished",
            )

        if not isinstance(decision, ToolCallDecision):
            return _failed(
                "Unsupported decision type.", trace, "invalid_model_decision"
            )

        # TODO 2.4: The model wants to use a tool. Check with the safety gate
        # first, and only run the tool if the gate says it's allowed.
        problems = validate_proposed_tool_call(decision, request=request, trace=trace)
        if problems:
            # Not allowed: don't run the tool, just note why it was blocked.
            result = ToolObservation(
                ok=False,
                tool=decision.tool,
                error_code="action_gate_denied",
                message="Blocked by the safety gate: " + "; ".join(problems),
            )
        else:
            # Allowed: actually run the tool and keep what it returns.
            result = environment.dispatch(decision.tool, decision.arguments)

        # TODO 2.5: Write this attempt in the trace (our logbook), whether it
        # worked, failed, or was blocked. MUTATING_TOOLS tells us if the tool
        # tries to change anything.
        trace.append(
            TraceEvent(
                step=step,
                decision=decision.model_dump(),
                observation=result,
                mutating=decision.tool in MUTATING_TOOLS,
            )
        )

        # TODO 2.6: Remind the model what it asked for and what happened,
        # so it can decide its next move.
        messages.append({"role": "assistant", "content": json.dumps(decision.model_dump())})
        messages.append({"role": "tool", "content": json.dumps(result.model_dump())})

    # TODO 2.7: We ran out of steps without the model finishing, so give up
    # with a clear failure instead of looping forever.
    return _failed(
        f"The agent exceeded its step budget of {max_steps}.",
        trace,
        "maximum_steps_exceeded",
    )
