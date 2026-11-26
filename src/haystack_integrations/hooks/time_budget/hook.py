# SPDX-FileCopyrightText: 2026-present Aryaman Godara
#
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Any
from weakref import WeakKeyDictionary

from haystack import logging
from haystack.components.agents.state import State
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage

logger = logging.getLogger(__name__)

EXIT_REASON = "time_budget_exceeded"
_FINAL_MESSAGE_TEXT = "The Agent stopped because the time budget was exceeded."


class TimeBudgetHook:
    """
    Stop an Agent run once it has been running for a configured number of seconds.

    The hook runs at the `before_llm` hook point. The first time it sees a run, it records the start time. Before
    every later LLM call, it checks how much time has passed, and once the budget is used up it ends the run with the
    exit reason `"time_budget_exceeded"`.

    The check runs between steps, so the hook never interrupts an LLM call or a tool call that is already running.
    A run can therefore go over the budget by up to one step.

    To also count the time spent in your `before_run` hooks, register the same hook instance under `before_run`
    too, so the clock starts there.

    A single hook instance can be shared by concurrent Agent runs: each run gets its own start time.

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator

    from haystack_integrations.hooks.time_budget import TimeBudgetHook

    agent = Agent(
        chat_generator=OpenAIChatGenerator(),
        tools=[web_search],
        hooks={"before_llm": [TimeBudgetHook(max_seconds=60)]},
    )

    result = agent.run(messages=[...])
    result["exit_reason"]
    # >> "time_budget_exceeded" if the run took longer than 60 seconds
    ```
    """

    allowed_hook_points = ("before_run", "before_llm")

    def __init__(self, *, max_seconds: float, add_final_message: bool = False) -> None:
        """
        Create a time budget hook.

        :param max_seconds: Maximum wall-clock time, in seconds, an Agent run may take before it is stopped.
        :param add_final_message: Whether to append an assistant message explaining why the Agent stopped.
        :raises ValueError: If `max_seconds` is not a positive number.
        """
        if max_seconds <= 0:
            msg = f"`max_seconds` must be a positive number of seconds, got {max_seconds}."
            raise ValueError(msg)
        self.max_seconds = max_seconds
        self.add_final_message = add_final_message
        # Keyed by the run's State object, so concurrent runs sharing this hook keep separate clocks, and entries go
        # away when a run's State is garbage collected.
        self._start_times: WeakKeyDictionary[State, float] = WeakKeyDictionary()

    def run(self, state: State) -> None:
        """
        Start the clock on the first call for a run, and stop the Agent once the time budget is used up.

        :param state: Agent state of the current run.
        """
        now = time.monotonic()
        start = self._start_times.setdefault(state, now)
        elapsed = now - start
        if elapsed < self.max_seconds:
            return
        logger.warning(
            "Agent reached its time budget of {max_seconds}s ({elapsed:.1f}s elapsed); requesting a stop.",
            max_seconds=self.max_seconds,
            elapsed=elapsed,
        )
        state.set("stop_run", EXIT_REASON)
        if self.add_final_message:
            state.set("messages", [ChatMessage.from_assistant(_FINAL_MESSAGE_TEXT)])

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this hook to a dictionary.

        :returns: Serialized representation of the hook.
        """
        return default_to_dict(self, max_seconds=self.max_seconds, add_final_message=self.add_final_message)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimeBudgetHook":
        """
        Create a hook from its serialized representation.

        :param data: Serialized hook data.
        :returns: The deserialized hook.
        """
        return default_from_dict(cls, data)
