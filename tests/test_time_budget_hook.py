# SPDX-FileCopyrightText: 2026-present Aryaman Godara
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from haystack.components.agents import Agent
from haystack.components.agents.state import State, replace_values
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool

from haystack_integrations.hooks.time_budget import EXIT_REASON, TimeBudgetHook
from haystack_integrations.hooks.time_budget import hook as hook_module


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(hook_module, "time", fake)
    return fake


def _make_state() -> State:
    return State(schema={"stop_run": {"type": str, "handler": replace_values}})


def _make_agent(clock: FakeClock, hook: TimeBudgetHook) -> Agent:
    def slow_search(query: str) -> str:
        clock.now += 2
        return f"results for {query}"

    search = Tool(
        name="search",
        description="Search the web.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        function=slow_search,
    )
    # The model keeps calling the tool, so only the time budget can end the run.
    generator = MockChatGenerator(
        responses=ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="search", arguments={"query": "q"})])
    )
    return Agent(chat_generator=generator, tools=[search], hooks={"before_llm": [hook]})


class TestTimeBudgetHook:
    def test_init_rejects_non_positive_budget(self):
        with pytest.raises(ValueError, match=r"`max_seconds` must be a positive number of seconds, got 0\."):
            TimeBudgetHook(max_seconds=0)

    def test_run_does_not_stop_within_budget(self, clock):
        hook = TimeBudgetHook(max_seconds=10)
        state = _make_state()

        hook.run(state)
        clock.now = 9.9
        hook.run(state)

        assert state.data.get("stop_run") is None

    def test_run_stops_once_budget_is_used_up(self, clock):
        hook = TimeBudgetHook(max_seconds=10, add_final_message=True)
        state = _make_state()

        hook.run(state)
        clock.now = 10
        hook.run(state)

        assert state.data["stop_run"] == EXIT_REASON
        assert state.data["messages"][-1].text == "The Agent stopped because the time budget was exceeded."

    def test_each_run_gets_its_own_clock(self, clock):
        hook = TimeBudgetHook(max_seconds=10)
        first_run, second_run = _make_state(), _make_state()

        hook.run(first_run)
        clock.now = 8
        hook.run(second_run)
        clock.now = 12
        hook.run(first_run)
        hook.run(second_run)

        assert first_run.data["stop_run"] == EXIT_REASON
        assert second_run.data.get("stop_run") is None

    def test_serialization_round_trip(self):
        hook = TimeBudgetHook(max_seconds=30, add_final_message=True)

        restored = TimeBudgetHook.from_dict(hook.to_dict())

        assert restored.max_seconds == 30
        assert restored.add_final_message is True

    def test_agent_serialization_round_trip(self):
        agent = Agent(chat_generator=MockChatGenerator(), hooks={"before_llm": [TimeBudgetHook(max_seconds=30)]})

        restored = Agent.from_dict(agent.to_dict())

        (restored_hook,) = restored.hooks["before_llm"]
        assert isinstance(restored_hook, TimeBudgetHook)
        assert restored_hook.max_seconds == 30

    def test_agent_run_stops_when_time_budget_is_exceeded(self, clock):
        agent = _make_agent(clock, TimeBudgetHook(max_seconds=5))

        result = agent.run(messages=[ChatMessage.from_user("Research this")])

        # Each tool call takes 2s: the checks before steps 2 and 3 see 2s and 4s, the one before step 4 sees 6s.
        assert result["exit_reason"] == EXIT_REASON
        assert result["step_count"] == 3

    def test_agent_run_async_stops_when_time_budget_is_exceeded(self, clock):
        agent = _make_agent(clock, TimeBudgetHook(max_seconds=5))

        result = asyncio.run(agent.run_async(messages=[ChatMessage.from_user("Research this")]))

        assert result["exit_reason"] == EXIT_REASON
        assert result["step_count"] == 3
