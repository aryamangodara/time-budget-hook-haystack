# SPDX-FileCopyrightText: 2026-present Aryaman Godara
#
# SPDX-License-Identifier: Apache-2.0

# Runs offline: MockChatGenerator keeps asking for a slow tool, so the time budget is what ends the run.

import time

from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool

from haystack_integrations.hooks.time_budget import TimeBudgetHook


def slow_search(query: str) -> str:
    time.sleep(1)
    return f"results for {query}"


search = Tool(
    name="search",
    description="Search the web.",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    function=slow_search,
)

agent = Agent(
    chat_generator=MockChatGenerator(
        responses=ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="search", arguments={"query": "haystack"})])
    ),
    tools=[search],
    hooks={"before_llm": [TimeBudgetHook(max_seconds=2.5, add_final_message=True)]},
)

result = agent.run(messages=[ChatMessage.from_user("Research Haystack")])

print(result["exit_reason"])
# >> time_budget_exceeded
print(result["step_count"])
# >> 3
print(result["last_message"].text)
# >> The Agent stopped because the time budget was exceeded.
