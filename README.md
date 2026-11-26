# time-budget-hook-haystack

A [Haystack](https://haystack.deepset.ai/) Agent hook that stops a run once it has been running for a set number of seconds.

`max_agent_steps` caps how many steps an Agent takes, and Haystack's `TokenBudgetHook` caps how many tokens it spends. Neither one limits how long a run takes, and that's usually the limit that matters when an Agent sits behind an API with a request timeout, or calls slow tools. `TimeBudgetHook` adds that limit.

## Installation

```bash
pip install time-budget-hook-haystack
```

Requires `haystack-ai>=3.2.0`.

## Usage

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator

from haystack_integrations.hooks.time_budget import TimeBudgetHook

agent = Agent(
    chat_generator=OpenAIChatGenerator(),  # needs OPENAI_API_KEY
    tools=[web_search],
    hooks={"before_llm": [TimeBudgetHook(max_seconds=60)]},
)

result = agent.run(messages=[...])
result["exit_reason"]
# >> "time_budget_exceeded" if the run took longer than 60 seconds
```

For a version that runs offline, with no API key, see [`examples/agent_with_time_budget.py`](examples/agent_with_time_budget.py).

### Parameters

| Parameter | Description |
|---|---|
| `max_seconds` | Maximum wall-clock time, in seconds, a run may take before it's stopped. Must be positive. |
| `add_final_message` | If `True`, adds an assistant message saying the Agent stopped because the time budget was exceeded. Defaults to `False`. |

## How it works

- The hook runs at the `before_llm` hook point. The first time it sees a run, it starts that run's clock.
- Before each later LLM call, it checks the elapsed time. Once the budget is used up, it sets the Agent's `stop_run` state key, and the run ends with `exit_reason == "time_budget_exceeded"` (also exported as `EXIT_REASON`).
- The check happens between steps, so an LLM call or tool call that's already running is never cut off. A run can go over the budget by up to one step.
- To also count time spent in your `before_run` hooks, register the same hook instance under `before_run` as well. The clock then starts there.
- One hook instance can be shared by concurrent runs, sync or async. Each run gets its own clock.
- The hook serializes with `Agent.to_dict()` / `Agent.from_dict()` like Haystack's built-in hooks.

Combine it with other budgets as needed:

```python
from haystack.hooks.budget import TokenBudgetHook

hooks = {"before_llm": [TimeBudgetHook(max_seconds=60), TokenBudgetHook(max_total_tokens=100_000)]}
```

## Development

This project uses [Hatch](https://hatch.pypa.io/).

```bash
hatch run test:all   # run the tests
hatch run fmt        # format and lint
```

## Publishing

Pushing a `v*` tag (for example `v0.1.0`) runs `.github/workflows/release.yml`, which publishes to PyPI. It needs a `PYPI_API_TOKEN` repository secret.

## License

`Apache-2.0`. See [LICENSE](LICENSE).
