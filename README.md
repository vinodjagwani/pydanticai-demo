# PydanticAI Support Agent — Demo

A ~90-line, fully-typed customer-support agent built with [PydanticAI](https://pydantic.dev/docs/ai/overview/).
It demonstrates the four core ideas of the framework:

| Idea | Where |
|------|-------|
| **Agent** — the typed agent loop | `agent = Agent(...)` |
| **Dependency injection** — pass a DB/services in, typed | `SupportDeps`, `RunContext` |
| **Tools** — plain Python the model can call | `@agent.tool customer_balance` |
| **Structured output** — validated Pydantic model, not a string | `SupportReply` |

## Run it (no API key needed)

```bash
pip install -r requirements.txt
python support_agent.py
```

This uses PydanticAI's built-in `TestModel`, which drives the *entire* agent
loop — calling every tool and returning a schema-valid `SupportReply` — with
zero API calls and zero cost.

## Run it against a real model

```bash
# Windows
set OPENAI_API_KEY=sk-...
set USE_REAL_MODEL=1
python support_agent.py

# macOS / Linux
export OPENAI_API_KEY=sk-...
USE_REAL_MODEL=1 python support_agent.py
```

Swap providers by changing one string in `support_agent.py`
(`"openai:gpt-4o"` → `"anthropic:claude-..."`, `"google:..."`, `"ollama:..."`).

## Test it

```bash
pip install pytest
pytest
```

The tests run the full agent loop deterministically — the payoff of
dependency injection: swap the real model for `TestModel`, keep everything else.
