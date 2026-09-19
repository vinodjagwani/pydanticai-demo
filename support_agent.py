"""
PydanticAI demo: a typed customer-support agent.

Shows the four ideas that make PydanticAI worth using:
  1. Agent          - the typed agent loop
  2. Dependency injection - pass a DB/services into tools with full typing
  3. Tools          - plain Python functions the model can call
  4. Structured output    - the model must return a validated Pydantic model

Run it two ways:

  # No API key needed - uses the built-in 'test' model:
  python support_agent.py

  # Real model (set your key first):
  #   set OPENAI_API_KEY=sk-...        (Windows)
  #   export OPENAI_API_KEY=sk-...     (macOS/Linux)
  USE_REAL_MODEL=1 python support_agent.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel


# --------------------------------------------------------------------------
# 1. A fake "database". In a real app this is Postgres, an HTTP client, etc.
#    The point: the agent never imports it directly - it arrives via DI.
# --------------------------------------------------------------------------
class FakeDB:
    _customers = {
        123: {"name": "Ada Lovelace", "balance": 1240.55, "pending": -75.00},
        456: {"name": "Alan Turing", "balance": 42.00, "pending": 0.0},
    }

    async def customer_name(self, customer_id: int) -> str:
        return self._customers[customer_id]["name"]

    async def balance(self, customer_id: int, include_pending: bool) -> float:
        row = self._customers[customer_id]
        total = row["balance"]
        if include_pending:
            total += row["pending"]
        return round(total, 2)


# --------------------------------------------------------------------------
# 2. Dependencies: whatever the agent's tools/instructions need at runtime.
#    Typed - so ctx.deps.db autocompletes and type-checks.
# --------------------------------------------------------------------------
@dataclass
class SupportDeps:
    customer_id: int
    db: FakeDB


# --------------------------------------------------------------------------
# 3. Structured output: the model MUST return this shape.
#    Invalid output -> PydanticAI auto-asks the model to fix it.
# --------------------------------------------------------------------------
class SupportReply(BaseModel):
    reply: str = Field(description="Message shown to the customer.")
    needs_human: bool = Field(description="True if a human agent should take over.")
    risk: int = Field(ge=0, le=10, description="Risk of the request, 0=safe 10=block card now.")


# --------------------------------------------------------------------------
# 4. The Agent. Generic over [SupportDeps, SupportReply] => fully typed.
#    Swap providers by changing the model string, e.g. 'anthropic:claude-...'.
# --------------------------------------------------------------------------
agent = Agent(
    "openai:gpt-4o",
    deps_type=SupportDeps,
    output_type=SupportReply,
    instructions=(
        "You are a support agent at a bank. Be concise and polite. "
        "Judge the risk of every request. Set needs_human when unsure."
    ),
)


# Dynamic instructions - runs each call, can use injected deps.
@agent.instructions
async def add_customer_name(ctx: RunContext[SupportDeps]) -> str:
    name = await ctx.deps.db.customer_name(ctx.deps.customer_id)
    return f"You are speaking with {name}."


# A tool - the model decides when to call it; args are validated first.
@agent.tool
async def customer_balance(ctx: RunContext[SupportDeps], include_pending: bool) -> float:
    """Return the customer's current account balance."""
    return await ctx.deps.db.balance(ctx.deps.customer_id, include_pending)


def main() -> None:
    deps = SupportDeps(customer_id=123, db=FakeDB())

    if os.getenv("USE_REAL_MODEL"):
        result = agent.run_sync("What is my balance, including pending charges?", deps=deps)
    else:
        with agent.override(model=TestModel()):
            result = agent.run_sync("What is my balance, including pending charges?", deps=deps)

    out = result.output
    print("reply      :", out.reply)
    print("needs_human:", out.needs_human)
    print("risk       :", out.risk)
    print("type       :", type(out).__name__)  # SupportReply - guaranteed


if __name__ == "__main__":
    main()
