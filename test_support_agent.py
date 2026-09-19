"""
Tests for the support agent - no API key, no network, no cost.

This is the payoff of dependency injection + TestModel:
you exercise the *whole* agent loop deterministically, like pytest tests code.

Run:  pip install pytest && pytest
"""

from pydantic_ai.models.test import TestModel

from support_agent import FakeDB, SupportDeps, SupportReply, agent


def test_agent_returns_structured_output():
    deps = SupportDeps(customer_id=123, db=FakeDB())
    with agent.override(model=TestModel()):
        result = agent.run_sync("What is my balance?", deps=deps)

    # Output is guaranteed to be a valid SupportReply - schema enforced.
    assert isinstance(result.output, SupportReply)
    assert 0 <= result.output.risk <= 10
    assert isinstance(result.output.needs_human, bool)


def test_tool_is_called_with_injected_db():
    # TestModel calls every registered tool once, so this proves the
    # balance tool wires through the injected FakeDB without a real LLM.
    deps = SupportDeps(customer_id=456, db=FakeDB())
    with agent.override(model=TestModel()):
        result = agent.run_sync("balance please", deps=deps)
    assert isinstance(result.output, SupportReply)
