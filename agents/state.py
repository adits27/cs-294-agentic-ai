"""State definitions for the multi-agent validation system."""

from typing import TypedDict, List, Optional, Annotated, Dict, Any
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class ABTestContext(TypedDict):
    """
    Context information for A/B testing validation.
    """
    hypothesis: str  # The A/B test hypothesis
    success_metrics: List[str]  # Metrics to evaluate success
    dataset_path: str  # Path to the dataset
    expected_effect_size: Optional[float]  # Expected effect size for power analysis
    significance_level: Optional[float]  # Statistical significance level (default: 0.05)
    power: Optional[float]  # Statistical power (default: 0.8)


class ValidationState(TypedDict):
    """
    State schema for the validation workflow.

    This state is shared across the orchestrator and all sub-agents,
    allowing them to communicate and coordinate validation tasks.
    """
    # Core task information
    task: str  # The task or content to validate

    # A/B Testing specific context
    ab_test_context: Optional[ABTestContext]  # A/B test validation context

    # Message history for LLM interactions
    messages: Annotated[List[BaseMessage], add_messages]

    # A2A protocol messages
    a2a_messages: List[Dict[str, Any]]  # A2A messages between agents

    # Sub-agent results
    data_validation_result: Optional[dict]  # Results from data validation agent
    sub_agent_2_result: Optional[dict]  # Results from second validation agent

    # Orchestrator decisions
    current_step: str  # Current step in the workflow
    next_action: str  # Next action to take

    # Final validation result
    validation_passed: Optional[bool]
    validation_summary: Optional[str]
    recommendations: Optional[List[str]]
