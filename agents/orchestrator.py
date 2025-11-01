"""
Main orchestrating agent for the multi-agent validation system.

This agent coordinates two sub-agents to validate tasks, managing
the overall workflow and synthesizing results using A2A protocol.
"""

from typing import Literal, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from agents.state import ValidationState, ABTestContext
from agents.a2a_protocol import A2AProtocolHandler, MessageStatus
from agents.data_validation_agent import DataValidationAgent


class OrchestratingAgent:
    """
    Main orchestrator that coordinates sub-agents for validation tasks.

    The orchestrator:
    1. Receives a validation task
    2. Delegates to sub-agents in sequence or parallel
    3. Synthesizes their results
    4. Produces a final validation decision
    """

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        """
        Initialize the orchestrating agent.

        Args:
            model: The Gemini model to use for orchestration decisions
                  (e.g., "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro")
            temperature: Temperature for LLM responses (0.0 for deterministic)
        """
        self.agent_id = "orchestrator"
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        self.a2a_handler = A2AProtocolHandler(self.agent_id)

        # Initialize sub-agents
        self.data_validation_agent = DataValidationAgent(model=model, temperature=temperature)

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow for orchestration.

        The graph structure:
        START -> orchestrator_entry -> delegate_to_agents -> synthesize_results -> END
        """
        workflow = StateGraph(ValidationState)

        # Add nodes
        workflow.add_node("orchestrator_entry", self._orchestrator_entry)
        workflow.add_node("delegate_to_agents", self._delegate_to_agents)
        workflow.add_node("synthesize_results", self._synthesize_results)

        # Define edges
        workflow.set_entry_point("orchestrator_entry")
        workflow.add_edge("orchestrator_entry", "delegate_to_agents")
        workflow.add_edge("delegate_to_agents", "synthesize_results")
        workflow.add_edge("synthesize_results", END)

        return workflow.compile()

    def _orchestrator_entry(self, state: ValidationState) -> ValidationState:
        """
        Entry point for the orchestrator.

        Analyzes the incoming task and sets up the validation workflow.
        """
        task = state["task"]

        # Use LLM to analyze the task and plan the validation approach
        ab_context = state.get("ab_test_context")

        system_prompt = """You are an orchestrating agent responsible for coordinating
        A/B testing validation tasks. Analyze the incoming task and determine the validation strategy.

        You will coordinate sub-agents:
        - Data Validation Agent: Validates dataset quality, relevance, and statistical adequacy
        - Additional validation agents as needed

        Provide a brief analysis of what needs to be validated."""

        context_info = ""
        if ab_context:
            context_info = f"""
A/B Test Context:
- Hypothesis: {ab_context.get('hypothesis', 'Not specified')}
- Success Metrics: {ab_context.get('success_metrics', [])}
- Dataset Path: {ab_context.get('dataset_path', 'Not specified')}
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Task to validate: {task}\n{context_info}")
        ]

        response = self.llm.invoke(messages)

        # Update state
        state["messages"] = messages + [response]
        state["current_step"] = "entry"
        state["next_action"] = "delegate"

        print(f"[ORCHESTRATOR] Entry point - Analyzed task")
        print(f"[ORCHESTRATOR] Strategy: {response.content}\n")

        return state

    def _delegate_to_agents(self, state: ValidationState) -> ValidationState:
        """
        Delegate validation work to sub-agents using A2A protocol.
        """
        print("[ORCHESTRATOR] Delegating to sub-agents via A2A protocol...")

        # Initialize A2A messages list if not present
        if "a2a_messages" not in state or state["a2a_messages"] is None:
            state["a2a_messages"] = []

        ab_context = state.get("ab_test_context")

        # Delegate to Data Validation Agent using A2A protocol
        if ab_context:
            # Create A2A request for data validation
            data_validation_request = self.a2a_handler.create_request(
                receiver="data_validation_agent",
                task="Validate dataset for A/B testing",
                data={
                    "hypothesis": ab_context.get("hypothesis", ""),
                    "success_metrics": ab_context.get("success_metrics", []),
                    "dataset_path": ab_context.get("dataset_path", ""),
                    "expected_effect_size": ab_context.get("expected_effect_size", 0.2),
                    "significance_level": ab_context.get("significance_level", 0.05),
                    "power": ab_context.get("power", 0.8)
                },
                metadata={"orchestrator_step": "delegation"}
            )

            # Store request
            state["a2a_messages"].append(self.a2a_handler.serialize_message(data_validation_request))

            # Process request with data validation agent
            data_validation_response = self.data_validation_agent.process_request(data_validation_request)

            # Store response
            state["a2a_messages"].append(self.a2a_handler.serialize_message(data_validation_response))

            # Extract result
            state["data_validation_result"] = data_validation_response.result
        else:
            state["data_validation_result"] = {
                "error": "No A/B test context provided"
            }

        # Placeholder for second sub-agent (to be implemented)
        state["sub_agent_2_result"] = {
            "agent": "sub_agent_2",
            "status": "pending",
            "message": "Sub-agent 2 - To be implemented"
        }

        state["current_step"] = "delegation"
        state["next_action"] = "synthesize"

        print("[ORCHESTRATOR] Sub-agent delegation complete\n")

        return state

    def _synthesize_results(self, state: ValidationState) -> ValidationState:
        """
        Synthesize results from sub-agents and produce final validation decision.
        """
        print("[ORCHESTRATOR] Synthesizing results...")

        # Gather sub-agent results
        data_validation = state.get("data_validation_result", {})
        sub_agent_2 = state.get("sub_agent_2_result", {})

        # Format data validation results for synthesis
        data_val_summary = self._format_data_validation_summary(data_validation)

        # Use LLM to synthesize results
        synthesis_prompt = f"""Based on the validation results from sub-agents,
        provide a final validation decision for the A/B testing setup.

        Task: {state['task']}

        Data Validation Agent Results:
        {data_val_summary}

        Additional Agent Results:
        {sub_agent_2.get('message', 'Pending')}

        Provide:
        1. Overall validation decision (PASS/FAIL)
        2. Brief summary of key findings
        3. Critical recommendations for the A/B test

        Format your response as:
        DECISION: [PASS/FAIL]
        SUMMARY: [Brief summary]
        RECOMMENDATIONS: [Bullet points or "None"]
        """

        response = self.llm.invoke([HumanMessage(content=synthesis_prompt)])

        # Parse the response (simple parsing for now)
        content = response.content
        validation_passed = "PASS" in content.split("\n")[0]

        state["validation_passed"] = validation_passed
        state["validation_summary"] = content
        state["current_step"] = "complete"
        state["messages"] = state.get("messages", []) + [response]

        print(f"[ORCHESTRATOR] Final Decision: {'PASSED' if validation_passed else 'FAILED'}")
        print(f"[ORCHESTRATOR] Summary:\n{content}\n")

        return state

    def _format_data_validation_summary(self, result: dict) -> str:
        """Format data validation results for synthesis."""
        if not result or "error" in result:
            return f"Error: {result.get('error', 'Unknown error')}"

        overall_status = result.get("overall_status", "unknown")
        checks_passed = result.get("checks_passed", [])
        checks_failed = result.get("checks_failed", [])

        summary = f"Overall Status: {overall_status.upper()}\n"
        summary += f"Checks Passed: {', '.join(checks_passed) if checks_passed else 'None'}\n"
        summary += f"Checks Failed: {', '.join(checks_failed) if checks_failed else 'None'}\n\n"

        details = result.get("details", {})
        for check_name, check_result in details.items():
            summary += f"{check_name}: {'PASSED' if check_result.get('passed') else 'FAILED'}\n"
            if "analysis" in check_result:
                summary += f"  {check_result['analysis'][:200]}...\n"

        return summary

    def validate(self, task: str, ab_test_context: Optional[ABTestContext] = None) -> dict:
        """
        Execute the validation workflow for a given task.

        Args:
            task: The task or content to validate
            ab_test_context: Optional A/B testing context with hypothesis, metrics, and dataset

        Returns:
            Dictionary containing validation results
        """
        # Initialize state
        initial_state = ValidationState(
            task=task,
            ab_test_context=ab_test_context,
            messages=[],
            a2a_messages=[],
            data_validation_result=None,
            sub_agent_2_result=None,
            current_step="start",
            next_action="entry",
            validation_passed=None,
            validation_summary=None,
            recommendations=None
        )

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        # Return results
        return {
            "task": task,
            "validation_passed": final_state["validation_passed"],
            "summary": final_state["validation_summary"],
            "data_validation": final_state["data_validation_result"],
            "sub_agent_2": final_state["sub_agent_2_result"],
            "a2a_messages": final_state["a2a_messages"]
        }
