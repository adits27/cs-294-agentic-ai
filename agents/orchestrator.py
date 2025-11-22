"""
Main orchestrating agent for the multi-agent validation system.

This agent coordinates two sub-agents to validate tasks, managing
the overall workflow and synthesizing results using A2A protocol.
"""

from typing import Literal, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from agents.state import ValidationState, ABTestContext, CodeValidationContext, ReportValidationContext
from agents.a2a_protocol import A2AProtocolHandler, MessageStatus
from agents.data_validation_agent import DataValidationAgent
from agents.code_validation_agent import CodeValidationAgent
from agents.report_validation_agent import ReportValidationAgent


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
        self.code_validation_agent = CodeValidationAgent(model=model, temperature=temperature)
        self.report_validation_agent = ReportValidationAgent(model=model, temperature=temperature)

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
        - Data Validation Agent: Validates dataset quality, relevance, and statistical adequacy for A/B tests
        - Code Validation Agent: Validates Python code for syntax, best practices, functionality, and readability
        - Report Validation Agent: Validates A/B testing reports for quality, completeness, and validity
        - Additional validation agents as needed

        Provide a brief analysis of what needs to be validated."""

        context_info = ""
        ab_context = state.get("ab_test_context")
        code_context = state.get("code_validation_context")
        report_context = state.get("report_validation_context")

        if ab_context:
            context_info += f"""
A/B Test Context:
- Hypothesis: {ab_context.get('hypothesis', 'Not specified')}
- Success Metrics: {ab_context.get('success_metrics', [])}
- Dataset Path: {ab_context.get('dataset_path', 'Not specified')}
"""
        if code_context:
            code_path = code_context.get('code_path', '')
            code_snippet = code_context.get('code', '')

            if code_path:
                context_info += f"""
Code Validation Context:
- Code File: {code_path}
- Description: {code_context.get('description', 'Not specified')}
"""
            elif code_snippet:
                code_preview = code_snippet[:200] + "..." if len(code_snippet) > 200 else code_snippet
                context_info += f"""
Code Validation Context:
- Code Length: {len(code_snippet)} characters
- Description: {code_context.get('description', 'Not specified')}
- Code Preview: {code_preview}
"""
        if report_context:
            report_path = report_context.get('report_path', '')
            report_content = report_context.get('report_content', '')

            if report_path:
                context_info += f"""
Report Validation Context:
- Report File: {report_path}
- Report Type: {report_context.get('report_type', 'ab_test')}
"""
            elif report_content:
                report_preview = report_content[:200] + "..." if len(report_content) > 200 else report_content
                context_info += f"""
Report Validation Context:
- Report Length: {len(report_content)} characters
- Report Type: {report_context.get('report_type', 'ab_test')}
- Report Preview: {report_preview}
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
        code_context = state.get("code_validation_context")
        report_context = state.get("report_validation_context")

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
            state["data_validation_result"] = None

        # Delegate to Code Validation Agent using A2A protocol
        if code_context:
            # Create A2A request for code validation
            code_validation_request = self.a2a_handler.create_request(
                receiver="code_validation_agent",
                task="Validate Python code for quality and correctness",
                data={
                    "code_path": code_context.get("code_path", ""),
                    "code": code_context.get("code", ""),
                    "description": code_context.get("description", ""),
                    "expected_behavior": code_context.get("expected_behavior", "")
                },
                metadata={"orchestrator_step": "delegation"}
            )

            # Store request
            state["a2a_messages"].append(self.a2a_handler.serialize_message(code_validation_request))

            # Process request with code validation agent
            code_validation_response = self.code_validation_agent.process_request(code_validation_request)

            # Store response
            state["a2a_messages"].append(self.a2a_handler.serialize_message(code_validation_response))

            # Extract result
            state["code_validation_result"] = code_validation_response.result
        else:
            state["code_validation_result"] = None

        # Delegate to Report Validation Agent using A2A protocol
        if report_context:
            # Create A2A request for report validation
            report_validation_request = self.a2a_handler.create_request(
                receiver="report_validation_agent",
                task="Validate A/B testing report for quality and completeness",
                data={
                    "report_path": report_context.get("report_path", ""),
                    "report_content": report_context.get("report_content", ""),
                    "report_type": report_context.get("report_type", "ab_test")
                },
                metadata={"orchestrator_step": "delegation"}
            )

            # Store request
            state["a2a_messages"].append(self.a2a_handler.serialize_message(report_validation_request))

            # Process request with report validation agent
            report_validation_response = self.report_validation_agent.process_request(report_validation_request)

            # Store response
            state["a2a_messages"].append(self.a2a_handler.serialize_message(report_validation_response))

            # Extract result
            state["report_validation_result"] = report_validation_response.result
        else:
            state["report_validation_result"] = None

        # Placeholder for additional sub-agent (to be implemented)
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
        data_validation = state.get("data_validation_result")
        code_validation = state.get("code_validation_result")
        report_validation = state.get("report_validation_result")
        sub_agent_2 = state.get("sub_agent_2_result", {})

        # Build synthesis prompt based on which agents were used
        results_summary = ""

        if data_validation:
            data_val_summary = self._format_data_validation_summary(data_validation)
            results_summary += f"\nData Validation Agent Results:\n{data_val_summary}\n"

        if code_validation:
            code_val_summary = self._format_code_validation_summary(code_validation)
            results_summary += f"\nCode Validation Agent Results:\n{code_val_summary}\n"

        if report_validation:
            report_val_summary = self._format_report_validation_summary(report_validation)
            results_summary += f"\nReport Validation Agent Results:\n{report_val_summary}\n"

        if not data_validation and not code_validation and not report_validation:
            results_summary = "No validation results available."

        # Use LLM to synthesize results
        synthesis_prompt = f"""Based on the validation results from sub-agents,
        provide a final validation decision.

        Task: {state['task']}

        {results_summary}

        Additional Agent Results:
        {sub_agent_2.get('message', 'Pending')}

        Provide:
        1. Overall validation decision (PASS/FAIL)
        2. Brief summary of key findings
        3. Critical recommendations

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

    def _format_code_validation_summary(self, result: dict) -> str:
        """Format code validation results for synthesis."""
        if not result or "error" in result:
            return f"Error: {result.get('error', 'Unknown error')}"

        overall_status = result.get("overall_status", "unknown")
        overall_score = result.get("overall_score", 0)

        summary = f"Overall Status: {overall_status.upper()}\n"
        summary += f"Overall Score: {overall_score}/10\n\n"

        summary += f"Scores:\n"
        summary += f"  - Syntax: {result.get('syntax_score', 0)}/10\n"
        summary += f"  - Best Practices: {result.get('best_practices_score', 0)}/10\n"
        summary += f"  - Functionality: {result.get('functionality_score', 0)}/10\n"
        summary += f"  - Readability: {result.get('readability_score', 0)}/10\n\n"

        feedback = result.get("feedback", {})
        if feedback:
            summary += "Feedback:\n"
            for category, fb in feedback.items():
                summary += f"  - {category.title()}: {fb[:150]}...\n"

        return summary

    def _format_report_validation_summary(self, result: dict) -> str:
        """Format report validation results for synthesis."""
        if not result or "error" in result:
            return f"Error: {result.get('error', 'Unknown error')}"

        overall_status = result.get("overall_status", "unknown")
        overall_score = result.get("overall_score", 0)

        summary = f"Overall Status: {overall_status.upper()}\n"
        summary += f"Overall Score: {overall_score}/100\n\n"

        summary += f"Category Assessments:\n"
        summary += f"  - Checks Passed: {', '.join(result.get('checks_passed', [])) if result.get('checks_passed') else 'None'}\n"
        summary += f"  - Checks Partial: {', '.join(result.get('checks_partial', [])) if result.get('checks_partial') else 'None'}\n"
        summary += f"  - Checks Failed: {', '.join(result.get('checks_failed', [])) if result.get('checks_failed') else 'None'}\n\n"

        # Show detailed scores for each category
        details = result.get("details", {})
        if details:
            summary += "Category Scores:\n"
            for category, detail in details.items():
                assessment = detail.get("assessment", "UNKNOWN")
                score = detail.get("score", 0)
                summary += f"  - {category}: {score}/100 ({assessment})\n"

        # Add summary and suggestions
        if result.get("summary"):
            summary += f"\nSummary: {result.get('summary')[:200]}...\n"

        if result.get("suggestions"):
            summary += f"\nKey Suggestions:\n"
            for i, suggestion in enumerate(result.get("suggestions", [])[:3], 1):
                summary += f"  {i}. {suggestion[:150]}...\n"

        return summary

    def validate(
        self,
        task: str,
        ab_test_context: Optional[ABTestContext] = None,
        code_validation_context: Optional[CodeValidationContext] = None,
        report_validation_context: Optional[ReportValidationContext] = None
    ) -> dict:
        """
        Execute the validation workflow for a given task.

        Args:
            task: The task or content to validate
            ab_test_context: Optional A/B testing context with hypothesis, metrics, and dataset
            code_validation_context: Optional Python code validation context
            report_validation_context: Optional A/B testing report validation context

        Returns:
            Dictionary containing validation results
        """
        # Initialize state
        initial_state = ValidationState(
            task=task,
            ab_test_context=ab_test_context,
            code_validation_context=code_validation_context,
            report_validation_context=report_validation_context,
            messages=[],
            a2a_messages=[],
            data_validation_result=None,
            code_validation_result=None,
            report_validation_result=None,
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
            "data_validation": final_state.get("data_validation_result"),
            "code_validation": final_state.get("code_validation_result"),
            "report_validation": final_state.get("report_validation_result"),
            "sub_agent_2": final_state["sub_agent_2_result"],
            "a2a_messages": final_state["a2a_messages"]
        }
