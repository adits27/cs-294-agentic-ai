"""
Report Validation Agent for A/B Testing Reports.

This agent validates A/B testing reports for quality, completeness, and validity,
ensuring reports contain necessary information for decision-making.
"""

import os
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.a2a_protocol import A2AProtocolHandler, A2AMessage, MessageStatus


class ReportValidationAgent:
    """
    Sub-agent responsible for validating A/B testing reports.

    Performs checks on:
    1. Experiment overview
    2. Experiment design
    3. Data and validity checks
    4. Results reporting
    5. Interpretation and insights
    6. Recommendations and next steps
    7. Completeness and clarity
    """

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        """
        Initialize the report validation agent.

        Args:
            model: The Gemini model to use
            temperature: Temperature for LLM responses
        """
        self.agent_id = "report_validation_agent"
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        self.a2a_handler = A2AProtocolHandler(self.agent_id)

    def process_request(self, request: A2AMessage) -> A2AMessage:
        """
        Process an A2A request for report validation.

        Args:
            request: A2A request message from orchestrator

        Returns:
            A2A response message with validation results
        """
        try:
            # Extract validation context
            data = request.data or {}
            report_path = data.get("report_path", "")
            report_content = data.get("report_content", "")
            report_type = data.get("report_type", "ab_test")

            # Load report from file if report_path is provided
            if report_path and not report_content:
                report_content = self._load_report_from_file(report_path)
                print(f"[{self.agent_id.upper()}] Processing report validation request")
                print(f"[{self.agent_id.upper()}] Report file: {report_path}")
                print(f"[{self.agent_id.upper()}] Report length: {len(report_content)} characters\n")
            else:
                print(f"[{self.agent_id.upper()}] Processing report validation request")
                print(f"[{self.agent_id.upper()}] Report length: {len(report_content)} characters\n")

            # Perform validation checks
            validation_results = self._perform_validation(
                report_content=report_content,
                report_type=report_type
            )

            print(f"[{self.agent_id.upper()}] Validation complete\n")

            # Create response
            response = self.a2a_handler.create_response(
                request_message=request,
                result=validation_results,
                status=MessageStatus.COMPLETED
            )

            return response

        except Exception as e:
            print(f"[{self.agent_id.upper()}] Error: {str(e)}\n")
            return self.a2a_handler.create_response(
                request_message=request,
                result={},
                status=MessageStatus.FAILED,
                error=str(e)
            )

    def _load_report_from_file(self, file_path: str) -> str:
        """
        Load report from a text file.

        Args:
            file_path: Path to the report file

        Returns:
            str: Contents of the report file

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is not a text file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Report file not found: {file_path}")

        if not file_path.endswith('.txt'):
            raise ValueError(f"File must be a text file (.txt): {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            raise ValueError(f"Error reading report file {file_path}: {str(e)}")

    def _perform_validation(
        self,
        report_content: str,
        report_type: str
    ) -> Dict[str, Any]:
        """
        Perform comprehensive report validation.

        Args:
            report_content: The report content to validate
            report_type: Type of report (default: "ab_test")

        Returns:
            Dictionary with validation results including scores and assessments
        """
        results = {
            "checks_passed": [],
            "checks_partial": [],
            "checks_failed": [],
            "overall_status": "unknown",
            "overall_score": 0,
            "details": {},
            "summary": "",
            "suggestions": []
        }

        # Check 1: Experiment Overview
        overview_check = self._check_experiment_overview(report_content)
        self._categorize_check_result(results, "experiment_overview", overview_check)
        results["details"]["experiment_overview"] = overview_check

        # Check 2: Experiment Design
        design_check = self._check_experiment_design(report_content)
        self._categorize_check_result(results, "experiment_design", design_check)
        results["details"]["experiment_design"] = design_check

        # Check 3: Data and Validity Checks
        validity_check = self._check_data_validity(report_content)
        self._categorize_check_result(results, "data_validity", validity_check)
        results["details"]["data_validity"] = validity_check

        # Check 4: Results Reporting
        results_check = self._check_results_reporting(report_content)
        self._categorize_check_result(results, "results_reporting", results_check)
        results["details"]["results_reporting"] = results_check

        # Check 5: Interpretation and Insights
        insights_check = self._check_interpretation_insights(report_content)
        self._categorize_check_result(results, "interpretation_insights", insights_check)
        results["details"]["interpretation_insights"] = insights_check

        # Check 6: Recommendations and Next Steps
        recommendations_check = self._check_recommendations(report_content)
        self._categorize_check_result(results, "recommendations", recommendations_check)
        results["details"]["recommendations"] = recommendations_check

        # Check 7: Completeness and Clarity
        clarity_check = self._check_completeness_clarity(report_content)
        self._categorize_check_result(results, "completeness_clarity", clarity_check)
        results["details"]["completeness_clarity"] = clarity_check

        # Calculate overall score (weighted average of all checks)
        total_score = sum([
            overview_check.get("score", 0),
            design_check.get("score", 0),
            validity_check.get("score", 0),
            results_check.get("score", 0),
            insights_check.get("score", 0),
            recommendations_check.get("score", 0),
            clarity_check.get("score", 0)
        ])
        results["overall_score"] = round(total_score / 7, 1)

        # Generate summary and suggestions using LLM
        summary_and_suggestions = self._generate_summary_and_suggestions(results, report_content)
        results["summary"] = summary_and_suggestions["summary"]
        results["suggestions"] = summary_and_suggestions["suggestions"]

        # Determine overall status
        if results["overall_score"] >= 80:
            results["overall_status"] = "passed"
        elif results["overall_score"] >= 60:
            results["overall_status"] = "passed_with_warnings"
        else:
            results["overall_status"] = "failed"

        return results

    def _categorize_check_result(self, results: Dict, check_name: str, check_result: Dict):
        """Categorize check result as PASS, PARTIAL, or FAIL."""
        assessment = check_result.get("assessment", "FAIL")
        if assessment == "PASS":
            results["checks_passed"].append(check_name)
        elif assessment == "PARTIAL":
            results["checks_partial"].append(check_name)
        else:
            results["checks_failed"].append(check_name)

    def _check_experiment_overview(self, report_content: str) -> Dict[str, Any]:
        """
        Check 1: Experiment Overview
        Check whether the report clearly states the purpose, hypothesis, and context.
        """
        print(f"[{self.agent_id.upper()}] Running Check 1: Experiment Overview")

        try:
            prompt = f"""Evaluate the Experiment Overview section of this A/B testing report.

Report Content:
{report_content}

Check whether the report clearly states:
1. **Purpose**: What is the experiment trying to achieve?
2. **Hypothesis**: What is the expected outcome or prediction?
3. **Context**: Why is this experiment being run? What problem does it address?

Assess the quality and completeness of the overview section.

Provide:
1. Assessment: PASS, PARTIAL, or FAIL
2. Score from 0-100 (100 = excellent overview, 0 = no overview)
3. Brief explanation of your assessment

Format your response as:
ASSESSMENT: [PASS/PARTIAL/FAIL]
SCORE: [0-100]
EXPLANATION: [Your detailed explanation]
"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse response
            assessment = self._extract_assessment(content)
            score = self._extract_score_100(content)
            explanation = self._extract_explanation(content)

            return {
                "assessment": assessment,
                "score": score,
                "explanation": explanation,
                "analysis": content
            }

        except Exception as e:
            return {
                "assessment": "FAIL",
                "score": 0,
                "explanation": f"Could not check experiment overview: {str(e)}",
                "error": str(e)
            }

    def _check_experiment_design(self, report_content: str) -> Dict[str, Any]:
        """
        Check 2: Experiment Design
        Verify whether variants, metrics, success criteria, sample size, targeting, and traffic allocation are defined.
        """
        print(f"[{self.agent_id.upper()}] Running Check 2: Experiment Design")

        try:
            prompt = f"""Evaluate the Experiment Design section of this A/B testing report.

Report Content:
{report_content}

Check whether the report properly defines:
1. **Variants**: Control and treatment groups clearly described
2. **Metrics**: Primary and secondary metrics identified
3. **Success Criteria**: Clear criteria for determining success/failure
4. **Sample Size**: Sample size requirements and actual samples
5. **Targeting**: Target audience or user segments
6. **Traffic Allocation**: How traffic is split between variants

Assess the completeness and clarity of the experiment design.

Provide:
1. Assessment: PASS, PARTIAL, or FAIL
2. Score from 0-100
3. Brief explanation noting which elements are present or missing

Format your response as:
ASSESSMENT: [PASS/PARTIAL/FAIL]
SCORE: [0-100]
EXPLANATION: [Your detailed explanation]
"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse response
            assessment = self._extract_assessment(content)
            score = self._extract_score_100(content)
            explanation = self._extract_explanation(content)

            return {
                "assessment": assessment,
                "score": score,
                "explanation": explanation,
                "analysis": content
            }

        except Exception as e:
            return {
                "assessment": "FAIL",
                "score": 0,
                "explanation": f"Could not check experiment design: {str(e)}",
                "error": str(e)
            }

    def _check_data_validity(self, report_content: str) -> Dict[str, Any]:
        """
        Check 3: Data and Validity Checks
        Assess if the report checks for data quality issues.
        """
        print(f"[{self.agent_id.upper()}] Running Check 3: Data and Validity Checks")

        try:
            prompt = f"""Evaluate the Data and Validity Checks section of this A/B testing report.

Report Content:
{report_content}

Check whether the report addresses:
1. **Sample Ratio Mismatch (SRM)**: Verification that traffic split matches expected ratios
2. **Tracking Errors**: Discussion of any tracking or instrumentation issues
3. **Anomalies**: Identification of unusual patterns or outliers in the data
4. **Outlier Handling**: How outliers or extreme values are handled
5. **Data Quality**: Overall assessment of data integrity and completeness

Assess whether the report adequately addresses data quality concerns.

Provide:
1. Assessment: PASS, PARTIAL, or FAIL
2. Score from 0-100
3. Brief explanation of what checks are present or missing

Format your response as:
ASSESSMENT: [PASS/PARTIAL/FAIL]
SCORE: [0-100]
EXPLANATION: [Your detailed explanation]
"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse response
            assessment = self._extract_assessment(content)
            score = self._extract_score_100(content)
            explanation = self._extract_explanation(content)

            return {
                "assessment": assessment,
                "score": score,
                "explanation": explanation,
                "analysis": content
            }

        except Exception as e:
            return {
                "assessment": "FAIL",
                "score": 0,
                "explanation": f"Could not check data validity: {str(e)}",
                "error": str(e)
            }

    def _check_results_reporting(self, report_content: str) -> Dict[str, Any]:
        """
        Check 4: Results Reporting
        Confirm that the report includes key metrics, effect sizes, statistical significance, etc.
        """
        print(f"[{self.agent_id.upper()}] Running Check 4: Results Reporting")

        try:
            prompt = f"""Evaluate the Results Reporting section of this A/B testing report.

Report Content:
{report_content}

Check whether the report includes:
1. **Key Metrics**: Clear reporting of primary and secondary metrics
2. **Effect Sizes**: Magnitude of differences between variants
3. **Statistical Significance**: P-values and significance levels
4. **Confidence Intervals**: Confidence intervals for estimates
5. **Supporting Data**: Tables, charts, or visualizations (described or referenced)

Assess the completeness and quality of results reporting.

Provide:
1. Assessment: PASS, PARTIAL, or FAIL
2. Score from 0-100
3. Brief explanation of what is well-reported or missing

Format your response as:
ASSESSMENT: [PASS/PARTIAL/FAIL]
SCORE: [0-100]
EXPLANATION: [Your detailed explanation]
"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse response
            assessment = self._extract_assessment(content)
            score = self._extract_score_100(content)
            explanation = self._extract_explanation(content)

            return {
                "assessment": assessment,
                "score": score,
                "explanation": explanation,
                "analysis": content
            }

        except Exception as e:
            return {
                "assessment": "FAIL",
                "score": 0,
                "explanation": f"Could not check results reporting: {str(e)}",
                "error": str(e)
            }

    def _check_interpretation_insights(self, report_content: str) -> Dict[str, Any]:
        """
        Check 5: Interpretation and Insights
        Evaluate whether the report explains why results occurred and discusses unexpected findings.
        """
        print(f"[{self.agent_id.upper()}] Running Check 5: Interpretation and Insights")

        try:
            prompt = f"""Evaluate the Interpretation and Insights section of this A/B testing report.

Report Content:
{report_content}

Check whether the report:
1. **Explains Why**: Provides reasoning for why the results occurred
2. **Unexpected Findings**: Discusses any surprising or unexpected outcomes
3. **Acknowledges Limitations**: Recognizes limitations of the experiment or analysis
4. **Context**: Places results in broader business or product context
5. **Depth of Analysis**: Goes beyond surface-level reporting to provide insights

Assess the quality and depth of interpretation.

Provide:
1. Assessment: PASS, PARTIAL, or FAIL
2. Score from 0-100
3. Brief explanation of the quality of insights

Format your response as:
ASSESSMENT: [PASS/PARTIAL/FAIL]
SCORE: [0-100]
EXPLANATION: [Your detailed explanation]
"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse response
            assessment = self._extract_assessment(content)
            score = self._extract_score_100(content)
            explanation = self._extract_explanation(content)

            return {
                "assessment": assessment,
                "score": score,
                "explanation": explanation,
                "analysis": content
            }

        except Exception as e:
            return {
                "assessment": "FAIL",
                "score": 0,
                "explanation": f"Could not check interpretation and insights: {str(e)}",
                "error": str(e)
            }

    def _check_recommendations(self, report_content: str) -> Dict[str, Any]:
        """
        Check 6: Recommendations and Next Steps
        Check if the report provides clear, justified conclusions and actionable next steps.
        """
        print(f"[{self.agent_id.upper()}] Running Check 6: Recommendations and Next Steps")

        try:
            prompt = f"""Evaluate the Recommendations and Next Steps section of this A/B testing report.

Report Content:
{report_content}

Check whether the report:
1. **Clear Conclusions**: Provides explicit conclusions (e.g., ship, don't ship, iterate)
2. **Justified Decisions**: Conclusions are well-supported by the data and analysis
3. **Actionable Next Steps**: Specific, concrete actions recommended
4. **Decision Criteria**: Clear explanation of how decision was reached
5. **Risk Assessment**: Considers risks or trade-offs of recommended actions

Assess the quality and actionability of recommendations.

Provide:
1. Assessment: PASS, PARTIAL, or FAIL
2. Score from 0-100
3. Brief explanation of recommendation quality

Format your response as:
ASSESSMENT: [PASS/PARTIAL/FAIL]
SCORE: [0-100]
EXPLANATION: [Your detailed explanation]
"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse response
            assessment = self._extract_assessment(content)
            score = self._extract_score_100(content)
            explanation = self._extract_explanation(content)

            return {
                "assessment": assessment,
                "score": score,
                "explanation": explanation,
                "analysis": content
            }

        except Exception as e:
            return {
                "assessment": "FAIL",
                "score": 0,
                "explanation": f"Could not check recommendations: {str(e)}",
                "error": str(e)
            }

    def _check_completeness_clarity(self, report_content: str) -> Dict[str, Any]:
        """
        Check 7: Completeness and Clarity
        Determine whether the report is well-structured, logically organized, and sufficiently detailed.
        """
        print(f"[{self.agent_id.upper()}] Running Check 7: Completeness and Clarity")

        try:
            prompt = f"""Evaluate the overall Completeness and Clarity of this A/B testing report.

Report Content:
{report_content}

Check whether the report is:
1. **Well-Structured**: Logical organization with clear sections
2. **Sufficiently Detailed**: Appropriate level of detail for decision-making
3. **Clear Writing**: Easy to understand, avoids jargon or explains technical terms
4. **Complete**: All necessary information is present
5. **Professional**: Follows reporting best practices

Assess the overall quality of the report as a decision-making document.

Provide:
1. Assessment: PASS, PARTIAL, or FAIL
2. Score from 0-100
3. Brief explanation of overall quality

Format your response as:
ASSESSMENT: [PASS/PARTIAL/FAIL]
SCORE: [0-100]
EXPLANATION: [Your detailed explanation]
"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse response
            assessment = self._extract_assessment(content)
            score = self._extract_score_100(content)
            explanation = self._extract_explanation(content)

            return {
                "assessment": assessment,
                "score": score,
                "explanation": explanation,
                "analysis": content
            }

        except Exception as e:
            return {
                "assessment": "FAIL",
                "score": 0,
                "explanation": f"Could not check completeness and clarity: {str(e)}",
                "error": str(e)
            }

    def _generate_summary_and_suggestions(
        self, results: Dict[str, Any], report_content: str
    ) -> Dict[str, Any]:
        """
        Generate overall summary and actionable suggestions using LLM.
        """
        print(f"[{self.agent_id.upper()}] Generating summary and suggestions")

        try:
            # Format check results
            checks_summary = f"""
Checks Passed: {', '.join(results['checks_passed']) if results['checks_passed'] else 'None'}
Checks Partial: {', '.join(results['checks_partial']) if results['checks_partial'] else 'None'}
Checks Failed: {', '.join(results['checks_failed']) if results['checks_failed'] else 'None'}
Overall Score: {results['overall_score']}/100

Category Scores:
"""
            for category, details in results['details'].items():
                checks_summary += f"- {category}: {details.get('score', 0)}/100 ({details.get('assessment', 'UNKNOWN')})\n"

            prompt = f"""Based on the validation results of this A/B testing report, provide:

1. A short summary (2-3 sentences) stating whether the experiment results are trustworthy
2. A list of 3-5 clear, actionable suggestions to improve the report or the experiment itself

Validation Results:
{checks_summary}

Report Content Preview:
{report_content[:1000]}...

Focus on the most critical issues and practical improvements.

Format your response as:
SUMMARY: [Your 2-3 sentence summary about trustworthiness]
SUGGESTIONS:
- [Suggestion 1]
- [Suggestion 2]
- [Suggestion 3]
...
"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse response
            summary = self._extract_summary(content)
            suggestions = self._extract_suggestions(content)

            return {
                "summary": summary,
                "suggestions": suggestions
            }

        except Exception as e:
            return {
                "summary": f"Could not generate summary: {str(e)}",
                "suggestions": []
            }

    def _extract_assessment(self, content: str) -> str:
        """Extract PASS/PARTIAL/FAIL assessment from LLM response."""
        try:
            for line in content.split("\n"):
                if "ASSESSMENT:" in line.upper():
                    assessment_str = line.split(":")[-1].strip().upper()
                    if "PASS" in assessment_str and "PARTIAL" not in assessment_str:
                        return "PASS"
                    elif "PARTIAL" in assessment_str:
                        return "PARTIAL"
                    elif "FAIL" in assessment_str:
                        return "FAIL"
            return "FAIL"  # Default to FAIL if not found
        except:
            return "FAIL"

    def _extract_score_100(self, content: str) -> float:
        """Extract score (0-100) from LLM response."""
        try:
            import re
            for line in content.split("\n"):
                if "SCORE:" in line.upper():
                    score_str = line.split(":")[-1].strip()
                    # Extract first number found
                    numbers = re.findall(r'\d+', score_str)
                    if numbers:
                        score = float(numbers[0])
                        return max(0, min(100, score))  # Clamp to 0-100
            return 50  # Default to 50 if no score found
        except:
            return 50

    def _extract_explanation(self, content: str) -> str:
        """Extract explanation from LLM response."""
        try:
            explanation_started = False
            explanation_lines = []
            for line in content.split("\n"):
                if "EXPLANATION:" in line.upper():
                    explanation_started = True
                    explanation_lines.append(line.split(":", 1)[-1].strip())
                elif explanation_started and not any(marker in line.upper() for marker in ["ASSESSMENT:", "SCORE:"]):
                    explanation_lines.append(line)
                elif explanation_started and any(marker in line.upper() for marker in ["SUMMARY:", "SUGGESTIONS:"]):
                    break

            if explanation_lines:
                return " ".join(explanation_lines).strip()
            else:
                return content.strip()
        except:
            return content.strip()

    def _extract_summary(self, content: str) -> str:
        """Extract summary from LLM response."""
        try:
            summary_started = False
            summary_lines = []
            for line in content.split("\n"):
                if "SUMMARY:" in line.upper():
                    summary_started = True
                    summary_lines.append(line.split(":", 1)[-1].strip())
                elif summary_started and "SUGGESTIONS:" not in line.upper():
                    summary_lines.append(line)
                elif summary_started and "SUGGESTIONS:" in line.upper():
                    break

            if summary_lines:
                return " ".join(summary_lines).strip()
            else:
                return "No summary available."
        except:
            return "No summary available."

    def _extract_suggestions(self, content: str) -> list:
        """Extract suggestions list from LLM response."""
        try:
            suggestions = []
            suggestions_started = False
            for line in content.split("\n"):
                if "SUGGESTIONS:" in line.upper():
                    suggestions_started = True
                    continue
                elif suggestions_started:
                    line = line.strip()
                    if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                        suggestions.append(line.lstrip("-•* ").strip())
                    elif line and not line.startswith("SUMMARY"):
                        suggestions.append(line)

            return suggestions if suggestions else ["No specific suggestions available."]
        except:
            return ["No specific suggestions available."]
