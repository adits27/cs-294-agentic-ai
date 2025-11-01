"""
Data Validation Agent for A/B Testing.

This agent validates datasets and data sources for A/B testing experiments,
ensuring data quality, relevance, and statistical adequacy.
"""

import os
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.a2a_protocol import A2AProtocolHandler, A2AMessage, MessageStatus


class DataValidationAgent:
    """
    Sub-agent responsible for validating datasets for A/B testing.

    Performs checks on:
    1. Data relevance to hypothesis
    2. Success metrics relevance
    3. Sample size adequacy (power analysis)
    4. Data suitability for A/B testing
    """

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        """
        Initialize the data validation agent.

        Args:
            model: The Gemini model to use
            temperature: Temperature for LLM responses
        """
        self.agent_id = "data_validation_agent"
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        self.a2a_handler = A2AProtocolHandler(self.agent_id)

    def process_request(self, request: A2AMessage) -> A2AMessage:
        """
        Process an A2A request for data validation.

        Args:
            request: A2A request message from orchestrator

        Returns:
            A2A response message with validation results
        """
        try:
            # Extract validation context
            data = request.data or {}
            hypothesis = data.get("hypothesis", "")
            success_metrics = data.get("success_metrics", [])
            dataset_path = data.get("dataset_path", "")
            expected_effect_size = data.get("expected_effect_size", 0.2)
            significance_level = data.get("significance_level", 0.05)
            power = data.get("power", 0.8)

            print(f"[{self.agent_id.upper()}] Processing validation request")
            print(f"[{self.agent_id.upper()}] Hypothesis: {hypothesis}")
            print(f"[{self.agent_id.upper()}] Dataset: {dataset_path}\n")

            # Perform validation checks
            validation_results = self._perform_validation(
                hypothesis=hypothesis,
                success_metrics=success_metrics,
                dataset_path=dataset_path,
                expected_effect_size=expected_effect_size,
                significance_level=significance_level,
                power=power
            )

            # Create response
            response = self.a2a_handler.create_response(
                request_message=request,
                result=validation_results,
                status=MessageStatus.COMPLETED
            )

            print(f"[{self.agent_id.upper()}] Validation complete\n")

            return response

        except Exception as e:
            print(f"[{self.agent_id.upper()}] Error: {str(e)}\n")
            return self.a2a_handler.create_response(
                request_message=request,
                result={},
                status=MessageStatus.FAILED,
                error=str(e)
            )

    def _perform_validation(
        self,
        hypothesis: str,
        success_metrics: List[str],
        dataset_path: str,
        expected_effect_size: float,
        significance_level: float,
        power: float
    ) -> Dict[str, Any]:
        """
        Perform comprehensive data validation.

        Args:
            hypothesis: A/B test hypothesis
            success_metrics: List of success metrics
            dataset_path: Path to dataset
            expected_effect_size: Expected effect size
            significance_level: Statistical significance level
            power: Statistical power

        Returns:
            Dictionary with validation results
        """
        results = {
            "checks_passed": [],
            "checks_failed": [],
            "warnings": [],
            "overall_status": "unknown",
            "details": {}
        }

        # Check 1: Data relevance to hypothesis
        relevance_check = self._check_data_relevance(hypothesis, dataset_path, success_metrics)
        if relevance_check["passed"]:
            results["checks_passed"].append("data_relevance")
        else:
            results["checks_failed"].append("data_relevance")
        results["details"]["data_relevance"] = relevance_check

        # Check 2: Success metrics relevance
        metrics_check = self._check_metrics_relevance(hypothesis, success_metrics, dataset_path)
        if metrics_check["passed"]:
            results["checks_passed"].append("metrics_relevance")
        else:
            results["checks_failed"].append("metrics_relevance")
        results["details"]["metrics_relevance"] = metrics_check

        # Check 3: Sample size adequacy (power analysis)
        power_check = self._check_sample_size(
            dataset_path, expected_effect_size, significance_level, power
        )
        if power_check["passed"]:
            results["checks_passed"].append("sample_size")
        else:
            results["checks_failed"].append("sample_size")
        results["details"]["sample_size"] = power_check

        # Check 4: Data suitability for A/B testing
        suitability_check = self._check_ab_test_suitability(dataset_path)
        if suitability_check["passed"]:
            results["checks_passed"].append("ab_test_suitability")
        else:
            results["checks_failed"].append("ab_test_suitability")
        results["details"]["ab_test_suitability"] = suitability_check

        # Determine overall status
        total_checks = len(results["checks_passed"]) + len(results["checks_failed"])
        if len(results["checks_failed"]) == 0:
            results["overall_status"] = "passed"
        elif len(results["checks_passed"]) >= total_checks * 0.75:
            results["overall_status"] = "passed_with_warnings"
        else:
            results["overall_status"] = "failed"

        return results

    def _check_data_relevance(
        self, hypothesis: str, dataset_path: str, success_metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Check if data is relevant to the hypothesis using LLM analysis.
        """
        print(f"[{self.agent_id.upper()}] Running Check 1: Data Relevance")

        try:
            # Load dataset info
            dataset_info = self._get_dataset_info(dataset_path)

            # Use LLM to analyze relevance
            prompt = f"""Analyze whether the provided dataset is relevant to the A/B testing hypothesis.

Hypothesis: {hypothesis}

Dataset Information:
- Columns: {dataset_info['columns']}
- Sample Size: {dataset_info['sample_size']}
- Data Types: {dataset_info['dtypes']}

Success Metrics: {', '.join(success_metrics)}

Assess if the dataset contains the necessary information to test this hypothesis and measure the success metrics.

Respond with:
1. RELEVANT or NOT_RELEVANT
2. Brief explanation (2-3 sentences)
3. Any missing data elements (if applicable)

Format:
ASSESSMENT: [RELEVANT/NOT_RELEVANT]
EXPLANATION: [Your explanation]
MISSING: [List any missing elements or "None"]
"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse response
            is_relevant = "RELEVANT" in content.split("\n")[0] and "NOT_RELEVANT" not in content.split("\n")[0]

            return {
                "passed": is_relevant,
                "analysis": content,
                "dataset_info": dataset_info
            }

        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "analysis": f"Could not analyze data relevance: {str(e)}"
            }

    def _check_metrics_relevance(
        self, hypothesis: str, success_metrics: List[str], dataset_path: str
    ) -> Dict[str, Any]:
        """
        Check if success metrics are relevant to hypothesis and dataset.
        """
        print(f"[{self.agent_id.upper()}] Running Check 2: Metrics Relevance")

        try:
            dataset_info = self._get_dataset_info(dataset_path)

            prompt = f"""Evaluate whether the success metrics are appropriate for the A/B testing hypothesis and available in the dataset.

Hypothesis: {hypothesis}

Success Metrics: {', '.join(success_metrics)}

Available Dataset Columns: {dataset_info['columns']}

Assess:
1. Are these metrics appropriate for testing the hypothesis?
2. Can these metrics be calculated from the dataset?
3. Are there better alternative metrics?

Format:
APPROPRIATE: [YES/NO]
CALCULABLE: [YES/NO]
EXPLANATION: [Your explanation]
ALTERNATIVES: [Suggest alternatives or "None"]
"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse response
            is_appropriate = "APPROPRIATE: YES" in content
            is_calculable = "CALCULABLE: YES" in content

            return {
                "passed": is_appropriate and is_calculable,
                "analysis": content
            }

        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "analysis": f"Could not analyze metrics relevance: {str(e)}"
            }

    def _check_sample_size(
        self,
        dataset_path: str,
        expected_effect_size: float,
        significance_level: float,
        power: float
    ) -> Dict[str, Any]:
        """
        Use LLM to evaluate if sample size is adequate based on power analysis.
        """
        print(f"[{self.agent_id.upper()}] Running Check 3: Sample Size (Power Analysis)")

        try:
            dataset_info = self._get_dataset_info(dataset_path)
            actual_sample_size = dataset_info['sample_size']

            # Perform power analysis calculation using scipy for context
            from scipy.stats import norm

            # Cohen's d effect size
            z_alpha = norm.ppf(1 - significance_level / 2)
            z_beta = norm.ppf(power)

            # Required sample size per group
            required_n_per_group = ((z_alpha + z_beta) ** 2) * 2 / (expected_effect_size ** 2)
            required_total = required_n_per_group * 2

            # Use LLM to evaluate sample size adequacy
            prompt = f"""Evaluate whether the sample size is adequate for this A/B test based on power analysis.

Power Analysis Context:
- Actual sample size: {actual_sample_size} total observations
- Required sample size: {int(required_total)} total ({int(required_n_per_group)} per group)
- Expected effect size (Cohen's d): {expected_effect_size}
- Significance level (α): {significance_level}
- Statistical power target: {power}
- Shortfall: {max(0, int(required_total - actual_sample_size))} observations

Guidelines:
- Standard practice requires meeting the calculated sample size for adequate power
- Small shortfalls (<10%) may be acceptable with caveats
- Large shortfalls significantly reduce the ability to detect real effects
- Consider the practical context and business constraints

Assess whether this sample size is adequate for the A/B test.

Format:
ADEQUATE: [YES/NO]
EXPLANATION: [Your reasoning about why the sample size is or isn't adequate]
RECOMMENDATION: [What should be done - proceed, collect more data, adjust parameters, etc.]
"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse LLM response
            is_adequate = "ADEQUATE: YES" in content

            return {
                "passed": is_adequate,
                "required_sample_size": int(required_total),
                "actual_sample_size": actual_sample_size,
                "analysis": content,
                "power_parameters": {
                    "effect_size": expected_effect_size,
                    "alpha": significance_level,
                    "power": power
                }
            }

        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "analysis": f"Could not perform power analysis: {str(e)}"
            }

    def _check_ab_test_suitability(self, dataset_path: str) -> Dict[str, Any]:
        """
        Use LLM to evaluate if data structure is suitable for A/B testing.
        """
        print(f"[{self.agent_id.upper()}] Running Check 4: A/B Test Suitability")

        try:
            dataset_info = self._get_dataset_info(dataset_path)
            df = self._load_dataset(dataset_path)

            # Gather dataset characteristics for LLM analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            # Identify potential group columns
            potential_group_cols = [col for col in df.columns if any(keyword in col.lower()
                                    for keyword in ['group', 'variant', 'treatment', 'control', 'test'])]

            # Calculate missing value percentages
            missing_info = {}
            for col in df.columns:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                if missing_pct > 0:
                    missing_info[col] = f"{missing_pct:.1f}%"

            # Get sample of data for context
            sample_data = df.head(3).to_dict('records')

            # Use LLM to evaluate A/B test suitability
            prompt = f"""Evaluate whether this dataset structure is suitable for A/B testing analysis.

Dataset Characteristics:
- Total rows: {len(df)}
- Total columns: {len(df.columns)}
- Column names: {list(df.columns)}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}
- Potential group/variant columns: {potential_group_cols if potential_group_cols else 'None identified'}
- Data types: {dict(df.dtypes.astype(str))}
- Missing values: {missing_info if missing_info else 'None'}

Sample data (first 3 rows):
{sample_data}

A/B Testing Requirements:
1. Must have a clear group assignment column (control vs treatment/variant)
2. Should have numeric metrics that can be analyzed
3. Should have reasonable data quality (not excessive missing values)
4. Should have appropriate data types for the metrics
5. Should be structured for comparative analysis between groups

Evaluate whether this dataset meets the requirements for A/B testing analysis.

Format:
SUITABLE: [YES/NO]
EXPLANATION: [Detailed explanation of why the dataset is or isn't suitable]
CRITICAL_ISSUES: [List any blocking issues, or "None"]
RECOMMENDATIONS: [Suggestions for improvement or "None needed"]
"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse LLM response
            is_suitable = "SUITABLE: YES" in content

            return {
                "passed": is_suitable,
                "analysis": content,
                "dataset_characteristics": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "numeric_columns": numeric_cols,
                    "potential_group_columns": potential_group_cols,
                    "missing_values": missing_info
                }
            }

        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "analysis": f"Could not check A/B test suitability: {str(e)}"
            }

    def _get_dataset_info(self, dataset_path: str) -> Dict[str, Any]:
        """Get basic information about the dataset."""
        if not os.path.exists(dataset_path):
            return {
                "exists": False,
                "error": f"Dataset not found at {dataset_path}",
                "columns": [],
                "sample_size": 0,
                "dtypes": {}
            }

        try:
            df = self._load_dataset(dataset_path)
            return {
                "exists": True,
                "columns": list(df.columns),
                "sample_size": len(df),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_values": df.isnull().sum().to_dict()
            }
        except Exception as e:
            return {
                "exists": True,
                "error": str(e),
                "columns": [],
                "sample_size": 0,
                "dtypes": {}
            }

    def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load dataset from file."""
        if dataset_path.endswith('.csv'):
            return pd.read_csv(dataset_path)
        elif dataset_path.endswith('.json'):
            return pd.read_json(dataset_path)
        elif dataset_path.endswith('.parquet'):
            return pd.read_parquet(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
