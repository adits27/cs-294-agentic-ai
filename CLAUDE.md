# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a course repository for CS 294: Agentic AI, implementing a multi-agent A/B testing validation system using LangGraph, LangChain, and Google's Gemini models. The system uses an orchestrating agent pattern with A2A (Agent-to-Agent) protocol for standardized inter-agent communication.

**Primary Use Case**: Validating A/B testing experiments by checking dataset quality, statistical adequacy, and experimental design.

## Development Setup

### Initial Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
# Get your API key from: https://makersuite.google.com/app/apikey
```

### Running the System

```bash
# Run the A/B testing validation demo
python main.py
```

### Project Structure

```
.
├── agents/
│   ├── __init__.py
│   ├── state.py                    # Shared state definitions (ValidationState, ABTestContext)
│   ├── a2a_protocol.py            # A2A protocol implementation for agent communication
│   ├── orchestrator.py            # Main orchestrating agent
│   └── data_validation_agent.py   # Sub-agent for data validation
├── datasource/
│   ├── data/
│   │   └── sample_ab_test.csv     # Sample A/B test dataset
│   └── README.md                   # Data source documentation
├── main.py                         # Entry point with A/B test example
├── requirements.txt                # Python dependencies
└── .env.example                    # Environment variable template
```

## Architecture Notes

### Multi-Agent System Design

The system implements a hierarchical multi-agent architecture with A2A protocol communication:

#### 1. Orchestrating Agent (agents/orchestrator.py)
- Entry point for A/B test validation tasks
- Coordinates sub-agents using A2A protocol for standardized communication
- Synthesizes results from sub-agents into final validation decision
- Built using LangGraph's StateGraph for workflow management
- Uses Google Gemini models (default: gemini-2.5-flash)

#### 2. Data Validation Agent (agents/data_validation_agent.py)
**First sub-agent** - Validates datasets for A/B testing experiments

**IMPORTANT**: All validation checks are **LLM-based**, not rule-based. The agent uses Gemini to make intelligent, context-aware decisions rather than rigid threshold checks.

Performs 4 critical checks:
1. **Data Relevance**: LLM analyzes if dataset aligns with the hypothesis by examining columns, data types, and content
2. **Metrics Relevance**: LLM evaluates if success metrics are appropriate for the hypothesis and calculable from the dataset
3. **Sample Size**: LLM evaluates adequacy based on power analysis calculations (scipy provides the math, LLM makes the judgment)
4. **A/B Test Suitability**: LLM assesses data structure, group columns, missing values, and overall fitness for A/B testing

Key features:
- **LLM-driven evaluation**: All pass/fail decisions made by Gemini with contextual understanding
- Supports CSV, JSON, and Parquet datasets
- Uses scipy for statistical calculations (input to LLM, not for decisions)
- Communicates via A2A protocol messages
- Returns detailed LLM analysis with structured results

#### 3. A2A Protocol (agents/a2a_protocol.py)
**Standardized agent communication protocol**

Components:
- `A2AMessage`: Pydantic model for structured messages
- `A2AProtocolHandler`: Creates, validates, and routes messages
- Message types: REQUEST, RESPONSE, ERROR, ACK
- Message status: PENDING, PROCESSING, COMPLETED, FAILED

All agent-to-agent communication flows through A2A messages, ensuring:
- Traceable message history
- Standardized request/response format
- Error handling and status tracking

#### 4. Shared State (agents/state.py)
Two main state schemas:

**ValidationState**: Main state that flows through the workflow
- Task description
- A/B test context (hypothesis, metrics, dataset path)
- Message history for LLM interactions
- A2A messages between agents
- Sub-agent results
- Final validation outcome

**ABTestContext**: A/B testing specific context
- Hypothesis
- Success metrics
- Dataset path
- Statistical parameters (effect size, significance, power)

### Workflow Graph

The orchestrator uses a linear workflow with A2A protocol:

```
START → orchestrator_entry → delegate_to_agents → synthesize_results → END
```

**Flow details:**
1. `orchestrator_entry`:
   - Analyzes the A/B test validation task
   - Plans validation strategy using LLM
   - Considers hypothesis, metrics, and dataset context

2. `delegate_to_agents`:
   - Creates A2A request messages for sub-agents
   - Sends to Data Validation Agent with full A/B test context
   - Receives and stores A2A response messages
   - Extracts validation results from responses

3. `synthesize_results`:
   - Formats data validation results
   - Uses LLM to synthesize final decision
   - Produces PASS/FAIL decision with recommendations

### Key Design Patterns

- **LLM-First Validation**: All evaluation decisions are made by LLM (Gemini), not hard-coded rules. This enables context-aware, nuanced judgments rather than rigid threshold checks.
- **A2A Protocol Communication**: All agents communicate through standardized A2A messages
- **State Management**: Shared `ValidationState` TypedDict flows through the graph
- **Message History**: LangChain messages for LLM interactions + A2A messages for agent communication
- **Graph-based Workflow**: LangGraph's StateGraph ensures deterministic execution
- **Statistical Context**: Uses scipy for calculations (power analysis, etc.) that provide context to the LLM, but the LLM makes the final decision
- **No Rule-Based Logic**: Avoided hard-coded thresholds or if/else validation logic in favor of LLM reasoning

### LLM Configuration

The system uses Google Gemini models via `langchain-google-genai`:
- **Default Model**: `gemini-2.5-flash` (latest, fast, cost-effective)
- **Alternative Models**: `gemini-1.5-flash`, `gemini-1.5-pro`
- **API Key**: Required via `GOOGLE_API_KEY` environment variable
- **Temperature**: Default 0.0 for deterministic outputs
- **Usage**: Both orchestrator and sub-agents use Gemini for analysis

### A/B Testing Validation

#### Supported Datasets
- **Formats**: CSV, JSON, Parquet
- **Location**: `datasource/data/` directory
- **Requirements**:
  - Group/variant column for test assignment
  - Columns for success metrics
  - Sufficient sample size
  - Clean, properly formatted data

#### Statistical Checks
- **Power Analysis**: Calculates required sample size using Cohen's d effect size
- **Default Parameters**: α=0.05, power=0.8
- **Configurable**: Effect size, significance level, and power can be adjusted

#### Example Usage

```python
from agents.orchestrator import OrchestratingAgent
from agents.state import ABTestContext

orchestrator = OrchestratingAgent()

ab_context = ABTestContext(
    hypothesis="New feature will increase conversion by 15%",
    success_metrics=["conversion", "revenue"],
    dataset_path="datasource/data/my_experiment.csv",
    expected_effect_size=0.3,
    significance_level=0.05,
    power=0.8
)

result = orchestrator.validate(
    task="Validate A/B test setup",
    ab_test_context=ab_context
)
```

### Adding New Sub-Agents

When implementing additional sub-agents:

1. **Create agent class** in `agents/` directory
2. **Implement A2A protocol**:
   - Initialize `A2AProtocolHandler` in `__init__`
   - Implement `process_request(request: A2AMessage)` method
   - Return `A2AMessage` response with results
3. **Update state** in `state.py` if new fields needed
4. **Update orchestrator** `_delegate_to_agents()` method:
   - Create A2A request with appropriate data
   - Call sub-agent's `process_request()`
   - Store request/response in `a2a_messages`
   - Extract results to state
5. **Update synthesis** to include new agent's results

Example sub-agent structure:
```python
class NewAgent:
    def __init__(self, model="gemini-2.5-flash"):
        self.agent_id = "new_agent"
        self.a2a_handler = A2AProtocolHandler(self.agent_id)
        self.llm = ChatGoogleGenerativeAI(model=model)

    def process_request(self, request: A2AMessage) -> A2AMessage:
        # Process request
        result = {"status": "completed"}
        return self.a2a_handler.create_response(
            request_message=request,
            result=result,
            status=MessageStatus.COMPLETED
        )
```

### Data Analysis Dependencies

The system uses:
- **pandas**: Dataset loading and manipulation (provides data context to LLM)
- **numpy**: Numerical operations (supports pandas operations)
- **scipy**: Statistical calculations (power analysis, distributions) - **calculations only, not decisions**

**Important**: These libraries provide mathematical context and data structure, but all validation decisions are made by the LLM, not by these libraries' output directly.

## Common Tasks

### Testing with Custom Datasets

1. Place dataset in `datasource/data/`
2. Update `ABTestContext` in `main.py`
3. Ensure dataset has required columns
4. Run `python main.py`

### Debugging A2A Messages

A2A messages are stored in the final state:
```python
result = orchestrator.validate(task, ab_test_context)
for msg in result['a2a_messages']:
    print(msg)  # View full message history
```

### Adjusting Statistical Parameters

Modify `ABTestContext` parameters:
- `expected_effect_size`: Cohen's d (0.2=small, 0.5=medium, 0.8=large)
- `significance_level`: Typically 0.05
- `power`: Typically 0.8 (80% power)
