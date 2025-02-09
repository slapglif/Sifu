"""Compiler prompts for LLM compiler system."""

from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field, validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

class Task(BaseModel):
    """Schema for a task in the execution plan."""
    idx: int = Field(..., description="Task index")
    tool: str = Field(..., description="Tool to use")
    args: Dict[str, Any] = Field(..., description="Tool arguments")
    dependencies: List[int] = Field(default_factory=list, description="Task dependencies")

class Plan(BaseModel):
    """Schema for an execution plan."""
    tasks: List[Task] = Field(..., description="List of tasks to execute")
    thought: Optional[str] = Field("", description="Planner's reasoning")

class TaskResult(BaseModel):
    """Schema for task execution results."""
    task_id: int = Field(..., description="Task ID")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result")
    error: Optional[str] = Field(None, description="Error if any")

    @validator("result")
    def validate_result(cls, v):
        """Validate that result includes a thought field if present."""
        if v is not None and "thought" not in v:
            raise ValueError("Result must include a thought field explaining the execution reasoning")
        return v

class JoinDecision(BaseModel):
    """Schema for join decisions."""
    complete: bool = Field(..., description="Whether execution is complete")
    thought: str = Field(..., description="Joiner's reasoning")
    replan: bool = Field(..., description="Whether replanning is needed")
    feedback: Optional[str] = Field(None, description="Feedback for replanning")

class CompilerState(TypedDict):
    """State for LLM compiler workflow."""
    content: str
    domain_name: str
    plan: Optional[Plan]
    results: List[TaskResult]
    join_decision: Optional[JoinDecision]
    final_result: Optional[Any]
    error: Optional[str]
    feedback: Optional[str]

def get_plan_generation_prompt() -> ChatPromptTemplate:
    """Get the prompt template for plan generation."""
    parser = PydanticOutputParser(pydantic_object=Plan)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are a planning expert that generates execution plans.
IMPORTANT: State variables MUST be referenced using {state[variable_name]} format, e.g. {state[domain_name]}.
Your plans should be efficient and well-organized.

{format_instructions}

CRITICAL: You MUST respond with ONLY a valid JSON object, no other text or explanation.
DO NOT wrap the JSON in markdown code blocks or any other formatting.
DO NOT include ```json or ``` markers.

Available tools and their required args:
- research_topics: {{"domain": "{state[domain_name]}"}} (MUST use {state[domain_name]} exactly like this)
- synthesize_knowledge: {{"sources": <array of source references>}}
- generate_examples: {{"knowledge": <array of knowledge references>}}
- train_model: {{"examples": <array of example objects with input_text and output_text>}}

Example valid response:
{{
  "tasks": [
    {{
      "idx": 0,
      "tool": "research_topics",
      "args": {{"domain": "{state[domain_name]}"}},
      "dependencies": []
    }},
    {{
      "idx": 1,
      "tool": "synthesize_knowledge",
      "args": {{"sources": [{{"id": 0}}]}},
      "dependencies": [0]
    }},
    {{
      "idx": 2,
      "tool": "generate_examples",
      "args": {{"knowledge": [{{"id": 1}}]}},
      "dependencies": [1]
    }},
    {{
      "idx": 3,
      "tool": "train_model",
      "args": {{"examples": [{{"id": 2}}]}},
      "dependencies": [2]
    }}
  ],
  "thought": "First research topics to gather sources, then synthesize knowledge from those sources, generate training examples from the knowledge, and finally train the model on those examples"
}}"""

    human_template = """Current state:
{state}

Generate a plan to execute the workflow based on the current state.

Remember:
1. Return ONLY a valid JSON object
2. Include both tasks and thought fields
3. Use {state[variable_name]} format for state variables (e.g. {state[domain_name]})
4. Make task IX start from 0 and increment sequentially
5. Ensure dependencies refer to valid task IDs
6. Include a clear thought explaining your plan
7. NEVER use {state} directly - always use {state[domain_name]} for the domain argument"""

    messages = [
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ]

    return ChatPromptTemplate(messages=messages)

def get_task_execution_prompt() -> ChatPromptTemplate:
    """Get the prompt template for task execution."""
    parser = PydanticOutputParser(pydantic_object=TaskResult)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are a task execution expert that carries out planned tasks.
Your executions should be precise and reliable.

{format_instructions}

CRITICAL: You MUST respond with ONLY a valid JSON object, no other text or explanation.
DO NOT wrap the JSON in markdown code blocks or any other formatting.
DO NOT include ```json or ``` markers.

Tool-specific result formats:
1. research_topics:
   "result": {{
     "knowledge_sources": [
       {{
         "content": <string>,
         "metadata": {{
           "source_type": "text",
           "confidence": <float>,
           "timestamp": <string in ISO format>
         }}
       }}
     ],
     "thought": <string explaining the research process>
   }}

2. synthesize_knowledge:
   "result": {{
     "synthetic_knowledge": [
       {{
         "content": <string>,
         "patterns": [<object>],
         "hypotheses": [<object>],
         "relationships": [<object>],
         "confidence": <float>,
         "metadata": <object>
       }}
     ],
     "thought": <string explaining the synthesis process>
   }}

3. generate_examples:
   "result": {{
     "training_examples": [
       {{
         "input_text": <string>,
         "output_text": <string>,
         "metadata": <object>,
         "quality_score": <float>
       }}
     ],
     "thought": <string explaining the example generation process>
   }}

4. train_model:
   "result": {{
     "model_metrics": {{
       "accuracy": <float>,
       "loss": <float>,
       "epochs": <integer>,
       "training_time": <float>
     }},
     "thought": <string explaining the training process>
   }}"""

    human_template = """Current state:
{state}

Execute the following task:
{task}

Remember to:
1. Return ONLY a valid JSON object
2. Include task_id, result, and error fields
3. Set task_id to the actual task index (not a template variable)
4. Include a thought field in the result explaining your execution
5. Set error to null if successful, or an error message if failed
6. Follow the tool-specific result format for the task's tool"""

    messages = [
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ]

    return ChatPromptTemplate(messages=messages)

def get_join_decision_prompt() -> ChatPromptTemplate:
    """Get the prompt template for join decisions."""
    parser = PydanticOutputParser(pydantic_object=JoinDecision)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are a decision-making expert that evaluates execution results.
Your decisions should be based on clear evidence and reasoning.

{format_instructions}

CRITICAL: You MUST respond with ONLY a valid JSON object, no other text or explanation.
DO NOT wrap the JSON in markdown code blocks or any other formatting.
DO NOT include ```json or ``` markers.

Example valid response for success case:
{{
  "complete": true,
  "thought": "All tasks completed successfully with expected results",
  "replan": false,
  "feedback": null
}}

Example valid response for failure case:
{{
  "complete": false,
  "thought": "Task 2 failed due to missing data",
  "replan": true,
  "feedback": "Need to add data_gathering task before task 2"
}}"""

    human_template = """Please evaluate this execution state and make a join decision:
{state}

Remember to:
1. Return ONLY a valid JSON object
2. Include complete, thought, replan, and feedback fields
3. Set complete=true ONLY if ALL tasks succeeded
4. Set replan=true if ANY task can be retried
5. Include clear thought explaining your decision
6. Provide feedback if replanning is needed
7. Check ALL task results and dependencies"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

    prompt = prompt.partial(format_instructions=format_instructions)

    return prompt 