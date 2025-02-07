"""Compiler prompts for LLM compiler system."""

from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field
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
    thought: str = Field(..., description="Planner's reasoning")

class TaskResult(BaseModel):
    """Schema for task execution results."""
    task_id: int = Field(..., description="Task ID")
    result: Any = Field(..., description="Task result")
    error: Optional[str] = Field(None, description="Error if any")

class JoinDecision(BaseModel):
    """Schema for join decisions."""
    complete: bool = Field(..., description="Whether execution is complete")
    thought: str = Field(..., description="Joiner's reasoning")
    replan: bool = Field(..., description="Whether replanning is needed")
    feedback: Optional[str] = Field(None, description="Feedback for replanning")

class CompilerState(TypedDict):
    """State for LLM compiler workflow."""
    content: str
    plan: Optional[Plan]
    results: List[TaskResult]
    join_decision: Optional[JoinDecision]
    final_result: Optional[Any]

def get_plan_generation_prompt() -> ChatPromptTemplate:
    """Get the prompt template for plan generation."""
    parser = PydanticOutputParser(pydantic_object=Plan)
    
    system_template = """You are a planning expert that generates execution plans.
Your plans should be efficient and well-organized.

CRITICAL: You MUST respond with ONLY a valid JSON object, no other text or explanation.
The JSON object MUST EXACTLY match this structure:
{{
  "tasks": [
    {{
      "idx": <integer>,
      "tool": <string>,
      "args": <object with string keys and any valid JSON values>,
      "dependencies": <array of integers>
    }},
  ],
  "thought": <string>
}}

Available tools and their required args:
- research_topics: {{"domain": <string>}} (MUST use state.domain_name for the domain value)
- synthesize_knowledge: {{"sources": <array>}}
- generate_examples: {{"knowledge": <array>}}
- train_model: {{"examples": <array>}}

CRITICAL FORMATTING RULES:
1. Use ONLY double quotes (") for strings and property names
2. Arrays must be comma-separated and enclosed in square brackets []
3. Objects must be comma-separated and enclosed in curly braces {{}}
4. No trailing commas after the last item in arrays or objects
5. No comments or explanatory text
6. No JavaScript/Python syntax - ONLY valid JSON
7. No extra fields or properties beyond what is specified
8. No malformed JSON or syntax errors
9. No single quotes (') - use double quotes (") only
10. No unescaped newlines in strings
11. No extra whitespace or indentation
12. No extra quotes around the entire JSON object
13. No extra quotes around individual fields
14. No extra quotes around arrays or objects
15. The "args" field MUST be an object/dictionary, not an array
16. Each task MUST have all required fields: idx, tool, args, dependencies
17. The "thought" field MUST be included and MUST be a string
18. For research_topics tool, you MUST use state.domain_name as the domain value
19. For research_topics tool, args MUST ONLY contain the "domain" field

Example valid response:
{{
  "tasks": [
    {{
      "idx": 0,
      "tool": "research_topics",
      "args": {{"domain": "test_domain"}},
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
      "args": {{"knowledge": [{{"id": 0}}]}},
      "dependencies": [1]
    }},
    {{
      "idx": 3,
      "tool": "train_model",
      "args": {{"examples": [{{"data": {{"value": "example"}}]}}]}},
      "dependencies": [2]
    }}
  ],
  "thought": "First research topics to gather sources, then synthesize knowledge from those sources, generate training examples from the knowledge, and finally train the model on those examples"
}}

{format_instructions}

Remember:
1. Return ONLY valid JSON with the EXACT structure shown above
2. No text before or after the JSON
3. No explanation, just the JSON object
4. Always include all required fields
5. The "args" field MUST be an object/dictionary
6. Follow the tool-specific args format exactly
7. Double-check your JSON is valid and properly formatted
8. Do not wrap the JSON in extra quotes
9. Do not quote individual fields that should not be quoted
10. The "thought" field MUST be included
11. For research_topics tool, you MUST use state.domain_name as the domain value
12. For research_topics tool, args MUST ONLY contain the "domain" field"""

    human_template = """Please generate an execution plan for this state:
{state}

Remember to respond with ONLY the JSON object, no other text."""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    return prompt

def get_task_execution_prompt() -> ChatPromptTemplate:
    """Get the prompt template for task execution."""
    parser = PydanticOutputParser(pydantic_object=TaskResult)
    
    system_template = """You are a task execution expert that carries out planned tasks.
Your executions should be precise and reliable.

CRITICAL: You MUST respond with ONLY a valid JSON object, no other text or explanation.
The JSON object MUST EXACTLY match this structure:
{{
  "task_id": <integer>,
  "result": <object with tool-specific format>,
  "error": <string or null>
}}

Tool-specific result formats:
1. research_topics:
   "result": {{
     "knowledge_sources": [
       {{"content": <string>, "metadata": <object>}}
     ]
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
     ]
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
     ]
   }}

4. train_model:
   "result": {{
     "model_metrics": {{
       "loss": <float>,
       "eval_loss": <float>,
       "train_samples": <integer>,
       "eval_samples": <integer>,
       "training_time": <float>
     }}
   }}

CRITICAL FORMATTING RULES:
1. Use ONLY double quotes (") for strings and property names
2. Arrays must be comma-separated and enclosed in square brackets []
3. Objects must be comma-separated and enclosed in curly braces {{}}
4. No trailing commas after the last item in arrays or objects
5. No comments or explanatory text
6. No JavaScript/Python syntax - ONLY valid JSON
7. No extra fields or properties beyond what is specified
8. No malformed JSON or syntax errors
9. No single quotes (') - use double quotes (") only
10. No unescaped newlines in strings
11. No extra whitespace or indentation
12. No extra quotes around the entire JSON object
13. No extra quotes around individual fields
14. No extra quotes around arrays or objects
15. ALWAYS include both task_id and result fields
16. NEVER return just a task_id without a result
17. If execution fails, set result=null and include error message
18. task_id MUST match the idx field from the task being executed
19. Dependencies MUST be checked - if any dependency task failed, this task should fail with "Dependencies not met"
20. The result field MUST match the tool-specific format exactly
21. The result field MUST be null if there is an error
22. The error field MUST be null if there is a result
23. NEVER include any text before or after the JSON object
24. NEVER include any comments or explanations
25. NEVER include any extra fields or properties

Example valid response for success:
{{
  "task_id": 0,
  "result": {{
    "knowledge_sources": [
      {{
        "content": "Example source content",
        "metadata": {{
          "source_type": "text",
          "confidence": 0.95,
          "timestamp": "2024-02-07T12:00:00Z"
        }}
      }}
    ]
  }},
  "error": null
}}

Example valid response for failure:
{{
  "task_id": 1,
  "result": null,
  "error": "Dependencies not met - task 0 failed"
}}

{format_instructions}

Remember:
1. Return ONLY valid JSON with the EXACT structure shown above
2. No text before or after the JSON
3. No explanation, just the JSON object
4. Always include all required fields
5. Set error=null for successful execution
6. Follow the tool-specific result format exactly
7. NEVER return just a task_id without a result
8. task_id MUST match the idx field from the task being executed
9. Dependencies MUST be checked - if any dependency task failed, this task should fail with "Dependencies not met"
10. The result field MUST match the tool-specific format exactly
11. The result field MUST be null if there is an error
12. The error field MUST be null if there is a result
13. NEVER include any text before or after the JSON object
14. NEVER include any comments or explanations
15. NEVER include any extra fields or properties"""

    human_template = """Please execute this task:

Task:
{{
  "idx": {task.idx},
  "tool": "{task.tool}",
  "args": {task.args},
  "dependencies": {task.dependencies}
}}

Remember to respond with ONLY the JSON object, no other text."""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    return prompt

def get_join_decision_prompt() -> ChatPromptTemplate:
    """Get the prompt template for join decisions."""
    parser = PydanticOutputParser(pydantic_object=JoinDecision)
    
    system_template = """You are a decision-making expert that evaluates execution results.
Your decisions should be based on clear evidence and reasoning.

CRITICAL: You MUST respond with ONLY a valid JSON object, no other text or explanation.
The JSON object MUST EXACTLY match this structure:
{{
  "complete": <boolean>,
  "thought": <string>,
  "replan": <boolean>,
  "feedback": <string or null>
}}

The response MUST:
1. Be ONLY valid JSON (no other text, no JavaScript/Python syntax)
2. Have ALL of these fields:
   - "complete": true/false indicating if execution is complete
   - "thought": string explaining your decision reasoning
   - "replan": true/false indicating if replanning is needed
   - "feedback": string with feedback for replanning, or null if no feedback
3. Set complete=true ONLY if ALL tasks succeeded (no errors and valid results)
4. Set replan=true if ANY task failed but can be retried
5. If replan=true, feedback MUST explain what needs to be fixed
6. Check task dependencies - if a task failed due to dependencies, replan=true

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
  "feedback": "Need to add data gathering task before task 2"
}}

{format_instructions}

Remember:
1. Return ONLY valid JSON with the EXACT structure shown above
2. No text before or after the JSON
3. No explanation, just the JSON object
4. Either complete=true OR replan=true must be true (both cannot be false)
5. If replan=true, feedback must be a non-null string
6. Check ALL task results and dependencies before deciding
7. Set complete=true ONLY if ALL tasks succeeded
8. Set replan=true if ANY task can be retried"""

    human_template = """Please evaluate this execution state and make a join decision:
{state}

Remember to respond with ONLY the JSON object, no other text."""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    return prompt 