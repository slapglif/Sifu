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
{
  "tasks": [
    {
      "idx": <integer>,
      "tool": <string>,
      "args": <object with string keys and any valid JSON values>,
      "dependencies": <array of integers>
    },
    ...more tasks...
  ],
  "thought": <string>
}

Available tools and their required args:
- research_topics: {"domain": <string>} (must be a valid domain name)
- synthesize_knowledge: {"sources": [{"id": <integer>}]} (must reference research results)
- generate_examples: {"knowledge": [{"id": <integer>}]} (must reference synthesis results)
- train_model: {"examples": [{"data": {"value": <string>}}]} (must reference example results)

CRITICAL TASK ORDER AND DEPENDENCY RULES:
1. Tasks MUST be in this exact order:
   a. research_topics (idx: 0, no dependencies)
   b. synthesize_knowledge (idx: 1, depends on research_topics)
   c. generate_examples (idx: 2, depends on synthesize_knowledge)
   d. train_model (idx: 3, depends on generate_examples)
2. Each task must have a unique integer index starting from 0
3. Tasks can only depend on tasks with lower indices
4. Dependencies must be an array of task indices
5. Dependencies must be valid task indices that exist in the plan
6. Dependencies must reflect the logical flow of data:
   - synthesize_knowledge needs results from research_topics
   - generate_examples needs results from synthesize_knowledge
   - train_model needs results from generate_examples

CRITICAL ARGUMENT RULES:
1. Each task MUST have the correct argument structure:
   a. research_topics: {"domain": "test_domain"}
   b. synthesize_knowledge: {"sources": [{"id": 0}]}
   c. generate_examples: {"knowledge": [{"id": 0}]}
   d. train_model: {"examples": [{"data": {"value": "example"}}]}
2. Arguments MUST be valid JSON objects
3. DO NOT use empty objects {} for arguments
4. Use the exact argument names and structure shown above
5. Use valid values for all arguments

CRITICAL FORMATTING RULES:
1. Use ONLY double quotes (") for strings and property names
2. Arrays must be comma-separated and enclosed in square brackets []
3. Objects must be comma-separated and enclosed in curly braces {}
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
18. DO NOT include any duplicate fields
19. DO NOT wrap the JSON in extra quotes
20. DO NOT include any explanatory comments in the JSON
21. DO NOT include any debugging or validation fields
22. The response MUST be parseable as a single valid JSON object
23. Each task MUST be a valid Task object with exactly these fields:
    - idx: integer
    - tool: string
    - args: object with required structure
    - dependencies: array of integers
24. The tasks array MUST be in order by idx starting from 0
25. Dependencies MUST be an array of integers, not an object or string
26. DO NOT include extra dependencies fields

Example valid response:
{
  "tasks": [
    {
      "idx": 0,
      "tool": "research_topics",
      "args": {"domain": "test_domain"},
      "dependencies": []
    },
    {
      "idx": 1,
      "tool": "synthesize_knowledge",
      "args": {"sources": [{"id": 0}]},
      "dependencies": [0]
    },
    {
      "idx": 2,
      "tool": "generate_examples",
      "args": {"knowledge": [{"id": 0}]},
      "dependencies": [1]
    },
    {
      "idx": 3,
      "tool": "train_model",
      "args": {"examples": [{"data": {"value": "example"}}]},
      "dependencies": [2]
    }
  ],
  "thought": "First research topics to gather sources, then synthesize knowledge from those sources, generate training examples from the knowledge, and finally train the model on those examples"
}

{format_instructions}

Remember:
1. Return ONLY valid JSON with the EXACT structure shown above
2. No text before or after the JSON
3. No explanation, just the JSON object
4. Always include all required fields
5. The "args" field MUST be an object/dictionary with required structure
6. Follow the tool-specific args format exactly
7. Double-check your JSON is valid and properly formatted
8. Do not wrap the JSON in extra quotes
9. Do not quote individual fields that should not be quoted
10. The "thought" field MUST be included
11. DO NOT include any duplicate fields
12. DO NOT include any debugging or validation fields
13. Tasks MUST have proper dependencies based on data flow
14. Task indices MUST be sequential starting from 0
15. Each task MUST be a valid Task object
16. Dependencies MUST be an array of integers
17. Tasks MUST be in the exact order specified above
18. Arguments MUST have the exact structure specified above"""

    human_template = """Generate a plan to process this state:

{{state}}

Remember:
1. Return ONLY valid JSON with the EXACT structure shown above
2. No text before or after the JSON
3. No explanation, just the JSON object
4. Always include all required fields
5. The "args" field MUST be an object/dictionary with required structure
6. Follow the tool-specific args format exactly
7. Double-check your JSON is valid and properly formatted
8. Do not wrap the JSON in extra quotes
9. Do not quote individual fields that should not be quoted
10. The "thought" field MUST be included
11. DO NOT include any duplicate fields
12. DO NOT include any debugging or validation fields
13. Tasks MUST have proper dependencies based on data flow
14. Task indices MUST be sequential starting from 0
15. Each task MUST be a valid Task object
16. Dependencies MUST be an array of integers
17. Tasks MUST be in the exact order specified above
18. Arguments MUST have the exact structure specified above"""

    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

def get_task_execution_prompt() -> ChatPromptTemplate:
    """Get the prompt template for task execution."""
    parser = PydanticOutputParser(pydantic_object=TaskResult)
    
    system_template = """You are a task execution expert that carries out planned tasks.
Your executions should be precise and reliable.

CRITICAL: You MUST respond with ONLY a valid JSON object, no other text or explanation.
The JSON object MUST EXACTLY match this structure:
{
  "task_id": <integer>,
  "result": <object with tool-specific format>,
  "error": <string or null>
}

Tool-specific result formats:
1. research_topics:
   "result": {
     "knowledge_sources": [
       {"content": <string>, "metadata": <object>}
     ]
   }

2. synthesize_knowledge:
   "result": {
     "synthetic_knowledge": [
       {
         "content": <string>,
         "patterns": [<object>],
         "hypotheses": [<object>],
         "relationships": [<object>],
         "confidence": <float>,
         "metadata": <object>
       }
     ]
   }

3. generate_examples:
   "result": {
     "training_examples": [
       {
         "input_text": <string>,
         "output_text": <string>,
         "metadata": <object>,
         "quality_score": <float>
       }
     ]
   }

4. train_model:
   "result": {
     "model_metrics": {
       "loss": <float>,
       "eval_loss": <float>,
       "train_samples": <integer>,
       "eval_samples": <integer>,
       "training_time": <float>
     }
   }

CRITICAL FORMATTING RULES:
1. Use ONLY double quotes (") for strings and property names
2. Arrays must be comma-separated and enclosed in square brackets []
3. Objects must be comma-separated and enclosed in curly braces {}
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

Example valid response for success:
{
  "task_id": 1,
  "result": {
    "knowledge_sources": [
      {
        "content": "Example source content",
        "metadata": {
          "source_type": "text",
          "confidence": 0.95,
          "timestamp": "2024-02-07T12:00:00Z"
        }
      }
    ]
  },
  "error": null
}

Example valid response for failure:
{
  "task_id": 2,
  "result": null,
  "error": "Failed to process input: invalid format"
}

{format_instructions}

Remember:
1. Return ONLY valid JSON with the EXACT structure shown above
2. No text before or after the JSON
3. No explanation, just the JSON object
4. Always include all required fields
5. Set error=null for successful execution
6. Follow the tool-specific result format exactly
7. NEVER return just a task_id without a result"""

    human_template = """Please execute this task:
{task}

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
{
  "complete": <boolean>,
  "thought": <string>,
  "replan": <boolean>,
  "feedback": <string or null>
}

The response MUST:
1. Be ONLY valid JSON (no other text, no JavaScript/Python syntax)
2. Have ALL of these fields:
   - "complete": true/false indicating if execution is complete
   - "thought": string explaining your decision reasoning
   - "replan": true/false indicating if replanning is needed
   - "feedback": string with feedback for replanning, or null if no feedback

Example valid response for success case:
{
  "complete": true,
  "thought": "All tasks completed successfully with expected results",
  "replan": false,
  "feedback": null
}

Example valid response for failure case:
{
  "complete": false,
  "thought": "Task 2 failed due to missing data",
  "replan": true,
  "feedback": "Need to add data gathering task before task 2"
}

{format_instructions}

Remember:
1. Return ONLY valid JSON with the EXACT structure shown above
2. No text before or after the JSON
3. No explanation, just the JSON object
4. Either complete=true OR replan=true must be true (both cannot be false)
5. If replan=true, feedback must be a non-null string"""

    human_template = """Please evaluate this execution state and make a join decision:
{state}

Remember to respond with ONLY the JSON object, no other text."""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    return prompt 