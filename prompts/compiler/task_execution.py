"""Task execution prompts."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

class TaskResult(BaseModel):
    """Schema for task execution results."""
    task_id: int = Field(..., description="Task ID")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result")
    error: Optional[str] = Field(None, description="Error if any")

def get_task_execution_prompt() -> ChatPromptTemplate:
    """Get the task execution prompt template."""
    parser = PydanticOutputParser(pydantic_object=TaskResult)
    format_instructions = parser.get_format_instructions()
    
    system_template = """Execute knowledge acquisition tasks.
{format_instructions}

CRITICAL: You MUST respond with ONLY a valid JSON object, no other text or explanation.
DO NOT wrap the JSON in markdown code blocks or any other formatting.
DO NOT include ```json or ``` markers.
The JSON object MUST EXACTLY match this structure:
{{
  "task_id": <integer - MUST be the idx value from the task>,
  "result": <object with tool-specific format or null>,
  "error": <string or null>
}}

CRITICAL: The task_id field MUST be an integer matching the idx value from the task.
DO NOT include the entire task object as the task_id.

CRITICAL: State variables MUST be referenced using {state[variable_name]} format.
For example, to reference the domain_name from state, use: {state[domain_name]}
DO NOT use {{state}} or {{state.variable_name}} formats - these are invalid and will cause errors.

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
       "loss": <float>,
       "eval_loss": <float>,
       "train_samples": <integer>,
       "eval_samples": <integer>,
       "training_time": <float>
     }},
     "thought": <string explaining the training process>
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
26. ALWAYS include a thought field in the result object explaining your reasoning
27. State variables in args (e.g. {state[domain_name]}) will be replaced with actual values
28. Function names MUST start with a letter or underscore
29. Function names MUST contain only letters, numbers, underscores, dots, or dashes
30. Function names MUST be at most 64 characters long
31. For train_model task, examples must be an array of objects with input_text and output_text fields
32. For train_model task, if examples are referenced by ID, fetch full examples from system state

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
    ],
    "thought": "Successfully researched topics and extracted knowledge"
  }},
  "error": null
}}

Example valid response for failure:
{{
  "task_id": 1,
  "result": null,
  "error": "Dependencies not met - task 0 failed"
}}"""

    human_template = """Execute this task:

Task:
{{
  "idx": {task.idx},
  "tool": "{task.tool}",
  "args": {task.args},
  "dependencies": {task.dependencies}
}}

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
15. NEVER include any extra fields or properties
16. ALWAYS include a thought field in the result object
17. State variables in args will be replaced with actual values
18. Function names MUST start with a letter or underscore
19. Function names MUST contain only letters, numbers, underscores, dots, or dashes
20. Function names MUST be at most 64 characters long"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])
    
    prompt = prompt.partial(format_instructions=format_instructions)
    return prompt 