"""LLM compiler system."""


from typing import Any, Dict, List
from langchain_core.output_parsers import PydanticOutputParser

from loguru import logger

from prompts.compiler import (
    Task,
    Plan,
    TaskResult,
    JoinDecision,
    CompilerState,
    get_plan_generation_prompt,
    get_task_execution_prompt,
    get_join_decision_prompt
)

from scripts.logging_config import (
    log_error_with_traceback,
)

class LLMCompiler:
    """LLM compiler system."""

    def __init__(self, llm):
        """Initialize with language model."""
        self.llm = llm

    async def generate_plan(self, state: CompilerState) -> Plan:
        """Generate execution plan."""
        try:
            prompt = get_plan_generation_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=Plan)
            result = await chain.ainvoke({
                "state": state
            })
            return result
            
        except Exception as e:
            log_error_with_traceback(e, "Error generating plan")
            raise

    async def execute_tasks(self, tasks: List[Task]) -> List[TaskResult]:
        """Execute planned tasks."""
        try:
            prompt = get_task_execution_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=TaskResult)
            results = []
            for task in tasks:
                result = await chain.ainvoke({
                    "task": task
                })
                results.append(result)
            return results
            
        except Exception as e:
            log_error_with_traceback(e, "Error executing tasks")
            raise

    async def make_join_decision(self, state: CompilerState) -> JoinDecision:
        """Make join decision."""
        try:
            prompt = get_join_decision_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=JoinDecision)
            result = await chain.ainvoke({
                "state": state
            })
            return result
            
        except Exception as e:
            log_error_with_traceback(e, "Error making join decision")
            raise

    async def run(self, initial_state: Dict[str, Any]) -> Any:
        """Run the LLM compiler workflow."""
        try:
            # Initialize state
            state = CompilerState(
                content=initial_state.get("content", ""),
                plan=None,
                results=[],
                join_decision=None,
                final_result=None
            )

            while True:
                try:
                    # Generate plan if needed
                    if not state["plan"]:
                        state["plan"] = await self.generate_plan(state)

                    # Execute tasks
                    if not state["results"]:
                        state["results"] = await self.execute_tasks(state["plan"].tasks)

                    # Make join decision
                    state["join_decision"] = await self.make_join_decision(state)

                    # Check completion
                    if state["join_decision"].complete:
                        return state["final_result"]

                    # Check replanning
                    if state["join_decision"].replan:
                        state["plan"] = None
                        state["results"] = []
                        continue

                    # Unexpected state
                    raise ValueError("Invalid join decision state")

                except Exception as e:
                    log_error_with_traceback(e, "Error in compiler workflow")
                    raise

        except Exception as e:
            log_error_with_traceback(e, "Error running compiler")
            raise 