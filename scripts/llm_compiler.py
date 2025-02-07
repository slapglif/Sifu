"""LLM compiler system."""

from typing import Any, Dict, List, Optional, Union, cast
from langchain_core.output_parsers import PydanticOutputParser
import json
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich import box
from loguru import logger
from pydantic import ValidationError

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
    log_info_with_context,
    create_progress,
    log_warning_with_context,
    console
)

class CompilerError(Exception):
    """Base class for compiler errors"""
    pass

class PlanningError(CompilerError):
    """Planning-related errors"""
    pass

class ExecutionError(CompilerError):
    """Task execution errors"""
    pass

class DecisionError(CompilerError):
    """Join decision errors"""
    pass

class LLMCompiler:
    """LLM compiler system."""

    def __init__(self, llm):
        """Initialize with language model."""
        try:
            if not llm:
                raise CompilerError("LLM instance is required")
                
            self.llm = llm
            log_info_with_context("Initialized LLM compiler", "Compiler")
            console.print(Panel("LLM Compiler Initialized", style="bold green"))
            
        except Exception as e:
            log_error_with_traceback(e, "Failed to initialize compiler", include_locals=True)
            raise CompilerError("Failed to initialize compiler") from e

    def _log_plan(self, plan: Plan):
        """Log plan details in a rich format"""
        try:
            table = Table(title="[bold]Execution Plan[/bold]", box=box.ROUNDED)
            table.add_column("Task ID", style="cyan")
            table.add_column("Tool", style="green")
            table.add_column("Dependencies", style="yellow")
            table.add_column("Arguments", style="magenta")
            
            for task in plan.tasks:
                table.add_row(
                    str(task.idx),
                    task.tool,
                    str(task.dependencies),
                    str(task.args)
                )
            
            console.print(table)
            console.print(Panel(f"Planning Thought: {plan.thought}", title="[bold]Planning Reasoning[/bold]", border_style="blue"))
            
        except Exception as e:
            log_error_with_traceback(e, "Failed to log plan", include_locals=True)

    def _log_task_result(self, result: TaskResult):
        """Log task result in a rich format"""
        try:
            if result.error:
                console.print(Panel(
                    f"[red]Error:[/red] {result.error}",
                    title=f"[bold red]Task {result.task_id} Failed[/bold red]",
                    border_style="red"
                ))
            else:
                console.print(Panel(
                    Syntax(str(result.result), "python", theme="monokai"),
                    title=f"[bold green]Task {result.task_id} Completed[/bold green]",
                    border_style="green"
                ))
                
        except Exception as e:
            log_error_with_traceback(e, "Failed to log task result", include_locals=True)

    def _log_join_decision(self, decision: JoinDecision):
        """Log join decision in a rich format"""
        try:
            status = "[green]Complete[/green]" if decision.complete else "[yellow]Replan[/yellow]" if decision.replan else "[red]Unknown[/red]"
            console.print(Panel(
                f"Status: {status}\nThought: {decision.thought}\nFeedback: {decision.feedback or 'None'}",
                title="[bold]Join Decision[/bold]",
                border_style="cyan"
            ))
            
        except Exception as e:
            log_error_with_traceback(e, "Failed to log join decision", include_locals=True)

    def _format_state(self, state: CompilerState) -> Dict[str, Any]:
        """Format state for LLM input."""
        plan = state.get("plan")
        join_decision = state.get("join_decision")
        
        return {
            "content": state.get("content", ""),
            "domain_name": state.get("domain_name", "test_domain"),
            "plan": plan.dict() if plan else None,
            "results": [r.dict() for r in state.get("results", [])],
            "join_decision": join_decision.dict() if join_decision else None,
            "final_result": state.get("final_result")
        }

    async def generate_plan(self, state: CompilerState) -> Plan:
        """Generate execution plan."""
        try:
            log_info_with_context("Starting plan generation", "Planning")
            console.print("\n[bold cyan]Generating Execution Plan...[/bold cyan]")
            
            # Format state for LLM
            try:
                formatted_state = self._format_state(state)
            except Exception as e:
                log_error_with_traceback(e, "Failed to format state", include_locals=True)
                raise PlanningError("Failed to format state for planning") from e
            
            # Get plan from LLM
            try:
                prompt = get_plan_generation_prompt()
                chain = prompt | self.llm | PydanticOutputParser(pydantic_object=Plan)
                response = await chain.ainvoke({"state": formatted_state})
            except Exception as e:
                log_error_with_traceback(e, "Failed to get plan from LLM", include_locals=True)
                raise PlanningError("Failed to generate plan") from e
            
            # Parse and validate response
            try:
                plan = self._parse_plan_response(response)
                if not plan.tasks:
                    log_warning_with_context("Generated plan has no tasks", "Planning", include_locals=True)
                    
                # Log the generated plan
                log_info_with_context(f"Generated plan with {len(plan.tasks)} tasks", "Planning")
                self._log_plan(plan)
                return plan
                
            except Exception as e:
                log_error_with_traceback(e, "Failed to parse plan response", include_locals=True)
                raise PlanningError("Failed to parse plan response") from e
            
        except Exception as e:
            log_error_with_traceback(e, "Error in plan generation", include_locals=True)
            if isinstance(e, (PlanningError, CompilerError)):
                raise
            raise PlanningError("Plan generation failed") from e

    def _parse_plan_response(self, response: Union[str, Plan, Dict[str, Any]]) -> Plan:
        """Parse and validate plan response."""
        try:
            if isinstance(response, Plan):
                return response
                
            if isinstance(response, str):
                # Handle JSON string
                try:
                    if response.startswith('"{') and response.endswith('}"'):
                        response = response[1:-1]
                    parsed = json.loads(response)
                except json.JSONDecodeError as e:
                    raise PlanningError(f"Invalid JSON format: {str(e)}")
            else:
                parsed = response
                
            # Validate parsed response
            if not isinstance(parsed, dict):
                raise PlanningError("Response must be a dictionary")
                
            if "tasks" not in parsed:
                raise PlanningError("Response missing 'tasks' field")
                
            if "thought" not in parsed:
                raise PlanningError("Response missing 'thought' field")
                
            # Validate tasks array
            if not isinstance(parsed["tasks"], list):
                raise PlanningError("'tasks' must be an array")
                
            # Validate and fix each task
            for task in parsed["tasks"]:
                if not isinstance(task, dict):
                    raise PlanningError("Each task must be a dictionary")
                    
                # Validate required task fields
                required_fields = ["idx", "tool", "args", "dependencies"]
                for field in required_fields:
                    if field not in task:
                        raise PlanningError(f"Task missing required field: {field}")
                        
                # Ensure args is a dictionary
                if not isinstance(task["args"], dict):
                    if isinstance(task["args"], list) and len(task["args"]) == 1:
                        try:
                            task["args"] = json.loads(task["args"][0])
                        except:
                            raise PlanningError(f"Invalid args format for task {task['idx']}")
                    else:
                        raise PlanningError(f"Args must be a dictionary for task {task['idx']}")
                        
                # Ensure dependencies is a list
                if not isinstance(task["dependencies"], list):
                    raise PlanningError(f"Dependencies must be a list for task {task['idx']}")
                    
            # Create Plan object
            try:
                return Plan(
                    tasks=[Task(**task) for task in parsed["tasks"]],
                    thought=parsed["thought"]
                )
            except ValidationError as e:
                raise PlanningError(f"Invalid plan format: {str(e)}")
                
        except Exception as e:
            log_error_with_traceback(e, "Failed to parse plan response", include_locals=True)
            if isinstance(e, PlanningError):
                raise
            raise PlanningError("Failed to parse plan response") from e

    async def execute_tasks(self, tasks: List[Task]) -> List[TaskResult]:
        """Execute planned tasks."""
        try:
            if not tasks:
                log_warning_with_context("No tasks to execute", "Execution", include_locals=True)
                return []
                
            log_info_with_context(f"Starting execution of {len(tasks)} tasks", "Execution")
            console.print("\n[bold yellow]Executing Tasks...[/bold yellow]")
            
            prompt = get_task_execution_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=TaskResult)
            results = []
            
            # Create progress bar
            progress = create_progress()
            task_progress = progress.add_task(
                "[yellow]Executing tasks...",
                total=len(tasks)
            )
            
            for task in tasks:
                try:
                    log_info_with_context(f"Executing task {task.idx}: {task.tool}", "Execution")
                    
                    # Check dependencies
                    deps_met = all(
                        any(r.task_id == dep_id and not r.error for r in results)
                        for dep_id in task.dependencies
                    )
                    if not deps_met:
                        log_warning_with_context(f"Dependencies not met for task {task.idx}", "Execution", include_locals=True)
                        results.append(TaskResult(
                            task_id=task.idx,
                            error="Dependencies not met",
                            result=None
                        ))
                        continue
                    
                    # Execute task
                    try:
                        response = await chain.ainvoke({"task": task})
                    except Exception as e:
                        log_error_with_traceback(e, f"Failed to execute task {task.idx}", include_locals=True)
                        results.append(TaskResult(
                            task_id=task.idx,
                            error=str(e),
                            result=None
                        ))
                        continue
                    
                    # Parse response
                    try:
                        result = self._parse_task_result(response, task.idx)
                        results.append(result)
                        self._log_task_result(result)
                    except Exception as e:
                        log_error_with_traceback(e, f"Failed to parse task {task.idx} result", include_locals=True)
                        results.append(TaskResult(
                            task_id=task.idx,
                            error=str(e),
                            result=None
                        ))
                        
                except Exception as e:
                    log_error_with_traceback(e, f"Error executing task {task.idx}", include_locals=True)
                    results.append(TaskResult(
                        task_id=task.idx,
                        error=str(e),
                        result=None
                    ))
                finally:
                    progress.update(task_progress, advance=1)
                    
            return results
            
        except Exception as e:
            log_error_with_traceback(e, "Error in task execution", include_locals=True)
            if isinstance(e, (ExecutionError, CompilerError)):
                raise
            raise ExecutionError("Task execution failed") from e

    def _parse_task_result(self, response: Union[str, TaskResult, Dict[str, Any]], task_id: int) -> TaskResult:
        """Parse and validate task result."""
        try:
            if isinstance(response, TaskResult):
                return response
                
            if isinstance(response, str):
                try:
                    # Handle double-encoded JSON
                    parsed = json.loads(response)
                    if isinstance(parsed, str):
                        parsed = json.loads(parsed)
                except json.JSONDecodeError as e:
                    raise ExecutionError(f"Invalid JSON format: {str(e)}")
            else:
                parsed = response
                
            # Validate parsed response
            if not isinstance(parsed, dict):
                raise ExecutionError("Response must be a dictionary")
                
            # Create TaskResult object
            try:
                result = TaskResult(
                    task_id=task_id,
                    result=parsed.get("result"),
                    error=parsed.get("error")
                )
                return result
            except ValidationError as e:
                raise ExecutionError(f"Invalid task result format: {str(e)}")
                
        except Exception as e:
            log_error_with_traceback(e, "Failed to parse task result", include_locals=True)
            if isinstance(e, ExecutionError):
                raise
            raise ExecutionError("Failed to parse task result") from e

    async def make_join_decision(self, state: CompilerState) -> JoinDecision:
        """Make join decision."""
        try:
            log_info_with_context("Making join decision", "Decision")
            console.print("\n[bold green]Making Join Decision...[/bold green]")
            
            # Format state for LLM
            try:
                formatted_state = self._format_state(state)
            except Exception as e:
                log_error_with_traceback(e, "Failed to format state", include_locals=True)
                raise DecisionError("Failed to format state for decision") from e
            
            # Get decision from LLM
            try:
                prompt = get_join_decision_prompt()
                chain = prompt | self.llm | PydanticOutputParser(pydantic_object=JoinDecision)
                response = await chain.ainvoke({"state": formatted_state})
            except Exception as e:
                log_error_with_traceback(e, "Failed to get decision from LLM", include_locals=True)
                raise DecisionError("Failed to make decision") from e
            
            # Parse and validate response
            try:
                decision = self._parse_join_decision(response)
                self._log_join_decision(decision)
                return decision
            except Exception as e:
                log_error_with_traceback(e, "Failed to parse decision response", include_locals=True)
                raise DecisionError("Failed to parse decision response") from e
            
        except Exception as e:
            log_error_with_traceback(e, "Error in join decision", include_locals=True)
            if isinstance(e, (DecisionError, CompilerError)):
                raise
            raise DecisionError("Join decision failed") from e

    def _parse_join_decision(self, response: Union[str, JoinDecision, Dict[str, Any]]) -> JoinDecision:
        """Parse and validate join decision."""
        try:
            if isinstance(response, JoinDecision):
                return response
                
            if isinstance(response, str):
                try:
                    if response.startswith('"{') and response.endswith('}"'):
                        response = response[1:-1]
                    parsed = json.loads(response)
                except json.JSONDecodeError as e:
                    raise DecisionError(f"Invalid JSON format: {str(e)}")
            else:
                parsed = response
                
            # Validate parsed response
            if not isinstance(parsed, dict):
                raise DecisionError("Response must be a dictionary")
                
            # Create JoinDecision object
            try:
                return JoinDecision(
                    complete=parsed.get("complete", False),
                    replan=parsed.get("replan", False),
                    thought=parsed.get("thought", ""),
                    feedback=parsed.get("feedback")
                )
            except ValidationError as e:
                raise DecisionError(f"Invalid join decision format: {str(e)}")
                
        except Exception as e:
            log_error_with_traceback(e, "Failed to parse join decision", include_locals=True)
            if isinstance(e, DecisionError):
                raise
            raise DecisionError("Failed to parse join decision") from e

    async def run(self, initial_state: Dict[str, Any]) -> Any:
        """Run the compiler workflow."""
        try:
            if not initial_state:
                raise CompilerError("Initial state is required")
                
            # Convert initial state to CompilerState
            state: CompilerState = {
                "content": initial_state.get("content", ""),
                "plan": None,
                "results": [],
                "join_decision": None,
                "final_result": None
            }
            
            max_iterations = 5
            iteration = 0
            
            while iteration < max_iterations:
                log_info_with_context(f"Starting iteration {iteration + 1}", "Compiler")
                
                try:
                    # Generate plan
                    plan = await self.generate_plan(state)
                    state["plan"] = plan
                    
                    # Execute tasks
                    results = await self.execute_tasks(plan.tasks)
                    state["results"] = results
                    
                    # Check for task failures
                    failed_tasks = [r for r in results if r.error]
                    if failed_tasks:
                        log_warning_with_context(
                            f"{len(failed_tasks)} tasks failed",
                            "Compiler",
                            include_locals=True
                        )
                    
                    # Make join decision
                    decision = await self.make_join_decision(state)
                    state["join_decision"] = decision
                    
                    if decision.complete:
                        # Generate final result
                        final_result = await self._generate_final_result(state)
                        state["final_result"] = final_result
                        return final_result
                        
                    if not decision.replan:
                        log_warning_with_context(
                            "Join decision neither complete nor replan",
                            "Compiler",
                            include_locals=True
                        )
                        break
                        
                except Exception as e:
                    log_error_with_traceback(e, f"Error in iteration {iteration + 1}", include_locals=True)
                    if isinstance(e, CompilerError):
                        raise
                    raise CompilerError(f"Iteration {iteration + 1} failed") from e
                    
                iteration += 1
                
            log_warning_with_context(
                f"Max iterations ({max_iterations}) reached",
                "Compiler",
                include_locals=True
            )
            return state.get("final_result")
            
        except Exception as e:
            log_error_with_traceback(e, "Error in compiler workflow", include_locals=True)
            if isinstance(e, CompilerError):
                raise
            raise CompilerError("Compiler workflow failed") from e

    async def _generate_final_result(self, state: CompilerState) -> Any:
        """Generate final result."""
        try:
            log_info_with_context("Generating final result", "Compiler")
            
            # Get successful results
            results = [r for r in state.get("results", []) if not r.error]
            if not results:
                log_warning_with_context("No successful results to generate final result", "Compiler", include_locals=True)
                return None
                
            # Combine results
            final_result = results[-1].result
            log_info_with_context("Generated final result", "Compiler")
            return final_result
            
        except Exception as e:
            log_error_with_traceback(e, "Error generating final result", include_locals=True)
            if isinstance(e, CompilerError):
                raise
            raise CompilerError("Failed to generate final result") from e 