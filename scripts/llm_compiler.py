"""LLM compiler system."""

import os
from typing import Any, Dict, List, Optional, Union, cast, Callable, Awaitable
from langchain_core.output_parsers import PydanticOutputParser
import json
from langchain_ollama import ChatOllama
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich import box
from loguru import logger
from pydantic import ValidationError, SecretStr
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel

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

from scripts.chat_langchain import ChatLangChain
from scripts.logging_config import (
    log_info_with_context,
    log_warning_with_context,
    log_error_with_context,
    log_error_with_traceback,
    create_progress,
    cleanup_progress,
    console
)

class CompilerError(Exception):
    """Base class for compiler errors"""
    pass

class PlanningError(CompilerError):
    """Error during plan generation"""
    pass

class ExecutionError(CompilerError):
    """Error during task execution"""
    pass

class DecisionError(CompilerError):
    """Error during join decision"""
    pass

class LLMCompiler:
    """LLM compiler system."""

    def __init__(self, llm: BaseChatModel):
        """Initialize with language model."""
        try:
            if not llm:
                raise CompilerError("LLM instance is required")
                
            self.llm = llm
            self.state: CompilerState = {
                "content": "",
                "domain_name": "",
                "plan": None,
                "results": [],
                "join_decision": None,
                "final_result": None,
                "error": None,
                "feedback": None
            }
            self.tools: Dict[str, Callable[..., Awaitable[Dict[str, Any]]]] = {}
            log_info_with_context("Initialized LLM compiler", "Compiler")
            console.print(Panel("LLM Compiler Initialized", style="bold green"))
            
        except Exception as e:
            log_error_with_traceback(e, "Failed to initialize compiler", include_locals=True)
            raise CompilerError("Failed to initialize compiler") from e

    def register_tool(self, name: str, func: Callable[..., Awaitable[Dict[str, Any]]]):
        """Register a tool with the compiler."""
        self.tools[name] = func
        log_info_with_context(f"Registered tool: {name}", "Compiler")

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

    def _format_state(self, state: Any) -> Dict[str, Any]:
        """Format state for LLM input."""
        if hasattr(state, "__fields__"):  # If it's a Pydantic model
            formatted_state = {}
            for field in state.__fields__:
                if hasattr(state, field):
                    value = getattr(state, field)
                    if isinstance(value, str):
                        formatted_state[field] = f"{{state[{field}]}}"
                    else:
                        formatted_state[field] = value
            return formatted_state
        elif isinstance(state, dict):  # If it's a dictionary
            formatted_state = {}
            for key, value in state.items():
                if isinstance(value, str):
                    formatted_state[key] = f"{{state[{key}]}}"
                else:
                    formatted_state[key] = value
            return formatted_state
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")

    async def generate_plan(self, state: CompilerState) -> Plan:
        """Generate a plan based on the current state."""
        try:
            # Format state for prompt using _format_state helper
            formatted_state = self._format_state(state)
            
            # Get predefined prompt template
            prompt = get_plan_generation_prompt()
            
            # Create chain with JSON formatting
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=Plan)
            
            # Generate plan with thought field
            response = await chain.ainvoke({
                "state": formatted_state,
                "domain_name": state.get("domain_name", "")  # Pass domain_name explicitly
            })
            
            # Validate and fix state variable format in task args
            for task in response.tasks:
                for key, value in task.args.items():
                    if isinstance(value, str):
                        if value == "{state}":
                            if task.tool == "research_topics" and key == "domain" and "domain_name" in state:
                                task.args[key] = "{state[domain_name]}"
                            else:
                                error_msg = f"Invalid state variable format in task {task.idx}. Use {{state[variable_name]}} instead of {{state}}"
                                log_error_with_context(error_msg, "Planning")
                                raise PlanningError(error_msg)
                        elif "{state." in value:
                            error_msg = f"Invalid state variable format in task {task.idx}. Use {{state[variable_name]}} instead of {value}"
                            log_error_with_context(error_msg, "Planning")
                            raise PlanningError(error_msg)
            
            return response

        except Exception as e:
            log_error_with_traceback(e, "Failed to generate plan")
            raise PlanningError("Failed to generate plan") from e

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
                        
                # Validate state variable format
                for key, value in task["args"].items():
                    if isinstance(value, str):
                        if value == "{state}":
                            error_msg = f"Invalid state variable format in task {task['idx']}. Use {{state[variable_name]}} instead of {{state}}"
                            log_error_with_context(error_msg, "Planning")
                            raise PlanningError(error_msg)
                        elif "{state." in value:
                            error_msg = f"Invalid state variable format in task {task['idx']}. Use {{state[variable_name]}} instead of {value}"
                            log_error_with_context(error_msg, "Planning")
                            raise PlanningError(error_msg)
                        
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

    async def execute_tasks(self, tasks: List[Task], state: CompilerState) -> List[TaskResult]:
        """Execute tasks in order."""
        try:
            log_info_with_context(f"Starting execution of {len(tasks)} tasks", "Execution")
            
            # Initialize progress
            progress = await create_progress()
            task_progress = progress.add_task("[cyan]Executing tasks...", total=len(tasks))
            
            # Track results
            results: List[TaskResult] = []
            result_map: Dict[int, Any] = {}  # Map task IDs to their results
            
            # Execute tasks in order
            for task in tasks:
                try:
                    log_info_with_context(f"Executing task {task.idx}: {task.tool}", "Execution")
                    
                    # Check dependencies and gather their results
                    dep_results = {}
                    deps_met = True
                    for dep in task.dependencies:
                        if dep not in result_map:
                            deps_met = False
                            error_msg = f"Dependency {dep} not found for task {task.idx}"
                            log_error_with_context(error_msg, "Execution")
                            break
                        dep_results[dep] = result_map[dep]
                    
                    if not deps_met:
                        error_msg = f"Dependencies not met for task {task.idx}"
                        log_error_with_context(error_msg, "Execution")
                        results.append(TaskResult(task_id=task.idx, result=None, error=error_msg))
                        continue
                    
                    # Format task args with state variables and dependency results
                    formatted_args = {}
                    for arg_name, arg_value in task.args.items():
                        if isinstance(arg_value, str):
                            # Handle dependency results
                            if arg_value.startswith("{") and arg_value.endswith("}"):
                                try:
                                    dep_id = int(arg_value.strip("{}"))
                                    if dep_id in result_map:
                                        formatted_args[arg_name] = result_map[dep_id].get("knowledge_sources", [])
                                    else:
                                        formatted_args[arg_name] = arg_value
                                except ValueError:
                                    # Not a dependency ID, use value as is
                                    formatted_args[arg_name] = arg_value
                            else:
                                formatted_args[arg_name] = arg_value
                        else:
                            formatted_args[arg_name] = arg_value
                    
                    # Execute task
                    try:
                        result = await getattr(self, f"_{task.tool}")(**formatted_args)
                        results.append(TaskResult(task_id=task.idx, result=result, error=None))
                        result_map[task.idx] = result  # Store result for dependencies
                    except Exception as e:
                        error_msg = f"Task {task.idx} failed: {str(e)}"
                        log_error_with_traceback(e, error_msg)
                        results.append(TaskResult(task_id=task.idx, result=None, error=error_msg))
                    
                    # Update progress
                    progress.update(task_progress, advance=1)
                    
                except Exception as e:
                    log_error_with_traceback(e, f"Error executing task {task.idx}")
                    results.append(TaskResult(task_id=task.idx, result=None, error=str(e)))
            
            # Clean up progress
            await cleanup_progress()
            return results
            
        except Exception as e:
            log_error_with_traceback(e, "Error in task execution")
            await cleanup_progress()
            raise

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
                
            # Convert initial state to CompilerState and set it
            self.state = {
                "content": initial_state.get("content", ""),
                "domain_name": initial_state.get("domain_name", ""),
                "plan": None,
                "results": [],
                "join_decision": None,
                "final_result": None,
                "error": None,
                "feedback": None
            }
            
            # Generate plan
            try:
                plan = await self.generate_plan(self.state)
                self.state["plan"] = plan
                self._log_plan(plan)
            except Exception as e:
                log_error_with_traceback(e, "Failed to generate plan", include_locals=True)
                if isinstance(e, PlanningError):
                    raise
                raise PlanningError("Failed to generate plan") from e
                
            # Execute tasks
            try:
                results = await self.execute_tasks(plan.tasks, self.state)
                self.state["results"] = results
                
                # Log failed tasks
                failed_tasks = [r for r in results if r.error]
                if failed_tasks:
                    console.print("\n[bold red]Task Failures:[/bold red]")
                    for task in failed_tasks:
                        console.print(f"Task {task.task_id} failed: {task.error}")
                        
            except Exception as e:
                log_error_with_traceback(e, "Failed to execute tasks", include_locals=True)
                if isinstance(e, ExecutionError):
                    raise
                raise ExecutionError("Failed to execute tasks") from e
                
            # Make join decision
            try:
                decision = await self.make_join_decision(self.state)
                self.state["join_decision"] = decision
                
                if decision.replan:
                    console.print(f"\nReplanning based on feedback: {decision.feedback}")
                    # Convert state back to dict for recursive call
                    return await self.run(dict(self.state))
                    
                if decision.complete:
                    return self.state
                    
            except Exception as e:
                log_error_with_traceback(e, "Failed to make join decision", include_locals=True)
                if isinstance(e, DecisionError):
                    raise
                raise DecisionError("Failed to make join decision") from e
                
            return self.state
            
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

    async def _execute_task(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task."""
        try:
            # Get the tool function
            tool_func = self.tools.get(tool)
            if not tool_func:
                raise ExecutionError(f"Tool {tool} not found")
            
            # Execute the tool
            try:
                result = await tool_func(**args)
                return result
            except Exception as e:
                raise ExecutionError(f"Tool execution failed: {str(e)}")
            
        except Exception as e:
            raise ExecutionError(f"Task execution failed: {str(e)}")

