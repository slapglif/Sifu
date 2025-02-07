"""LLM compiler system."""

from typing import Any, Dict, List
from langchain_core.output_parsers import PydanticOutputParser
import json
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich import box

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
    log_info_with_context,
    create_progress,
    log_warning_with_context,
    console
)

class LLMCompiler:
    """LLM compiler system."""

    def __init__(self, llm):
        """Initialize with language model."""
        self.llm = llm
        log_info_with_context("Initialized LLM compiler", "Compiler")
        console.print(Panel("LLM Compiler Initialized", style="bold green"))

    def _log_plan(self, plan: Plan):
        """Log plan details in a rich format"""
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

    def _log_task_result(self, result: TaskResult):
        """Log task result in a rich format"""
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

    def _log_join_decision(self, decision: JoinDecision):
        """Log join decision in a rich format"""
        status = "[green]Complete[/green]" if decision.complete else "[yellow]Replan[/yellow]" if decision.replan else "[red]Unknown[/red]"
        console.print(Panel(
            f"Status: {status}\nThought: {decision.thought}\nFeedback: {decision.feedback or 'None'}",
            title="[bold]Join Decision[/bold]",
            border_style="cyan"
        ))

    async def generate_plan(self, state: CompilerState) -> Plan:
        """Generate execution plan."""
        try:
            log_info_with_context("Starting plan generation", "Planning")
            console.print("\n[bold cyan]Generating Execution Plan...[/bold cyan]")
            
            prompt = get_plan_generation_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=Plan)
            
            # Format state for LLM
            formatted_state = {
                "content": state.content,
                "plan": state.plan.model_dump() if state.plan else None,
                "results": [r.model_dump() if hasattr(r, "model_dump") else r.dict() for r in state.results],
                "join_decision": state.join_decision.model_dump() if state.join_decision else None,
                "final_result": state.final_result
            }
            
            # Get response from LLM
            response = await chain.ainvoke({
                "state": json.dumps(formatted_state, indent=2)
            })
            
            # Handle response
            if isinstance(response, Plan):
                plan = response
            elif isinstance(response, dict) and "tasks" in response and "thought" in response:
                # Validate tasks
                for task in response["tasks"]:
                    if not isinstance(task, dict):
                        raise ValueError("Each task must be a dictionary")
                    
                    # Validate required task fields
                    required_fields = ["idx", "tool", "args", "dependencies"]
                    for field in required_fields:
                        if field not in task:
                            raise ValueError(f"Task missing required field: {field}")
                    
                    # Remove any extra fields
                    task_clean = {
                        "idx": task["idx"],
                        "tool": task["tool"],
                        "args": task["args"],
                        "dependencies": task["dependencies"]
                    }
                    task.clear()
                    task.update(task_clean)
                    
                    # Ensure args is a dictionary
                    if not isinstance(task["args"], dict):
                        raise ValueError(f"Args must be a dictionary for task {task['idx']}")
                    
                    # Ensure dependencies is a list
                    if not isinstance(task["dependencies"], list):
                        raise ValueError(f"Dependencies must be a list for task {task['idx']}")
                    
                    # Validate dependencies
                    for dep in task["dependencies"]:
                        if not isinstance(dep, int):
                            raise ValueError(f"Dependencies must be integers for task {task['idx']}")
                        if dep >= task["idx"]:
                            raise ValueError(f"Task {task['idx']} cannot depend on future task {dep}")
                        if dep < 0:
                            raise ValueError(f"Task {task['idx']} has invalid negative dependency {dep}")
                
                # Create Plan object
                plan = Plan(
                    tasks=[Task(**task) for task in response["tasks"]],
                    thought=response["thought"]
                )
            elif isinstance(response, str):
                try:
                    # Try parsing JSON response
                    parsed = json.loads(response)
                    if isinstance(parsed, dict) and "tasks" in parsed and "thought" in parsed:
                        # Validate tasks
                        for task in parsed["tasks"]:
                            if not isinstance(task, dict):
                                raise ValueError("Each task must be a dictionary")
                            
                            # Validate required task fields
                            required_fields = ["idx", "tool", "args", "dependencies"]
                            for field in required_fields:
                                if field not in task:
                                    raise ValueError(f"Task missing required field: {field}")
                            
                            # Remove any extra fields
                            task_clean = {
                                "idx": task["idx"],
                                "tool": task["tool"],
                                "args": task["args"],
                                "dependencies": task["dependencies"]
                            }
                            task.clear()
                            task.update(task_clean)
                            
                            # Ensure args is a dictionary
                            if not isinstance(task["args"], dict):
                                raise ValueError(f"Args must be a dictionary for task {task['idx']}")
                            
                            # Ensure dependencies is a list
                            if not isinstance(task["dependencies"], list):
                                raise ValueError(f"Dependencies must be a list for task {task['idx']}")
                            
                            # Validate dependencies
                            for dep in task["dependencies"]:
                                if not isinstance(dep, int):
                                    raise ValueError(f"Dependencies must be integers for task {task['idx']}")
                                if dep >= task["idx"]:
                                    raise ValueError(f"Task {task['idx']} cannot depend on future task {dep}")
                                if dep < 0:
                                    raise ValueError(f"Task {task['idx']} has invalid negative dependency {dep}")
                        
                        # Create Plan object
                        plan = Plan(
                            tasks=[Task(**task) for task in parsed["tasks"]],
                            thought=parsed["thought"]
                        )
                    else:
                        raise ValueError(f"Invalid response format: {response}")
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid response format: {response}")
            else:
                raise ValueError(f"Invalid response type: {type(response)}")
            
            # Log the generated plan
            log_info_with_context(f"Generated plan with {len(plan.tasks)} tasks", "Planning")
            self._log_plan(plan)
            return plan
            
        except Exception as e:
            log_error_with_traceback(e, "Error generating plan")
            raise

    async def execute_tasks(self, tasks: List[Task]) -> List[TaskResult]:
        """Execute planned tasks."""
        try:
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
                    # Get response from LLM
                    response = await chain.ainvoke({
                        "task": task.model_dump() if hasattr(task, "model_dump") else task
                    })
                    
                    # Handle response
                    if isinstance(response, TaskResult):
                        results.append(response)
                        self._log_task_result(response)
                    elif isinstance(response, dict) and "task_id" in response:
                        result = TaskResult(**response)
                        results.append(result)
                        self._log_task_result(result)
                    elif isinstance(response, str):
                        try:
                            # Try parsing JSON response
                            parsed = json.loads(response)
                            if isinstance(parsed, dict) and "task_id" in parsed:
                                result = TaskResult(**parsed)
                                results.append(result)
                                self._log_task_result(result)
                            else:
                                raise ValueError(f"Invalid response format: {response}")
                        except json.JSONDecodeError:
                            raise ValueError(f"Invalid response format: {response}")
                    else:
                        raise ValueError(f"Invalid response type: {type(response)}")
                    
                except Exception as e:
                    log_error_with_traceback(e, f"Error executing task {task.idx}")
                    results.append(TaskResult(
                        task_id=task.idx,
                        result=None,
                        error=str(e)
                    ))
                finally:
                    progress.update(task_progress, advance=1)
                    progress.refresh()
                    
            return results
            
        except Exception as e:
            log_error_with_traceback(e, "Error executing tasks")
            raise

    async def make_join_decision(self, state: CompilerState) -> JoinDecision:
        """Make join decision."""
        try:
            log_info_with_context("Making join decision", "Decision")
            console.print("\n[bold green]Making Join Decision...[/bold green]")
            
            prompt = get_join_decision_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=JoinDecision)
            
            # Format state for LLM
            formatted_state = {
                "content": state.content,
                "plan": state.plan.model_dump() if state.plan else None,
                "results": [r.model_dump() if hasattr(r, "model_dump") else r.dict() for r in state.results],
                "join_decision": state.join_decision.model_dump() if state.join_decision else None,
                "final_result": state.final_result
            }
            
            # Get response from LLM
            response = await chain.ainvoke({
                "state": json.dumps(formatted_state, indent=2)
            })
            
            # Handle response
            if isinstance(response, JoinDecision):
                self._log_join_decision(response)
                return response
            elif isinstance(response, dict) and "complete" in response and "thought" in response:
                decision = JoinDecision(**response)
                self._log_join_decision(decision)
                return decision
            elif isinstance(response, str):
                try:
                    # Try parsing JSON response
                    parsed = json.loads(response)
                    if isinstance(parsed, dict) and "complete" in parsed and "thought" in parsed:
                        decision = JoinDecision(**parsed)
                        self._log_join_decision(decision)
                        return decision
                    else:
                        raise ValueError(f"Invalid response format: {response}")
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid response format: {response}")
            else:
                raise ValueError(f"Invalid response type: {type(response)}")
            
        except Exception as e:
            log_error_with_traceback(e, "Error making join decision")
            raise

    async def run(self, initial_state: Dict[str, Any]) -> Any:
        """Run the LLM compiler workflow."""
        try:
            # Initialize state
            state = CompilerState(
                content=initial_state.get("content", ""),
                plan=initial_state.get("plan"),
                results=initial_state.get("results", []),
                join_decision=initial_state.get("join_decision"),
                final_result=initial_state.get("final_result")
            )

            log_info_with_context("Starting LLM compiler workflow", "Compiler")
            progress = create_progress()

            # Add tasks for tracking
            plan_task = progress.add_task("[cyan]Generating Plan...", total=1)
            exec_task = progress.add_task("[yellow]Executing Tasks...", total=1)
            join_task = progress.add_task("[green]Making Decisions...", total=1)

            while True:
                try:
                    # Generate plan if needed
                    if not state.plan:
                        log_info_with_context("Generating execution plan", "Planning")
                        state.plan = await self.generate_plan(state)
                        progress.update(plan_task, advance=1)
                        progress.refresh()
                        log_info_with_context(f"Generated plan with {len(state.plan.tasks)} tasks", "Planning")

                    # Execute tasks
                    if not state.results:
                        log_info_with_context("Executing planned tasks", "Execution")
                        progress.update(exec_task, total=len(state.plan.tasks))
                        progress.refresh()
                        
                        results = []
                        for idx, task in enumerate(state.plan.tasks):
                            try:
                                log_info_with_context(f"Executing task {task.idx}: {task.tool}", "Execution")
                                result = await self.execute_tasks([task])
                                results.extend(result)
                                progress.update(exec_task, advance=1)
                                progress.refresh()
                                log_info_with_context(f"Task {task.idx} completed successfully", "Execution")
                            except Exception as e:
                                log_error_with_traceback(e, f"Error executing task {task.idx}")
                                results.append(TaskResult(
                                    task_id=task.idx,
                                    result=None,
                                    error=str(e)
                                ))
                                progress.update(exec_task, advance=1)
                                progress.refresh()
                        
                        state.results = results

                    # Make join decision
                    log_info_with_context("Making join decision", "Decision")
                    state.join_decision = await self.make_join_decision(state)
                    progress.update(join_task, advance=1)
                    progress.refresh()
                    log_info_with_context(
                        f"Join decision: complete={state.join_decision.complete}, replan={state.join_decision.replan}",
                        "Decision"
                    )

                    # Check completion
                    if state.join_decision.complete:
                        log_info_with_context("Workflow completed successfully", "Compiler")
                        
                        # Combine results into final state
                        final_state = await self._generate_final_result(state)
                        if final_state:
                            state.final_result = final_state
                            return state.final_result
                        else:
                            log_warning_with_context("No valid results produced", "Compiler")
                            return None

                    # Check replanning
                    if state.join_decision.replan:
                        log_info_with_context(
                            f"Replanning required: {state.join_decision.feedback}",
                            "Planning"
                        )
                        state.plan = None
                        state.results = []
                        # Reset progress for new iteration
                        progress.update(plan_task, completed=0)
                        progress.update(exec_task, completed=0)
                        progress.update(join_task, completed=0)
                        progress.refresh()
                        continue

                    # Unexpected state
                    raise ValueError("Invalid join decision state")

                except Exception as e:
                    log_error_with_traceback(e, "Error in compiler workflow")
                    raise

        except Exception as e:
            log_error_with_traceback(e, "Fatal error in compiler")
            raise

    async def _generate_final_result(self, state: CompilerState) -> Any:
        """Generate final result from task results."""
        try:
            # Initialize final state
            final_state = {
                "knowledge_sources": [],
                "synthetic_knowledge": [],
                "training_examples": [],
                "model_metrics": {}
            }
            
            # Process each task result
            for result in state.results:
                if result.error or not result.result:
                    continue
                    
                if isinstance(result.result, dict):
                    # Research task results
                    if "knowledge_sources" in result.result:
                        final_state["knowledge_sources"].extend(result.result["knowledge_sources"])
                        
                    # Knowledge synthesis results
                    if "synthetic_knowledge" in result.result:
                        if isinstance(result.result["synthetic_knowledge"], list):
                            final_state["synthetic_knowledge"].extend(result.result["synthetic_knowledge"])
                        else:
                            final_state["synthetic_knowledge"].append(result.result["synthetic_knowledge"])
                        
                    # Training example results
                    if "training_examples" in result.result:
                        final_state["training_examples"].extend(result.result["training_examples"])
                        
                    # Model training results
                    if "model_metrics" in result.result:
                        final_state["model_metrics"].update(result.result["model_metrics"])
                        
            # Only return final state if we have valid results
            if any(len(v) > 0 if isinstance(v, list) else bool(v) for v in final_state.values()):
                return final_state
            else:
                log_warning_with_context("No valid results produced", "Compiler")
                return None
                
        except Exception as e:
            log_error_with_traceback(e, "Error generating final result")
            raise 