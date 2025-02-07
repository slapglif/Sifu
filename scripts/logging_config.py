import sys
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.traceback import install as install_rich_traceback
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.tree import Tree
from rich import box
import json
from typing import Any, Dict

# Install rich traceback handler
install_rich_traceback(show_locals=True, width=120, word_wrap=True)

# Create console for rich output
console = Console()

# Create a single shared progress instance
_shared_progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(complete_style="green", finished_style="bright_green"),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeRemainingColumn(),
    console=console,
    auto_refresh=False,
    expand=True
)

def create_progress() -> Progress:
    """Get the shared progress instance"""
    return _shared_progress

def setup_logging(log_file: str = "debug.log"):
    """Configure logging with both file and console output"""
    # Remove default handler
    logger.remove()
    
    # Add rich console handler with custom format
    logger.add(
        lambda msg: console.print(Panel(
            msg,
            border_style="blue",
            title=f"[cyan]{msg.record['time'].strftime('%Y-%m-%d %H:%M:%S')}[/cyan]",
            subtitle=f"[yellow]{msg.record['name']}:{msg.record['function']}[/yellow]"
        )),
        format="<level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # Add file handler with detailed format including tracebacks
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="1 day",
        retention="7 days",
        backtrace=True,
        diagnose=True,
        enqueue=True,
        catch=True
    )
    
    # Start the shared progress
    _shared_progress.start()

def log_error_with_traceback(e: Exception, context: str = ""):
    """Log an error with full traceback and context using both loguru and rich"""
    if context:
        error_panel = Panel(
            f"[red bold]{context}[/red bold]\n[white]{str(e)}[/white]",
            title="[red]Error[/red]",
            border_style="red"
        )
    else:
        error_panel = Panel(
            f"[white]{str(e)}[/white]",
            title="[red]Error[/red]",
            border_style="red"
        )
    
    console.print(error_panel)
    logger.opt(depth=1).error(str(e))
    logger.opt(depth=1).debug("Full traceback:", exc_info=e)
    console.print_exception(show_locals=True, width=120, word_wrap=True)

def log_warning_with_context(msg: str, context: str = ""):
    """Log a warning with context"""
    warning_panel = Panel(
        f"[yellow]{msg}[/yellow]",
        title=f"[yellow]Warning: {context}[/yellow]" if context else "[yellow]Warning[/yellow]",
        border_style="yellow"
    )
    console.print(warning_panel)
    if context:
        logger.opt(depth=1).warning(f"{context}: {msg}")
    else:
        logger.opt(depth=1).warning(msg)

def log_info_with_context(msg: str, context: str = ""):
    """Log an info message with context"""
    info_panel = Panel(
        f"[cyan]{msg}[/cyan]",
        title=f"[blue]Info: {context}[/blue]" if context else "[blue]Info[/blue]",
        border_style="blue"
    )
    console.print(info_panel)
    if context:
        logger.opt(depth=1).info(f"{context}: {msg}")
    else:
        logger.opt(depth=1).info(msg)

def log_extraction_results(knowledge: Any):
    """Log knowledge extraction results in a beautiful format"""
    # Create a tree for visualization
    tree = Tree("[bold blue]Extracted Knowledge[/bold blue]")
    
    # Add content
    content_tree = tree.add("[bold cyan]Content[/bold cyan]")
    content_tree.add(knowledge.content[:200] + "..." if len(knowledge.content) > 200 else knowledge.content)
    
    # Add entities
    entities_tree = tree.add("[bold green]Entities[/bold green]")
    for entity in knowledge.entities:
        entities_tree.add(f"[green]• {entity}[/green]")
    
    # Add relationships
    relationships_tree = tree.add("[bold yellow]Relationships[/bold yellow]")
    for rel in knowledge.relationships:
        relationships_tree.add(f"[yellow]{rel.source} → {rel.relation} → {rel.target}[/yellow]")
    
    # Add metadata
    if knowledge.metadata:
        metadata_tree = tree.add("[bold magenta]Metadata[/bold magenta]")
        for key, value in knowledge.metadata.dict().items():
            metadata_tree.add(f"[magenta]{key}: {value}[/magenta]")
    
    # Create stats table
    stats_table = Table(
        title="[bold]Extraction Statistics[/bold]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold"
    )
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Content Length", str(len(knowledge.content)))
    stats_table.add_row("Entity Count", str(len(knowledge.entities)))
    stats_table.add_row("Relationship Count", str(len(knowledge.relationships)))
    stats_table.add_row("Confidence", f"{knowledge.confidence:.2f}")
    
    # Print results
    console.print("\n")
    console.print(Panel(tree, title="[bold]Knowledge Extraction Results[/bold]", border_style="blue"))
    console.print(stats_table)
    console.print("\n")

def log_confidence_evaluation(result: Dict):
    """Log confidence evaluation results in a beautiful format"""
    # Create confidence table
    table = Table(
        title="[bold]Confidence Evaluation[/bold]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold"
    )
    
    table.add_column("Factor", style="cyan")
    table.add_column("Score", style="green")
    
    factors = result.get("factors", {})
    table.add_row("Content Quality", f"{factors.get('content_quality', 0.0):.2f}")
    table.add_row("Entity Confidence", f"{factors.get('entity_confidence', 0.0):.2f}")
    table.add_row("Relationship Validity", f"{factors.get('relationship_validity', 0.0):.2f}")
    table.add_row("Source Reliability", f"{factors.get('source_reliability', 0.0):.2f}")
    table.add_row("Context Relevance", f"{factors.get('context_relevance', 0.0):.2f}")
    table.add_row("Overall Confidence", f"{result.get('confidence', 0.0):.2f}")
    
    # Create reasoning panel
    reasoning_panel = Panel(
        result.get("reasoning", "No reasoning provided"),
        title="[bold]Confidence Reasoning[/bold]",
        border_style="blue"
    )
    
    # Print results
    console.print("\n")
    console.print(table)
    console.print(reasoning_panel)
    console.print("\n") 