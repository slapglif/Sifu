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
_shared_progress = None

def create_progress() -> Progress:
    """Create a new progress instance"""
    global _shared_progress
    if _shared_progress is None:
        _shared_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="bright_green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            expand=True,
            refresh_per_second=10  # Increase refresh rate
        )
        _shared_progress.start()
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
        backtrace=True,
        diagnose=True,
        colorize=True,
        catch=True  # Catch exceptions in the handlers
    )
    
    # Add file handler with detailed format
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        backtrace=True,
        diagnose=True,
        rotation="1 day",
        retention="7 days",
        compression="zip",
        catch=True,  # Catch exceptions in the handlers
        enqueue=True  # Thread-safe logging
    )
    
    # Add error file handler for critical errors
    logger.add(
        "error.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="ERROR",
        backtrace=True,
        diagnose=True,
        rotation="1 day",
        retention="30 days",
        compression="zip",
        catch=True,
        enqueue=True
    )

def log_error_with_traceback(e: Exception, context: str = ""):
    """Log error with full traceback"""
    logger.opt(exception=True).error(f"{context}: {str(e)}")
    
    # Also print rich traceback
    console.print()
    console.print(Panel(
        Syntax(
            str(e.__traceback__),
            "python",
            theme="monokai",
            line_numbers=True,
            word_wrap=True
        ),
        title="[red]Full Traceback[/red]",
        border_style="red"
    ))
    
    # Log full exception info to error log
    logger.opt(exception=True).error(f"""
    Error Context: {context}
    Error Type: {type(e).__name__}
    Error Message: {str(e)}
    Stack Trace:
    {e.__traceback__}
    """)

def log_warning_with_context(msg: str, context: str = ""):
    """Log warning with context"""
    logger.warning(f"{context}: {msg}")
    console.print(f"[yellow]WARNING[/yellow] [{context}] {msg}")

def log_info_with_context(msg: str, context: str = ""):
    """Log info with context"""
    logger.info(f"{context}: {msg}")
    console.print(f"[cyan]INFO[/cyan] [{context}] {msg}")

def log_debug_with_context(msg: str, context: str = ""):
    """Log debug with context"""
    logger.debug(f"{context}: {msg}")

def log_extraction_results(knowledge: Any):
    """Log knowledge extraction results"""
    if not knowledge:
        return
        
    # Create tree view of results
    tree = Tree("Knowledge Extraction Results")
    
    # Add entities
    entities = tree.add("Entities")
    for entity in getattr(knowledge, "entities", []):
        entities.add(f"[cyan]{entity}[/cyan]")
        
    # Add relationships
    relationships = tree.add("Relationships")
    for rel in getattr(knowledge, "relationships", []):
        rel_str = f"[yellow]{rel.source}[/yellow] -> [green]{rel.relation}[/green] -> [yellow]{rel.target}[/yellow]"
        relationships.add(rel_str)
        
    # Add metadata
    if hasattr(knowledge, "metadata"):
        metadata = tree.add("Metadata")
        for k, v in knowledge.metadata.dict().items():
            metadata.add(f"[blue]{k}[/blue]: {v}")
            
    # Print tree
    console.print()
    console.print(Panel(tree, title="Extraction Results", border_style="cyan"))

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