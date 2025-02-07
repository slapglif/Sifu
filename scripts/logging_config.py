import sys
from pathlib import Path
from traceback import format_exception
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
from typing import Any, Dict, Optional
from datetime import datetime

# Install rich traceback handler with more detailed settings
install_rich_traceback(
    show_locals=True,
    width=120,
    word_wrap=True,
    extra_lines=3,
    theme=None,
    max_frames=20
)

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
            refresh_per_second=10
        )
        _shared_progress.start()
    return _shared_progress

def setup_logging(log_file: Optional[str] = None):
    """Configure logging with both file and console output"""
    # Remove default handler
    logger.remove()
    
    # Generate log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/debug_{timestamp}.log"
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Add rich console handler with enhanced format
    logger.add(
        lambda msg: console.print(Panel(
            msg,
            border_style="blue",
            title=f"[cyan]{msg.record['time'].strftime('%Y-%m-%d %H:%M:%S')}[/cyan]",
            subtitle=f"[yellow]{msg.record['name']}:{msg.record['function']}:{msg.record['line']}[/yellow]"
        )),
        format="<level>{message}</level>",
        level="INFO",
        backtrace=True,
        diagnose=True,
        colorize=True,
        catch=True  # Catch exceptions in handlers
    )
    
    # Add file handler with detailed format including process and thread IDs
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {process}:{thread} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        backtrace=True,
        diagnose=True,
        rotation="1 day",
        retention="30 days",
        compression="zip",
        catch=True,
        enqueue=True  # Thread-safe logging
    )
    
    # Add error file handler for critical errors
    error_log = log_file.replace("debug", "error")
    logger.add(
        error_log,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {process}:{thread} | {level: <8} | {name}:{function}:{line} | {message}",
        level="ERROR",
        backtrace=True,
        diagnose=True,
        rotation="1 day",
        retention="30 days",
        compression="zip",
        catch=True,
        enqueue=True
    )

def log_error_with_traceback(e: Exception, context: str = "", include_locals: bool = True):
    """Log error with full traceback and local variables"""
    error_msg = f"{context}: {str(e)}"
    exc_info = (type(e), e, e.__traceback__)
    
    # Log to loguru with full context
    logger.opt(exception=exc_info, depth=1).error(error_msg)
    
    # Create detailed error panel
    error_details = []
    error_details.append(f"[red]Error Type:[/red] {type(e).__name__}")
    error_details.append(f"[red]Error Message:[/red] {str(e)}")
    error_details.append(f"[red]Context:[/red] {context}")
    
    if include_locals and hasattr(e, "__traceback__"):
        tb = e.__traceback__
        while tb:
            frame = tb.tb_frame
            error_details.append("\n[yellow]Local Variables:[/yellow]")
            for key, value in frame.f_locals.items():
                try:
                    value_str = str(value)
                    if len(value_str) > 200:
                        value_str = value_str[:200] + "..."
                    error_details.append(f"  [blue]{key}[/blue] = {value_str}")
                except:
                    error_details.append(f"  [blue]{key}[/blue] = <unprintable value>")
            tb = tb.tb_next
    
    # Print rich error panel
    console.print()
    console.print(Panel(
        "\n".join(error_details),
        title="[red]Error Details[/red]",
        border_style="red",
        padding=(1, 2)
    ))
    
    # Print traceback
    console.print()
    console.print(Panel(
        Syntax(
            "".join(format_exception(*exc_info)),
            "python",
            theme="monokai",
            line_numbers=True,
            word_wrap=True
        ),
        title="[red]Full Traceback[/red]",
        border_style="red"
    ))

def log_warning_with_context(msg: str, context: str = "", include_locals: bool = False):
    """Log warning with context and optionally local variables"""
    warning_msg = f"{context}: {msg}"
    
    # Log to loguru
    logger.opt(depth=1).warning(warning_msg)
    
    if include_locals:
        # Get caller's frame
        frame = sys._getframe(1)
        locals_msg = "\nLocal Variables:\n"
        for key, value in frame.f_locals.items():
            try:
                value_str = str(value)
                if len(value_str) > 200:
                    value_str = value_str[:200] + "..."
                locals_msg += f"  {key} = {value_str}\n"
            except:
                locals_msg += f"  {key} = <unprintable value>\n"
        logger.opt(depth=1).warning(locals_msg)

def log_info_with_context(msg: str, context: str = "", include_locals: bool = False):
    """Log info with context and optionally local variables"""
    info_msg = f"{context}: {msg}"
    
    # Log to loguru
    logger.opt(depth=1).info(info_msg)
    
    if include_locals:
        # Get caller's frame
        frame = sys._getframe(1)
        locals_msg = "\nLocal Variables:\n"
        for key, value in frame.f_locals.items():
            try:
                value_str = str(value)
                if len(value_str) > 200:
                    value_str = value_str[:200] + "..."
                locals_msg += f"  {key} = {value_str}\n"
            except:
                locals_msg += f"  {key} = <unprintable value>\n"
        logger.opt(depth=1).info(locals_msg)

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