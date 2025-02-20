"""Example usage of the AI co-scientist system."""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from rich.console import Console
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser

from agents.co_scientist.main import AICoScientist

async def main():
    """Run an example research process."""
    # Initialize console
    console = Console()
    console.print("[bold blue]Starting AI Co-scientist Example[/bold blue]")
    
    try:
        # Initialize language model
        llm = ChatOllama(
            model="deepscaler",  # Using deepscaler as base model
            format="json",  # Ensure JSON output format
            temperature=0.7,  # Add some variability
            stop=["\n\n"],  # Add stop sequence for better output control
            seed=42  # For reproducibility
        )
        
        # Initialize AI co-scientist
        scientist = AICoScientist(llm, console)
        
        # Define research goal
        research_goal = {
            "goal": "Investigate potential drug repurposing candidates for treating acute myeloid leukemia (AML)",
            "domain": "drug_repurposing",
            "constraints": [
                "Focus on FDA-approved drugs",
                "Consider only small molecule drugs",
                "Must have known safety profiles",
                "Should be cost-effective"
            ],
            "preferences": {
                "prioritize_novel_mechanisms": True,
                "consider_combination_therapy": True,
                "focus_on_targeted_therapy": True,
                "min_evidence_level": "preclinical",
                "max_candidates": 5
            }
        }
        
        # Run the research process
        console.print("\n[bold green]Starting Research Process[/bold green]")
        results = await scientist.run(research_goal)
        
        # Print summary
        console.print("\n[bold blue]Research Summary[/bold blue]")
        
        # Print key findings from each cycle
        if results.get("cycles"):
            console.print("\n[bold]Key Findings by Cycle:[/bold]")
            for i, cycle in enumerate(results["cycles"], 1):
                console.print(f"\n[bold]Cycle {i}:[/bold]")
                overview = cycle["overview"]
                for finding in overview["key_findings"]:
                    console.print(f"- {finding}")
                    
            # Print top hypotheses
            if results["cycles"]:
                console.print("\n[bold]Top Ranked Hypotheses:[/bold]")
                final_cycle = results["cycles"][-1]
                for h_id, score in final_cycle["rankings"][:3]:
                    for h in final_cycle["hypotheses"]:
                        if h["id"] == h_id:
                            console.print(f"\nHypothesis: {h['statement']}")
                            console.print(f"Score: {score:.2f}")
                            console.print(f"Evidence: {', '.join(h['evidence'])}")
                            
                # Print research directions
                console.print("\n[bold]Promising Research Directions:[/bold]")
                for direction in final_cycle["overview"]["promising_directions"]:
                    console.print(f"\n- {direction['title']}")
                    if "rationale" in direction:
                        console.print(f"  Rationale: {direction['rationale']}")
        else:
            console.print("\n[bold red]No research cycles were completed.[/bold red]")
            console.print("Please check the system status and logs for more information.")
                
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        import traceback
        console.print(traceback.format_exc())
        raise
        
if __name__ == "__main__":
    asyncio.run(main()) 