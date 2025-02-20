"""Main entry point for the AI co-scientist system."""

from typing import Any, Dict, List, Optional, Literal, cast
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain.graphs import Graph
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import asyncio

from .supervisor.supervisor_agent import SupervisorAgent, ResearchGoal
from .generation.generation_agent import GenerationAgent, Hypothesis
from .reflection.reflection_agent import ReflectionAgent, Review
from .ranking.ranking_agent import RankingAgent, TournamentMatch
from .evolution.evolution_agent import EvolutionAgent, RefinementResult
from .proximity.proximity_agent import ProximityAgent, HypothesisCluster
from .meta_review.meta_review_agent import MetaReviewAgent, ResearchOverview

# Type definitions
ReviewType = Literal["initial", "full", "deep_verification", "observation", "simulation", "tournament"]

class AICoScientist:
    """Main AI co-scientist system."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        console: Optional[Console] = None
    ):
        """Initialize the AI co-scientist system."""
        self.llm = llm
        self.console = console or Console()
        
        # Initialize agents
        self.supervisor = SupervisorAgent(llm)
        self.generation = GenerationAgent(llm)
        self.reflection = ReflectionAgent(llm)
        self.ranking = RankingAgent(llm)
        self.evolution = EvolutionAgent(llm)
        self.proximity = ProximityAgent(llm)
        self.meta_review = MetaReviewAgent(llm)
        
        # Initialize system graph
        self.system_graph = Graph()
        self._initialize_system_graph()
        
        # Print initialization status
        self._print_initialization_status()
        
    def _initialize_system_graph(self) -> None:
        """Initialize the system graph structure."""
        # Add agent nodes
        for agent in [
            self.supervisor,
            self.generation,
            self.reflection,
            self.ranking,
            self.evolution,
            self.proximity,
            self.meta_review
        ]:
            self.system_graph.add_node(
                agent.state.agent_id,
                properties={
                    "type": agent.state.agent_type,
                    "state": agent.state.dict()
                }
            )
            
        # Add relationships
        self.system_graph.add_edge(
            self.supervisor.state.agent_id,
            self.generation.state.agent_id,
            "coordinates"
        )
        self.system_graph.add_edge(
            self.supervisor.state.agent_id,
            self.reflection.state.agent_id,
            "coordinates"
        )
        self.system_graph.add_edge(
            self.supervisor.state.agent_id,
            self.ranking.state.agent_id,
            "coordinates"
        )
        self.system_graph.add_edge(
            self.supervisor.state.agent_id,
            self.evolution.state.agent_id,
            "coordinates"
        )
        self.system_graph.add_edge(
            self.supervisor.state.agent_id,
            self.proximity.state.agent_id,
            "coordinates"
        )
        self.system_graph.add_edge(
            self.supervisor.state.agent_id,
            self.meta_review.state.agent_id,
            "coordinates"
        )
        
        # Add data flow relationships
        self.system_graph.add_edge(
            self.generation.state.agent_id,
            self.reflection.state.agent_id,
            "sends_hypotheses"
        )
        self.system_graph.add_edge(
            self.reflection.state.agent_id,
            self.ranking.state.agent_id,
            "sends_reviews"
        )
        self.system_graph.add_edge(
            self.ranking.state.agent_id,
            self.evolution.state.agent_id,
            "sends_rankings"
        )
        self.system_graph.add_edge(
            self.evolution.state.agent_id,
            self.proximity.state.agent_id,
            "sends_refinements"
        )
        self.system_graph.add_edge(
            self.proximity.state.agent_id,
            self.meta_review.state.agent_id,
            "sends_clusters"
        )
        
    def _print_initialization_status(self) -> None:
        """Print system initialization status."""
        table = Table(title="AI Co-scientist System Status", box=box.ROUNDED)
        table.add_column("Agent", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("State", style="yellow")
        
        for agent in [
            self.supervisor,
            self.generation,
            self.reflection,
            self.ranking,
            self.evolution,
            self.proximity,
            self.meta_review
        ]:
            table.add_row(
                agent.state.agent_type,
                "Initialized",
                f"Memory: {len(agent.state.memory)} items"
            )
            
        self.console.print(table)
        self.console.print(Panel(
            "[bold green]AI Co-scientist System Initialized[/bold green]\n"
            "Ready to begin research exploration",
            title="System Status"
        ))
        
    async def set_research_goal(self, goal: Dict[str, Any]) -> None:
        """Set the research goal for the system."""
        research_goal = ResearchGoal(**goal)
        self.supervisor.state.research_goal = research_goal
        
        self.console.print(Panel(
            f"[bold blue]Research Goal Set[/bold blue]\n"
            f"Goal: {research_goal.goal}\n"
            f"Domain: {research_goal.domain}",
            title="Research Configuration"
        ))
        
    async def run_research_cycle(self) -> Dict[str, Any]:
        """Run a complete research cycle."""
        if not self.supervisor.state.research_goal:
            raise ValueError("Research goal must be set before running a cycle")
            
        # Create research plan
        plan = await self.supervisor.create_research_plan()
        
        # Generate hypotheses
        hypotheses = []
        for _ in range(plan.tasks[0].get("num_hypotheses", 3)):
            hypothesis = await self.generation.generate_hypothesis(
                self.supervisor.state.research_goal.dict(),
                {"plan": plan.dict()}
            )
            hypotheses.append(hypothesis)
            
        # Review hypotheses
        reviews = []
        review_types: List[ReviewType] = ["initial", "full", "deep_verification"]
        for hypothesis in hypotheses:
            for review_type in review_types:
                review = await self.reflection.review_hypothesis(
                    hypothesis,
                    review_type,
                    {"plan": plan.dict()}
                )
                reviews.append(review)
                
        # Conduct tournament
        for i, h1 in enumerate(hypotheses):
            for h2 in hypotheses[i+1:]:
                match = await self.ranking.conduct_match(
                    h1, h2,
                    {"reviews": [r.dict() for r in reviews]}
                )
                
        # Refine top hypotheses
        rankings = self.ranking.get_rankings()
        refinements = []
        for h_id, _ in rankings[:3]:  # Refine top 3
            hypothesis = next(h for h in hypotheses if h.id == h_id)
            refinement = await self.evolution.refine_hypothesis(
                hypothesis,
                strategy_id="literature_enhancement"
            )
            refinements.append(refinement)
            
        # Cluster hypotheses
        clusters = await self.proximity.cluster_hypotheses(
            hypotheses + [r.refined_hypothesis for r in refinements]
        )
        
        # Generate overview
        overview = await self.meta_review.generate_overview(
            hypotheses + [r.refined_hypothesis for r in refinements],
            reviews,
            {
                "plan": plan.dict(),
                "rankings": rankings,
                "clusters": [c.dict() for c in clusters]
            }
        )
        
        return {
            "hypotheses": [h.dict() for h in hypotheses],
            "reviews": [r.dict() for r in reviews],
            "rankings": rankings,
            "refinements": [r.dict() for r in refinements],
            "clusters": [c.dict() for c in clusters],
            "overview": overview.dict()
        }
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "research_goal": self.supervisor.state.research_goal.dict() if self.supervisor.state.research_goal else None,
            "generation_stats": {
                "total_hypotheses": len(self.generation.state.hypotheses),
                "generation_history": len(self.generation.state.generation_history)
            },
            "reflection_stats": {
                "total_reviews": sum(len(reviews) for reviews in self.reflection.state.reviews.values()),
                "review_history": len(self.reflection.state.review_history)
            },
            "ranking_stats": {
                "total_matches": len(self.ranking.state.matches),
                "match_history": len(self.ranking.state.match_history)
            },
            "evolution_stats": {
                "total_refinements": len(self.evolution.state.refinement_history),
                "active_strategies": len(self.evolution.state.strategies)
            },
            "proximity_stats": {
                "total_clusters": len(self.proximity.state.clusters),
                "similarity_cache_size": len(self.proximity.state.similarity_cache)
            },
            "meta_review_stats": {
                "total_overviews": len(self.meta_review.state.research_overviews),
                "synthesis_history": len(self.meta_review.state.synthesis_history)
            }
        }
        
    def print_system_status(self) -> None:
        """Print current system status."""
        status = self.get_system_status()
        
        table = Table(title="System Status Overview", box=box.ROUNDED)
        table.add_column("Component", style="cyan")
        table.add_column("Metric", style="yellow")
        table.add_column("Value", style="green")
        
        if status["research_goal"]:
            table.add_row(
                "Research Goal",
                "Domain",
                status["research_goal"]["domain"]
            )
            
        for agent, stats in [
            ("Generation", status["generation_stats"]),
            ("Reflection", status["reflection_stats"]),
            ("Ranking", status["ranking_stats"]),
            ("Evolution", status["evolution_stats"]),
            ("Proximity", status["proximity_stats"]),
            ("Meta-review", status["meta_review_stats"])
        ]:
            for metric, value in stats.items():
                table.add_row(agent, metric, str(value))
                
        self.console.print(table)
        
    async def run(self, research_goal: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete research process."""
        # Set research goal
        await self.set_research_goal(research_goal)
        
        # Run research cycles
        results = []
        try:
            while True:
                cycle_result = await self.run_research_cycle()
                results.append(cycle_result)
                
                # Check if we should continue
                overview = cycle_result["overview"]
                if len(overview["key_findings"]) >= 10:  # Example stopping condition
                    break
                    
        except Exception as e:
            self.console.print(f"[bold red]Error in research cycle:[/bold red] {str(e)}")
            
        finally:
            # Print final status
            self.print_system_status()
            
        return {
            "research_goal": research_goal,
            "cycles": results,
            "final_status": self.get_system_status()
        } 