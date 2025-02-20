"""Evolution agent for hypothesis refinement and improvement."""

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from ..base_agent import BaseAgent, AgentState
from ..generation.generation_agent import Hypothesis

class RefinementStrategy(BaseModel):
    """Strategy for hypothesis refinement."""
    strategy_id: str = Field(description="Unique identifier for this strategy")
    name: str = Field(description="Name of the strategy")
    description: str = Field(description="Description of how the strategy works")
    target_aspects: List[str] = Field(description="Aspects of hypotheses this strategy targets")
    success_criteria: List[str] = Field(description="Criteria for successful application")

class RefinementResult(BaseModel):
    """Result of applying a refinement strategy."""
    result_id: str = Field(description="Unique identifier for this result")
    original_hypothesis: str = Field(description="ID of original hypothesis")
    refined_hypothesis: Hypothesis = Field(description="The refined hypothesis")
    strategy_used: str = Field(description="ID of strategy used")
    improvements: List[str] = Field(description="List of improvements made")
    rationale: str = Field(description="Reasoning behind refinements")
    metrics: Dict[str, float] = Field(description="Improvement metrics")

class EvolutionState(AgentState):
    """Evolution agent state."""
    strategies: Dict[str, RefinementStrategy] = Field(default_factory=dict)
    refinement_history: List[RefinementResult] = Field(default_factory=list)
    evolution_metrics: Dict[str, Any] = Field(default_factory=dict)
    current_strategy: Optional[str] = None

class EvolutionAgent(BaseAgent):
    """Agent responsible for refining and improving hypotheses."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        agent_id: str = "evolution",
        system_prompt: Optional[str] = None
    ):
        """Initialize the evolution agent."""
        if system_prompt is None:
            system_prompt = """You are the evolution agent responsible for refining research hypotheses.
Your role is to:
1. Apply various refinement strategies
2. Improve hypothesis quality and testability
3. Address identified weaknesses
4. Enhance scientific rigor
5. Maintain hypothesis novelty
6. Track improvement metrics

Refinement Strategies:
- Enhancement through Literature
- Coherence Improvement
- Feasibility Enhancement
- Inspiration from Success
- Combination of Strengths
- Simplification
- Out-of-box Thinking

Follow these guidelines:
- Preserve valuable aspects
- Address specific weaknesses
- Maintain scientific validity
- Track all modifications
- Justify refinements
- Consider multiple approaches
- Balance improvement goals"""

        super().__init__(
            llm=llm,
            agent_id=agent_id,
            agent_type="evolution",
            system_prompt=system_prompt,
            output_parser=PydanticOutputParser(pydantic_object=RefinementResult)
        )
        
        # Initialize evolution-specific state
        self.state = EvolutionState(
            agent_id=agent_id,
            agent_type="evolution",
            strategies={},
            refinement_history=[],
            evolution_metrics={},
            current_strategy=None
        )
        
        # Initialize default strategies
        self._initialize_default_strategies()
        
    def _initialize_default_strategies(self) -> None:
        """Initialize default refinement strategies."""
        default_strategies = [
            RefinementStrategy(
                strategy_id="literature_enhancement",
                name="Enhancement through Literature",
                description="Strengthen hypothesis by incorporating additional literature evidence",
                target_aspects=["evidence", "grounding", "support"],
                success_criteria=["increased_citations", "stronger_evidence", "better_grounding"]
            ),
            RefinementStrategy(
                strategy_id="coherence_improvement",
                name="Coherence Improvement",
                description="Enhance logical flow and internal consistency",
                target_aspects=["logic", "consistency", "clarity"],
                success_criteria=["reduced_contradictions", "clearer_logic", "better_structure"]
            ),
            RefinementStrategy(
                strategy_id="feasibility_enhancement",
                name="Feasibility Enhancement",
                description="Improve practical testability and implementation",
                target_aspects=["testability", "practicality", "resources"],
                success_criteria=["more_testable", "resource_efficient", "practical_methods"]
            ),
            RefinementStrategy(
                strategy_id="success_inspiration",
                name="Inspiration from Success",
                description="Learn from successful hypotheses and adapt their strengths",
                target_aspects=["methodology", "structure", "approach"],
                success_criteria=["adapted_strengths", "novel_combination", "improved_approach"]
            ),
            RefinementStrategy(
                strategy_id="strength_combination",
                name="Combination of Strengths",
                description="Combine strong elements from multiple hypotheses",
                target_aspects=["integration", "synthesis", "complementarity"],
                success_criteria=["successful_integration", "preserved_strengths", "added_value"]
            )
        ]
        
        for strategy in default_strategies:
            self.register_strategy(strategy)
            
    def register_strategy(self, strategy: RefinementStrategy) -> None:
        """Register a new refinement strategy."""
        self.state.strategies[strategy.strategy_id] = strategy
        
    async def refine_hypothesis(
        self,
        hypothesis: Hypothesis,
        strategy_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RefinementResult:
        """Refine a hypothesis using specified strategy."""
        # Use specified or current strategy
        strategy_id = strategy_id or self.state.current_strategy
        if not strategy_id or strategy_id not in self.state.strategies:
            raise ValueError("Valid strategy_id required")
            
        strategy = self.state.strategies[strategy_id]
        context = context or {}
        
        # Generate refinement using LLM
        result = await self.arun({
            "hypothesis": hypothesis.dict(),
            "strategy": strategy.dict(),
            "context": context,
            "previous_refinements": [
                r.dict() for r in self.state.refinement_history 
                if r.original_hypothesis == hypothesis.id
            ]
        })
        
        # Create refinement result
        refinement = RefinementResult(**result)
        
        # Update state
        self.state.refinement_history.append(refinement)
        self._update_evolution_metrics(refinement)
        
        return refinement
        
    def set_strategy(self, strategy_id: str) -> None:
        """Set the current refinement strategy."""
        if strategy_id not in self.state.strategies:
            raise ValueError(f"Unknown strategy: {strategy_id}")
        self.state.current_strategy = strategy_id
        
    def get_strategies(self) -> List[RefinementStrategy]:
        """Get all available refinement strategies."""
        return list(self.state.strategies.values())
        
    def get_refinement_history(
        self,
        hypothesis_id: Optional[str] = None,
        strategy_id: Optional[str] = None
    ) -> List[RefinementResult]:
        """Get refinement history, optionally filtered."""
        history = self.state.refinement_history
        
        if hypothesis_id:
            history = [r for r in history if r.original_hypothesis == hypothesis_id]
            
        if strategy_id:
            history = [r for r in history if r.strategy_used == strategy_id]
            
        return history
        
    def _update_evolution_metrics(self, result: RefinementResult) -> None:
        """Update evolution metrics with new refinement result."""
        metrics = self.state.evolution_metrics
        
        # Update strategy usage
        strategy_usage = metrics.get("strategy_usage", {})
        strategy_id = result.strategy_used
        strategy_usage[strategy_id] = strategy_usage.get(strategy_id, 0) + 1
        metrics["strategy_usage"] = strategy_usage
        
        # Update improvement tracking
        improvements = metrics.get("improvements", {})
        for improvement in result.improvements:
            improvements[improvement] = improvements.get(improvement, 0) + 1
        metrics["improvements"] = improvements
        
        # Update metric averages
        for metric, value in result.metrics.items():
            if "metric_averages" not in metrics:
                metrics["metric_averages"] = {}
            if metric not in metrics["metric_averages"]:
                metrics["metric_averages"][metric] = {"sum": 0.0, "count": 0}
            metrics["metric_averages"][metric]["sum"] += value
            metrics["metric_averages"][metric]["count"] += 1
            
        self.state.evolution_metrics = metrics
        
    def analyze_evolution_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in hypothesis evolution."""
        metrics = self.state.evolution_metrics
        
        # Calculate metric averages
        metric_averages = {}
        for metric, data in metrics.get("metric_averages", {}).items():
            if data["count"] > 0:
                metric_averages[metric] = data["sum"] / data["count"]
                
        # Get most successful strategies
        strategy_success = {}
        for result in self.state.refinement_history:
            strategy_id = result.strategy_used
            if strategy_id not in strategy_success:
                strategy_success[strategy_id] = {
                    "total_improvements": 0,
                    "total_metrics": 0.0,
                    "count": 0
                }
            stats = strategy_success[strategy_id]
            stats["total_improvements"] += len(result.improvements)
            stats["total_metrics"] += sum(result.metrics.values())
            stats["count"] += 1
            
        top_strategies = sorted(
            [
                (s_id, stats)
                for s_id, stats in strategy_success.items()
                if stats["count"] >= 3  # Minimum applications threshold
            ],
            key=lambda x: (x[1]["total_improvements"] / x[1]["count"]),
            reverse=True
        )[:5]
        
        return {
            "total_refinements": len(self.state.refinement_history),
            "strategy_usage": metrics.get("strategy_usage", {}),
            "metric_averages": metric_averages,
            "top_strategies": [
                {
                    "strategy_id": s_id,
                    "name": self.state.strategies[s_id].name,
                    "avg_improvements": stats["total_improvements"] / stats["count"],
                    "avg_metrics": stats["total_metrics"] / stats["count"],
                    "times_used": stats["count"]
                }
                for s_id, stats in top_strategies
            ],
            "common_improvements": sorted(
                metrics.get("improvements", {}).items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        } 