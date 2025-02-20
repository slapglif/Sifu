"""Generation agent for hypothesis generation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from ..base_agent import BaseAgent, AgentState

class Hypothesis(BaseModel):
    """Research hypothesis model."""
    id: str = Field(description="Unique identifier for this hypothesis")
    statement: str = Field(description="The hypothesis statement")
    rationale: str = Field(description="Reasoning behind the hypothesis")
    evidence: List[str] = Field(description="Supporting evidence from literature")
    novelty_score: float = Field(description="Estimated novelty (0-1)", ge=0.0, le=1.0)
    feasibility_score: float = Field(description="Estimated feasibility (0-1)", ge=0.0, le=1.0)
    assumptions: List[str] = Field(description="Key assumptions")
    testability: Dict[str, Any] = Field(description="Information about how to test the hypothesis")
    references: List[str] = Field(description="Literature references")

class GenerationState(AgentState):
    """Generation agent state."""
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    literature_context: Dict[str, Any] = Field(default_factory=dict)
    generation_history: List[Dict[str, Any]] = Field(default_factory=list)
    current_strategy: Optional[str] = None

class GenerationAgent(BaseAgent):
    """Agent responsible for generating research hypotheses."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        agent_id: str = "generation",
        system_prompt: Optional[str] = None
    ):
        """Initialize the generation agent."""
        if system_prompt is None:
            system_prompt = """You are the generation agent responsible for creating novel research hypotheses.
Your role is to:
1. Generate original and testable research hypotheses
2. Ground hypotheses in existing scientific literature
3. Ensure hypotheses are aligned with research goals
4. Provide clear rationale and evidence
5. Assess novelty and feasibility
6. Break down complex hypotheses into testable components

Follow these guidelines:
- Focus on generating truly novel ideas
- Ensure all hypotheses are scientifically sound
- Provide detailed supporting evidence
- Consider multiple research directions
- Break down complex ideas into testable parts
- Maintain clear documentation of your reasoning
- Assess practical feasibility of testing

Your output must be a JSON object with the following structure:
{
    "id": "unique_hypothesis_id",
    "statement": "clear hypothesis statement",
    "rationale": "detailed reasoning behind the hypothesis",
    "evidence": [
        "Evidence point 1: Description of supporting evidence",
        "Evidence point 2: Additional supporting evidence"
    ],
    "novelty_score": 0.85,  # between 0 and 1
    "feasibility_score": 0.75,  # between 0 and 1
    "assumptions": [
        "Assumption 1: Description of key assumption",
        "Assumption 2: Description of another assumption"
    ],
    "testability": {
        "methods": ["method1", "method2"],
        "required_resources": ["resource1", "resource2"],
        "estimated_duration": "duration estimate"
    },
    "references": [
        "Author et al. (2023) Title of paper, Journal Name",
        "Author et al. (2022) Another paper title, Journal Name"
    ]
}

IMPORTANT: All evidence, assumptions, and references must be simple strings, not objects or dictionaries."""

        super().__init__(
            llm=llm,
            agent_id=agent_id,
            agent_type="generation",
            system_prompt=system_prompt,
            output_parser=PydanticOutputParser(pydantic_object=Hypothesis)
        )
        
        # Initialize generation-specific state
        self.state = GenerationState(
            agent_id=agent_id,
            agent_type="generation",
            hypotheses=[],
            literature_context={},
            generation_history=[],
            current_strategy=None
        )
        
    async def generate_hypothesis(self, research_goal: Dict[str, Any], context: Dict[str, Any]) -> Hypothesis:
        """Generate a new research hypothesis."""
        # Update context
        self.state.literature_context.update(context.get("literature", {}))
        
        # Generate hypothesis using LLM
        result = await self.arun({
            "goal": research_goal,
            "context": context,
            "previous_hypotheses": [h.dict() for h in self.state.hypotheses],
            "strategy": self.state.current_strategy
        })
        
        # Create hypothesis object
        if isinstance(result, dict):
            hypothesis = Hypothesis(**result)
        else:
            # If result is already a Hypothesis (from output parser), use it directly
            hypothesis = result
        
        # Update state
        self.state.hypotheses.append(hypothesis)
        self.state.generation_history.append({
            "hypothesis_id": hypothesis.id,
            "goal": research_goal,
            "context": context,
            "strategy": self.state.current_strategy,
            "timestamp": "TODO: Add timestamp"
        })
        
        return hypothesis
        
    def set_generation_strategy(self, strategy: str) -> None:
        """Set the current hypothesis generation strategy."""
        self.state.current_strategy = strategy
        self.update_memory("current_strategy", strategy)
        
    def get_hypotheses(self, filters: Optional[Dict[str, Any]] = None) -> List[Hypothesis]:
        """Get generated hypotheses, optionally filtered."""
        if not filters:
            return self.state.hypotheses
            
        filtered = self.state.hypotheses
        
        # Apply filters
        if "min_novelty" in filters:
            filtered = [h for h in filtered if h.novelty_score >= filters["min_novelty"]]
            
        if "min_feasibility" in filters:
            filtered = [h for h in filtered if h.feasibility_score >= filters["min_feasibility"]]
            
        if "keywords" in filters:
            keywords = filters["keywords"]
            filtered = [
                h for h in filtered 
                if any(k.lower() in h.statement.lower() for k in keywords)
            ]
            
        return filtered
        
    def get_generation_history(self) -> List[Dict[str, Any]]:
        """Get the history of hypothesis generation."""
        return self.state.generation_history
        
    def analyze_hypothesis_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in generated hypotheses."""
        if not self.state.hypotheses:
            return {}
            
        # Calculate basic statistics
        avg_novelty = sum(h.novelty_score for h in self.state.hypotheses) / len(self.state.hypotheses)
        avg_feasibility = sum(h.feasibility_score for h in self.state.hypotheses) / len(self.state.hypotheses)
        
        # Collect common themes
        all_assumptions = [a for h in self.state.hypotheses for a in h.assumptions]
        assumption_counts = {}
        for assumption in all_assumptions:
            assumption_counts[assumption] = assumption_counts.get(assumption, 0) + 1
            
        return {
            "total_hypotheses": len(self.state.hypotheses),
            "average_novelty": avg_novelty,
            "average_feasibility": avg_feasibility,
            "common_assumptions": sorted(
                assumption_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "strategies_used": list(set(h["strategy"] for h in self.state.generation_history if h["strategy"]))
        } 