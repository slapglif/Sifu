"""Reflection agent for hypothesis evaluation."""

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from ..base_agent import BaseAgent, AgentState
from ..generation.generation_agent import Hypothesis

class Review(BaseModel):
    """Model for hypothesis reviews."""
    hypothesis_id: str = Field(description="ID of the hypothesis being reviewed")
    review_type: str = Field(description="Type of review conducted")
    score: float = Field(description="Overall review score")
    confidence: float = Field(description="Confidence in the review")
    key_points: List[str] = Field(default_factory=list, description="Key points from the review")
    strengths: List[str] = Field(description="Identified strengths")
    weaknesses: List[str] = Field(description="Identified weaknesses")
    suggestions: List[str] = Field(description="Suggestions for improvement")
    timestamp: str = Field(description="When the review was conducted")

class ReflectionState(AgentState):
    """Reflection agent state."""
    reviews: Dict[str, List[Review]] = Field(default_factory=dict)  # hypothesis_id -> reviews
    review_history: List[Dict[str, Any]] = Field(default_factory=list)
    verification_tools: Dict[str, Any] = Field(default_factory=dict)
    review_metrics: Dict[str, Any] = Field(default_factory=dict)

class ReflectionAgent(BaseAgent):
    """Agent responsible for evaluating research hypotheses."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        agent_id: str = "reflection",
        system_prompt: Optional[str] = None
    ):
        """Initialize the reflection agent."""
        if system_prompt is None:
            system_prompt = """You are the reflection agent responsible for evaluating research hypotheses.
Your role is to:
1. Conduct thorough reviews of research hypotheses
2. Identify strengths and weaknesses
3. Verify assumptions and claims
4. Assess practical feasibility
5. Suggest concrete improvements
6. Maintain high scientific standards

Review Types:
- Initial Review: Quick assessment of basic validity
- Full Review: Comprehensive evaluation with literature
- Deep Verification: Detailed analysis of assumptions
- Observation Review: Check against existing evidence
- Simulation Review: Mental simulation of mechanisms
- Tournament Review: Comparative evaluation

Follow these guidelines:
- Be thorough and systematic
- Support all claims with evidence
- Consider multiple perspectives
- Identify potential issues early
- Suggest concrete improvements
- Maintain scientific rigor
- Be constructive in criticism

Your output must be a JSON object with the following structure:
{
    "review_id": "unique_review_id",
    "hypothesis_id": "id_of_hypothesis_being_reviewed",
    "review_type": "initial",  # one of: initial, full, deep_verification, observation, simulation, tournament
    "score": 0.85,  # between 0 and 1
    "strengths": [
        "strength point 1",
        "strength point 2"
    ],
    "weaknesses": [
        "weakness point 1",
        "weakness point 2"
    ],
    "suggestions": [
        "improvement suggestion 1",
        "improvement suggestion 2"
    ],
    "verification_results": {
        "assumption_checks": ["check1", "check2"],
        "evidence_validation": ["validation1", "validation2"],
        "feasibility_assessment": ["assessment1", "assessment2"]
    },
    "confidence": 0.9  # between 0 and 1
}"""

        super().__init__(
            llm=llm,
            agent_id=agent_id,
            agent_type="reflection",
            system_prompt=system_prompt,
            output_parser=PydanticOutputParser(pydantic_object=Review)
        )
        
        # Initialize reflection-specific state
        self.state = ReflectionState(
            agent_id=agent_id,
            agent_type="reflection",
            reviews={},
            review_history=[],
            verification_tools={},
            review_metrics={}
        )
        
    async def review_hypothesis(
        self,
        hypothesis: Hypothesis,
        review_type: Literal["initial", "full", "deep_verification", "observation", "simulation", "tournament"],
        context: Dict[str, Any]
    ) -> Review:
        """Review a research hypothesis."""
        # Generate review using LLM
        result = await self.arun({
            "hypothesis": hypothesis.dict(),
            "review_type": review_type,
            "context": context,
            "previous_reviews": [r.dict() for r in self.state.reviews.get(hypothesis.id, [])],
            "verification_tools": list(self.state.verification_tools.keys())
        })
        
        # Create review object
        if isinstance(result, dict):
            review = Review(**result)
        else:
            # If result is already a Review (from output parser), use it directly
            review = result
        
        # Update state
        if hypothesis.id not in self.state.reviews:
            self.state.reviews[hypothesis.id] = []
        self.state.reviews[hypothesis.id].append(review)
        
        self.state.review_history.append({
            "review_id": review.review_id,
            "hypothesis_id": hypothesis.id,
            "review_type": review_type,
            "context": context,
            "timestamp": "TODO: Add timestamp"
        })
        
        # Update metrics
        self._update_review_metrics(review)
        
        return review
        
    def register_verification_tool(self, name: str, tool: Any) -> None:
        """Register a verification tool."""
        self.state.verification_tools[name] = tool
        
    def get_reviews(
        self,
        hypothesis_id: Optional[str] = None,
        review_type: Optional[str] = None
    ) -> List[Review]:
        """Get reviews, optionally filtered by hypothesis and type."""
        if hypothesis_id and hypothesis_id not in self.state.reviews:
            return []
            
        reviews = []
        if hypothesis_id:
            reviews = self.state.reviews[hypothesis_id]
        else:
            reviews = [r for rs in self.state.reviews.values() for r in rs]
            
        if review_type:
            reviews = [r for r in reviews if r.review_type == review_type]
            
        return reviews
        
    def get_review_metrics(self) -> Dict[str, Any]:
        """Get current review metrics."""
        return self.state.review_metrics
        
    def _update_review_metrics(self, review: Review) -> None:
        """Update review metrics with new review."""
        metrics = self.state.review_metrics
        
        # Update review type counts
        type_counts = metrics.get("review_type_counts", {})
        type_counts[review.review_type] = type_counts.get(review.review_type, 0) + 1
        metrics["review_type_counts"] = type_counts
        
        # Update average scores
        scores = metrics.get("average_scores", {})
        review_type = review.review_type
        current_avg = scores.get(review_type, {"sum": 0, "count": 0})
        current_avg["sum"] += review.score
        current_avg["count"] += 1
        scores[review_type] = current_avg
        metrics["average_scores"] = scores
        
        # Update common issues
        all_weaknesses = metrics.get("common_weaknesses", {})
        for weakness in review.weaknesses:
            all_weaknesses[weakness] = all_weaknesses.get(weakness, 0) + 1
        metrics["common_weaknesses"] = all_weaknesses
        
        # Update improvement suggestions
        all_suggestions = metrics.get("common_suggestions", {})
        for suggestion in review.suggestions:
            all_suggestions[suggestion] = all_suggestions.get(suggestion, 0) + 1
        metrics["common_suggestions"] = all_suggestions
        
        self.state.review_metrics = metrics
        
    def analyze_review_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in reviews."""
        metrics = self.state.review_metrics
        
        # Calculate averages
        avg_scores = {}
        for review_type, data in metrics.get("average_scores", {}).items():
            if data["count"] > 0:
                avg_scores[review_type] = data["sum"] / data["count"]
                
        # Get top issues and suggestions
        top_weaknesses = sorted(
            metrics.get("common_weaknesses", {}).items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        top_suggestions = sorted(
            metrics.get("common_suggestions", {}).items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "total_reviews": sum(metrics.get("review_type_counts", {}).values()),
            "review_type_distribution": metrics.get("review_type_counts", {}),
            "average_scores_by_type": avg_scores,
            "top_weaknesses": top_weaknesses,
            "top_suggestions": top_suggestions
        } 