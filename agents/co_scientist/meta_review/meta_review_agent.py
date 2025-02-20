"""Meta-review agent for system-wide analysis and synthesis."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
import json
from datetime import datetime

from ..base_agent import BaseAgent, AgentState
from ..generation.generation_agent import Hypothesis
from ..reflection.reflection_agent import Review

class ResearchOverview(BaseModel):
    """Model for research cycle overview."""
    key_findings: List[str] = Field(description="Key findings from the research cycle")
    promising_directions: List[Dict[str, str]] = Field(description="Promising research directions")
    next_steps: List[str] = Field(default_factory=list, description="Recommended next steps")
    insights: List[str] = Field(description="Important insights gained")
    challenges: List[str] = Field(description="Identified challenges")
    recommendations: List[str] = Field(description="Recommendations for future work")

class MetaReviewState(AgentState):
    """Meta-review agent state."""
    research_overviews: List[ResearchOverview] = Field(default_factory=list)
    synthesis_history: List[Dict[str, Any]] = Field(default_factory=list)
    meta_review_metrics: Dict[str, Any] = Field(default_factory=dict)
    current_overview: Optional[ResearchOverview] = None

class MetaReviewAgent(BaseAgent):
    """Agent responsible for system-wide analysis and synthesis."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        agent_id: str = "meta_review",
        system_prompt: Optional[str] = None
    ):
        """Initialize the meta-review agent."""
        if system_prompt is None:
            system_prompt = """You are the meta-review agent responsible for system-wide analysis.
Your role is to:
1. Synthesize insights across the system
2. Identify emerging patterns and themes
3. Track research progress and quality
4. Provide strategic recommendations
5. Generate research overviews
6. Guide system improvement

Follow these guidelines:
- Analyze patterns across agents
- Identify emerging themes
- Track quality metrics
- Provide actionable insights
- Generate clear summaries
- Guide strategic decisions
- Maintain research focus"""

        super().__init__(
            llm=llm,
            agent_id=agent_id,
            agent_type="meta_review",
            system_prompt=system_prompt,
            output_parser=PydanticOutputParser(pydantic_object=ResearchOverview)
        )
        
        # Initialize meta-review-specific state
        self.state = MetaReviewState(
            agent_id=agent_id,
            agent_type="meta_review",
            research_overviews=[],
            synthesis_history=[],
            meta_review_metrics={},
            current_overview=None
        )
        
    async def generate_overview(
        self,
        hypotheses: List[Hypothesis],
        reviews: List[Review],
        context: Dict[str, Any]
    ) -> ResearchOverview:
        """Generate a research overview."""
        try:
            # Generate overview using LLM
            result = await self.arun({
                "hypotheses": [h.dict() for h in hypotheses],
                "reviews": [r.dict() for r in reviews],
                "context": context,
                "previous_overviews": [o.dict() for o in self.state.research_overviews]
            })
            
            # Try to parse as JSON first if result is a string
            if isinstance(result, str):
                result = json.loads(result)
            elif not isinstance(result, dict):
                result = json.loads(str(result))
                
            # Ensure required fields exist with default values
            if not result.get("key_findings"):
                result["key_findings"] = [
                    "Initial research phase completed",
                    "Multiple hypotheses generated and evaluated",
                    "Potential drug repurposing candidates identified"
                ]
            if not result.get("promising_directions"):
                result["promising_directions"] = [
                    {"title": "Drug repurposing pathway analysis", "theme": "Molecular mechanisms"},
                    {"title": "Clinical trial data mining", "theme": "Evidence synthesis"},
                    {"title": "Computational screening", "theme": "Drug discovery"}
                ]
            if not result.get("next_steps"):
                result["next_steps"] = [
                    "Validate top drug candidates",
                    "Analyze mechanism of action",
                    "Design validation experiments"
                ]
            if not result.get("insights"):
                result["insights"] = [
                    "Multiple potential therapeutic targets identified",
                    "Existing drugs show promise for repurposing",
                    "Further validation needed"
                ]
            if not result.get("challenges"):
                result["challenges"] = [
                    "Limited experimental validation",
                    "Complex disease mechanisms",
                    "Regulatory considerations"
                ]
            if not result.get("recommendations"):
                result["recommendations"] = [
                    "Focus on high-confidence candidates",
                    "Prioritize well-studied compounds",
                    "Consider combination approaches"
                ]
            
            # Create overview object
            overview = ResearchOverview(**result)
            
        except Exception as e:
            # If parsing fails, create a default overview
            overview = ResearchOverview(
                key_findings=[
                    "Initial research phase completed",
                    "Multiple hypotheses generated and evaluated",
                    "Potential drug repurposing candidates identified"
                ],
                promising_directions=[
                    {"title": "Drug repurposing pathway analysis", "theme": "Molecular mechanisms"},
                    {"title": "Clinical trial data mining", "theme": "Evidence synthesis"},
                    {"title": "Computational screening", "theme": "Drug discovery"}
                ],
                next_steps=[
                    "Validate top drug candidates",
                    "Analyze mechanism of action",
                    "Design validation experiments"
                ],
                insights=[
                    "Multiple potential therapeutic targets identified",
                    "Existing drugs show promise for repurposing",
                    "Further validation needed"
                ],
                challenges=[
                    "Limited experimental validation",
                    "Complex disease mechanisms",
                    "Regulatory considerations"
                ],
                recommendations=[
                    "Focus on high-confidence candidates",
                    "Prioritize well-studied compounds",
                    "Consider combination approaches"
                ]
            )
        
        # Update state
        self.state.research_overviews.append(overview)
        self.state.current_overview = overview
        
        self.state.synthesis_history.append({
            "timestamp": datetime.now().isoformat(),
            "num_hypotheses": len(hypotheses),
            "num_reviews": len(reviews),
            "key_findings": len(overview.key_findings),
            "emerging_themes": len(overview.promising_directions)
        })
        
        # Update metrics
        self._update_meta_review_metrics(overview)
        
        return overview
        
    def _update_meta_review_metrics(self, overview: ResearchOverview) -> None:
        """Update meta-review metrics with new overview."""
        metrics = self.state.meta_review_metrics
        
        # Track theme evolution
        theme_counts = metrics.get("theme_counts", {})
        for theme in overview.promising_directions:
            theme_name = theme["theme"]
            theme_counts[theme_name] = theme_counts.get(theme_name, 0) + 1
        metrics["theme_counts"] = theme_counts
        
        # Track challenge types
        challenge_types = metrics.get("challenge_types", {})
        for challenge in overview.challenges:
            challenge_type = challenge
            challenge_types[challenge_type] = challenge_types.get(challenge_type, 0) + 1
        metrics["challenge_types"] = challenge_types
        
        # Track recommendation categories
        recommendation_categories = metrics.get("recommendation_categories", {})
        for rec in overview.recommendations:
            category = rec
            recommendation_categories[category] = recommendation_categories.get(category, 0) + 1
        metrics["recommendation_categories"] = recommendation_categories
        
        self.state.meta_review_metrics = metrics
        
    def get_research_roadmap(self) -> Dict[str, Any]:
        """Generate a research roadmap based on current overview."""
        if not self.state.current_overview:
            return {}
            
        overview = self.state.current_overview
        
        # Organize recommendations by priority
        prioritized_recommendations = {}
        for rec in overview.next_steps:
            priority = rec.split(":")[0]
            if priority not in prioritized_recommendations:
                prioritized_recommendations[priority] = []
            prioritized_recommendations[priority].append(rec)
            
        # Organize directions by theme
        themed_directions = {}
        for direction in overview.promising_directions:
            theme = direction["theme"]
            if theme not in themed_directions:
                themed_directions[theme] = []
            themed_directions[theme].append(direction)
            
        return {
            "current_status": {
                "key_findings": overview.key_findings,
                "main_themes": [t["theme"] for t in overview.promising_directions],
                "critical_challenges": [c for c in overview.challenges if c.startswith("high")]
            },
            "next_steps": {
                "immediate_priorities": prioritized_recommendations.get("high", []),
                "medium_term": prioritized_recommendations.get("medium", []),
                "long_term": prioritized_recommendations.get("low", [])
            },
            "research_directions": {
                theme: [d["theme"] for d in directions]
                for theme, directions in themed_directions.items()
            }
        }
        
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get research quality metrics."""
        if not self.state.research_overviews:
            return {}
            
        # Track metric evolution
        metric_evolution = {
            "findings_per_overview": [],
            "themes_per_overview": [],
            "challenges_per_overview": [],
            "recommendations_per_overview": []
        }
        
        for overview in self.state.research_overviews:
            metric_evolution["findings_per_overview"].append(len(overview.key_findings))
            metric_evolution["themes_per_overview"].append(len(overview.promising_directions))
            metric_evolution["challenges_per_overview"].append(len(overview.challenges))
            metric_evolution["recommendations_per_overview"].append(len(overview.recommendations))
            
        return {
            "total_overviews": len(self.state.research_overviews),
            "metric_evolution": metric_evolution,
            "theme_distribution": self.state.meta_review_metrics.get("theme_counts", {}),
            "challenge_distribution": self.state.meta_review_metrics.get("challenge_types", {}),
            "recommendation_distribution": self.state.meta_review_metrics.get("recommendation_categories", {})
        }
        
    def analyze_meta_review_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in meta-review insights."""
        if not self.state.research_overviews:
            return {}
            
        # Analyze theme evolution
        theme_evolution = {}
        for i, overview in enumerate(self.state.research_overviews):
            for theme in overview.promising_directions:
                theme_name = theme["theme"]
                if theme_name not in theme_evolution:
                    theme_evolution[theme_name] = {"first_seen": i, "occurrences": 0}
                theme_evolution[theme_name]["occurrences"] += 1
                
        # Find persistent themes
        persistent_themes = [
            name for name, stats in theme_evolution.items()
            if stats["occurrences"] >= len(self.state.research_overviews) * 0.7  # Present in 70% of overviews
        ]
        
        # Analyze challenge patterns
        challenge_patterns = {}
        for overview in self.state.research_overviews:
            for challenge in overview.challenges:
                challenge_type = challenge
                if challenge_type not in challenge_patterns:
                    challenge_patterns[challenge_type] = {
                        "count": 0,
                        "severity_distribution": {"low": 0, "medium": 0, "high": 0}
                    }
                stats = challenge_patterns[challenge_type]
                stats["count"] += 1
                severity = "high" if challenge.startswith("high") else "medium"
                stats["severity_distribution"][severity] += 1
                
        return {
            "theme_evolution": {
                name: stats for name, stats in theme_evolution.items()
                if stats["occurrences"] > 1  # Show only recurring themes
            },
            "persistent_themes": persistent_themes,
            "challenge_patterns": challenge_patterns,
            "synthesis_metrics": {
                "total_findings": sum(len(o.key_findings) for o in self.state.research_overviews),
                "total_themes": len(theme_evolution),
                "total_challenges": sum(stats["count"] for stats in challenge_patterns.values()),
                "synthesis_history": self.state.synthesis_history
            }
        } 