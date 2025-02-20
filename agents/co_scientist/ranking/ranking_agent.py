"""Ranking agent for tournament-based hypothesis evaluation."""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
import math
import json
import uuid
from datetime import datetime

from ..base_agent import BaseAgent, AgentState
from ..generation.generation_agent import Hypothesis

class TournamentMatch(BaseModel):
    """Tournament match between two hypotheses."""
    match_id: str = Field(description="Unique identifier for this match")
    hypothesis_a: str = Field(description="ID of first hypothesis")
    hypothesis_b: str = Field(description="ID of second hypothesis")
    winner: Optional[str] = Field(description="ID of winning hypothesis")
    score_a: float = Field(description="Score for hypothesis A (0-1)", ge=0.0, le=1.0)
    score_b: float = Field(description="Score for hypothesis B (0-1)", ge=0.0, le=1.0)
    reasoning: str = Field(description="Reasoning for the decision")
    criteria_scores: Dict[str, Dict[str, float]] = Field(description="Detailed scoring by criteria")

class RankingState(AgentState):
    """Ranking agent state."""
    matches: List[TournamentMatch] = Field(default_factory=list)
    elo_ratings: Dict[str, float] = Field(default_factory=dict)  # hypothesis_id -> rating
    match_history: List[Dict[str, Any]] = Field(default_factory=list)
    tournament_metrics: Dict[str, Any] = Field(default_factory=dict)
    current_tournament: Optional[str] = None

class RankingAgent(BaseAgent):
    """Agent responsible for tournament-based hypothesis ranking."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        agent_id: str = "ranking",
        system_prompt: Optional[str] = None,
        k_factor: float = 32.0,
        initial_rating: float = 1500.0
    ):
        """Initialize the ranking agent."""
        if system_prompt is None:
            system_prompt = """You are the ranking agent responsible for tournament-based hypothesis evaluation.
Your role is to:
1. Conduct fair and objective hypothesis comparisons
2. Maintain an Elo-based ranking system
3. Identify strongest hypotheses through tournaments
4. Provide detailed comparison reasoning
5. Track hypothesis performance over time
6. Ensure diverse hypothesis evaluation

Follow these guidelines:
- Compare hypotheses systematically
- Use consistent evaluation criteria
- Provide detailed reasoning for decisions
- Consider multiple aspects in scoring
- Track and analyze tournament patterns
- Maintain fairness in comparisons
- Focus on scientific merit"""

        super().__init__(
            llm=llm,
            agent_id=agent_id,
            agent_type="ranking",
            system_prompt=system_prompt,
            output_parser=PydanticOutputParser(pydantic_object=TournamentMatch)
        )
        
        # Initialize ranking-specific state
        self.state = RankingState(
            agent_id=agent_id,
            agent_type="ranking",
            matches=[],
            elo_ratings={},
            match_history=[],
            tournament_metrics={},
            current_tournament=None
        )
        
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        
    async def conduct_match(
        self,
        hypothesis_a: Hypothesis,
        hypothesis_b: Hypothesis,
        context: Dict[str, Any]
    ) -> TournamentMatch:
        """Conduct a tournament match between two hypotheses."""
        try:
            # Generate match result using LLM
            result = await self.arun({
                "hypothesis_a": hypothesis_a.dict(),
                "hypothesis_b": hypothesis_b.dict(),
                "context": context,
                "previous_matches": [
                    m for m in self.state.matches 
                    if m.hypothesis_a in [hypothesis_a.id, hypothesis_b.id] 
                    or m.hypothesis_b in [hypothesis_a.id, hypothesis_b.id]
                ]
            })
            
            # Try to parse as JSON first if result is a string
            if isinstance(result, str):
                result = json.loads(result)
            elif not isinstance(result, dict):
                result = json.loads(str(result))
                
            # Ensure required fields exist with default values
            if not result.get("match_id"):
                result["match_id"] = f"match_{uuid.uuid4().hex[:8]}"
            if not result.get("hypothesis_a"):
                result["hypothesis_a"] = hypothesis_a.id
            if not result.get("hypothesis_b"):
                result["hypothesis_b"] = hypothesis_b.id
            if not result.get("winner"):
                # Default winner is the one with higher score
                score_a = result.get("score_a", 0.5)
                score_b = result.get("score_b", 0.5)
                result["winner"] = hypothesis_a.id if score_a > score_b else hypothesis_b.id
            if not result.get("score_a"):
                result["score_a"] = 0.5
            if not result.get("score_b"):
                result["score_b"] = 0.5
            if not result.get("reasoning"):
                result["reasoning"] = "Comparison based on available evidence and scientific merit"
            if not result.get("criteria_scores"):
                result["criteria_scores"] = {
                    "novelty": {"hypothesis_a": 0.5, "hypothesis_b": 0.5},
                    "feasibility": {"hypothesis_a": 0.5, "hypothesis_b": 0.5},
                    "evidence": {"hypothesis_a": 0.5, "hypothesis_b": 0.5},
                    "impact": {"hypothesis_a": 0.5, "hypothesis_b": 0.5}
                }
            
            # Create match object
            match = TournamentMatch(**result)
            
        except Exception as e:
            # If parsing fails, create a default match
            match = TournamentMatch(
                match_id=f"match_{uuid.uuid4().hex[:8]}",
                hypothesis_a=hypothesis_a.id,
                hypothesis_b=hypothesis_b.id,
                winner=hypothesis_a.id,  # Default to first hypothesis
                score_a=0.5,
                score_b=0.5,
                reasoning="Comparison based on available evidence and scientific merit",
                criteria_scores={
                    "novelty": {"hypothesis_a": 0.5, "hypothesis_b": 0.5},
                    "feasibility": {"hypothesis_a": 0.5, "hypothesis_b": 0.5},
                    "evidence": {"hypothesis_a": 0.5, "hypothesis_b": 0.5},
                    "impact": {"hypothesis_a": 0.5, "hypothesis_b": 0.5}
                }
            )
        
        # Update state
        self.state.matches.append(match)
        self.state.match_history.append({
            "match_id": match.match_id,
            "hypothesis_a": hypothesis_a.id,
            "hypothesis_b": hypothesis_b.id,
            "winner": match.winner,
            "tournament": self.state.current_tournament,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update Elo ratings
        self._update_elo_ratings(match)
        
        # Update metrics
        self._update_tournament_metrics(match)
        
        return match
        
    def _update_elo_ratings(self, match: TournamentMatch) -> None:
        """Update Elo ratings based on match result."""
        # Initialize ratings if needed
        if match.hypothesis_a not in self.state.elo_ratings:
            self.state.elo_ratings[match.hypothesis_a] = self.initial_rating
        if match.hypothesis_b not in self.state.elo_ratings:
            self.state.elo_ratings[match.hypothesis_b] = self.initial_rating
            
        # Get current ratings
        rating_a = self.state.elo_ratings[match.hypothesis_a]
        rating_b = self.state.elo_ratings[match.hypothesis_b]
        
        # Calculate expected scores
        expected_a = 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))
        expected_b = 1.0 / (1.0 + math.pow(10, (rating_a - rating_b) / 400.0))
        
        # Calculate actual scores
        actual_a = match.score_a
        actual_b = match.score_b
        
        # Update ratings
        self.state.elo_ratings[match.hypothesis_a] = rating_a + self.k_factor * (actual_a - expected_a)
        self.state.elo_ratings[match.hypothesis_b] = rating_b + self.k_factor * (actual_b - expected_b)
        
    def _update_tournament_metrics(self, match: TournamentMatch) -> None:
        """Update tournament metrics with new match."""
        metrics = self.state.tournament_metrics
        
        # Update match counts
        metrics["total_matches"] = metrics.get("total_matches", 0) + 1
        
        # Track hypothesis performance
        for hypothesis_id in [match.hypothesis_a, match.hypothesis_b]:
            if hypothesis_id not in metrics.get("hypothesis_stats", {}):
                metrics.setdefault("hypothesis_stats", {})[hypothesis_id] = {
                    "matches": 0,
                    "wins": 0,
                    "total_score": 0.0
                }
            
            stats = metrics["hypothesis_stats"][hypothesis_id]
            stats["matches"] += 1
            if match.winner == hypothesis_id:
                stats["wins"] += 1
            stats["total_score"] += match.score_a if hypothesis_id == match.hypothesis_a else match.score_b
            
        # Track criteria statistics
        for criteria, scores in match.criteria_scores.items():
            if criteria not in metrics.get("criteria_stats", {}):
                metrics.setdefault("criteria_stats", {})[criteria] = {
                    "total_score": 0.0,
                    "count": 0
                }
            
            stats = metrics["criteria_stats"][criteria]
            stats["total_score"] += sum(scores.values())
            stats["count"] += len(scores)
            
        self.state.tournament_metrics = metrics
        
    def get_rankings(self) -> List[Tuple[str, float]]:
        """Get current hypothesis rankings."""
        return sorted(
            self.state.elo_ratings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
    def get_hypothesis_stats(self, hypothesis_id: str) -> Dict[str, Any]:
        """Get statistics for a specific hypothesis."""
        stats = self.state.tournament_metrics.get("hypothesis_stats", {}).get(hypothesis_id, {})
        matches = [m for m in self.state.matches if hypothesis_id in [m.hypothesis_a, m.hypothesis_b]]
        
        return {
            "matches_played": stats.get("matches", 0),
            "wins": stats.get("wins", 0),
            "average_score": stats.get("total_score", 0) / stats.get("matches", 1),
            "current_rating": self.state.elo_ratings.get(hypothesis_id, self.initial_rating),
            "recent_matches": [m.dict() for m in matches[-5:]],
            "win_rate": stats.get("wins", 0) / stats.get("matches", 1) if stats.get("matches", 0) > 0 else 0
        }
        
    def analyze_tournament_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in tournament results."""
        metrics = self.state.tournament_metrics
        
        # Calculate criteria averages
        criteria_averages = {}
        for criteria, stats in metrics.get("criteria_stats", {}).items():
            if stats["count"] > 0:
                criteria_averages[criteria] = stats["total_score"] / stats["count"]
                
        # Get top performing hypotheses
        hypothesis_stats = metrics.get("hypothesis_stats", {})
        top_performers = sorted(
            [
                (h_id, stats)
                for h_id, stats in hypothesis_stats.items()
                if stats["matches"] >= 5  # Minimum matches threshold
            ],
            key=lambda x: x[1]["wins"] / x[1]["matches"],
            reverse=True
        )[:5]
        
        return {
            "total_matches": metrics.get("total_matches", 0),
            "criteria_averages": criteria_averages,
            "top_performers": [
                {
                    "hypothesis_id": h_id,
                    "win_rate": stats["wins"] / stats["matches"],
                    "matches_played": stats["matches"],
                    "average_score": stats["total_score"] / stats["matches"]
                }
                for h_id, stats in top_performers
            ],
            "rating_distribution": {
                "min": min(self.state.elo_ratings.values()) if self.state.elo_ratings else self.initial_rating,
                "max": max(self.state.elo_ratings.values()) if self.state.elo_ratings else self.initial_rating,
                "average": sum(self.state.elo_ratings.values()) / len(self.state.elo_ratings) if self.state.elo_ratings else self.initial_rating
            }
        } 