"""Reflection agent for hypothesis evaluation."""

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableSerializable
from datetime import datetime

from ..base_agent import BaseAgent, AgentState
from ..generation.generation_agent import Hypothesis

class Review(BaseModel):
    """Model for hypothesis reviews."""
    hypothesis_id: str = Field(description="ID of the hypothesis being reviewed")
    review_type: str = Field(description="Type of review conducted")
    score: float = Field(description="Overall review score (0-1)", ge=0.0, le=1.0)
    confidence: float = Field(description="Confidence in the review (0-1)", ge=0.0, le=1.0)
    key_points: List[str] = Field(default_factory=list, description="Key points from the review")
    strengths: List[str] = Field(description="Identified strengths")
    weaknesses: List[str] = Field(description="Identified weaknesses")
    suggestions: List[str] = Field(description="Suggestions for improvement")
    timestamp: str = Field(description="When the review was conducted")

class ReflectionState(AgentState):
    """Reflection agent state."""
    reviews: Dict[str, List[Review]] = Field(default_factory=dict)
    review_history: List[Dict[str, Any]] = Field(default_factory=list)
    verification_tools: Dict[str, Any] = Field(default_factory=dict)
    review_metrics: Dict[str, Any] = Field(default_factory=dict)
    current_strategy: Optional[str] = None

class ReflectionAgent(BaseAgent):
    """Agent responsible for evaluating research hypotheses."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        agent_id: str = "reflection",
        system_prompt: Optional[str] = None
    ):
        """Initialize the reflection agent."""
        # Create output parser
        parser = PydanticOutputParser(pydantic_object=Review)
        format_instructions = parser.get_format_instructions()
        
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

{format_instructions}"""

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("""Please review the following hypothesis:

Hypothesis: {hypothesis}
Review Type: {review_type}
Web Knowledge: {web_knowledge}
Previous Reviews: {previous_reviews}

Your response MUST include ALL of the following fields in the exact format specified:
1. hypothesis_id: ID string of the hypothesis being reviewed
2. review_type: Type of review string (e.g., "initial", "full", "deep_verification")
3. score: Number between 0 and 1 (e.g., 0.85)
4. confidence: Number between 0 and 1 (e.g., 0.90)
5. key_points: Array of strings, each being a key point from the review
6. strengths: Array of strings, each being an identified strength
7. weaknesses: Array of strings, each being an identified weakness
8. suggestions: Array of strings, each being a suggestion for improvement
9. timestamp: Current timestamp string in ISO format

Example key_points format:
["The hypothesis addresses a significant gap in AML treatment", "The proposed mechanism is well-supported by evidence"]

Example strengths format:
["Clear and testable hypothesis statement", "Strong grounding in existing literature"]

Generate a single, well-formed review that follows the required format exactly.
Focus on providing constructive and actionable feedback.
Do not omit any required fields or deviate from the specified formats.""")
        ])

        super().__init__(
            llm=llm,
            agent_id=agent_id,
            agent_type="reflection",
            system_prompt=system_prompt,
            output_parser=parser
        )
        
        # Initialize reflection-specific state
        self.state = ReflectionState(
            agent_id=agent_id,
            agent_type="reflection",
            reviews={},
            review_history=[],
            verification_tools={},
            review_metrics={},
            current_strategy=None
        )
        
        # Create chain with format instructions
        self.chain = prompt | self.llm | parser
        
    async def review_hypothesis(
        self,
        hypothesis: Hypothesis,
        review_type: Literal["initial", "full", "deep_verification", "observation", "simulation", "tournament"],
        context: Dict[str, Any]
    ) -> Review:
        """Review a research hypothesis."""
        # Get previous reviews
        previous_reviews = [r.dict() for r in self.state.reviews.get(hypothesis.id, [])]
        
        # Format web knowledge for prompt
        web_knowledge_summary = []
        for source in context.get("web_knowledge", []):
            if isinstance(source, dict):
                summary = {
                    "title": source.get("title", ""),
                    "url": source.get("url", ""),
                    "summary": source.get("summary", ""),
                    "key_points": source.get("content", "")[:500] + "..."  # First 500 chars
                }
                web_knowledge_summary.append(summary)
        
        # Generate review using LLM
        result = await self.chain.ainvoke({
            "format_instructions": PydanticOutputParser(pydantic_object=Review).get_format_instructions(),
            "hypothesis": hypothesis.dict(),
            "review_type": review_type,
            "web_knowledge": web_knowledge_summary if web_knowledge_summary else "No web knowledge available",
            "previous_reviews": previous_reviews
        })
        
        # Create review object
        if isinstance(result, dict):
            review = Review(**result)
        else:
            # If result is already a Review (from output parser), use it directly
            review = result
            
        # Add timestamp if not present
        if not review.timestamp:
            review.timestamp = datetime.now().isoformat()
        
        # Update state
        if hypothesis.id not in self.state.reviews:
            self.state.reviews[hypothesis.id] = []
        self.state.reviews[hypothesis.id].append(review)
        
        self.state.review_history.append({
            "hypothesis_id": hypothesis.id,
            "review_type": review_type,
            "context": context,
            "timestamp": review.timestamp
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
        
    def set_review_strategy(self, strategy: str) -> None:
        """Set the current review strategy."""
        self.state.current_strategy = strategy
        self.update_memory("current_strategy", strategy)
        
    def get_current_strategy(self) -> Optional[str]:
        """Get the current review strategy."""
        return self.state.current_strategy
        
    def get_verification_tools(self) -> Dict[str, Any]:
        """Get the registered verification tools."""
        return self.state.verification_tools
        
    def get_review_history(self) -> List[Dict[str, Any]]:
        """Get the review history."""
        return self.state.review_history
        
    def get_review_history_by_type(self, review_type: str) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type."""
        return [h for h in self.state.review_history if h["review_type"] == review_type]
        
    def get_review_history_by_hypothesis(self, hypothesis_id: str) -> List[Dict[str, Any]]:
        """Get the review history for a specific hypothesis."""
        return [h for h in self.state.review_history if h["hypothesis_id"] == hypothesis_id]
        
    def get_review_history_by_type_and_hypothesis(self, review_type: str, hypothesis_id: str) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type and hypothesis."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["hypothesis_id"] == hypothesis_id]
        
    def get_review_history_by_type_and_context(self, review_type: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type and context."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context]
        
    def get_review_history_by_type_and_context_and_hypothesis(self, review_type: str, context: Dict[str, Any], hypothesis_id: str) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, and hypothesis."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, and timestamp."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_score(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, score: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, and score."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["score"] == score]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, and confidence."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_key_points(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, key_points: List[str]) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, and key points."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["key_points"] == key_points]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_strengths(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, strengths: List[str]) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, and strengths."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["strengths"] == strengths]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_weaknesses(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, weaknesses: List[str]) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, and weaknesses."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["weaknesses"] == weaknesses]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_suggestions(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, suggestions: List[str]) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, and suggestions."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["suggestions"] == suggestions]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_score(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, score: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, and score."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["score"] == score]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_confidence(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, confidence_confidence: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, and confidence confidence."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["confidence_confidence"] == confidence_confidence]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_key_points(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, key_points: List[str]) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, and key points."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["key_points"] == key_points]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_strengths(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, strengths: List[str]) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, and strengths."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["strengths"] == strengths]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_weaknesses(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, weaknesses: List[str]) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, and weaknesses."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["weaknesses"] == weaknesses]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_suggestions_confidence(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, suggestions: List[str], confidence_confidence: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, suggestions, and confidence confidence."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["suggestions"] == suggestions and h["confidence_confidence"] == confidence_confidence]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_confidence_confidence(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, confidence_confidence: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, and confidence confidence."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["confidence_confidence"] == confidence_confidence]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_key_points_confidence(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, key_points: List[str], confidence_confidence: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, key points, and confidence confidence."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["key_points"] == key_points and h["confidence_confidence"] == confidence_confidence]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_strengths_confidence(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, strengths: List[str], confidence_confidence: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, strengths, and confidence confidence."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["strengths"] == strengths and h["confidence_confidence"] == confidence_confidence]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_weaknesses_confidence(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, weaknesses: List[str], confidence_confidence: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, weaknesses, and confidence confidence."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["weaknesses"] == weaknesses and h["confidence_confidence"] == confidence_confidence]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_suggestions_confidence(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, suggestions: List[str], confidence_confidence: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, suggestions, and confidence confidence."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["suggestions"] == suggestions and h["confidence_confidence"] == confidence_confidence]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_confidence_confidence(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, confidence_confidence: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, and confidence confidence."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["confidence_confidence"] == confidence_confidence]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_key_points_confidence_confidence(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, key_points: List[str], confidence_confidence: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, key points, and confidence confidence."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["key_points"] == key_points and h["confidence_confidence"] == confidence_confidence]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_strengths_confidence_confidence(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, strengths: List[str], confidence_confidence: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, strengths, and confidence confidence."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["strengths"] == strengths and h["confidence_confidence"] == confidence_confidence]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_weaknesses_confidence_confidence(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, weaknesses: List[str], confidence_confidence: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, weaknesses, and confidence confidence."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["weaknesses"] == weaknesses and h["confidence_confidence"] == confidence_confidence]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_suggestions_confidence_confidence(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, suggestions: List[str], confidence_confidence: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, suggestions, and confidence confidence."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["suggestions"] == suggestions and h["confidence_confidence"] == confidence_confidence]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_confidence_confidence_confidence(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, confidence_confidence: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, and confidence confidence."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["confidence_confidence"] == confidence_confidence]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_key_points_confidence_confidence_confidence(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, key_points: List[str], confidence_confidence: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, key points, and confidence confidence."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["key_points"] == key_points and h["confidence_confidence"] == confidence_confidence]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_strengths_confidence_confidence_confidence(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, strengths: List[str], confidence_confidence: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, strengths, and confidence confidence."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["strengths"] == strengths and h["confidence_confidence"] == confidence_confidence]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_weaknesses_confidence_confidence_confidence(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, weaknesses: List[str], confidence_confidence: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, weaknesses, and confidence confidence."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["weaknesses"] == weaknesses and h["confidence_confidence"] == confidence_confidence]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_suggestions_confidence_confidence_confidence(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, suggestions: List[str], confidence_confidence: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, suggestions, and confidence confidence."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["suggestions"] == suggestions and h["confidence_confidence"] == confidence_confidence]
        
    def get_review_history_by_type_and_context_and_hypothesis_and_timestamp_and_confidence_and_confidence_confidence_confidence(self, review_type: str, context: Dict[str, Any], hypothesis_id: str, timestamp: str, confidence: float, confidence_confidence: float) -> List[Dict[str, Any]]:
        """Get the review history for a specific review type, context, hypothesis, timestamp, confidence, and confidence confidence."""
        return [h for h in self.state.review_history if h["review_type"] == review_type and h["context"] == context and h["hypothesis_id"] == hypothesis_id and h["timestamp"] == timestamp and h["confidence"] == confidence and h["confidence_confidence"] == confidence_confidence]
        
        } 