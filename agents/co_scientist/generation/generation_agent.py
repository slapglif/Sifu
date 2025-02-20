"""Generation agent for hypothesis generation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableSerializable

from ..base_agent import BaseAgent, AgentState

class Hypothesis(BaseModel):
    """Model for research hypotheses."""
    id: str = Field(description="Unique identifier for this hypothesis")
    statement: str = Field(description="Clear hypothesis statement")
    rationale: str = Field(description="Detailed reasoning behind the hypothesis")
    evidence: List[str] = Field(description="Supporting evidence points")
    novelty_score: float = Field(description="Novelty score (0-1)", ge=0.0, le=1.0)
    feasibility_score: float = Field(description="Feasibility score (0-1)", ge=0.0, le=1.0)
    assumptions: List[str] = Field(description="Key assumptions")
    testability: Dict[str, Any] = Field(description="Testability information")
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
        # Create output parser
        parser = PydanticOutputParser(pydantic_object=Hypothesis)
        format_instructions = parser.get_format_instructions()
        
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

{format_instructions}"""

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("""Please generate a research hypothesis based on the following:

Research Goal: {goal}
Domain: {domain}
Web Knowledge: {web_knowledge}
Previous Hypotheses: {previous_hypotheses}
Current Strategy: {strategy}

Use the web knowledge to inform your hypothesis generation. Each web source contains:
- title: Title of the source
- url: URL of the source
- content: Main content
- summary: Brief summary
- metadata: Additional metadata

Your response MUST include ALL of the following fields:
1. id: A unique identifier for the hypothesis
2. statement: A clear hypothesis statement
3. rationale: Detailed reasoning behind the hypothesis
4. evidence: List of supporting evidence points
5. novelty_score: A number between 0 and 1
6. feasibility_score: A number between 0 and 1
7. assumptions: List of key assumptions
8. testability: Object with methods, required_resources, and estimated_duration
9. references: List of literature references

Generate a single, well-formed hypothesis that follows the required format.
Focus on drug repurposing opportunities for treating the specified condition.
Ensure the hypothesis is novel, testable, and grounded in the available literature.
Do not omit any required fields.""")
        ])

        super().__init__(
            llm=llm,
            agent_id=agent_id,
            agent_type="generation",
            system_prompt=system_prompt,
            output_parser=parser
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
        
        # Create chain with format instructions
        self.chain = prompt | self.llm | parser
        
    async def generate_hypothesis(self, research_goal: Dict[str, Any], context: Dict[str, Any]) -> Hypothesis:
        """Generate a new research hypothesis."""
        # Update context
        self.state.literature_context.update(context.get("literature", {}))
        
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
        
        # Generate hypothesis using LLM
        result = await self.chain.ainvoke({
            "format_instructions": PydanticOutputParser(pydantic_object=Hypothesis).get_format_instructions(),
            "goal": research_goal.get("goal", ""),
            "domain": research_goal.get("domain", ""),
            "web_knowledge": web_knowledge_summary if web_knowledge_summary else "No web knowledge available",
            "previous_hypotheses": [h.dict() for h in self.state.hypotheses],
            "strategy": self.state.current_strategy or "Generate novel hypotheses based on available literature"
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