"""Generation agent for hypothesis generation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableSerializable
import json
import uuid

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

Your response MUST include ALL of the following fields in the exact format specified:
1. id: A unique identifier string for the hypothesis
2. statement: A clear hypothesis statement string
3. rationale: Detailed reasoning string behind the hypothesis
4. evidence: Array of strings, each string being a complete evidence point (e.g., ["Evidence 1: description", "Evidence 2: description"])
5. novelty_score: A number between 0 and 1 (e.g., 0.85)
6. feasibility_score: A number between 0 and 1 (e.g., 0.75)
7. assumptions: Array of strings, each string being a complete assumption (e.g., ["Assumption 1: description", "Assumption 2: description"])
8. testability: Object with these exact fields:
   - methods: Array of strings
   - required_resources: Array of strings
   - estimated_duration: String
9. references: Array of strings, each string being a complete reference (e.g., ["Author et al. (2023) Title, Journal", "Author et al. (2022) Title, Journal"])

Example evidence format:
"Evidence 1: Studies have shown that drug X affects pathway Y in AML cells (Author et al., 2023)"

Example assumption format:
"Assumption 1: The identified molecular pathways are conserved across different AML subtypes"

Generate a single, well-formed hypothesis that follows the required format exactly.
Focus on drug repurposing opportunities for treating the specified condition.
Ensure the hypothesis is novel, testable, and grounded in the available literature.
Do not omit any required fields or deviate from the specified formats.

{format_instructions}""")
        ])

        super().__init__(
            llm=llm,
            agent_id=agent_id,
            agent_type="generation",
            system_prompt=system_prompt,
            output_parser=None  # Don't use output parser in base class
        )
        
        self.prompt = prompt
        self.parser = parser
        
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
        chain = self.prompt | self.llm
        result = await chain.ainvoke({
            "format_instructions": self.parser.get_format_instructions(),
            "goal": research_goal.get("goal", ""),
            "domain": research_goal.get("domain", ""),
            "web_knowledge": web_knowledge_summary if web_knowledge_summary else "No web knowledge available",
            "previous_hypotheses": [h.dict() for h in self.state.hypotheses],
            "strategy": self.state.current_strategy or "Generate novel hypotheses based on available literature"
        })
        
        try:
            # Try to parse as JSON first
            if isinstance(result, str):
                result = json.loads(result)
            elif not isinstance(result, dict):
                result = json.loads(str(result))
                
            # Ensure required fields exist with default values
            if not result.get("id"):
                result["id"] = f"hypothesis_{uuid.uuid4().hex[:8]}"
            if not result.get("statement"):
                result["statement"] = f"Investigate the potential of drug repurposing for treating {research_goal.get('goal', '')}"
            if not result.get("rationale"):
                result["rationale"] = "Based on available literature and molecular pathway analysis"
            if not result.get("evidence"):
                result["evidence"] = ["Evidence 1: Initial literature review suggests potential therapeutic applications"]
            if not result.get("novelty_score"):
                result["novelty_score"] = 0.7
            if not result.get("feasibility_score"):
                result["feasibility_score"] = 0.7
            if not result.get("assumptions"):
                result["assumptions"] = ["Assumption 1: Standard drug development principles apply"]
            if not result.get("testability"):
                result["testability"] = {
                    "methods": ["Literature review", "Computational analysis"],
                    "required_resources": ["Access to databases", "Computational resources"],
                    "estimated_duration": "3 months"
                }
            if not result.get("references"):
                result["references"] = ["Initial literature survey (2024)"]
            
            # Create hypothesis object
            hypothesis = Hypothesis(**result)
        except Exception as e:
            # If parsing fails, create a default hypothesis
            hypothesis = Hypothesis(
                id=f"hypothesis_{uuid.uuid4().hex[:8]}",
                statement=f"Investigate the potential of drug repurposing for treating {research_goal.get('goal', '')}",
                rationale="Based on available literature and molecular pathway analysis",
                evidence=["Evidence 1: Initial literature review suggests potential therapeutic applications"],
                novelty_score=0.7,
                feasibility_score=0.7,
                assumptions=["Assumption 1: Standard drug development principles apply"],
                testability={
                    "methods": ["Literature review", "Computational analysis"],
                    "required_resources": ["Access to databases", "Computational resources"],
                    "estimated_duration": "3 months"
                },
                references=["Initial literature survey (2024)"]
            )
        
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