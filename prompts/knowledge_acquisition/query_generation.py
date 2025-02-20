"""Query generation prompts."""
from typing import List, Tuple
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

class QueryGroup(BaseModel):
    """Group of related search queries"""
    group_name: str = Field(description="Name of the query group")
    queries: List[str] = Field(description="List of search queries in this group")
    strategy: str = Field(description="Search strategy/approach for this group")

class QueryGenerationResponse(BaseModel):
    """Response for query generation"""
    query_groups: List[QueryGroup] = Field(description="Generated query groups")
    reasoning: str = Field(description="Explanation of query generation strategy")

def get_query_generation_prompt() -> ChatPromptTemplate:
    """Get the query generation prompt template."""
    system_template = """You are an expert at generating diverse and effective search queries for academic and medical research.
Given a base query and domain, generate groups of related search queries that will help gather comprehensive information.

{format_instructions}

CRITICAL RULES:
1. You MUST output ONLY a valid JSON object
2. The JSON MUST match the schema exactly
3. Each query group must have 3-5 unique queries
4. Queries must be specific and targeted
5. No duplicate queries across groups
6. Each query must be clear and well-formed
7. Use proper medical/scientific terminology
8. Include different search strategies per group

Query Group Types to Generate:
1. Overview & Background:
   - Current understanding
   - Historical context
   - Key concepts
   
2. Clinical Research:
   - Clinical trials
   - Patient outcomes
   - Treatment efficacy
   
3. Mechanisms & Pathways:
   - Biological mechanisms
   - Molecular pathways
   - Physiological processes
   
4. Treatment Approaches:
   - Therapeutic strategies
   - Intervention methods
   - Treatment protocols
   
5. Reviews & Meta-analyses:
   - Systematic reviews
   - Meta-analyses
   - Literature reviews

6. Latest Developments:
   - Recent advances
   - New findings
   - Emerging research

CRITICAL: You MUST respond with ONLY a valid JSON object, no other text.
DO NOT include ```json or ``` markers.
DO NOT include any explanatory text.
ENSURE all JSON is properly escaped and formatted."""

    human_template = """Generate diverse search queries for this topic:

Base Query: {base_query}
Domain: {domain}

Remember to:
1. Generate unique, specific queries
2. Use proper terminology
3. Cover different aspects
4. Ensure queries are well-formed
5. Include various search strategies"""

    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ]) 