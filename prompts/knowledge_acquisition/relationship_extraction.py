"""Relationship extraction prompts."""
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

class RelationshipResponse(BaseModel):
    """Schema for relationship extraction response"""
    relationships: List[Dict[str, str]] = Field(description="Extracted relationships")

def get_relationship_extraction_prompt() -> ChatPromptTemplate:
    """Get the relationship extraction prompt template."""
    parser = PydanticOutputParser(pydantic_object=RelationshipResponse)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are an expert at extracting medical and therapeutic relationships from scientific text.
{format_instructions}

CRITICAL RULES:
1. You MUST output ONLY a valid JSON object
2. The JSON MUST match the schema exactly
3. The relationships field MUST be an array of objects
4. Each relationship object MUST have exactly these fields:
   - source: string (medical entity)
   - relation: string (one of the valid types)
   - target: string (medical entity)
5. All strings MUST be properly escaped if they contain special characters
6. Do not include any text before or after the JSON object
7. Focus on medical/therapeutic relationships only

VALID RELATIONSHIP TYPES:
1. Medical Relationships:
   - treats (for treatment relationships)
   - causes (for causative relationships)
   - prevents (for preventive relationships)
   - affects (for impact relationships)
   - regulates (for regulatory relationships)
   - part_of (for anatomical relationships)
   - interacts_with (for drug/therapy interactions)

2. Therapeutic Relationships:
   - improves (for therapeutic benefits)
   - reduces (for symptom reduction)
   - increases (for enhancement effects)
   - modulates (for biological modulation)
   - targets (for therapeutic targeting)

3. Research Relationships:
   - studied_in (for research context)
   - measured_by (for assessment methods)
   - associated_with (for correlations)
   - supported_by (for evidence basis)
   - compared_to (for comparative studies)

4. Mechanism Relationships:
   - activates (for activation pathways)
   - inhibits (for inhibitory effects)
   - mediates (for mediating processes)
   - signals_through (for signaling pathways)
   - binds_to (for molecular binding)

RELATIONSHIP VALIDATION RULES:
1. Medical Validity:
   - Both source and target must be valid medical entities
   - Relationship must be supported by medical literature
   - Direction of relationship must be biologically plausible

2. Therapeutic Relevance:
   - Focus on relationships relevant to treatment
   - Include mechanism of action where possible
   - Consider patient safety and outcomes

3. Evidence Quality:
   - Prefer relationships from clinical studies
   - Note strength of evidence
   - Consider replication status

4. Domain Specificity:
   - Focus on PANDAS-relevant relationships
   - Include gut-brain axis connections
   - Consider vagus nerve pathways
   - Include plant-based therapeutic mechanisms"""

    human_template = """Extract medical and therapeutic relationships from this text:

{content}

Remember:
1. Return ONLY a valid JSON object
2. Focus on PANDAS, gut-brain axis, and therapeutic relationships
3. Include mechanism of action where possible
4. Validate relationships against medical knowledge
5. Consider evidence quality"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])
    
    prompt = prompt.partial(format_instructions=format_instructions)
    return prompt