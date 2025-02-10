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
    
    system_template = """You are an expert at extracting relationships between concepts from text.
{format_instructions}

CRITICAL RULES:
1. You MUST output ONLY a valid JSON object
2. The JSON MUST match the schema exactly
3. The relationships field MUST be an array of objects
4. Each relationship object MUST have exactly these fields:
   - source: string
   - relation: string (one of the valid types)
   - target: string
5. All strings MUST be properly escaped if they contain special characters
6. Do not include any text before or after the JSON object
7. Do not include any explanations or notes
8. The response should look exactly like this:
{{
    "relationships": [
        {{
            "source": "Neural Networks",
            "relation": "is_a",
            "target": "Machine Learning Model"
        }},
        {{
            "source": "Data Preprocessing",
            "relation": "precedes",
            "target": "Model Training"
        }}
    ]
}}

VALID RELATIONSHIP TYPES:
1. Methodology relationships:
   - uses (for tool/method usage)
   - applies (for applying concepts/techniques)
   - implements (for implementation relationships)
2. Performance relationships:
   - improves (for enhancement relationships)
   - outperforms (for comparison relationships)
   - achieves (for accomplishment relationships)
3. Component relationships:
   - contains (for containment relationships)
   - consists_of (for composition relationships)
   - part_of (for part-whole relationships)
4. Comparison relationships:
   - better_than (for superiority relationships)
   - similar_to (for similarity relationships)
   - different_from (for contrast relationships)
5. Causal relationships:
   - leads_to (for consequence relationships)
   - causes (for direct causation)
   - affects (for influence relationships)
6. Temporal relationships:
   - precedes (for sequential relationships)
   - follows (for order relationships)
   - concurrent_with (for parallel relationships)
7. Legacy relationships:
   - is_a (for type/class relationships)
   - has_part (for composition relationships)
   - related_to (for general relationships)

RELATIONSHIP TYPE MAPPING:
- For usage relationships like "used_for", "used_in", "utilizes", map to "uses"
- For application relationships like "applied_to", "applied_in", map to "applies"
- For implementation relationships like "implemented_by", "implemented_in", map to "implements"
- For improvement relationships like "enhances", "boosts", map to "improves"
- For performance relationships like "performs_better", map to "outperforms"
- For achievement relationships like "reaches", "attains", map to "achieves"
- For containment relationships like "includes", "incorporates", map to "contains"
- For composition relationships like "made_of", "composed_of", map to "consists_of"
- For part relationships like "belongs_to", map to "part_of"
- For comparison relationships like "superior_to", map to "better_than"
- For similarity relationships like "resembles", map to "similar_to"
- For difference relationships like "differs_from", map to "different_from"
- For consequence relationships like "results_in", map to "leads_to"
- For causation relationships like "produces", map to "causes"
- For influence relationships like "impacts", map to "affects"
- For sequence relationships like "comes_before", map to "precedes"
- For order relationships like "comes_after", map to "follows"
- For parallel relationships like "happens_with", map to "concurrent_with"
- For type relationships like "type_of", "kind_of", map to "is_a"
- For composition relationships like "contains_part", map to "has_part"
- For general relationships like "connected_to", map to "related_to"

GUIDELINES for relationship extraction:
1. Extract meaningful relationships between entities
2. Ensure both source and target are valid entities
3. Use the most specific relationship type that applies
4. Consider both explicit and implicit relationships
5. Maintain logical relationship direction
6. Avoid redundant or duplicate relationships
7. Map any non-standard relationship types to the closest valid type using the mapping above"""

    human_template = """Extract relationships from this text:

{content}

Remember:
1. Return ONLY a valid JSON object
2. Include all relationships you find
3. Use proper JSON formatting
4. Do not include any text before or after the JSON
5. Map any non-standard relationship types to valid ones"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])
    
    prompt = prompt.partial(format_instructions=format_instructions)
    return prompt