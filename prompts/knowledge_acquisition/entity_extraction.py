"""Entity extraction prompts."""
from typing import List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

class EntityResponse(BaseModel):
    """Schema for entity extraction response"""
    entities: List[str] = Field(
        description="List of extracted entities",
        default_factory=list
    )

def get_entity_extraction_prompt() -> ChatPromptTemplate:
    """Get the entity extraction prompt template."""
    parser = PydanticOutputParser(pydantic_object=EntityResponse)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are an expert at extracting medical and scientific entities from research text.
{format_instructions}

CRITICAL RULES:
1. You MUST output ONLY a valid JSON object
2. The JSON MUST match the schema exactly
3. The entities field MUST be an array of strings
4. Each entity MUST be properly escaped if it contains special characters
5. Do not include any text before or after the JSON object
6. Do not include any explanations or notes
7. The response should look exactly like this:
{{
    "entities": [
        "PANDAS",
        "Streptococcal Infection",
        "Gut-Brain Axis",
        "Vagus Nerve",
        "Plant-Based Therapy"
    ]
}}

GUIDELINES for medical entity extraction:
1. Extract meaningful medical and scientific terms:
   - Medical conditions and diseases
   - Anatomical structures and systems
   - Biological mechanisms and pathways
   - Therapeutic approaches and treatments
   - Biomarkers and clinical indicators
   - Research methodologies and protocols
   - Patient outcomes and symptoms
   - Drug classes and compounds
   - Plant-based and alternative therapies

2. Entity Categories to Extract:
   - Diseases and Conditions:
     * Primary conditions (e.g., "PANDAS", "Autoimmune Encephalitis")
     * Related disorders
     * Comorbidities
   
   - Anatomical/Biological:
     * Body systems (e.g., "Gut-Brain Axis", "Immune System")
     * Organs and tissues
     * Neural pathways
     * Cellular components
   
   - Therapeutic:
     * Treatment modalities
     * Medications and compounds
     * Natural remedies
     * Therapeutic approaches
   
   - Clinical:
     * Symptoms and signs
     * Diagnostic tests
     * Biomarkers
     * Clinical outcomes

3. Entity Validation Rules:
   - Must be recognized medical/scientific terms
   - Should be specific rather than general
   - Must be relevant to the medical domain
   - Should be supported by context
   - Must be properly normalized (e.g., "IL-6" for "Interleukin 6")

4. Formatting Guidelines:
   - Use standard medical terminology
   - Maintain proper capitalization for proper nouns
   - Keep acronyms in uppercase (e.g., "TNF-α", "IL-6")
   - Use full names for clarity
   - Include both common and scientific names where relevant

5. Quality Requirements:
   - Extract at least 10-15 entities per text
   - Ensure balanced coverage across categories
   - Focus on domain-relevant entities
   - Include both specific and general terms
   - Capture key relationships and hierarchies"""

    human_template = """Extract medical and scientific entities from this text, focusing on PANDAS, gut-brain axis, and therapeutic relationships:

{content}

Remember:
1. Return ONLY a valid JSON object
2. Extract medical/scientific entities only
3. Use proper medical terminology
4. Ensure entities are domain-relevant
5. Include all entity categories (diseases, anatomical, therapeutic, clinical)"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])
    
    prompt = prompt.partial(format_instructions=format_instructions)
    return prompt 