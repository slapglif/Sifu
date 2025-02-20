import logging
from typing import List, Dict, Any
from prompts.example_generation import get_example_generation_prompt

logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self, config, llm, example_parser):
        self.config = config
        self.llm = llm
        self.example_parser = example_parser
        self.state = None  # Will be set during initialization
        
    async def _generate_examples(self, knowledge: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate examples from knowledge entries."""
        examples = []
        
        for entry in knowledge:
            if "id" not in entry:
                logger.warning(f"Knowledge entry missing ID: {entry}")
                continue
            
            if entry["id"] not in self.state.synthetic_knowledge:
                logger.warning(f"Knowledge entry {entry['id']} not found in synthetic knowledge")
                continue
            
            full_entry = self.state.synthetic_knowledge[entry["id"]]
            
            # Extract content and metadata from full entry
            content = full_entry.get("content", "")
            metadata = full_entry.get("metadata", {})
            
            # Generate examples using content
            prompt = get_example_generation_prompt()
            chain = prompt | self.llm | self.example_parser
            result = await chain.ainvoke({"content": content})
            
            if not result or not hasattr(result, "examples"):
                logger.warning(f"No examples generated for knowledge {entry['id']}")
                continue
            
            # Add metadata to examples
            for example in result.examples:
                if hasattr(example, "metadata") and isinstance(example.metadata, dict):
                    example.metadata.update(metadata)
                examples.append(example)
            
        return examples 