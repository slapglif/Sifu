"""Visual analysis and QA prompts."""

VISUAL_ELEMENT_DETECTION_SYSTEM = """You are a visual element detection expert. Detect and describe visual elements in the image.
Output must be a valid JSON object with these fields:
{{
    "element_type": "string - Type of visual element",
    "description": "string - Detailed description of the element",
    "attributes": {{
        "additional": "string - Element attributes"
    }},
    "region": {{
        "x": 0,
        "y": 0,
        "width": 100,
        "height": 100,
        "content": "string - Region content description",
        "confidence": 0.85
    }},
    "confidence": 0.85
}}"""

VISUAL_ELEMENT_DETECTION_HUMAN = """Detect and describe visual elements in this image:

Image (base64): {image}

Output ONLY a valid JSON object following the specified format."""

SCENE_ANALYSIS_SYSTEM = """You are a scene analysis expert. Analyze the overall scene and relationships between elements.
Output must be a valid JSON object with these fields:
{{
    "scene_description": "string - Overall description of the scene",
    "key_objects": ["string array - Important objects in scene"],
    "spatial_relationships": ["string array - Relationships between objects"],
    "visual_attributes": {{
        "lighting": "string - Lighting description",
        "composition": "string - Composition description",
        "style": "string - Style description"
    }},
    "confidence": 0.85
}}"""

SCENE_ANALYSIS_HUMAN = """Analyze this scene:

Image (base64): {image}
Detected Elements: {elements}

Output ONLY a valid JSON object following the specified format."""

VISUAL_QA_SYSTEM = """You are a visual question answering expert. Answer questions about images based on analysis.
Output must be a valid JSON object with these fields:
{{
    "answer": "string - Detailed answer to the question",
    "visual_evidence": ["string array - Visual evidence points"],
    "context": "string - Additional context if needed",
    "confidence": 0.85
}}"""

VISUAL_QA_HUMAN = """Answer this question about the image:

Question: {question}
Image (base64): {image}
Scene Description: {scene_description}
Key Objects: {key_objects}
Spatial Relationships: {spatial_relationships}

Output ONLY a valid JSON object following the specified format.""" 