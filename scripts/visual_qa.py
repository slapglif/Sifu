import base64
import json
import mimetypes
import os
from io import BytesIO
from typing import Optional, List, Dict, Any
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.graph import StateGraph
from loguru import logger

from prompts.visual_qa import (
    VISUAL_ELEMENT_DETECTION_SYSTEM,
    VISUAL_ELEMENT_DETECTION_HUMAN,
    SCENE_ANALYSIS_SYSTEM,
    SCENE_ANALYSIS_HUMAN,
    VISUAL_QA_SYSTEM,
    VISUAL_QA_HUMAN
)

load_dotenv(override=True)

class ImageAnalysisResult(BaseModel):
    """Schema for image analysis results"""
    answer: str = Field(description="Answer to the question")
    visual_evidence: List[str] = Field(description="Visual evidence supporting the answer")
    context: Optional[str] = Field(None, description="Additional context")
    confidence: float = Field(description="Confidence in the analysis")

class ImageRegion(BaseModel):
    """Schema for image regions"""
    x: int = Field(description="X coordinate")
    y: int = Field(description="Y coordinate")
    width: int = Field(description="Region width")
    height: int = Field(description="Region height")
    content: str = Field(description="Region content description")
    confidence: float = Field(description="Detection confidence")

class VisualElement(BaseModel):
    """Schema for visual elements"""
    element_type: str = Field(description="Type of visual element")
    description: str = Field(description="Element description")
    attributes: Dict = Field(description="Element attributes")
    region: Optional[ImageRegion] = Field(None, description="Element region")
    confidence: float = Field(description="Detection confidence")

class ImageAnalysis(BaseModel):
    """Schema for image analysis results"""
    elements: List[VisualElement] = Field(description="Detected visual elements")
    scene_description: str = Field(description="Overall scene description")
    key_objects: List[str] = Field(description="Key objects in the image")
    spatial_relationships: List[str] = Field(description="Spatial relationships")
    visual_attributes: Dict = Field(description="Visual attributes")
    confidence: float = Field(description="Overall confidence")

class VisualAnswer(BaseModel):
    """Schema for visual question answering"""
    answer: str = Field(description="Answer to the question")
    reasoning: str = Field(description="Reasoning process")
    relevant_elements: List[str] = Field(description="Relevant visual elements")
    confidence: float = Field(description="Answer confidence")

class VisualState(BaseModel):
    """State for visual QA workflow"""
    image_path: str = Field(description="Path to image")
    image_data: Optional[str] = Field(None, description="Base64 encoded image")
    question: Optional[str] = Field(None, description="Question about the image")
    analysis: Optional[ImageAnalysis] = None
    answer: Optional[VisualAnswer] = None

def encode_image(image_path: str) -> str:
    """Convert image to Base64 encoded string."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Save to BytesIO in JPEG format
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        raise

def resize_image(image_path: str) -> str:
    """Resize an image to half its dimensions."""
    try:
        # Convert to Path object for proper path handling
        path = Path(image_path)
        img = Image.open(path)
    width, height = img.size
    img = img.resize((int(width / 2), int(height / 2)))
        
        # Create resized directory if it doesn't exist
        resized_dir = path.parent / "resized"
        resized_dir.mkdir(exist_ok=True)
        
        # Create new path for resized image
        new_image_path = resized_dir / f"resized_{path.name}"
        img.save(new_image_path)
        return str(new_image_path)
        except Exception as e:
        logger.error(f"Error resizing image {image_path}: {e}")
        raise

@tool
async def visualizer(image_path: str, question: str) -> str:
    """Analyze an image and answer questions about it using the minicpm-v model."""

    logger.info(f"Analyzing image: {image_path}")
    
    # Convert image to base64
    image_b64 = encode_image(image_path)
    
    # Create Ollama client for minicpm-v
    llm = ChatOllama(
        model="minicpm-v:latest",
        temperature=0.7,
        mirostat=2,
        mirostat_eta=0.1,
        mirostat_tau=5.0,
        format="json"
    )
    
    # Create parser with proper format instructions
    parser = PydanticOutputParser(pydantic_object=ImageAnalysisResult)
    format_instructions = parser.get_format_instructions()

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""You are a visual analysis expert. Analyze the image and answer the question.
Your response must be a valid JSON object that EXACTLY matches this schema:
{{
    "answer": "Your detailed answer to the question",
    "visual_evidence": ["List of visual elements that support your answer"],
    "context": "Additional context or observations",
    "confidence": 0.85
}}

IMPORTANT:
1. All fields are required
2. confidence must be a number between 0.0 and 1.0
3. visual_evidence must be a non-empty array of strings
4. Do not include any text before or after the JSON
5. Use proper JSON formatting with double quotes

{format_instructions}"""),
        HumanMessage(content=[
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
            {"type": "text", "text": f"""
                Question: {question}
                Format Instructions: {format_instructions}
            """}
        ])
    ])
    
    # Create chain with parser
    chain = prompt | llm | parser
    
    try:
        # Get response from model
        result = await chain.ainvoke({})
        
        # Convert to JSON string
        return json.dumps(result.model_dump(), indent=2)
        
    except Exception as e:
        logger.error(f"Error in visual analysis: {e}")
        # Return error result
        error_result = ImageAnalysisResult(
            answer=f"Error analyzing image: {str(e)}",
            visual_evidence=["Error occurred during analysis"],
            context="Analysis failed",
            confidence=0.0
        )
        return json.dumps(error_result.model_dump(), indent=2)

class VisualAnalyzer:
    """Visual analysis system using LLMCompiler pattern"""
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm if llm else ChatOllama(
            model="minicpm-v:latest",
            temperature=0.7,
            mirostat=2,
            mirostat_eta=0.1,
            mirostat_tau=5.0,
            format="json"
        )
        
    async def analyze_image(self, image_path: str, question: str) -> ImageAnalysisResult:
        """Analyze an image and answer questions about it"""
        try:
            # Convert image to base64
            image_b64 = encode_image(image_path)
            
            # Create parser with proper format instructions
            parser = PydanticOutputParser(pydantic_object=ImageAnalysisResult)
            format_instructions = parser.get_format_instructions()
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=f"""You are a visual analysis expert. Analyze the image and answer the question.
Your response must be a valid JSON object that EXACTLY matches this schema:
{{
    "answer": "Your detailed answer to the question",
    "visual_evidence": ["List of visual elements that support your answer"],
    "context": "Additional context or observations",
    "confidence": 0.85
}}

IMPORTANT:
1. All fields are required
2. confidence must be a number between 0.0 and 1.0
3. visual_evidence must be a non-empty array of strings
4. Do not include any text before or after the JSON
5. Use proper JSON formatting with double quotes

{format_instructions}"""),
                HumanMessage(content=[
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                    {"type": "text", "text": f"""
                        Question: {question}
                        Format Instructions: {format_instructions}
                    """}
                ])
            ])
            
            # Create chain with parser
            chain = prompt | self.llm | parser
            
            # Get response from model
            result = await chain.ainvoke({})
            return result
            
        except Exception as e:
            logger.error(f"Error in visual analysis: {e}")
            return ImageAnalysisResult(
                answer=f"Error analyzing image: {str(e)}",
                visual_evidence=["Error occurred during analysis"],
                context="Analysis failed",
                confidence=0.0
            )
            
    async def detect_elements(self, image_path: str) -> List[VisualElement]:
        """Detect visual elements in an image"""
        try:
            # Convert image to base64
            image_b64 = encode_image(image_path)
            
            # Create parser with proper format instructions
            parser = PydanticOutputParser(pydantic_object=VisualElement)
            format_instructions = parser.get_format_instructions()
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=VISUAL_ELEMENT_DETECTION_SYSTEM + "\n\n" + format_instructions),
                HumanMessage(content=[
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                    {"type": "text", "text": format_instructions}
                ])
            ])
            
            # Create chain with parser
            chain = prompt | self.llm | parser
            
            # Get response from model
            result = await chain.ainvoke({})
            return [result] if isinstance(result, VisualElement) else []
            
        except Exception as e:
            logger.error(f"Error detecting elements: {e}")
            return []
            
    async def analyze_scene(self, image_path: str, elements: List[VisualElement]) -> ImageAnalysis:
        """Analyze the overall scene"""
        try:
            # Convert image to base64
            image_b64 = encode_image(image_path)
            
            # Create parser with proper format instructions
            parser = PydanticOutputParser(pydantic_object=ImageAnalysis)
            format_instructions = parser.get_format_instructions()
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=SCENE_ANALYSIS_SYSTEM + "\n\n" + format_instructions),
                HumanMessage(content=[
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                    {"type": "text", "text": f"""
                        Detected Elements: {json.dumps([e.model_dump() for e in elements], indent=2)}
                        Format Instructions: {format_instructions}
                    """}
                ])
            ])
            
            # Create chain with parser
            chain = prompt | self.llm | parser
            
            # Get response from model
            result = await chain.ainvoke({})
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing scene: {e}")
            return ImageAnalysis(
                elements=[],
                scene_description="Error analyzing scene",
                key_objects=[],
                spatial_relationships=[],
                visual_attributes={},
                confidence=0.0
            )
            
    async def answer_question(self, image_path: str, question: str, scene_analysis: ImageAnalysis) -> VisualAnswer:
        """Answer a question about the image"""
        try:
            # Convert image to base64
            image_b64 = encode_image(image_path)
            
            # Create parser with proper format instructions
            parser = PydanticOutputParser(pydantic_object=VisualAnswer)
            format_instructions = parser.get_format_instructions()
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=VISUAL_QA_SYSTEM + "\n\n" + format_instructions),
                HumanMessage(content=[
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                    {"type": "text", "text": f"""
                        Question: {question}
                        Scene Description: {scene_analysis.scene_description}
                        Key Objects: {scene_analysis.key_objects}
                        Spatial Relationships: {scene_analysis.spatial_relationships}
                        Format Instructions: {format_instructions}
                    """}
                ])
            ])
            
            # Create chain with parser
            chain = prompt | self.llm | parser
            
            # Get response from model
            result = await chain.ainvoke({})
            return result
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return VisualAnswer(
                answer=f"Error answering question: {str(e)}",
                reasoning="Error occurred",
                relevant_elements=[],
                confidence=0.0
            )

