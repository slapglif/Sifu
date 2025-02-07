from typing import Optional, Dict, Any, List
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langgraph.graph import StateGraph
from loguru import logger

from smolagents import Tool
from smolagents.models import MessageRole, Model

from .mdconvert import MarkdownConverter


class TextSegment(BaseModel):
    """Schema for text segments"""
    content: str = Field(description="The text content")
    start_char: int = Field(description="Starting character position")
    end_char: int = Field(description="Ending character position")
    metadata: Dict = Field(description="Segment metadata")

class TextAnalysis(BaseModel):
    """Schema for text analysis results"""
    segments: List[TextSegment] = Field(description="Analyzed text segments")
    key_points: List[str] = Field(description="Key points from the text")
    entities: List[Dict] = Field(description="Extracted entities")
    relationships: List[Dict] = Field(description="Identified relationships")
    summary: str = Field(description="Text summary")

class InspectionState(BaseModel):
    """State for text inspection workflow"""
    text: str = Field(description="Input text")
    segments: List[TextSegment] = Field(default_factory=list)
    analysis: Optional[TextAnalysis] = None
    metadata: Dict = Field(default_factory=dict)

class TextInspector:
    def __init__(self, llm):
        self.llm = llm
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.workflow = self._create_workflow()
        
    def _create_workflow(self) -> StateGraph:
        """Create the text inspection workflow"""
        workflow = StateGraph(InspectionState)
        
        # Text segmentation node
        def segment_text(state: InspectionState) -> InspectionState:
            splits = self.text_splitter.split_text(state.text)
            state.segments = [
                TextSegment(
                    content=split,
                    start_char=state.text.find(split),
                    end_char=state.text.find(split) + len(split),
                    metadata={}
                )
                for split in splits
            ]
            return state
            
        workflow.add_node("segment_text", RunnableLambda(segment_text))
        
        # Key points extraction node
        def extract_key_points(state: InspectionState) -> InspectionState:
            template = ChatPromptTemplate.from_messages([
                ("system", "Extract key points from the text segment."),
                ("human", "Text: {text}")
            ])
            
            key_points = []
            for segment in state.segments:
                response = template | self.llm
                result = response.invoke({"text": segment.content})
                points = [p.strip() for p in result.content.split("\n") if p.strip()]
                key_points.extend(points)
                
            if not state.analysis:
                state.analysis = TextAnalysis(
                    segments=state.segments,
                    key_points=key_points,
                    entities=[],
                    relationships=[],
                    summary=""
                )
            else:
                state.analysis.key_points = key_points
                
            return state
            
        workflow.add_node("extract_key_points", RunnableLambda(extract_key_points))
        
        # Entity extraction node
        def extract_entities(state: InspectionState) -> InspectionState:
            template = ChatPromptTemplate.from_messages([
                ("system", "Extract entities and their attributes from the text."),
                ("human", "Text: {text}")
            ])
            
            entities = []
            for segment in state.segments:
                response = template | self.llm
                result = response.invoke({"text": segment.content})
                try:
                    segment_entities = eval(result.content)
                    entities.extend(segment_entities)
                except:
                    logger.warning(f"Failed to parse entities from: {result.content}")
                    
            if state.analysis:
                state.analysis.entities = entities
                
            return state
            
        workflow.add_node("extract_entities", RunnableLambda(extract_entities))
        
        # Relationship extraction node
        def extract_relationships(state: InspectionState) -> InspectionState:
            template = ChatPromptTemplate.from_messages([
                ("system", "Extract relationships between entities in the text."),
                ("human", "Text: {text}\nEntities: {entities}")
            ])
            
            relationships = []
            for segment in state.segments:
                response = template | self.llm
                result = response.invoke({
                    "text": segment.content,
                    "entities": str(state.analysis.entities if state.analysis else [])
                })
                try:
                    segment_relationships = eval(result.content)
                    relationships.extend(segment_relationships)
                except:
                    logger.warning(f"Failed to parse relationships from: {result.content}")
                    
            if state.analysis:
                state.analysis.relationships = relationships
                
            return state
            
        workflow.add_node("extract_relationships", RunnableLambda(extract_relationships))
        
        # Summarization node
        def summarize_text(state: InspectionState) -> InspectionState:
            template = ChatPromptTemplate.from_messages([
                ("system", "Generate a comprehensive summary of the text."),
                ("human", """
                Text: {text}
                Key Points: {key_points}
                Entities: {entities}
                Relationships: {relationships}
                """)
            ])
            
            response = template | self.llm
            result = response.invoke({
                "text": state.text,
                "key_points": "\n".join(state.analysis.key_points) if state.analysis else "",
                "entities": str(state.analysis.entities if state.analysis else []),
                "relationships": str(state.analysis.relationships if state.analysis else [])
            })
            
            if state.analysis:
                state.analysis.summary = result.content
                
            return state
            
        workflow.add_node("summarize_text", RunnableLambda(summarize_text))
        
        # Define edges
        workflow.add_edge("segment_text", "extract_key_points")
        workflow.add_edge("extract_key_points", "extract_entities")
        workflow.add_edge("extract_entities", "extract_relationships")
        workflow.add_edge("extract_relationships", "summarize_text")
        
        workflow.set_entry_point("segment_text")
        return workflow
        
    def inspect_text(self, text: str, metadata: Dict = None) -> TextAnalysis:
        """Inspect text content"""
        try:
            initial_state = InspectionState(
                text=text,
                metadata=metadata or {}
            )
            
            final_state = self.workflow.invoke(initial_state.dict())
            return final_state.analysis
            
        except Exception as e:
            logger.error(f"Error inspecting text: {e}")
            raise
            
    def inspect_file(self, file_path: str, metadata: Dict = None) -> TextAnalysis:
        """Inspect text file"""
        try:
            loader = TextLoader(file_path)
            docs = loader.load()
            text = "\n".join(doc.page_content for doc in docs)
            
            file_metadata = {
                "source": file_path,
                "type": "file",
                **(metadata or {})
            }
            
            return self.inspect_text(text, file_metadata)
            
        except Exception as e:
            logger.error(f"Error inspecting file: {e}")
            raise

def inspect_file_as_text(file_path: str, llm) -> Dict:
    """Convenience function for file inspection"""
    inspector = TextInspector(llm)
    result = inspector.inspect_file(file_path)
    return result.dict()
