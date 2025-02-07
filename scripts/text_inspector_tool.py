from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator


from langchain.text_splitter import RecursiveCharacterTextSplitter

from loguru import logger
import json

from .mdconvert import MarkdownConverter
from scripts.logging_config import log_error_with_traceback
from langchain_core.language_models.chat_models import BaseChatModel
from scripts.llm_compiler import LLMCompiler, Task, Plan, TaskResult, JoinDecision, CompilerState

# Import only the prompt functions and use our own model definitions
from prompts.text_inspection.text_analysis import get_text_analysis_prompt
from prompts.text_inspection.text_inspector_prompts import (
    get_plan_generation_prompt,
    get_join_decision_prompt
)

# Local model definitions that match the schema but are independent
class TextSegment(BaseModel):
    """Schema for text segments"""
    content: str = Field(description="Segment text")
    start_char: int = Field(description="Start character position")
    end_char: int = Field(description="End character position")
    metadata: Dict = Field(description="Segment metadata")

    @validator("content")
    def validate_content(cls, v: Any) -> str:
        if isinstance(v, str):
            return v
        if hasattr(v, "content"):
            return str(v.content)
        return str(v)

class Entity(BaseModel):
    """Schema for named entities"""
    text: str = Field(description="Entity text")
    title: str = Field(description="Entity title")

class TextAnalysis(BaseModel):
    """Schema for text analysis results"""
    content: str = Field(description="Original text content")
    segments: List[TextSegment] = Field(description="Text segments")
    key_points: List[str] = Field(description="Key points from text")
    entities: List[Entity] = Field(description="Extracted entities")
    relationships: List[str] = Field(description="Identified relationships")
    summary: str = Field(description="Text summary")

class InspectionState(BaseModel):
    """State for text inspection workflow"""
    text: str = Field(description="Input text")
    segments: List[TextSegment] = Field(default_factory=list)
    analysis: Optional[TextAnalysis] = None
    metadata: Dict = Field(default_factory=dict)

class TextInspector(LLMCompiler):
    """Text inspection system using LLMCompiler pattern"""
    def __init__(self, llm: Optional[BaseChatModel] = None):
        super().__init__(llm)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    async def _generate_plan(self, state: CompilerState) -> Plan:
        """Generate text inspection plan"""
        try:
            prompt, parser = get_plan_generation_prompt()
            chain = prompt | self.llm | parser
            plan = await chain.ainvoke({"content": state.content})
            return plan

        except Exception as e:
            log_error_with_traceback(e, "Error generating inspection plan")
            raise

    async def _execute_tasks(self, tasks: List[Task]) -> List[TaskResult]:
        """Execute text inspection tasks"""
        try:
            results = []
            for task in tasks:
                try:
                    # Check dependencies
                    deps_met = all(
                        any(r.task_id == dep and not r.error for r in results)
                        for dep in task.dependencies
                    )
                    if not deps_met:
                        continue

                    # Execute task
                    result = None
                    if task.tool == "analyze_text":
                        result = await self._analyze_text(task.args["content"])
                    elif task.tool == "identify_segments":
                        result = await self._identify_segments(task.args["content"])
                    elif task.tool == "extract_entities":
                        result = await self._extract_entities(task.args["content"])
                    elif task.tool == "identify_relationships":
                        result = await self._identify_relationships(task.args["content"])

                    results.append(TaskResult(
                        task_id=task.idx,
                        result=result,
                        error=None
                    ))

                except Exception as e:
                    results.append(TaskResult(
                        task_id=task.idx,
                        result=None,
                        error=str(e)
                    ))

            return results

        except Exception as e:
            log_error_with_traceback(e, "Error executing inspection tasks")
            raise

    async def _make_join_decision(self, state: CompilerState) -> JoinDecision:
        """Decide whether to complete or replan"""
        try:
            # Create join prompt
            plan_json = "{}"
            if state.plan:
                plan_json = json.dumps(state.plan.model_dump(), indent=2)

            results_json = "[]"
            if state.results:
                results_json = json.dumps([r.model_dump() if hasattr(r, "model_dump") else r.dict() for r in state.results], indent=2)

            prompt, parser = get_join_decision_prompt()
            chain = prompt | self.llm | parser
            decision = await chain.ainvoke({
                "plan": plan_json,
                "results": results_json
            })
            return decision

        except Exception as e:
            log_error_with_traceback(e, "Error making join decision")
            raise

    async def _generate_final_result(self, state: CompilerState) -> TextAnalysis:
        """Generate final text analysis result"""
        try:
            # Combine results into TextAnalysis
            analysis = TextAnalysis(
                content=state.content,
                segments=[],
                key_points=[],
                entities=[],
                relationships=[],
                summary=""
            )

            # Extract results from tasks
            for result in state.results:
                if result and result.result:
                    if isinstance(result.result, dict):
                        if 'segments' in result.result:
                            analysis.segments.extend(result.result['segments'])
                        if 'key_points' in result.result:
                            analysis.key_points.extend(result.result['key_points'])
                        if 'entities' in result.result:
                            analysis.entities.extend(result.result['entities'])
                        if 'relationships' in result.result:
                            analysis.relationships.extend(result.result['relationships'])
                        if 'summary' in result.result:
                            analysis.summary = result.result['summary']

            return analysis

        except Exception as e:
            log_error_with_traceback(e, "Error generating final result")
            raise

    async def inspect_text(self, text: str) -> TextAnalysis:
        """Inspect text content and extract structured information"""
        try:
            # Create initial state
            state = {
                "content": text,
                "plan": None,
                "results": [],
                "join_decision": None,
                "final_result": None
            }

            # Run LLMCompiler workflow
            result = await self.run(state)
            return result if isinstance(result, TextAnalysis) else TextAnalysis(
                content=text,
                segments=[TextSegment(
                    content=text,
                    start_char=0,
                    end_char=len(text),
                    metadata={}
                )],
                key_points=[],
                entities=[],
                relationships=[],
                summary=""
            )

        except Exception as e:
            log_error_with_traceback(e, "Error inspecting text")
            raise

    async def inspect_file(self, file_path: str) -> TextAnalysis:
        """Inspect a file and extract structured information"""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Extract metadata from file path
            file_metadata = {
                "file_path": file_path,
                "file_type": file_path.split('.')[-1] if '.' in file_path else 'unknown'
            }
            
            return await self.inspect_text(text)
            
        except Exception as e:
            log_error_with_traceback(e, f"Error inspecting file: {file_path}")
            return TextAnalysis(
                content="",
                segments=[TextSegment(
                    content="",
                    start_char=0,
                    end_char=0,
                    metadata={}
                )],
                key_points=[],
                entities=[],
                relationships=[],
                summary=f"Failed to analyze file: {file_path}"
            )

    async def _analyze_text(self, content: str) -> Dict[str, Any]:
        """Analyze text content"""
        try:
            prompt, parser = get_text_analysis_prompt()
            chain = prompt | self.llm | parser
            result = await chain.ainvoke({"text": content})
            return result.model_dump()
            
        except Exception as e:
            log_error_with_traceback(e, "Error analyzing text")
            raise

    async def _identify_segments(self, content: str) -> Dict[str, Any]:
        """Identify logical segments in text"""
        try:
            # Split text into segments
            segments = []
            current_pos = 0
            
            # Use text splitter to get chunks
            chunks = self.text_splitter.split_text(content)
            
            # Convert chunks to segments
            for chunk in chunks:
                start_pos = content.find(chunk, current_pos)
                if start_pos != -1:
                    segments.append(TextSegment(
                        content=chunk,
                        start_char=start_pos,
                        end_char=start_pos + len(chunk),
                        metadata={
                            "type": "chunk",
                            "length": len(chunk)
                        }
                    ))
                    current_pos = start_pos + len(chunk)
            
            return {"segments": [s.model_dump() for s in segments]}
            
        except Exception as e:
            log_error_with_traceback(e, "Error identifying segments")
            raise

    async def _extract_entities(self, content: str) -> Dict[str, Any]:
        """Extract named entities from text"""
        try:
            prompt, parser = get_text_analysis_prompt()
            chain = prompt | self.llm | parser
            result = await chain.ainvoke({"text": content})
            return {"entities": [e.model_dump() for e in result.entities]}
            
        except Exception as e:
            log_error_with_traceback(e, "Error extracting entities")
            raise

    async def _identify_relationships(self, content: str) -> Dict[str, Any]:
        """Identify relationships between concepts"""
        try:
            prompt, parser = get_text_analysis_prompt()
            chain = prompt | self.llm | parser
            result = await chain.ainvoke({"text": content})
            return {"relationships": result.relationships}
            
        except Exception as e:
            log_error_with_traceback(e, "Error identifying relationships")
            raise

async def inspect_file_as_text(file_path: str, llm) -> Dict:
    """Convenience function for file inspection"""
    inspector = TextInspector(llm)
    result = await inspector.inspect_file(file_path)
    return result.model_dump()
