"""Visual QA system."""

from typing import List, Dict, Optional, Any
from pathlib import Path
from io import BytesIO
import base64
import json
import requests
from dotenv import load_dotenv

# Core imports
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# UI imports
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table

# Image processing
from PIL import Image

# Logging
from loguru import logger

# Local imports
from prompts.visual_qa.element_detection import (
    Region,
    VisualElement,
    get_element_detection_prompt
)
from prompts.visual_qa.scene_analysis import (
    VisualAttributes,
    SceneAnalysis,
    get_scene_analysis_prompt
)
from prompts.visual_qa.visual_qa_prompts import (
    VisualAnswer,
    get_visual_qa_prompt
)
from prompts.visual_qa.plan_generation import (
    Plan,
    get_plan_generation_prompt
)
from prompts.visual_qa.join_decision import (
    JoinDecision,
    get_join_decision_prompt
)

from scripts.logging_config import (
    log_error_with_traceback,
)

from scripts.llm_compiler import LLMCompiler, Task, TaskResult, CompilerState

load_dotenv(override=True)

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

class VisualAnalyzer(LLMCompiler):
    """Visual analysis system."""

    def __init__(self, llm):
        """Initialize with language model."""
        super().__init__(llm)

    async def _detect_elements(self, image_path: str) -> List[VisualElement]:
        """Detect visual elements in the image."""
        try:
            # Encode image
            encoded_image = encode_image(image_path)
            
            # Get detection prompt
            prompt = get_element_detection_prompt()
            
            # Run detection
            chain = prompt | self.llm | RunnableLambda(lambda x: [VisualElement(**e) for e in json.loads(x.content if hasattr(x, 'content') else x).get('elements', [])])
            elements = await chain.ainvoke({"image": encoded_image})
            return elements
            
        except Exception as e:
            log_error_with_traceback(e, "Error detecting elements")
            return []

    async def _analyze_scene(self, image_path: str, elements: List[VisualElement]) -> SceneAnalysis:
        """Analyze the scene composition."""
        try:
            # Encode image
            encoded_image = encode_image(image_path)
            
            # Get scene analysis prompt
            prompt = get_scene_analysis_prompt()
            
            # Run analysis
            chain = prompt | self.llm | RunnableLambda(lambda x: SceneAnalysis(**json.loads(x.content if hasattr(x, 'content') else x)))
            scene = await chain.ainvoke({
                "image": encoded_image,
                "elements": [e.model_dump() for e in elements]
            })
            return scene
            
        except Exception as e:
            log_error_with_traceback(e, "Error analyzing scene")
            return SceneAnalysis(
                scene_description="Failed to analyze scene",
                key_objects=[],
                spatial_relationships=[],
                visual_attributes=VisualAttributes(
                    lighting="unknown",
                    composition="unknown",
                    style="unknown"
                ),
                confidence=0.0
            )

    async def _answer_question(self, image_path: str, question: str, scene: SceneAnalysis) -> VisualAnswer:
        """Answer a question about the image."""
        try:
            # Encode image
            encoded_image = encode_image(image_path)
            
            # Get QA prompt
            prompt = get_visual_qa_prompt()
            
            # Run QA
            chain = prompt | self.llm | RunnableLambda(lambda x: VisualAnswer(**json.loads(x.content if hasattr(x, 'content') else x)))
            answer = await chain.ainvoke({
                "image": encoded_image,
                "question": question,
                "scene_description": scene.scene_description,
                "key_objects": scene.key_objects,
                "spatial_relationships": scene.spatial_relationships
            })
            return answer
            
        except Exception as e:
            log_error_with_traceback(e, "Error answering question")
            return VisualAnswer(
                answer="Failed to answer question",
                visual_evidence=[],
                context="Analysis failed",
                confidence=0.0
            )

    async def _generate_plan(self, state: CompilerState) -> Plan:
        """Generate visual analysis plan."""
        try:
            prompt = get_plan_generation_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=Plan)
            plan = await chain.ainvoke({
                "image_path": state.get('image_path', ''),
                "question": state.get('question', '')
            })
            return plan

        except Exception as e:
            log_error_with_traceback(e, "Error generating visual analysis plan")
            raise

    async def _execute_tasks(self, tasks: List[Task]) -> List[TaskResult]:
        """Execute visual analysis tasks."""
        try:
            results = []
            elements = []
            scene = None
            
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
                    if task.tool == "detect_elements":
                        elements = await self._detect_elements(task.args["image_path"])
                        result = {"elements": [e.model_dump() for e in elements]}
                        
                    elif task.tool == "analyze_scene":
                        scene = await self._analyze_scene(task.args["image_path"], elements)
                        result = scene.model_dump()
                        
                    elif task.tool == "answer_question":
                        if scene:
                            answer = await self._answer_question(
                                task.args["image_path"],
                                task.args["question"],
                                scene
                            )
                            result = answer.model_dump()

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
            log_error_with_traceback(e, "Error executing visual analysis tasks")
            raise

    async def _make_join_decision(self, state: CompilerState) -> JoinDecision:
        """Decide whether to complete or replan"""
        try:
            # Create join prompt
            plan_json = "{}"
            plan = state.get('plan')
            if plan is not None:
                plan_json = json.dumps(plan.dict() if hasattr(plan, 'dict') else plan, indent=2)

            results_json = "[]"
            results = state.get('results')
            if results:
                results_json = json.dumps([r.dict() if hasattr(r, 'dict') else r for r in results], indent=2)

            prompt = get_join_decision_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=JoinDecision)
            decision = await chain.ainvoke({
                "plan": plan_json,
                "results": results_json
            })
            return decision

        except Exception as e:
            log_error_with_traceback(e, "Error making join decision")
            raise

    async def _generate_final_result(self, state: CompilerState) -> VisualAnswer:
        """Generate final visual analysis result."""
        try:
            # Extract result from tasks
            for result in state.get('results', []):
                if result and result.result:
                    if isinstance(result.result, dict):
                        return VisualAnswer(
                            answer=result.result.get('answer', 'Failed to analyze image'),
                            visual_evidence=result.result.get('visual_evidence', []),
                            context=result.result.get('context', 'Analysis failed'),
                            confidence=result.result.get('confidence', 0.0)
                        )

            # Return default result if no valid results found
            return VisualAnswer(
                answer="Failed to analyze image",
                visual_evidence=[],
                context="Analysis failed",
                confidence=0.0
            )

        except Exception as e:
            log_error_with_traceback(e, "Error generating final result")
            raise

    async def analyze_image(self, image_path: str, question: str) -> VisualAnswer:
        """Analyze an image and answer questions about it."""
        try:
            # Create initial state
            state = {
                "image_path": image_path,
                "question": question,
                "plan": None,
                "results": [],
                "join_decision": None,
                "final_result": None
            }

            # Run LLM compiler workflow
            result = await self.run(state)
            return result if isinstance(result, VisualAnswer) else VisualAnswer(
                answer="Failed to analyze image",
                visual_evidence=[],
                context="Analysis failed",
                confidence=0.0
            )

        except Exception as e:
            log_error_with_traceback(e, "Error analyzing image")
            raise 