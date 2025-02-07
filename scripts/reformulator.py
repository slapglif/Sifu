"""Reformulator for converting conversation outputs into clear, concise answers."""

from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser

from prompts.reformulation.reformulation_prompts import (
    ReformulatedAnswer,
    get_reformulation_prompt
)

from scripts.logging_config import log_error_with_traceback
from scripts.text_web_browser import SimpleTextBrowser, web_search

async def prepare_response(
    original_task: str,
    inner_messages: List[BaseMessage],
    reformulation_model: BaseLanguageModel,
    include_web_search: bool = False
) -> str:
    """Prepare a reformulated response from conversation messages."""
    try:
        # Get prompt and parser
        prompt, parser = get_reformulation_prompt()
        
        # Convert messages to string format
        conversation = ""
        for msg in inner_messages:
            if not msg.content:
                continue
            role = msg.type if hasattr(msg, 'type') else 'unknown'
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            conversation += f"{role}: {content}\n\n"
        
        # Optionally perform web search for additional context
        web_context = ""
        if include_web_search:
            try:
                search_results = await web_search(original_task)
                if search_results and not search_results.startswith("Error"):
                    web_context = f"\nRelevant web search results:\n{search_results}\n"
            except Exception as e:
                log_error_with_traceback(e, "Error performing web search")
        
        # Run reformulation chain
        chain = (
            {"original_task": RunnablePassthrough(), 
             "conversation": RunnableLambda(lambda x: x["conversation"] + x.get("web_context", ""))} 
            | prompt 
            | reformulation_model 
            | parser
        )
        
        result = await chain.ainvoke({
            "original_task": original_task,
            "conversation": conversation,
            "web_context": web_context
        })
        
        # Return just the answer part
        return result.answer
        
    except Exception as e:
        log_error_with_traceback(e, "Error reformulating response")
        return f"Error reformulating response: {str(e)}"
