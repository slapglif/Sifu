# https://github.com/microsoft/autogen/blob/gaia_multiagent_v01_march_1st/autogen/browser_utils.py
import mimetypes
import os
import pathlib
import re
import time
import uuid
import json
from typing import Any, Dict, List, Optional, Tuple, Union, TypedDict, Annotated, Coroutine
from urllib.parse import unquote, urljoin, urlparse

import pathvalidate
import requests
from bs4 import BeautifulSoup
from loguru import logger
from langchain_core.tools import BaseTool, StructuredTool, Tool
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_community.utilities import SearxSearchWrapper
from pydantic import BaseModel, Field
import PyPDF2
import aiohttp
import asyncio

from .cookies import COOKIES
from .mdconvert import FileConversionException, MarkdownConverter, UnsupportedFormatException
from scripts.logging_config import log_error_with_traceback, log_info_with_context

# Initialize SearxNG search
searx = SearxSearchWrapper(searx_host="https://searchapi.anuna.dev")

class BrowserState(TypedDict):
    current_page: str
    current_url: Optional[str]
    current_position: int
    find_position: int
    viewport_size: int
    last_search: Optional[str]

class SimpleTextBrowser:
    """Simple browser for text extraction."""
    
    def __init__(self):
        """Initialize browser."""
        self.session = None
        
    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
    async def get_content(self, url: str) -> Optional[str]:
        """Get text content from URL."""
        try:
            await self._ensure_session()
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                        
                    # Get text content
                    text = soup.get_text()
                    
                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    log_info_with_context(f"Successfully extracted content from {url}", "Web Browser")
                    return text
                else:
                    log_error_with_traceback(
                        Exception(f"HTTP {response.status}"),
                        f"Failed to fetch content from {url}"
                    )
                    return None
                    
        except Exception as e:
            log_error_with_traceback(e, f"Error fetching content from {url}")
            return None
            
    async def close(self):
        """Close browser session."""
        if self.session:
            await self.session.close()
            self.session = None

class WebSearchInput(BaseModel):
    query: str = Field(description="The web search query to perform")
    filter_year: Optional[str] = Field(None, description="Filter results to a specific year")

class VisitInput(BaseModel):
    url: str = Field(description="The URL to visit")

class FindInput(BaseModel):
    search_string: str = Field(description="The text to search for on the page")

async def web_search(query: str) -> str:
    """Perform web search using SearxNG."""
    try:
        if not query or not query.strip():
            return "Error: Empty search query"
            
        logger.info(f"Searching for: {query}")
        
        # Configure SearxNG search
        searx_host = os.getenv("SEARX_HOST", "https://searchapi.anuna.dev")
        headers = {
            "User-Agent": os.getenv("USER_AGENT", "Mozilla/5.0"),
            "Accept": "application/json"
        }
        
        # Prepare search parameters
        params = {
            "q": query,
            "format": "json",
            "engines": "google,duckduckgo,bing",
            "language": "en",
            "time_range": "",
            "category": "general"
        }
        
        # Make request to SearxNG
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{searx_host}/search",
                    params=params,
                    headers=headers,
                    ssl=False,  # Disable SSL verification
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    response.raise_for_status()
                    results = await response.json()
                    
                    # Extract URLs and snippets
                    formatted_results = []
                    if "results" in results and results["results"]:
                        for result in results["results"][:5]:  # Limit to top 5 results
                            url = result.get("url", "").strip()
                            title = result.get("title", "").strip()
                            snippet = result.get("content", "").strip()
                            
                            if url:  # Only include results with valid URLs
                                formatted_results.extend([
                                    f"URL: {url}",
                                    f"Title: {title}",
                                    f"Summary: {snippet}",
                                    "---"
                                ])
                        
                        if formatted_results:
                            return "\n".join(formatted_results)
                        else:
                            return "Error: No valid results found"
     
                    
        except aiohttp.ClientError as e:
            logger.error(f"Error making search request: {e}")
            return f"Error: Search request failed - {str(e)}"
            
    except Exception as e:
        logger.error(f"Error in web search: {e}")
        return f"Error: Web search failed - {str(e)}"
    
    return "Error: No results found in search response"

web_search_tool = StructuredTool.from_function(
    func=web_search,
    name="web_search",
    description="Perform a web search query (think a google search) and returns the search results.",
    args_schema=WebSearchInput,
    coroutine=web_search
)

async def visit_page(url: str) -> str:
    """Visit a webpage and return its text content."""
    browser = SimpleTextBrowser()
    content = await browser.get_content(url)
    if content:
        return content
    else:
        return "Error: Unable to fetch content from the webpage"

visit_tool = StructuredTool.from_function(
    func=visit_page,
    name="visit_page",
    description="Visit a webpage at a given URL and return its text. Given a url to a YouTube video, this returns the transcript.",
    args_schema=VisitInput,
    coroutine=visit_page
)

async def page_up() -> str:
    """Move the viewport up one page."""
    browser = SimpleTextBrowser()
    content = await browser.get_content(browser._state['current_url'])
    if content:
        return content
    else:
        return "Error: Unable to fetch content from the current webpage"

page_up_tool = StructuredTool.from_function(
    func=page_up,
    name="page_up",
    description="Scroll the viewport UP one page-length in the current webpage and return the new viewport content.",
    coroutine=page_up
)

async def page_down() -> str:
    """Move the viewport down one page."""
    browser = SimpleTextBrowser()
    content = await browser.get_content(browser._state['current_url'])
    if content:
        return content
    else:
        return "Error: Unable to fetch content from the current webpage"

page_down_tool = StructuredTool.from_function(
    func=page_down,
    name="page_down",
    description="Scroll the viewport DOWN one page-length in the current webpage and return the new viewport content.",
    coroutine=page_down
)

async def find_on_page(search_string: str) -> str:
    """Search for text on the current page."""
    browser = SimpleTextBrowser()
    content = await browser.get_content(browser._state['current_url'])
    if content:
        find_position = browser._state.get('find_position', 0)
        position = content.find(search_string, find_position)
        if position == -1:
            return "Text not found"
            
        browser._state['find_position'] = position
        browser._state['current_position'] = max(0, position - 100)
        browser._state['last_search'] = search_string
        return browser._get_current_viewport()
    else:
        return "Error: Unable to fetch content from the current webpage"

find_tool = StructuredTool.from_function(
    func=find_on_page,
    name="find_on_page",
    description="Scroll the viewport to the first occurrence of the search string. This is equivalent to Ctrl+F.",
    args_schema=FindInput,
    coroutine=find_on_page
)

async def find_next() -> str:
    """Find the next occurrence of the last search term."""
    browser = SimpleTextBrowser()
    content = await browser.get_content(browser._state['current_url'])
    if content:
        last_search = browser._state.get('last_search')
        if not last_search:
            return "No page loaded or no previous search"
            
        find_position = browser._state.get('find_position', 0)
        position = content.find(last_search, find_position + 1)
        if position == -1:
            return "No more occurrences found"
            
        browser._state['find_position'] = position
        browser._state['current_position'] = max(0, position - 100)
        return browser._get_current_viewport()
    else:
        return "Error: Unable to fetch content from the current webpage"

find_next_tool = StructuredTool.from_function(
    func=find_next,
    name="find_next",
    description="Scroll the viewport to next occurrence of the search string. This is equivalent to finding the next match in a Ctrl+F search.",
    coroutine=find_next
)

async def find_archived_url(url: str, date: str) -> str:
    """Search the Wayback Machine for an archived version of a URL."""
    try:
        no_timestamp_url = f"https://archive.org/wayback/available?url={url}"
        archive_url = no_timestamp_url + f"&timestamp={date}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(archive_url) as response:
                response.raise_for_status()
                response_data = await response.json()
                
            async with session.get(no_timestamp_url) as response:
                response.raise_for_status()
                response_notimestamp = await response.json()
        
        if "archived_snapshots" in response_data and "closest" in response_data["archived_snapshots"]:
            closest = response_data["archived_snapshots"]["closest"]
        elif "archived_snapshots" in response_notimestamp and "closest" in response_notimestamp["archived_snapshots"]:
            closest = response_notimestamp["archived_snapshots"]["closest"]
        else:
            return f"Error: URL {url} was not archived on Wayback Machine"
            
        target_url = closest["url"]
        browser = SimpleTextBrowser()
        content = await browser.get_content(target_url)
        return f"Web archive for url {url}, snapshot taken at date {closest['timestamp'][:8]}:\n{content}"
        
    except Exception as e:
        return f"Error accessing Wayback Machine: {str(e)}"

archive_tool = StructuredTool.from_function(
    func=find_archived_url,
    name="find_archived_url",
    description="Given a url, searches the Wayback Machine and returns the archived version of the url that's closest in time to the desired date.",
    coroutine=find_archived_url
)

# Export all tools
tools = [
    web_search_tool,
    visit_tool,
    page_up_tool,
    page_down_tool,
    find_tool,
    find_next_tool,
    archive_tool
]
