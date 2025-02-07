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
from .logging_config import (
    log_error_with_traceback,
    log_warning_with_context,
    log_info_with_context
)

# Initialize SearxNG search
searx = SearxSearchWrapper(searx_host="https://searchapi.anuna.dev")

class BrowserError(Exception):
    """Base class for browser errors"""
    pass

class NetworkError(BrowserError):
    """Network-related errors"""
    pass

class ContentExtractionError(BrowserError):
    """Content extraction errors"""
    pass

class BrowserState(TypedDict):
    """Browser state"""
    current_page: str
    current_url: Optional[str]
    current_position: int
    find_position: int
    viewport_size: int
    last_search: Optional[str]

class SimpleTextBrowser:
    """Simple text-based browser for web scraping."""
    
    def __init__(self, viewport_size: int = 5120, downloads_folder: str = "downloads", request_kwargs: Optional[Dict[str, Any]] = None):
        """Initialize browser."""
        try:
            self.viewport_size = viewport_size
            self.downloads_folder = downloads_folder
            self.request_kwargs = request_kwargs or {}
            self._state: Dict[str, Any] = {
                'current_page': '',
                'current_url': None,
                'current_position': 0,
                'find_position': 0,
                'viewport_size': viewport_size,
                'last_search': None
            }
            
            # Create downloads folder if it doesn't exist
            os.makedirs(downloads_folder, exist_ok=True)
            
            log_info_with_context("Browser initialized", "Browser", include_locals=True)
            
        except Exception as e:
            log_error_with_traceback(e, "Failed to initialize browser", include_locals=True)
            raise BrowserError("Failed to initialize browser") from e
        
    async def visit(self, url: str) -> str:
        """Visit a URL and return the text content."""
        try:
            log_info_with_context(f"Visiting URL: {url}", "Browser")
            
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise NetworkError("Invalid URL format")
            
            # Configure timeout and headers
            timeout = aiohttp.ClientTimeout(total=10)
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5"
            }
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(url, timeout=timeout, headers=headers) as response:
                        response.raise_for_status()
                        
                        # Store response state
                        state_update = {
                            'url': url,
                            'status_code': response.status,
                            'headers': dict(response.headers),
                            'content_type': response.headers.get('content-type', '')
                        }
                        self._state.update(state_update)
                        
                        # Extract text content
                        content = await self._extract_text(response)
                        if not content:
                            log_warning_with_context("No content extracted from page", "Browser", include_locals=True)
                            return "No content found on page"
                        
                        self._state['current_page'] = content
                        self._state['current_url'] = url
                        self._state['current_position'] = 0
                        self._state['find_position'] = 0
                        
                        log_info_with_context(f"Successfully visited {url}", "Browser")
                        return self._get_current_viewport()
                        
                except aiohttp.ClientError as e:
                    log_error_with_traceback(e, f"Network error visiting {url}", include_locals=True)
                    raise NetworkError(f"Failed to fetch URL: {str(e)}") from e
                    
        except Exception as e:
            log_error_with_traceback(e, f"Error visiting {url}", include_locals=True)
            if isinstance(e, (NetworkError, BrowserError)):
                raise
            raise BrowserError(f"Failed to visit URL: {str(e)}") from e
            
    async def _extract_text(self, response: aiohttp.ClientResponse) -> str:
        """Extract text content from response."""
        try:
            content_type = response.headers.get('content-type', '').lower()
            
            if content_type and 'text/html' in content_type:
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                    
                text = soup.get_text()
                if not text.strip():
                    log_warning_with_context("No text content found in HTML", "Browser", include_locals=True)
                return text
                
            elif content_type and 'application/pdf' in content_type:
                # Save PDF and extract text
                pdf_path = os.path.join(self.downloads_folder, f"{uuid.uuid4()}.pdf")
                
                try:
                    raw_content = await response.read()
                    with open(pdf_path, 'wb') as f:
                        f.write(raw_content)
                        
                    with open(pdf_path, 'rb') as f:
                        pdf = PyPDF2.PdfReader(f)
                        text = '\n'.join(page.extract_text() for page in pdf.pages)
                        
                    if not text.strip():
                        log_warning_with_context("No text content extracted from PDF", "Browser", include_locals=True)
                    return text
                    
                except Exception as e:
                    log_error_with_traceback(e, "Failed to extract text from PDF", include_locals=True)
                    raise ContentExtractionError("Failed to extract text from PDF") from e
                    
                finally:
                    if os.path.exists(pdf_path):
                        try:
                            os.remove(pdf_path)
                        except Exception as e:
                            log_warning_with_context(f"Failed to remove temporary PDF file: {e}", "Browser")
                            
            else:
                # Return raw text for other content types
                content = await response.text()
                if not content.strip():
                    log_warning_with_context("No content found in response", "Browser", include_locals=True)
                return content
                
        except Exception as e:
            log_error_with_traceback(e, "Failed to extract text content", include_locals=True)
            if isinstance(e, ContentExtractionError):
                raise
            raise ContentExtractionError("Failed to extract text content") from e
            
    @property
    def state(self) -> Dict[str, Any]:
        """Get current browser state."""
        return dict(self._state)  # Create a new dict to avoid modifying the original
            
    def page_up(self) -> str:
        """Move viewport up."""
        try:
            current_position = self._state.get('current_position', 0)
            self._state['current_position'] = max(0, current_position - self.viewport_size)
            return self._get_current_viewport()
        except Exception as e:
            log_error_with_traceback(e, "Failed to move viewport up", include_locals=True)
            raise BrowserError("Failed to move viewport up") from e
        
    def page_down(self) -> str:
        """Move viewport down."""
        try:
            current_page = self._state.get('current_page', '')
            if not current_page:
                log_warning_with_context("No page loaded", "Browser", include_locals=True)
                return "No page loaded"
                
            current_position = self._state.get('current_position', 0)
            self._state['current_position'] = min(
                len(current_page) - self.viewport_size,
                current_position + self.viewport_size
            )
            return self._get_current_viewport()
        except Exception as e:
            log_error_with_traceback(e, "Failed to move viewport down", include_locals=True)
            raise BrowserError("Failed to move viewport down") from e
        
    def find(self, text: str) -> str:
        """Find text in current page."""
        try:
            if not text:
                log_warning_with_context("Empty search text", "Browser", include_locals=True)
                return "Search text cannot be empty"
                
            current_page = self._state.get('current_page', '')
            if not current_page:
                log_warning_with_context("No page loaded", "Browser", include_locals=True)
                return "No page loaded"
                
            find_position = self._state.get('find_position', 0)
            position = current_page.find(text, find_position)
            if position == -1:
                log_info_with_context(f"Text not found: {text}", "Browser")
                return "Text not found"
                
            self._state['find_position'] = position
            self._state['current_position'] = max(0, position - 100)
            self._state['last_search'] = text
            
            log_info_with_context(f"Found text at position {position}", "Browser")
            return self._get_current_viewport()
            
        except Exception as e:
            log_error_with_traceback(e, "Failed to find text", include_locals=True)
            raise BrowserError("Failed to find text") from e
        
    def find_next(self) -> str:
        """Find next occurrence of the last search."""
        try:
            current_page = self._state.get('current_page', '')
            last_search = self._state.get('last_search')
            
            if not current_page:
                log_warning_with_context("No page loaded", "Browser", include_locals=True)
                return "No page loaded"
                
            if not last_search:
                log_warning_with_context("No previous search", "Browser", include_locals=True)
                return "No previous search"
                
            find_position = self._state.get('find_position', 0)
            position = current_page.find(last_search, find_position + 1)
            if position == -1:
                log_info_with_context("No more occurrences found", "Browser")
                return "No more occurrences found"
                
            self._state['find_position'] = position
            self._state['current_position'] = max(0, position - 100)
            
            log_info_with_context(f"Found next occurrence at position {position}", "Browser")
            return self._get_current_viewport()
            
        except Exception as e:
            log_error_with_traceback(e, "Failed to find next occurrence", include_locals=True)
            raise BrowserError("Failed to find next occurrence") from e
        
    def _get_current_viewport(self) -> str:
        """Get the current viewport of text."""
        try:
            current_page = self._state.get('current_page', '')
            if not current_page:
                log_warning_with_context("No page loaded", "Browser", include_locals=True)
                return "No page loaded"
                
            current_position = self._state.get('current_position', 0)
            end_pos = min(
                len(current_page), 
                current_position + self.viewport_size
            )
            
            viewport = current_page[current_position:end_pos]
            if not viewport.strip():
                log_warning_with_context("Empty viewport", "Browser", include_locals=True)
                
            return viewport
            
        except Exception as e:
            log_error_with_traceback(e, "Failed to get viewport", include_locals=True)
            raise BrowserError("Failed to get viewport") from e

class SearchError(BrowserError):
    """Search-related errors"""
    pass

class WebSearchInput(BaseModel):
    """Input model for web search"""
    query: str = Field(description="The web search query to perform")
    filter_year: Optional[str] = Field(None, description="Filter results to a specific year")

class VisitInput(BaseModel):
    """Input model for visiting URLs"""
    url: str = Field(description="The URL to visit")

class FindInput(BaseModel):
    """Input model for text search"""
    search_string: str = Field(description="The text to search for on the page")

async def web_search(query: str) -> str:
    """Perform web search using SearxNG."""
    try:
        if not query or not query.strip():
            log_warning_with_context("Empty search query", "Search", include_locals=True)
            return "Error: Empty search query"
            
        log_info_with_context(f"Searching for: {query}", "Search")
        
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
                try:
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
                                        f"Title: {title}",
                                        f"URL: {url}",
                                        f"Summary: {snippet}",
                                        "---"
                                    ])
                                    
                            log_info_with_context(f"Found {len(formatted_results)//4} results", "Search")
                            return "\n".join(formatted_results)
                        else:
                            log_warning_with_context("No search results found", "Search", include_locals=True)
                            return "No results found"
                            
                except aiohttp.ClientError as e:
                    log_error_with_traceback(e, "Network error during search", include_locals=True)
                    raise NetworkError(f"Failed to connect to search service: {str(e)}") from e
                    
                except json.JSONDecodeError as e:
                    log_error_with_traceback(e, "Invalid JSON response from search service", include_locals=True)
                    raise SearchError("Invalid response from search service") from e
                    
        except Exception as e:
            log_error_with_traceback(e, "Error performing search", include_locals=True)
            if isinstance(e, (NetworkError, SearchError)):
                raise
            raise SearchError(f"Search failed: {str(e)}") from e
            
    except Exception as e:
        log_error_with_traceback(e, "Fatal error in web search", include_locals=True)
        if isinstance(e, (NetworkError, SearchError, BrowserError)):
            raise
        raise SearchError(f"Search failed: {str(e)}") from e

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
    return await browser.visit(url)

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
    return browser.page_up()

page_up_tool = StructuredTool.from_function(
    func=page_up,
    name="page_up",
    description="Scroll the viewport UP one page-length in the current webpage and return the new viewport content.",
    coroutine=page_up
)

async def page_down() -> str:
    """Move the viewport down one page."""
    browser = SimpleTextBrowser()
    return browser.page_down()

page_down_tool = StructuredTool.from_function(
    func=page_down,
    name="page_down",
    description="Scroll the viewport DOWN one page-length in the current webpage and return the new viewport content.",
    coroutine=page_down
)

async def find_on_page(search_string: str) -> str:
    """Search for text on the current page."""
    browser = SimpleTextBrowser()
    return browser.find(search_string)

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
    return browser.find_next()

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
        content = await browser.visit(target_url)
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
