# https://github.com/microsoft/autogen/blob/gaia_multiagent_v01_march_1st/autogen/browser_utils.py
import mimetypes
import os
import pathlib
import re
import time
import uuid
import json
import heapq
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, TypedDict, Annotated, Coroutine, Set
from urllib.parse import unquote, urljoin, urlparse

import pathvalidate
import requests
from bs4 import BeautifulSoup, Comment
from loguru import logger
from langchain_core.tools import BaseTool, StructuredTool, Tool
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.utilities import SearxSearchWrapper
from pydantic import BaseModel, Field
import PyPDF2
import aiohttp
import asyncio
from rich import box
import random
from fake_useragent import UserAgent  # For better user agent rotation
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
import tiktoken
from playwright.async_api import async_playwright, Browser, Page, Error as PlaywrightError

from .cookies import COOKIES
from .mdconvert import FileConversionException, MarkdownConverter, UnsupportedFormatException
from .logging_config import (
    log_error_with_traceback,
    log_warning_with_context,
    log_info_with_context,
    log_error_with_context,
    console,
    create_progress,
    cleanup_progress
)

# Initialize tiktoken encoder
try:
    enc = tiktoken.get_encoding("cl100k_base")
except:
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Initialize fake user agent with fallback and rotation
class UserAgentManager:
    """Manages user agent rotation with fallback"""
    FALLBACK_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/120.0.0.0",
        "Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0.0.0"
    ]
    
    def __init__(self):
        """Initialize with fake-useragent and fallback"""
        try:
            self.ua = UserAgent(browsers=['chrome', 'firefox', 'safari', 'edge'])
            self.using_fallback = False
        except Exception as e:
            logger.warning(f"Failed to initialize fake-useragent: {e}. Using fallback user agents.")
            self.using_fallback = True
            
        self.last_rotation = time.time()
        self.current_agent = None
        self.rotation_interval = 60  # Rotate every minute
        
    def get_agent(self) -> str:
        """Get a user agent string, rotating if needed"""
        current_time = time.time()
        if not self.current_agent or (current_time - self.last_rotation) > self.rotation_interval:
            if self.using_fallback:
                self.current_agent = random.choice(self.FALLBACK_AGENTS)
            else:
                try:
                    self.current_agent = self.ua.random
                except Exception as e:
                    logger.warning(f"Error getting random user agent: {e}. Using fallback.")
                    self.current_agent = random.choice(self.FALLBACK_AGENTS)
            self.last_rotation = current_time
            logger.debug(f"Rotated user agent to: {self.current_agent}")
        return self.current_agent

# Initialize user agent manager
ua_manager = UserAgentManager()

class BrowserError(Exception):
    """Base class for browser errors"""
    pass

class NetworkError(BrowserError):
    """Network-related errors"""
    pass

class ContentExtractionError(BrowserError):
    """Content extraction errors"""
    pass

class WebSearchInput(BaseModel):
    """Input model for web search"""
    query: str = Field(description="The web search query to perform")

class VisitInput(BaseModel):
    """Input model for visiting URLs"""
    url: str = Field(description="The URL to visit")

class BrowserState(TypedDict):
    """Browser state"""
    current_page: str
    current_url: Optional[str]
    current_position: int
    find_position: int
    viewport_size: int
    last_search: Optional[str]

class SimpleTextBrowser:
    """Browser using browserless.io with Playwright"""
    def __init__(self, viewport_size: int = 5120, downloads_folder: str = "downloads"):
        self.viewport_size = viewport_size
        self.downloads_folder = downloads_folder
        self.browserless_url = "https://browserless.anuna.dev"
        
        # Initialize state
        self._state = {
            'current_page': '',
            'current_url': '',
            'current_position': 0,
            'find_position': 0,
            'visited_urls': set(),
            'extracted_content': {},
            'domain_delays': {},
        }
        
        # Create downloads folder if it doesn't exist
        os.makedirs(downloads_folder, exist_ok=True)
        
        # Initialize counters
        self.total_tokens = 0
        self.total_chunks = 0
        
        # Configure crawling settings
        self.max_depth = 3
        self.max_pages = 50
        self.same_domain_only = True
        self.min_delay = 1.0
        self.max_retries = 3
        
        # Update excluded patterns
        self.excluded_patterns = [
            r'\.(jpg|jpeg|png|gif|ico|exe|dmg|iso|bin)$',
            r'/(ads|analytics|tracking|cdn)/',
            r'\?(utm_source|utm_medium|utm_campaign|fbclid)='
        ]

    async def _get_browser(self) -> None:
        """No longer needed with direct API calls"""
        pass

    async def visit(self, url: str) -> str:
        """Visit URL and extract text content using MarkdownConverter"""
        retries = 3
        retry_delay = 1.0
        
        for attempt in range(retries):
            try:
                # Use MarkdownConverter for better content handling
                converter = MarkdownConverter()
                result = converter.convert(url)
                
                if not result or not result.text_content:
                    log_warning_with_context("No text content found", "Browser")
                    if attempt < retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    return "Error: No content found"
                
                # Clean up the text
                text = result.text_content
                text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
                text = re.sub(r'\n\s*\n', '\n\n', text)  # Replace multiple newlines
                text = text.strip()
                
                return text
                
            except Exception as e:
                log_error_with_traceback(e, f"Error visiting {url}")
                if attempt < retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                return f"Error accessing content: {str(e)}"
        
        return "Error: Maximum retries exceeded"

    async def crawl_website(self, start_url: str, max_depth: Optional[int] = None, max_pages: Optional[int] = None, query: Optional[str] = None) -> Dict[str, str]:
        """Crawl a website using browserless scrape API"""
        try:
            # Reset crawling state
            self._state['visited_urls'] = set()
            self._state['extracted_content'] = {}
            self._state['domain_delays'] = {}
            
            # Override defaults if provided
            if max_depth is not None:
                self.max_depth = max_depth
            if max_pages is not None:
                self.max_pages = max_pages
            
            # Visit start URL
            content = await self.visit(start_url)
            if content and "Error accessing content:" not in content:
                self._state['extracted_content'][start_url] = content
                self._state['visited_urls'].add(start_url)
            
            # Get links using content API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.browserless_url}/scrape",
                    json={
                        "url": start_url,
                        "elements": [{"selector": "a[href]"}],
                        "waitFor": 2000
                    },
                    headers={
                        "Content-Type": "application/json",
                        "Cache-Control": "no-cache",
                        "x-browserless-token": os.getenv("BROWSERLESS_TOKEN", "")
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        links = []
                        for element in data.get("data", []):
                            if "attributes" in element and "href" in element["attributes"]:
                                link = element["attributes"]["href"]
                                if link.startswith("http"):
                                    links.append(link)
                        
                        # Process links
                        for link in links:
                            if len(self._state['visited_urls']) >= self.max_pages:
                                break
                                
                            try:
                                # Skip if already visited
                                if link in self._state['visited_urls']:
                                    continue
                                    
                                # Skip if matches excluded patterns
                                if any(re.search(pattern, link) for pattern in self.excluded_patterns):
                                    continue
                                
                                # Visit link
                                content = await self.visit(link)
                                if content and "Error accessing content:" not in content:
                                    self._state['extracted_content'][link] = content
                                    self._state['visited_urls'].add(link)
                                    
                            except Exception as e:
                                log_error_with_traceback(e, f"Error processing link: {link}")
                                continue
            
            return self._state['extracted_content']
            
        except Exception as e:
            log_error_with_traceback(e, f"Error crawling website: {start_url}")
            return {}

# Export tools
async def web_search(query: str) -> str:
    """Perform web search using SearxNG and browserless for content extraction"""
    try:
        if not query or not query.strip():
            return "Error: Empty search query"
        
        # Configure SearxNG search
        searx_host = os.getenv("SEARX_HOST", "https://searchapi.anuna.dev")
        
        # Make search request with expanded parameters
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{searx_host}/search",
                params={
                    "q": query,
                    "format": "json",
                    "engines": "google,bing,duckduckgo,yahoo,brave,qwant,google_scholar,arxiv,pubmed,crossref,github,gitlab,stackoverflow,wikipedia,semantic_scholar",
                    "language": "en",
                    "time_range": "year",
                    "categories": "general,science,it,news,social media,research",
                    "pageno": "1",
                    "results": "50",
                    "safesearch": 0,
                    "engine_type": "general,scientific,research",
                    "page_size": "50",
                    "deep_search": "true",
                    "full_content": "true"
                },
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                },
                ssl=False
            ) as response:
                results = await response.json()
        
        # Process results with browserless
        browser = SimpleTextBrowser()
        formatted_results = []
        
        if "results" in results and results["results"]:
            for result in results["results"][:10]:
                url = result.get("url", "").strip()
                title = result.get("title", "").strip()
                snippet = result.get("content", "").strip()
                
                if url:
                    try:
                        # Use MarkdownConverter for better content extraction
                        converter = MarkdownConverter()
                        content = converter.convert(url).text_content
                        
                        if content and "Error accessing content:" not in content:
                            # Clean the content
                            # Remove HTML tags and attributes
                            content = re.sub(r'<[^>]+>', '', content)
                            # Remove navigation elements
                            content = re.sub(r'Navigation.*?Menu', '', content, flags=re.IGNORECASE | re.DOTALL)
                            # Remove common UI elements
                            content = re.sub(r'(Search|Menu|Navigation|Footer|Header|Copyright|Privacy Policy|Terms of Service).*?\n', '', content, flags=re.IGNORECASE)
                            # Remove URLs and paths
                            content = re.sub(r'https?://\S+', '', content)
                            content = re.sub(r'/\S+', '', content)
                            # Remove square brackets with links
                            content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
                            # Remove remaining square brackets
                            content = re.sub(r'\[(.*?)\]', r'\1', content)
                            # Remove extra whitespace
                            content = re.sub(r'\s+', ' ', content)
                            content = re.sub(r'\n\s*\n', '\n\n', content)
                            # Remove very short lines (likely UI elements)
                            content = '\n'.join(line for line in content.split('\n') if len(line.strip()) > 30)
                            # Remove lines that are just punctuation or special characters
                            content = '\n'.join(line for line in content.split('\n') if re.search(r'[a-zA-Z]', line))
                            # Remove lines that are just navigation elements
                            content = '\n'.join(line for line in content.split('\n') if not re.match(r'^(next|previous|page|chapter|section|home|back|forward)\s*$', line, re.IGNORECASE))
                            content = content.strip()
                            
                            # Create a more detailed result entry
                            table = Table(show_header=False, box=box.ROUNDED)
                            table.add_column("Field", style="cyan")
                            table.add_column("Value", style="white")
                            
                            table.add_row("Title", title)
                            table.add_row("URL", url)
                            table.add_row("Summary", snippet)
                            
                            # Save full content to file
                            filename = f"web/source_{len(formatted_results) + 1}.txt"
                            with open(filename, "w", encoding="utf-8") as f:
                                f.write(f"Title: {title}\n")
                                f.write(f"URL: {url}\n")
                                f.write(f"Query: {query}\n")
                                f.write(f"Summary: {snippet}\n")
                                f.write(f"Domain: test_domain\n")
                                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                                f.write("---\n")
                                f.write(content)
                            
                            # Add preview to formatted results
                            table.add_row("Content Preview", content[:500] + "...")
                            formatted_results.extend([
                                str(table),
                                "---"
                            ])
                            
                    except Exception as e:
                        log_error_with_traceback(e, f"Error processing result {url}")
                        continue
        
        return "\n".join(formatted_results) if formatted_results else "No results found"
        
    except Exception as e:
        log_error_with_traceback(e, "Error in web search")
        return f"Search failed: {str(e)}"

web_search_tool = StructuredTool.from_function(
    func=web_search,
    name="web_search",
    description="Perform a web search query and returns the search results.",
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
    description="Visit a webpage at a given URL and return its text content.",
    args_schema=VisitInput,
    coroutine=visit_page
)

# Export all tools
tools = [
    web_search_tool,
    visit_tool
]
