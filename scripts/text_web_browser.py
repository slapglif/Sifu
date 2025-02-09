# https://github.com/microsoft/autogen/blob/gaia_multiagent_v01_march_1st/autogen/browser_utils.py
import mimetypes
import os
import pathlib
import re
import time
import uuid
import json
from typing import Any, Dict, List, Optional, Tuple, Union, TypedDict, Annotated, Coroutine, Set
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
from pydantic import BaseModel, Field, OutputParser
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

from .cookies import COOKIES
from .mdconvert import FileConversionException, MarkdownConverter, UnsupportedFormatException
from .logging_config import (
    log_error_with_traceback,
    log_warning_with_context,
    log_info_with_context,
    console,
    create_progress
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

class BrowserState(TypedDict):
    """Browser state"""
    current_page: str
    current_url: Optional[str]
    current_position: int
    find_position: int
    viewport_size: int
    last_search: Optional[str]

class SimpleTextBrowser:
    """Agentic browser for intelligent web crawling and text extraction"""
    def __init__(self, viewport_size: int = 5120, downloads_folder: str = "downloads", request_kwargs: Optional[Dict] = None):
        self.viewport_size = viewport_size
        self.downloads_folder = downloads_folder
        self.request_kwargs = request_kwargs or {}
        
        # Initialize state
        self._state = {
            'current_page': '',
            'current_url': '',
            'current_position': 0,
            'find_position': 0,
            'visited_urls': set(),  # Track visited URLs
            'url_queue': set(),     # URLs to visit
            'extracted_content': {}  # Map of URL to extracted content
        }
        
        # Initialize session with default headers and cookies
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=30)
        
        # Create downloads folder if it doesn't exist
        os.makedirs(downloads_folder, exist_ok=True)
        
        # Initialize counters
        self.total_tokens = 0
        self.total_chunks = 0
        
        # Configure crawling settings
        self.max_depth = 3  # Maximum crawl depth
        self.max_pages = 50  # Maximum pages to crawl
        self.same_domain_only = True  # Only crawl same domain
        self.excluded_patterns = [  # Patterns to exclude
            r'\.(jpg|jpeg|png|gif|ico|css|js|xml|pdf)$',
            r'(mailto:|tel:)',
            r'#.*$',
            r'\?.*$'
        ]
        
    async def __aenter__(self):
        """Async context manager entry"""
        if not self.session:
            # Initialize session with cookies and default headers
            self.session = aiohttp.ClientSession(
                cookies=COOKIES,
                headers=self._get_default_headers()
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            self.session = None

    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers with rotated user agent"""
        return {
            "User-Agent": ua_manager.get_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "DNT": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate", 
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
            "Connection": "keep-alive"
        }

    async def crawl_website(self, start_url: str, max_depth: Optional[int] = None, max_pages: Optional[int] = None) -> Dict[str, str]:
        """Intelligently crawl a website starting from the given URL."""
        try:
            # Reset crawling state
            self._state['visited_urls'] = set()
            self._state['url_queue'] = {start_url}
            self._state['extracted_content'] = {}
            
            # Override defaults if provided
            if max_depth is not None:
                self.max_depth = max_depth
            if max_pages is not None:
                self.max_pages = max_pages
                
            # Get base domain for same-domain checking
            base_domain = urlparse(start_url).netloc
            
            # Initialize progress tracking
            progress = create_progress()
            crawl_task = progress.add_task(
                f"[cyan]Crawling {base_domain}...", 
                total=self.max_pages
            )
            
            # Process URL queue
            current_depth = 0
            while (current_depth < self.max_depth and 
                   len(self._state['url_queue']) > 0 and 
                   len(self._state['visited_urls']) < self.max_pages):
                
                # Get URLs for current depth
                current_urls = self._state['url_queue'].copy()
                self._state['url_queue'] = set()
                
                # Process URLs at current depth in parallel
                async def process_url(url: str) -> None:
                    try:
                        # Skip if already visited
                        if url in self._state['visited_urls']:
                            return
                            
                        # Skip if different domain and same_domain_only is True
                        if self.same_domain_only and urlparse(url).netloc != base_domain:
                            return
                            
                        # Skip if matches excluded patterns
                        if any(re.search(pattern, url) for pattern in self.excluded_patterns):
                            return
                            
                        # Visit URL and extract content
                        content = await self.visit(url)
                        if content and not any(x in content for x in ["Content not accessible", "Error accessing"]):
                            self._state['extracted_content'][url] = content
                            
                            # Extract new URLs from content
                            new_urls = await self._extract_urls(content, url)
                            self._state['url_queue'].update(new_urls)
                            
                        # Mark as visited
                        self._state['visited_urls'].add(url)
                        
                        # Update progress
                        progress.update(crawl_task, advance=1)
                        
                        # Log progress
                        log_info_with_context(
                            f"Crawled {url} - Found {len(new_urls)} new URLs",
                            "Crawler"
                        )
                        
                    except Exception as e:
                        log_error_with_traceback(e, f"Error processing URL: {url}")
                
                # Process URLs with rate limiting
                semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
                async def bounded_process(url: str) -> None:
                    async with semaphore:
                        await process_url(url)
                
                # Process all URLs at current depth
                tasks = [bounded_process(url) for url in current_urls]
                await asyncio.gather(*tasks)
                
                current_depth += 1
                
                # Log depth completion
                console.print(f"[green]✓ Completed depth {current_depth}[/green]")
                console.print(f"  Pages crawled: {len(self._state['visited_urls'])}")
                console.print(f"  URLs in queue: {len(self._state['url_queue'])}")
            
            # Return extracted content
            return self._state['extracted_content']
            
        except Exception as e:
            log_error_with_traceback(e, f"Error crawling website: {start_url}")
            raise
            
    async def _extract_urls(self, content: str, base_url: str) -> Set[str]:
        """Extract and normalize URLs from content."""
        try:
            urls = set()
            
            # Parse HTML
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract URLs from various tags
            for tag in soup.find_all(['a', 'link', 'area', 'iframe']):
                url = tag.get('href') or tag.get('src')
                if url:
                    # Normalize URL
                    try:
                        normalized = urljoin(base_url, url)
                        parsed = urlparse(normalized)
                        
                        # Skip invalid URLs
                        if not parsed.scheme or not parsed.netloc:
                            continue
                            
                        # Skip non-HTTP(S) URLs
                        if parsed.scheme not in ['http', 'https']:
                            continue
                            
                        # Skip excluded patterns
                        if any(re.search(pattern, normalized) for pattern in self.excluded_patterns):
                            continue
                            
                        urls.add(normalized)
                        
                    except Exception as e:
                        log_warning_with_context(f"Error normalizing URL {url}: {e}")
                        continue
            
            return urls
            
        except Exception as e:
            log_error_with_traceback(e, f"Error extracting URLs from {base_url}")
            return set()

    async def visit(self, url: str) -> str:
        """Visit URL and extract text content with enhanced error handling"""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        if not self.session:
            self.session = aiohttp.ClientSession(
                cookies=COOKIES,
                headers=self._get_default_headers()
            )
            
        try:
            # Make request with timeout and headers
            async with self.session.get(
                url,
                headers=self._get_default_headers(),
                timeout=self.timeout,
                allow_redirects=True,
                max_redirects=5,
                **self.request_kwargs
            ) as response:
                
                # Handle various status codes
                if response.status == 403:
                    logger.warning(f"Access denied (403) for {url}, skipping...")
                    return f"Content not accessible (403 Forbidden) from {url}"
                elif response.status == 429:
                    logger.warning(f"Rate limited (429) for {url}, skipping...")
                    return f"Rate limited (429 Too Many Requests) from {url}"
                elif response.status != 200:
                    logger.warning(f"HTTP {response.status} for {url}, skipping...")
                    return f"Content not accessible (HTTP {response.status}) from {url}"
                
                content_type = response.headers.get('content-type', 'text/plain')
                
                # Handle different content types
                if 'application/pdf' in content_type:
                    content = await response.read()  # Read raw bytes for PDF
                else:
                    content = await response.text()  # Read text for other types
                
                # Extract text
                text = await self._extract_text(content, url, content_type)
                if text:
                    # Count tokens and log
                    tokens = len(enc.encode(text))
                    self.total_tokens += tokens
                    
                    # Log successful extraction with token count
                    logger.info(f"Successfully extracted {len(text)} chars from {url}")
                    console.print(Panel(
                        f"[green]✓ Extracted content from {url}[/green]\n"
                        f"Length: {len(text)} chars\n"
                        f"Tokens: {tokens}\n"
                        f"Preview: {text[:200]}...",
                        title="Content Extraction"
                    ))
                    return text
                    
                return f"No content could be extracted from {url}"
                
        except aiohttp.ClientError as e:
            logger.warning(f"Network error accessing {url}: {str(e)}")
            return f"Network error: {str(e)}"
        except Exception as e:
            logger.warning(f"Error accessing {url}: {str(e)}")
            return f"Error accessing content: {str(e)}"

    async def _extract_text(self, content: Union[str, bytes], url: str, content_type: str) -> str:
        """Extract text content with chunk tracking and logging"""
        try:
            if 'text/html' in content_type:
                if isinstance(content, bytes):
                    try:
                        content = content.decode('utf-8')
                    except UnicodeDecodeError:
                        content = content.decode('utf-8', errors='ignore')
                    
                soup = BeautifulSoup(content, 'html.parser')
                
                # Remove script, style, and other non-content elements
                for element in soup(['script', 'style', 'meta', 'link', 'noscript', 'header', 'footer', 'nav']):
                    element.decompose()
                
                # Extract text from remaining elements with chunk tracking
                text_parts = []
                chunk_count = 0
                
                for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div']):
                    text = element.get_text(strip=True)
                    if text:
                        chunk_count += 1
                        # Log chunk preview
                        logger.debug(f"Chunk {chunk_count}: {text[:100]}...")
                        text_parts.append(text)
                
                self.total_chunks += chunk_count
                
                # Join with newlines between paragraphs
                text = '\n\n'.join(text_parts)
                
                if not text.strip():
                    log_warning_with_context("No text content found in HTML", "Browser", include_locals=True)
                    # Try getting all text as fallback
                    text = soup.get_text(separator='\n\n', strip=True)
                
                # Log chunk statistics
                logger.info(f"Extracted {chunk_count} text chunks")
                return text if text else "No text content found"
                
            elif 'application/pdf' in content_type:
                # Handle PDF extraction
                if not isinstance(content, bytes):
                    if isinstance(content, str):
                        content = content.encode('utf-8')
                    else:
                        raise ContentExtractionError("Invalid content type for PDF extraction")
                return await self._extract_pdf(content, url)
            else:
                # Try extracting as plain text
                if isinstance(content, bytes):
                    try:
                        content = content.decode('utf-8')
                    except UnicodeDecodeError:
                        content = content.decode('utf-8', errors='ignore')
                elif not isinstance(content, str):
                    content = str(content)
                    
                text = content.strip()
                return text if text else "No text content found"
                
        except Exception as e:
            log_error_with_traceback(e, "Error extracting text", include_locals=True)
            return f"Error extracting text: {str(e)}"

    async def _extract_pdf(self, content: bytes, url: str) -> str:
        """Extract text from a PDF file"""
        try:
            pdf_path = os.path.join(self.downloads_folder, f"{uuid.uuid4()}.pdf")
            
            with open(pdf_path, 'wb') as f:
                f.write(content)
            
            with open(pdf_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                text_parts = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                text = '\n\n'.join(text_parts)
                
            if not text.strip():
                log_warning_with_context("No text content extracted from PDF", "Browser", include_locals=True)
                return text
            
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
    """Perform web search using SearxNG with intelligent crawling."""
    try:
        if not query or not query.strip():
            log_warning_with_context("Empty search query", "Search", include_locals=True)
            return "Error: Empty search query"
            
        log_info_with_context(f"Searching for: {query}", "Search")
        
        # Get shared progress instance
        progress = create_progress()
        
        console.print(Panel(f"[bold cyan]Web Search[/bold cyan]\nQuery: {query}"))
        
        # Configure SearxNG search
        searx_host = os.getenv("SEARX_HOST", "https://searchapi.anuna.dev")
        headers = {
            "User-Agent": ua_manager.get_agent(),
            "Accept": "application/json"
        }
        
        # Prepare search parameters
        params = {
            "q": query,
            "format": "json",
            "engines": "google,bing,duckduckgo,yahoo,brave,qwant,google_scholar,arxiv,pubmed,crossref,github,gitlab,stackoverflow,wikipedia",
            "language": "en",
            "time_range": "",
            "categories": "general,science,it,news,social media",
            "pageno": "1",
            "results": "25",
            "safesearch": 0,
            "engine_type": "general,scientific"
        }
        
        # Make request to SearxNG
        try:
            async with aiohttp.ClientSession(cookies=COOKIES) as session:
                try:
                    # Create progress tasks
                    search_task = progress.add_task("[cyan]Searching...", total=100)
                    fetch_task = progress.add_task("[yellow]Fetching results...", total=100, visible=False)
                    process_task = progress.add_task("[green]Processing content...", total=100, visible=False)
                    
                    async with session.get(
                        f"{searx_host}/search",
                        params=params,
                        headers=headers,
                        ssl=False,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        response.raise_for_status()
                        results = await response.json()
                        
                        # Update search progress
                        progress.update(search_task, completed=100)
                        progress.update(fetch_task, visible=True)
                        
                        # Extract URLs and snippets
                        formatted_results = []
                        if "results" in results and results["results"]:
                            total_results = len(results["results"])
                            console.print(f"[green]✓ Found {total_results} results[/green]")
                            
                            # Process results with intelligent crawling
                            async with SimpleTextBrowser() as browser:
                                # Process each result URL
                                for i, result in enumerate(results["results"][:10]):  # Limit to top 10 for crawling
                                    try:
                                        url = result.get("url", "").strip()
                                        title = result.get("title", "").strip()
                                        snippet = result.get("content", "").strip()
                                        
                                        if url:
                                            # Crawl the website
                                            console.print(f"\n[cyan]Crawling {url}...[/cyan]")
                                            crawled_content = await browser.crawl_website(
                                                url,
                                                max_depth=2,  # Limit depth for search results
                                                max_pages=10  # Limit pages per result
                                            )
                                            
                                            # Update fetch progress
                                            progress.update(fetch_task, advance=100/total_results)
                                            
                                            if crawled_content:
                                                # Create rich table for result
                                                table = Table(show_header=False, box=box.ROUNDED)
                                                table.add_column("Field", style="cyan")
                                                table.add_column("Value", style="white")
                                                
                                                table.add_row("Title", title)
                                                table.add_row("URL", url)
                                                table.add_row("Summary", snippet)
                                                
                                                # Add crawled pages summary
                                                crawled_summary = f"Crawled {len(crawled_content)} pages:"
                                                for crawled_url, content in list(crawled_content.items())[:3]:
                                                    crawled_summary += f"\n- {crawled_url}"
                                                    if content:
                                                        preview = content[:200].replace('\n', ' ').strip()
                                                        crawled_summary += f"\n  {preview}..."
                                                if len(crawled_content) > 3:
                                                    crawled_summary += f"\n... and {len(crawled_content) - 3} more pages"
                                                
                                                table.add_row("Crawled Content", crawled_summary)
                                                
                                                formatted_results.extend([
                                                    str(table),
                                                    "---"
                                                ])
                                                
                                                # Update processing progress
                                                progress.update(process_task, advance=100/total_results)
                                                
                                    except Exception as e:
                                        log_error_with_traceback(e, f"Error processing result {url}")
                                        continue
                                
                                progress.update(process_task, completed=100)
                                log_info_with_context(f"Processed {len(formatted_results)//2} results", "Search")
                                return "\n".join(formatted_results)
                        else:
                            log_warning_with_context("No results found", "Search", include_locals=True)
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
