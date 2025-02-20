"""Web browser utilities."""
import re
import os
import asyncio
from typing import Dict, Any, Optional, List, Set, Tuple
from bs4 import BeautifulSoup, Comment
from scripts.mdconvert import MarkdownConverter
from scripts.logging_config import log_error_with_traceback, log_warning_with_context
import aiohttp
from rich.table import Table
from rich import box
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_community.utilities import SearxSearchWrapper
from urllib.parse import urlparse
import hashlib
from tqdm.rich import tqdm
import numpy as np
from collections import defaultdict
from scripts.chat_langchain import ChatLangChain
from prompts.knowledge_acquisition.query_generation import (
    QueryGenerationResponse,
    get_query_generation_prompt
)
from pydantic import SecretStr
from langchain.output_parsers import PydanticOutputParser
import xml.etree.ElementTree as ET
import json
import ssl
from rich.progress import Progress, track
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class SearchMetadata(BaseModel):
    """Metadata for a search result"""
    query: str = Field(description="Original search query")
    url: str = Field(description="Source URL")
    title: str = Field(description="Page title")
    snippet: str = Field(description="Search result snippet")
    source: str = Field(description="Source type (e.g., pubmed, arxiv)")
    published_date: Optional[str] = Field(None, description="Publication date if available")
    score: float = Field(default=0.0, description="Search result score")
    timestamp: str = Field(description="When the result was processed")
    hash: str = Field(description="Content hash for deduplication")
    query_group: str = Field(description="Group of related queries this came from")

class WebContent(BaseModel):
    """Model for processed web content"""
    title: str = Field(description="Title of the webpage")
    url: str = Field(description="URL of the webpage")
    content: str = Field(description="Cleaned and processed content")
    summary: str = Field(description="Brief summary of the content")
    domain: str = Field(description="Domain/topic this content belongs to")
    timestamp: str = Field(description="When the content was processed")
    metadata: SearchMetadata = Field(description="Search and source metadata")
    chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Chunked content with embeddings")

class URLTracker:
    """Track and manage URLs to avoid duplicates"""
    def __init__(self):
        self.urls: Set[str] = set()
        self.content_hashes: Set[str] = set()
        self.domain_counts: Dict[str, int] = {}
        self.query_groups: Dict[str, Set[str]] = defaultdict(set)
        
    def add_url(self, url: str, query_group: str) -> bool:
        """Add URL if not seen before. Returns True if URL is new."""
        if url in self.urls:
            return False
            
        # Check domain limits
        domain = urlparse(url).netloc
        if self.domain_counts.get(domain, 0) >= 3:  # Max 3 URLs per domain
            return False
            
        # Check query group limits
        if len(self.query_groups[query_group]) >= 5:  # Max 5 URLs per query group
            return False
            
        self.urls.add(url)
        self.domain_counts[domain] = self.domain_counts.get(domain, 0) + 1
        self.query_groups[query_group].add(url)
        return True
        
    def add_content(self, content: str) -> bool:
        """Add content if not seen before. Returns True if content is new."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if content_hash in self.content_hashes:
            return False
        self.content_hashes.add(content_hash)
        return True

class SimpleTextBrowser:
    """Simple browser for extracting text from web pages."""
    
    def __init__(self):
        """Initialize browser."""
        self.browserless_url = "https://browserless.anuna.dev"
        self.session = aiohttp.ClientSession()
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
        
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

async def generate_diverse_queries(base_query: str, domain: str) -> List[Tuple[str, str]]:
    """Generate diverse search queries with group labels using LLM."""
    try:
        # Initialize LLM
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY environment variable must be set")
            
        llm = ChatLangChain(
            model="gemini-1.5-flash",
            temperature=0.7,
            api_key=SecretStr(api_key),
            pydantic_schema=QueryGenerationResponse,
            format='json',
            response_format={"type": "json_object"}
        )
        
        # Extract key terms if queries are too long
        if len(domain) > 50:
            key_terms = [term.strip() for term in domain.split() if len(term) > 3][:3]
            domain = " ".join(key_terms)
            
        if len(base_query) > 50:
            key_terms = [term.strip() for term in base_query.split() if len(term) > 3][:3]
            base_query = " ".join(key_terms)
            
        # Get prompt template
        prompt = get_query_generation_prompt()
        
        # Generate queries using LLM
        chain = prompt | llm | PydanticOutputParser(pydantic_object=QueryGenerationResponse)
        response = await chain.ainvoke({
            "base_query": base_query,
            "domain": domain,
            "format_instructions": PydanticOutputParser(pydantic_object=QueryGenerationResponse).get_format_instructions()
        })
        
        # Convert to list of (query, group) tuples
        queries = []
        seen_queries = set()  # Track unique queries
        
        for group in response.query_groups:
            for query in group.queries:
                # Skip duplicates and ensure reasonable length
                if query in seen_queries or len(query) > 100:
                    continue
                    
                # Clean and format query
                query = query.strip()
                if not query:
                    continue
                    
                # Add to results
                group_id = f"{group.group_name}_{len(queries)}"
                queries.append((query, group_id))
                seen_queries.add(query)
                
        # Shuffle while maintaining group associations
        indices = np.random.permutation(len(queries))
        return [queries[i] for i in indices]
        
    except Exception as e:
        log_error_with_traceback(e, "Error generating queries")
        # Fallback to basic queries if LLM fails
        return [
            (f"{domain} {base_query} overview", "overview_0"),
            (f"{domain} {base_query} research", "research_0"),
            (f"{domain} {base_query} treatment", "treatment_0")
        ]

async def clean_content(content: str) -> str:
    """Clean and normalize content."""
    # Remove HTML
    soup = BeautifulSoup(content, "html.parser")
    
    # Remove unwanted elements
    for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
        element.decompose()
        
    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
        
    # Get main content
    main_content = ""
    main_tags = soup.find_all(["article", "main", "div[role='main']"])
    if main_tags:
        main_content = " ".join(tag.get_text(strip=True, separator=" ") for tag in main_tags)
    else:
        body = soup.find("body")
        if body:
            main_content = body.get_text(strip=True, separator=" ")
        else:
            main_content = soup.get_text(strip=True, separator=" ")
            
    # Clean up text
    main_content = re.sub(r'\s+', ' ', main_content)  # Normalize whitespace
    main_content = re.sub(r'\[.*?\]', '', main_content)  # Remove square bracket content
    main_content = re.sub(r'[^\x00-\x7F]+', '', main_content)  # Remove non-ASCII
    main_content = re.sub(r'(\w)\1{3,}', r'\1\1', main_content)  # Normalize repeated chars
    main_content = main_content.strip()
    
    return main_content

async def fetch_content(browser: SimpleTextBrowser, result: Dict[str, Any], query: str, query_group: str) -> Optional[WebContent]:
    """Fetch and process content from a search result."""
    try:
        url = result.get("link", "").strip()
        if not url:
            return None
            
        # Get content
        content = await browser.visit(url)
        if not content or "Error accessing content:" in content:
            return None
            
        # Clean content
        cleaned_content = await clean_content(content)
        if not cleaned_content:
            return None
            
        # Create metadata
        metadata = SearchMetadata(
            query=query,
            url=url,
            title=result.get("title", ""),
            snippet=result.get("snippet", ""),
            source=result.get("source", ""),
            published_date=result.get("published_date"),
            score=float(result.get("score", 0)),
            timestamp=datetime.now().isoformat(),
            hash=hashlib.md5(cleaned_content.encode()).hexdigest(),
            query_group=query_group
        )
        
        # Create web content object
        return WebContent(
            title=result.get("title", ""),
            url=url,
            content=cleaned_content,
            summary=result.get("snippet", ""),
            domain=result.get("domain", ""),
            timestamp=datetime.now().isoformat(),
            metadata=metadata,
            chunks=[]  # Will be populated later
        )
        
    except Exception as e:
        log_error_with_traceback(e, f"Error processing result {url}")
        return None

async def web_search(query: str, config: Dict[str, Any]) -> str:
    """Perform web search using multiple fallback strategies."""
    try:
        if not query or not query.strip():
            return "Error: Empty search query"
            
        # Get domain from config and extract key terms
        domain = config.get("domain_name", "general")
        disable_progress = config.get("disable_progress", False)
        
        # Generate diverse search queries using LLM
        search_queries = await generate_diverse_queries(query, domain)
        
        # Initialize URL tracker
        url_tracker = URLTracker()
        
        # Initialize search with multiple fallback engines
        search_engines = [
            {
                "name": "pubmed",
                "url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                "params": {"db": "pubmed", "term": "{query}", "retmax": "10", "format": "json"},
                "rate_limit": 3.0  # Requests per second
            },
            {
                "name": "arxiv",
                "url": "http://export.arxiv.org/api/query",
                "params": {"search_query": "{query}", "max_results": "10"},
                "rate_limit": 1.0  # Requests per second
            }
        ]
        
        formatted_results = []
        web_contents = []
        
        # Process queries in parallel with rate limiting
        async with SimpleTextBrowser() as browser:
            semaphore = asyncio.Semaphore(2)  # Limit concurrent requests
            
            async def process_query(query: str, query_group: str) -> List[WebContent]:
                async with semaphore:
                    try:
                        results = []
                        # Try each search engine with rate limiting
                        for engine in search_engines:
                            try:
                                await asyncio.sleep(1.0 / engine["rate_limit"])  # Rate limiting
                                engine_results = await _search_with_engine(engine, query)
                                if engine_results:
                                    results.extend(engine_results)
                            except Exception as e:
                                log_error_with_traceback(e, f"Error with {engine['name']}")
                                continue
                        
                        # Process results in parallel with rate limiting
                        tasks = []
                        for result in results[:10]:
                            # Check URL uniqueness
                            url = result.get("link", "").strip()
                            if not url or not url_tracker.add_url(url, query_group):
                                continue
                                
                            # Add delay between requests
                            await asyncio.sleep(0.5)  # Rate limiting for content fetching
                            tasks.append(fetch_content(browser, result, query, query_group))
                            
                        # Wait for all content fetching to complete
                        contents = await asyncio.gather(*tasks)
                        return [c for c in contents if c is not None]
                        
                    except Exception as e:
                        log_error_with_traceback(e, f"Error processing query: {query}")
                        return []
            
            # Process queries with tqdm progress
            tasks = []
            if disable_progress:
                for query, group in search_queries:
                    tasks.append(process_query(query, group))
            else:
                for query, group in tqdm(search_queries, desc="Processing search queries"):
                    tasks.append(process_query(query, group))
                
            # Wait for all queries to complete
            all_results = await asyncio.gather(*tasks)
            
            # Flatten results and filter duplicates
            for results in all_results:
                for content in results:
                    if url_tracker.add_content(content.content):
                        web_contents.append(content)
            
            # Save results to files with progress bar
            if disable_progress:
                for i, content in enumerate(web_contents, 1):
                    filename = f"web/source_{i}.txt"
                    os.makedirs("web", exist_ok=True)  # Ensure directory exists
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(f"title: {content.title}\n")
                        f.write(f"url: {content.url}\n")
                        f.write(f"query: {content.metadata.query}\n")
                        f.write(f"query_group: {content.metadata.query_group}\n")
                        f.write(f"source: {content.metadata.source}\n")
                        f.write(f"published_date: {content.metadata.published_date or ''}\n")
                        f.write(f"timestamp: {content.timestamp}\n")
                        f.write("---\n")
                        f.write(content.content)
                    
                    formatted_results.append(str(content.model_dump()))
                    formatted_results.append("---")
            else:
                for i, content in enumerate(tqdm(web_contents, desc="Saving results"), 1):
                    filename = f"web/source_{i}.txt"
                    os.makedirs("web", exist_ok=True)  # Ensure directory exists
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(f"title: {content.title}\n")
                        f.write(f"url: {content.url}\n")
                        f.write(f"query: {content.metadata.query}\n")
                        f.write(f"query_group: {content.metadata.query_group}\n")
                        f.write(f"source: {content.metadata.source}\n")
                        f.write(f"published_date: {content.metadata.published_date or ''}\n")
                        f.write(f"timestamp: {content.timestamp}\n")
                        f.write("---\n")
                        f.write(content.content)
                    
                    formatted_results.append(str(content.model_dump()))
                    formatted_results.append("---")
                
        return "\n".join(formatted_results) if formatted_results else "No results found"
            
    except Exception as e:
        log_error_with_traceback(e, "Error in web search")
        return f"Search failed: {str(e)}"

async def _search_with_engine(engine: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    """Search using a specific engine with retries and error handling."""
    try:
        # Format query for engine
        params = engine["params"].copy()
        for k, v in params.items():
            if isinstance(v, str):
                params[k] = v.format(query=query)
                
        # Configure SSL verification
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        # Configure connection pooling and retries
        connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            limit=10,  # Connection pool limit
            force_close=False,  # Keep connections alive
            enable_cleanup_closed=True  # Clean up closed connections
        )
        
        # Configure timeout and retry settings
        timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=10)
        
        # Make request with retries
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            raise_for_status=True
        ) as session:
            for attempt in range(3):
                try:
                    async with session.get(engine["url"], params=params) as response:
                        if response.status == 200:
                            data = await response.text()
                            # Parse response based on engine
                            if engine["name"] == "pubmed":
                                return _parse_pubmed_response(data)
                            elif engine["name"] == "arxiv":
                                return _parse_arxiv_response(data)
                        # Rate limiting
                        await asyncio.sleep(1.0 / engine.get("rate_limit", 1.0))
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        log_error_with_traceback(e, f"Error searching {engine['name']}")
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                    continue
        return []
    except Exception as e:
        log_error_with_traceback(e, f"Error in {engine['name']} search")
        return []

def _parse_pubmed_response(data: str) -> List[Dict[str, Any]]:
    """Parse PubMed API response."""
    try:
        results = []
        # Parse JSON response
        response = json.loads(data)
        
        # Extract article IDs
        if "esearchresult" in response and "idlist" in response["esearchresult"]:
            for pmid in response["esearchresult"]["idlist"]:
                results.append({
                    "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "title": f"PubMed Article {pmid}",
                    "snippet": "",
                    "source": "pubmed"
                })
        return results
    except Exception as e:
        log_error_with_traceback(e, "Error parsing PubMed response")
        return []

def _parse_arxiv_response(data: str) -> List[Dict[str, Any]]:
    """Parse arXiv API response."""
    try:
        results = []
        root = ET.fromstring(data)
        for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
            title = entry.find(".//{http://www.w3.org/2005/Atom}title")
            link = entry.find(".//{http://www.w3.org/2005/Atom}id")
            summary = entry.find(".//{http://www.w3.org/2005/Atom}summary")
            if title is not None and link is not None:
                results.append({
                    "link": link.text,
                    "title": title.text,
                    "snippet": summary.text if summary is not None else "",
                    "source": "arxiv"
                })
        return results
    except Exception as e:
        log_error_with_traceback(e, "Error parsing arXiv response")
        return []

async def _fetch_url_content(session: aiohttp.ClientSession, url: str) -> str:
    """Base function to fetch content from a URL."""
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return ""

async def _fetch_url_content_with_ssl(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch content from a URL with SSL verification disabled."""
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=connector) as ssl_session:
        try:
            async with ssl_session.get(url, timeout=timeout) as response:
                return await response.text()
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return ""

async def fetch_url_content(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch content from a URL, handling both regular and SSL requests."""
    if "pubmed.ncbi.nlm.nih.gov" in url:
        return await _fetch_url_content_with_ssl(session, url)
    return await _fetch_url_content(session, url)

async def process_search_query(session: aiohttp.ClientSession, query: str, engine: str = "google") -> List[Dict[str, str]]:
    """Process a search query and return results."""
    try:
        if engine == "google":
            results = await google_search(session, query)
        elif engine == "arxiv":
            results = await arxiv_search(session, query)
        else:
            results = []
        
        web_contents = []
        for result in results:
            content = await fetch_url_content(session, result["url"])
            if content:
                web_contents.append({
                    "title": result.get("title", ""),
                    "url": result["url"],
                    "content": content
                })
        return web_contents
    except Exception as e:
        print(f"Error processing query {query}: {str(e)}")
        return []

async def google_search(session: aiohttp.ClientSession, query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """Perform a Google search and return results."""
    try:
        service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_API_KEY"))
        result = service.cse().list(q=query, cx=os.getenv("GOOGLE_CSE_ID"), num=num_results).execute()
        
        search_results = []
        for item in result.get("items", []):
            search_results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", "")
            })
        return search_results
    except HttpError as e:
        if e.resp.status == 429:  # Rate limit exceeded
            print("Google API rate limit exceeded. Waiting before retrying...")
            await asyncio.sleep(60)  # Wait for 60 seconds
            return await google_search(session, query, num_results)
        else:
            print(f"Google search error: {str(e)}")
            return []
    except Exception as e:
        print(f"Error in Google search: {str(e)}")
        return []

async def arxiv_search(session: aiohttp.ClientSession, query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """Perform an arXiv search and return results."""
    try:
        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": query,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        async with session.get(base_url, params=params) as response:
            content = await response.text()
            
            # Parse XML response
            results = []
            # Add XML parsing logic here if needed
            return results
    except Exception as e:
        print(f"Error in arXiv search: {str(e)}")
        return [] 