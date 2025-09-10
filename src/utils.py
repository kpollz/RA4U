# RA4U - Utility Functions
# Helper functions for the research assistant system

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for the RA4U system"""
    # Model configuration
    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    
    # Research parameters
    DEFAULT_MAX_PAPERS: int = 10
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.7
    DEFAULT_MIN_CITATIONS: int = 5
    
    # Timeout settings
    AGENT_TIMEOUT: int = 300
    REQUEST_TIMEOUT: int = 30
    
    # Cache settings
    CACHE_TTL: int = 3600  # 1 hour
    
    def __post_init__(self):
        """Load configuration from environment variables"""
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def setup_environment(api_key: str, model_provider: str) -> None:
    """Setup environment variables for API access"""
    if model_provider == "OpenAI":
        os.environ["OPENAI_API_KEY"] = api_key
    elif model_provider == "Google":
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")

def validate_api_key(api_key: str, model_provider: str) -> bool:
    """Validate API key format"""
    if not api_key or not isinstance(api_key, str):
        return False
    
    if model_provider == "OpenAI":
        return api_key.startswith("sk-") and len(api_key) > 20
    elif model_provider == "Google":
        return len(api_key) > 20
    else:
        return False

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
    
    return text.strip()

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using simple heuristics"""
    if not text:
        return []
    
    # Convert to lowercase and split
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove common stop words
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
        'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'can', 'shall', 'a', 'an', 'some', 'any', 'all', 'both',
        'each', 'every', 'few', 'many', 'much', 'more', 'most', 'other',
        'another', 'such', 'no', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 'just', 'now', 'here', 'there', 'when', 'where', 'why',
        'how', 'what', 'which', 'who', 'whom', 'whose', 'if', 'then', 'else',
        'because', 'as', 'until', 'while', 'whereas', 'although', 'though',
        'unless', 'since', 'once', 'twice', 'always', 'never', 'sometimes',
        'often', 'usually', 'rarely', 'seldom', 'hardly', 'barely', 'scarcely'
    }
    
    # Filter out stop words and count frequency
    word_freq = {}
    for word in words:
        if word not in stop_words and len(word) > 3:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_keywords]]

def calculate_similarity_score(text1: str, text2: str) -> float:
    """Calculate simple similarity score between two texts"""
    if not text1 or not text2:
        return 0.0
    
    # Extract keywords from both texts
    keywords1 = set(extract_keywords(text1))
    keywords2 = set(extract_keywords(text2))
    
    if not keywords1 or not keywords2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(keywords1.intersection(keywords2))
    union = len(keywords1.union(keywords2))
    
    return intersection / union if union > 0 else 0.0

def format_citation(paper_title: str, authors: List[str], year: str, venue: str = None) -> str:
    """Format paper citation in academic style"""
    if not authors:
        authors_str = "Unknown Authors"
    elif len(authors) == 1:
        authors_str = authors[0]
    elif len(authors) <= 3:
        authors_str = ", ".join(authors[:-1]) + f" and {authors[-1]}"
    else:
        authors_str = f"{authors[0]} et al."
    
    citation = f"{authors_str} ({year}). {paper_title}."
    
    if venue:
        citation += f" {venue}."
    
    return citation

def parse_date_range(date_range: str) -> Tuple[datetime, datetime]:
    """Parse date range string to datetime objects"""
    now = datetime.now()
    
    if date_range == "Last 2 years":
        start_date = now - timedelta(days=730)
    elif date_range == "Last 5 years":
        start_date = now - timedelta(days=1825)
    elif date_range == "All time":
        start_date = datetime(2000, 1, 1)  # Arbitrary start date
    else:
        start_date = now - timedelta(days=730)  # Default to 2 years
    
    return start_date, now

def validate_research_query(topic: str, max_papers: int, min_citations: int) -> Tuple[bool, str]:
    """Validate research query parameters"""
    if not topic or len(topic.strip()) < 3:
        return False, "Research topic must be at least 3 characters long"
    
    if max_papers < 1 or max_papers > 50:
        return False, "Maximum papers must be between 1 and 50"
    
    if min_citations < 0:
        return False, "Minimum citations cannot be negative"
    
    return True, "Valid query"

def create_research_prompt(topic: str, domain: str, max_papers: int, 
                          min_citations: int, date_range: str) -> str:
    """Create comprehensive research prompt for agents"""
    return f"""
    Research Topic: {topic}
    Domain: {domain}
    Maximum Papers: {max_papers}
    Minimum Citations: {min_citations}
    Date Range: {date_range}
    
    Please conduct a comprehensive research analysis following these steps:
    
    1. SEARCH: Find {max_papers} most relevant academic papers on "{topic}" in the {domain} domain
    2. VERIFY: Verify the accuracy and existence of each paper
    3. ANALYZE: Analyze each paper's content, contributions, and relevance
    4. LIMITATIONS: Identify specific limitations in each paper
    5. GAPS: Synthesize limitations to identify research gaps and opportunities
    
    Provide a structured report with:
    - List of verified papers with metadata
    - Analysis of each paper's contributions
    - Identified limitations categorized by type
    - Research gaps and future opportunities
    - Recommendations for future research directions
    """

def log_agent_activity(agent_name: str, activity: str, details: Dict[str, Any] = None) -> None:
    """Log agent activity for debugging and monitoring"""
    log_data = {
        "agent": agent_name,
        "activity": activity,
        "timestamp": datetime.now().isoformat(),
        "details": details or {}
    }
    
    logger.info(f"Agent Activity: {json.dumps(log_data, indent=2)}")

def create_error_response(error_message: str, error_type: str = "general") -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "success": False,
        "error_type": error_type,
        "error_message": error_message,
        "timestamp": datetime.now().isoformat()
    }

def create_success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
    """Create standardized success response"""
    return {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }

def format_paper_list(papers: List[Dict[str, Any]]) -> str:
    """Format list of papers for display"""
    if not papers:
        return "No papers found."
    
    formatted_papers = []
    for i, paper in enumerate(papers, 1):
        title = paper.get('title', 'Unknown Title')
        authors = paper.get('authors', [])
        year = paper.get('published', 'Unknown Year')
        
        authors_str = ", ".join(authors[:3])
        if len(authors) > 3:
            authors_str += " et al."
        
        formatted_papers.append(f"{i}. {title} - {authors_str} ({year})")
    
    return "\n".join(formatted_papers)

def extract_arxiv_id(url: str) -> Optional[str]:
    """Extract arXiv ID from URL"""
    if not url:
        return None
    
    # Pattern for arXiv URLs
    patterns = [
        r'arxiv\.org/abs/(\d+\.\d+)',
        r'arxiv\.org/pdf/(\d+\.\d+)',
        r'arxiv\.org/e-print/(\d+\.\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def is_valid_arxiv_id(arxiv_id: str) -> bool:
    """Validate arXiv ID format"""
    if not arxiv_id:
        return False
    
    # arXiv ID pattern: YYMM.NNNNN
    pattern = r'^\d{4}\.\d{4,5}$'
    return bool(re.match(pattern, arxiv_id))

def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to specified length with ellipsis"""
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."

def create_progress_message(stage: str, progress: int, total: int) -> str:
    """Create progress message for UI"""
    percentage = (progress / total) * 100 if total > 0 else 0
    return f"{stage}: {progress}/{total} ({percentage:.1f}%)"

# Configuration instance
config = Config()
