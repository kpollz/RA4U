# RA4U - Search Agent
# Specialized agent for discovering relevant academic papers

from agno.agent import Agent
from agno.tools.arxiv import ArxivTools
from agno.tools.duckduckgo import DuckDuckGoTools
from typing import List, Dict, Any, Optional
import logging

from models import Paper, ResearchQuery, AgentConfig
from utils import log_agent_activity, calculate_similarity_score, extract_keywords

logger = logging.getLogger(__name__)

class SearchAgent:
    """Search Agent for discovering relevant academic papers"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the search agent with appropriate model and tools"""
        if self.config.model_provider == "OpenAI":
            from agno.models.openai import OpenAIChat
            model = OpenAIChat(
                id=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                api_key=self.config.api_key
            )
        elif self.config.model_provider == "Google":
            from agno.models.google import Gemini
            model = Gemini(
                id=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                api_key=self.config.api_key
            )
        else:
            raise ValueError(f"Unsupported model provider: {self.config.model_provider}")
        
        return Agent(
            name="Search Agent",
            model=model,
            role="""You are a specialized academic search agent. Your role is to:
            1. Search for relevant academic papers using arXiv and other academic databases
            2. Filter papers based on relevance to the research topic
            3. Rank papers by quality metrics (citations, venue, recency)
            4. Provide structured paper information including title, authors, abstract, and metadata
            """,
            tools=[ArxivTools(), DuckDuckGoTools()],
            instructions=[
                "When given a research topic, search for the most relevant academic papers",
                "Focus on recent papers (last 3 years) with high citation counts",
                "Prioritize papers from top-tier conferences and journals",
                "Provide detailed abstracts and author information",
                "Rank papers by relevance score (0-1 scale)",
                "Use arXiv as primary source, DuckDuckGo for additional verification",
                "Extract key information: title, authors, abstract, publication date, arXiv ID, URL"
            ]
        )
    
    async def search_papers(self, query: ResearchQuery) -> List[Paper]:
        """Search for relevant academic papers based on the research query"""
        try:
            log_agent_activity("Search Agent", "Starting paper search", {
                "topic": query.topic,
                "domain": query.domain,
                "max_papers": query.max_papers
            })
            
            # Create search prompt
            search_prompt = self._create_search_prompt(query)
            
            # Run the search agent
            response = self.agent.run(search_prompt, stream=False)
            
            # Parse response and extract papers
            papers = self._parse_search_results(response.content, query)
            
            log_agent_activity("Search Agent", "Search completed", {
                "papers_found": len(papers),
                "topic": query.topic
            })
            
            return papers
            
        except Exception as e:
            logger.error(f"Search Agent error: {str(e)}")
            log_agent_activity("Search Agent", "Search failed", {
                "error": str(e),
                "topic": query.topic
            })
            return []
    
    def _create_search_prompt(self, query: ResearchQuery) -> str:
        """Create search prompt for the agent"""
        return f"""
        Search for academic papers with the following criteria:
        
        Research Topic: {query.topic}
        Domain: {query.domain}
        Maximum Papers: {query.max_papers}
        Minimum Citations: {query.min_citations}
        Date Range: {query.date_range}
        
        Please search for papers using arXiv and provide the following information for each paper:
        1. Title
        2. Authors (list all authors)
        3. Abstract (full abstract)
        4. Publication date
        5. arXiv ID
        6. URL
        7. DOI (if available)
        8. Venue/Journal (if available)
        9. Citation count (if available)
        10. Keywords (extract from abstract)
        
        Focus on:
        - Recent papers (preferably within the specified date range)
        - High-quality papers from reputable sources
        - Papers directly relevant to the research topic
        - Papers with good citation counts
        
        Rank the papers by relevance and quality.
        """
    
    def _parse_search_results(self, response_content: str, query: ResearchQuery) -> List[Paper]:
        """Parse search results from agent response"""
        papers = []
        
        try:
            # This is a simplified parser - in a real implementation,
            # you would use more sophisticated parsing or structured output
            lines = response_content.split('\n')
            current_paper = {}
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('Title:') or line.startswith('**Title:'):
                    if current_paper:
                        papers.append(self._create_paper_from_dict(current_paper, query))
                    current_paper = {'title': line.replace('Title:', '').replace('**Title:', '').strip()}
                
                elif line.startswith('Authors:') or line.startswith('**Authors:'):
                    authors_text = line.replace('Authors:', '').replace('**Authors:', '').strip()
                    current_paper['authors'] = [author.strip() for author in authors_text.split(',')]
                
                elif line.startswith('Abstract:') or line.startswith('**Abstract:'):
                    abstract_text = line.replace('Abstract:', '').replace('**Abstract:', '').strip()
                    current_paper['abstract'] = abstract_text
                
                elif line.startswith('Published:') or line.startswith('**Published:'):
                    current_paper['published'] = line.replace('Published:', '').replace('**Published:', '').strip()
                
                elif line.startswith('arXiv ID:') or line.startswith('**arXiv ID:'):
                    current_paper['arxiv_id'] = line.replace('arXiv ID:', '').replace('**arXiv ID:', '').strip()
                
                elif line.startswith('URL:') or line.startswith('**URL:'):
                    current_paper['url'] = line.replace('URL:', '').replace('**URL:', '').strip()
                
                elif line.startswith('DOI:') or line.startswith('**DOI:'):
                    current_paper['doi'] = line.replace('DOI:', '').replace('**DOI:', '').strip()
                
                elif line.startswith('Venue:') or line.startswith('**Venue:'):
                    current_paper['venue'] = line.replace('Venue:', '').replace('**Venue:', '').strip()
                
                elif line.startswith('Citations:') or line.startswith('**Citations:'):
                    try:
                        current_paper['citation_count'] = int(line.replace('Citations:', '').replace('**Citations:', '').strip())
                    except ValueError:
                        current_paper['citation_count'] = 0
                
                elif line.startswith('Keywords:') or line.startswith('**Keywords:'):
                    keywords_text = line.replace('Keywords:', '').replace('**Keywords:', '').strip()
                    current_paper['keywords'] = [kw.strip() for kw in keywords_text.split(',')]
            
            # Add the last paper if exists
            if current_paper:
                papers.append(self._create_paper_from_dict(current_paper, query))
            
            # Limit to requested number of papers
            papers = papers[:query.max_papers]
            
            # Calculate relevance scores
            for paper in papers:
                paper.relevance_score = self._calculate_relevance_score(paper, query)
            
            # Sort by relevance score
            papers.sort(key=lambda x: x.relevance_score, reverse=True)
            
        except Exception as e:
            logger.error(f"Error parsing search results: {str(e)}")
        
        return papers
    
    def _create_paper_from_dict(self, paper_dict: Dict[str, Any], query: ResearchQuery) -> Paper:
        """Create Paper object from dictionary"""
        return Paper(
            title=paper_dict.get('title', 'Unknown Title'),
            authors=paper_dict.get('authors', []),
            abstract=paper_dict.get('abstract', ''),
            published=paper_dict.get('published', ''),
            arxiv_id=paper_dict.get('arxiv_id', ''),
            url=paper_dict.get('url', ''),
            doi=paper_dict.get('doi'),
            venue=paper_dict.get('venue'),
            citation_count=paper_dict.get('citation_count', 0),
            keywords=paper_dict.get('keywords', [])
        )
    
    def _calculate_relevance_score(self, paper: Paper, query: ResearchQuery) -> float:
        """Calculate relevance score for a paper"""
        if not paper.abstract:
            return 0.0
        
        # Calculate topic similarity
        topic_similarity = calculate_similarity_score(paper.abstract, query.topic)
        
        # Calculate keyword overlap
        paper_keywords = set(extract_keywords(paper.abstract))
        query_keywords = set(extract_keywords(query.topic))
        
        if query_keywords:
            keyword_overlap = len(paper_keywords.intersection(query_keywords)) / len(query_keywords)
        else:
            keyword_overlap = 0.0
        
        # Citation score (normalized)
        citation_score = min(paper.citation_count / 100, 1.0) if paper.citation_count else 0.0
        
        # Combine scores with weights
        relevance_score = (
            topic_similarity * 0.5 +
            keyword_overlap * 0.3 +
            citation_score * 0.2
        )
        
        return min(relevance_score, 1.0)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the search agent"""
        return {
            "name": "Search Agent",
            "role": "Academic paper discovery and filtering",
            "tools": ["arXiv API", "DuckDuckGo Search"],
            "capabilities": [
                "Search academic databases",
                "Filter by relevance and quality",
                "Extract paper metadata",
                "Calculate relevance scores",
                "Rank papers by importance"
            ]
        }
