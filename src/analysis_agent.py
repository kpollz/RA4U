# RA4U - Analysis Agent
# Specialized agent for processing and analyzing academic papers

from agno.agent import Agent
from typing import List, Dict, Any, Optional
import logging

from models import Paper, AnalysisResult, AgentConfig
from utils import log_agent_activity, extract_keywords, calculate_similarity_score

logger = logging.getLogger(__name__)

class AnalysisAgent:
    """Analysis Agent for processing and analyzing academic papers"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the analysis agent with appropriate model and tools"""
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
            name="Analysis Agent",
            model=model,
            role="""You are an academic analysis expert. Your role is to:
            1. Extract key concepts and methodologies from papers
            2. Identify research contributions and innovations
            3. Analyze paper structure and content quality
            4. Calculate semantic similarity scores with the research topic
            """,
            instructions=[
                "Analyze each paper's abstract and content for key concepts",
                "Identify the main research contributions and methodologies",
                "Extract important keywords and technical terms",
                "Calculate how well each paper matches the research topic",
                "Provide structured analysis results for each paper",
                "Focus on methodological approaches and novel contributions",
                "Assess the quality and rigor of the research",
                "Identify the paper's place in the research landscape"
            ]
        )
    
    async def analyze_papers(self, papers: List[Paper], research_topic: str) -> List[AnalysisResult]:
        """Analyze a list of papers for content and contributions"""
        try:
            log_agent_activity("Analysis Agent", "Starting paper analysis", {
                "papers_count": len(papers),
                "research_topic": research_topic
            })
            
            analysis_results = []
            
            for paper in papers:
                analysis_result = await self._analyze_single_paper(paper, research_topic)
                analysis_results.append(analysis_result)
            
            log_agent_activity("Analysis Agent", "Analysis completed", {
                "papers_analyzed": len(analysis_results),
                "research_topic": research_topic
            })
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Analysis Agent error: {str(e)}")
            log_agent_activity("Analysis Agent", "Analysis failed", {
                "error": str(e),
                "research_topic": research_topic
            })
            return []
    
    async def _analyze_single_paper(self, paper: Paper, research_topic: str) -> AnalysisResult:
        """Analyze a single paper for content and contributions"""
        try:
            # Create analysis prompt
            analysis_prompt = self._create_analysis_prompt(paper, research_topic)
            
            # Run the analysis agent
            response = self.agent.run(analysis_prompt, stream=False)
            
            # Parse analysis results
            analysis_result = self._parse_analysis_response(response.content, paper, research_topic)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing paper {paper.title}: {str(e)}")
            return AnalysisResult(
                paper_id=paper.arxiv_id,
                key_concepts=[],
                methodology="",
                contributions=[],
                limitations=[],
                relevance_score=0.0,
                quality_score=0.0
            )
    
    def _create_analysis_prompt(self, paper: Paper, research_topic: str) -> str:
        """Create analysis prompt for a specific paper"""
        return f"""
        Please analyze the following academic paper in detail:
        
        Research Topic: {research_topic}
        
        Paper Details:
        - Title: {paper.title}
        - Authors: {', '.join(paper.authors) if paper.authors else 'Unknown'}
        - Abstract: {paper.abstract}
        - Published: {paper.published}
        - Venue: {paper.venue if paper.venue else 'Not specified'}
        - Citation Count: {paper.citation_count}
        
        Please provide a comprehensive analysis including:
        
        1. KEY CONCEPTS: Extract the main concepts, theories, and technical terms used in this paper
        2. METHODOLOGY: Describe the research methodology, approach, and techniques used
        3. CONTRIBUTIONS: List the main contributions and innovations of this paper
        4. LIMITATIONS: Identify any limitations or constraints mentioned in the paper
        5. RELEVANCE: Assess how relevant this paper is to the research topic "{research_topic}"
        6. QUALITY: Evaluate the overall quality and rigor of the research
        
        Provide your analysis in the following format:
        - Key Concepts: [List of main concepts, one per line]
        - Methodology: [Description of research methodology]
        - Contributions: [List of main contributions, one per line]
        - Limitations: [List of limitations mentioned, one per line]
        - Relevance Score: [0.0-1.0 score for relevance to research topic]
        - Quality Score: [0.0-1.0 score for overall research quality]
        - Summary: [Brief summary of the paper's significance]
        """
    
    def _parse_analysis_response(self, response_content: str, paper: Paper, research_topic: str) -> AnalysisResult:
        """Parse analysis response from agent"""
        try:
            # Initialize default values
            key_concepts = []
            methodology = ""
            contributions = []
            limitations = []
            relevance_score = 0.0
            quality_score = 0.0
            
            # Parse response content
            lines = response_content.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('Key Concepts:') or line.startswith('**Key Concepts:'):
                    current_section = 'concepts'
                    continue
                elif line.startswith('Methodology:') or line.startswith('**Methodology:'):
                    current_section = 'methodology'
                    methodology = line.replace('Methodology:', '').replace('**Methodology:', '').strip()
                    continue
                elif line.startswith('Contributions:') or line.startswith('**Contributions:'):
                    current_section = 'contributions'
                    continue
                elif line.startswith('Limitations:') or line.startswith('**Limitations:'):
                    current_section = 'limitations'
                    continue
                elif line.startswith('Relevance Score:') or line.startswith('**Relevance Score:'):
                    try:
                        relevance_score = float(line.replace('Relevance Score:', '').replace('**Relevance Score:', '').strip())
                    except ValueError:
                        relevance_score = 0.0
                    continue
                elif line.startswith('Quality Score:') or line.startswith('**Quality Score:'):
                    try:
                        quality_score = float(line.replace('Quality Score:', '').replace('**Quality Score:', '').strip())
                    except ValueError:
                        quality_score = 0.0
                    continue
                
                # Process content based on current section
                if current_section == 'concepts' and line and not line.startswith('-'):
                    key_concepts.append(line.strip('- '))
                elif current_section == 'contributions' and line and not line.startswith('-'):
                    contributions.append(line.strip('- '))
                elif current_section == 'limitations' and line and not line.startswith('-'):
                    limitations.append(line.strip('- '))
            
            # Calculate additional relevance score if not provided
            if relevance_score == 0.0:
                relevance_score = self._calculate_relevance_score(paper, research_topic)
            
            # Calculate quality score if not provided
            if quality_score == 0.0:
                quality_score = self._calculate_quality_score(paper, key_concepts, contributions)
            
            return AnalysisResult(
                paper_id=paper.arxiv_id,
                key_concepts=key_concepts,
                methodology=methodology,
                contributions=contributions,
                limitations=limitations,
                relevance_score=relevance_score,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"Error parsing analysis response: {str(e)}")
            return AnalysisResult(
                paper_id=paper.arxiv_id,
                key_concepts=[],
                methodology="",
                contributions=[],
                limitations=[],
                relevance_score=0.0,
                quality_score=0.0
            )
    
    def _calculate_relevance_score(self, paper: Paper, research_topic: str) -> float:
        """Calculate relevance score for a paper"""
        if not paper.abstract or not research_topic:
            return 0.0
        
        # Calculate similarity between abstract and research topic
        similarity = calculate_similarity_score(paper.abstract, research_topic)
        
        # Boost score for papers with high citation counts
        citation_boost = min(paper.citation_count / 100, 0.2) if paper.citation_count else 0.0
        
        # Combine similarity and citation boost
        relevance_score = min(similarity + citation_boost, 1.0)
        
        return relevance_score
    
    def _calculate_quality_score(self, paper: Paper, key_concepts: List[str], contributions: List[str]) -> float:
        """Calculate quality score for a paper"""
        quality_factors = []
        
        # Abstract quality (length and content)
        if paper.abstract:
            abstract_quality = min(len(paper.abstract) / 500, 1.0)  # Normalize by expected length
            quality_factors.append(abstract_quality)
        
        # Citation count
        if paper.citation_count:
            citation_quality = min(paper.citation_count / 50, 1.0)  # Normalize by expected citations
            quality_factors.append(citation_quality)
        
        # Key concepts richness
        if key_concepts:
            concept_quality = min(len(key_concepts) / 10, 1.0)  # Normalize by expected concepts
            quality_factors.append(concept_quality)
        
        # Contributions richness
        if contributions:
            contribution_quality = min(len(contributions) / 5, 1.0)  # Normalize by expected contributions
            quality_factors.append(contribution_quality)
        
        # Venue quality (if available)
        if paper.venue:
            venue_quality = self._assess_venue_quality(paper.venue)
            quality_factors.append(venue_quality)
        
        # Calculate average quality score
        if quality_factors:
            return sum(quality_factors) / len(quality_factors)
        else:
            return 0.5  # Default medium quality
    
    def _assess_venue_quality(self, venue: str) -> float:
        """Assess the quality of the publication venue"""
        if not venue:
            return 0.5
        
        venue_lower = venue.lower()
        
        # High-quality venues
        high_quality_keywords = [
            'nature', 'science', 'cell', 'lancet', 'nejm', 'ieee', 'acm', 'springer',
            'elsevier', 'plos', 'pnas', 'jama', 'bmj', 'nejm'
        ]
        
        # Medium-quality venues
        medium_quality_keywords = [
            'conference', 'symposium', 'workshop', 'proceedings', 'journal',
            'transactions', 'letters', 'communications'
        ]
        
        for keyword in high_quality_keywords:
            if keyword in venue_lower:
                return 0.9
        
        for keyword in medium_quality_keywords:
            if keyword in venue_lower:
                return 0.7
        
        return 0.5  # Default medium quality
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the analysis agent"""
        return {
            "name": "Analysis Agent",
            "role": "Academic paper content analysis and evaluation",
            "tools": ["Text Analysis", "Concept Extraction"],
            "capabilities": [
                "Extract key concepts and methodologies",
                "Identify research contributions",
                "Analyze paper quality and rigor",
                "Calculate relevance scores",
                "Assess research significance",
                "Evaluate methodological approaches"
            ]
        }
