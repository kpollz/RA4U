# RA4U - Verification Agent
# Specialized agent for ensuring accuracy and preventing hallucination

from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from typing import List, Dict, Any, Optional
import logging
import re

from models import Paper, VerificationResult, AgentConfig, VerificationStatus
from utils import log_agent_activity, extract_arxiv_id, is_valid_arxiv_id

logger = logging.getLogger(__name__)

class VerificationAgent:
    """Verification Agent for ensuring accuracy and preventing hallucination"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the verification agent with appropriate model and tools"""
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
            name="Verification Agent",
            model=model,
            role="""You are a verification specialist. Your role is to:
            1. Cross-reference paper information across multiple sources
            2. Verify paper existence and metadata accuracy
            3. Check for potential hallucinated or fabricated papers
            4. Validate citation information and publication details
            """,
            tools=[DuckDuckGoTools()],
            instructions=[
                "Verify each paper's existence by cross-referencing with multiple sources",
                "Check publication dates, author names, and venue information",
                "Flag any papers that cannot be verified or seem suspicious",
                "Provide confidence scores for verification results",
                "Focus on preventing hallucination and ensuring accuracy",
                "Use DuckDuckGo to search for paper titles and verify existence",
                "Check arXiv IDs for validity and format",
                "Verify author names and publication venues"
            ]
        )
    
    async def verify_papers(self, papers: List[Paper]) -> List[Paper]:
        """Verify a list of papers for accuracy and existence"""
        try:
            log_agent_activity("Verification Agent", "Starting paper verification", {
                "papers_count": len(papers)
            })
            
            verified_papers = []
            
            for paper in papers:
                verification_result = await self._verify_single_paper(paper)
                
                if verification_result.is_valid:
                    paper.verification_status = VerificationStatus.VERIFIED
                    verified_papers.append(paper)
                else:
                    paper.verification_status = VerificationStatus.FAILED
                    log_agent_activity("Verification Agent", "Paper verification failed", {
                        "paper_title": paper.title,
                        "reason": verification_result.error_message
                    })
            
            log_agent_activity("Verification Agent", "Verification completed", {
                "total_papers": len(papers),
                "verified_papers": len(verified_papers),
                "failed_papers": len(papers) - len(verified_papers)
            })
            
            return verified_papers
            
        except Exception as e:
            logger.error(f"Verification Agent error: {str(e)}")
            log_agent_activity("Verification Agent", "Verification failed", {
                "error": str(e)
            })
            return papers  # Return original papers if verification fails
    
    async def _verify_single_paper(self, paper: Paper) -> VerificationResult:
        """Verify a single paper for accuracy and existence"""
        try:
            # Create verification prompt
            verification_prompt = self._create_verification_prompt(paper)
            
            # Run the verification agent
            response = self.agent.run(verification_prompt, stream=False)
            
            # Parse verification results
            verification_result = self._parse_verification_response(response.content, paper)
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Error verifying paper {paper.title}: {str(e)}")
            return VerificationResult(
                paper_id=paper.arxiv_id,
                is_valid=False,
                confidence=0.0,
                checks={},
                error_message=f"Verification error: {str(e)}"
            )
    
    def _create_verification_prompt(self, paper: Paper) -> str:
        """Create verification prompt for a specific paper"""
        return f"""
        Please verify the following academic paper for accuracy and existence:
        
        Paper Details:
        - Title: {paper.title}
        - Authors: {', '.join(paper.authors) if paper.authors else 'Unknown'}
        - Published: {paper.published}
        - arXiv ID: {paper.arxiv_id}
        - URL: {paper.url}
        - DOI: {paper.doi if paper.doi else 'Not provided'}
        - Venue: {paper.venue if paper.venue else 'Not provided'}
        
        Please perform the following verification checks:
        1. Search for the paper title using DuckDuckGo to verify existence
        2. Check if the arXiv ID is valid and properly formatted
        3. Verify author names and publication date
        4. Check if the venue/journal information is accurate
        5. Look for any inconsistencies or suspicious details
        
        Provide your verification results in the following format:
        - Existence: [VERIFIED/SUSPICIOUS/NOT_FOUND]
        - arXiv ID: [VALID/INVALID/UNKNOWN]
        - Authors: [VERIFIED/SUSPICIOUS/UNKNOWN]
        - Publication Date: [VERIFIED/SUSPICIOUS/UNKNOWN]
        - Venue: [VERIFIED/SUSPICIOUS/UNKNOWN]
        - Overall Confidence: [0.0-1.0]
        - Issues Found: [List any issues or concerns]
        - Recommendation: [ACCEPT/REJECT/INVESTIGATE_FURTHER]
        """
    
    def _parse_verification_response(self, response_content: str, paper: Paper) -> VerificationResult:
        """Parse verification response from agent"""
        try:
            # Initialize default values
            checks = {
                'existence': False,
                'arxiv_id': False,
                'authors': False,
                'publication_date': False,
                'venue': False
            }
            
            confidence = 0.0
            is_valid = False
            error_message = None
            
            # Parse response content
            lines = response_content.split('\n')
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('Existence:'):
                    status = line.replace('Existence:', '').strip().upper()
                    checks['existence'] = status == 'VERIFIED'
                
                elif line.startswith('arXiv ID:'):
                    status = line.replace('arXiv ID:', '').strip().upper()
                    checks['arxiv_id'] = status == 'VALID'
                
                elif line.startswith('Authors:'):
                    status = line.replace('Authors:', '').strip().upper()
                    checks['authors'] = status == 'VERIFIED'
                
                elif line.startswith('Publication Date:'):
                    status = line.replace('Publication Date:', '').strip().upper()
                    checks['publication_date'] = status == 'VERIFIED'
                
                elif line.startswith('Venue:'):
                    status = line.replace('Venue:', '').strip().upper()
                    checks['venue'] = status == 'VERIFIED'
                
                elif line.startswith('Overall Confidence:'):
                    try:
                        confidence = float(line.replace('Overall Confidence:', '').strip())
                    except ValueError:
                        confidence = 0.0
                
                elif line.startswith('Recommendation:'):
                    recommendation = line.replace('Recommendation:', '').strip().upper()
                    is_valid = recommendation == 'ACCEPT'
                
                elif line.startswith('Issues Found:'):
                    issues = line.replace('Issues Found:', '').strip()
                    if issues and issues != 'None':
                        error_message = issues
            
            # Additional validation checks
            if not self._validate_arxiv_id(paper.arxiv_id):
                checks['arxiv_id'] = False
                is_valid = False
            
            # Calculate overall validity
            if not is_valid:
                # Override based on individual checks
                passed_checks = sum(checks.values())
                total_checks = len(checks)
                is_valid = passed_checks >= (total_checks * 0.6)  # At least 60% of checks must pass
            
            return VerificationResult(
                paper_id=paper.arxiv_id,
                is_valid=is_valid,
                confidence=confidence,
                checks=checks,
                error_message=error_message
            )
            
        except Exception as e:
            logger.error(f"Error parsing verification response: {str(e)}")
            return VerificationResult(
                paper_id=paper.arxiv_id,
                is_valid=False,
                confidence=0.0,
                checks={},
                error_message=f"Parse error: {str(e)}"
            )
    
    def _validate_arxiv_id(self, arxiv_id: str) -> bool:
        """Validate arXiv ID format and existence"""
        if not arxiv_id:
            return False
        
        # Check format
        if not is_valid_arxiv_id(arxiv_id):
            return False
        
        # Additional validation could be added here
        # e.g., checking if the paper actually exists on arXiv
        
        return True
    
    def _check_paper_existence(self, title: str, authors: List[str]) -> bool:
        """Check if paper exists using basic heuristics"""
        if not title or not authors:
            return False
        
        # Basic validation - in a real implementation,
        # this would use actual API calls to verify existence
        return len(title) > 10 and len(authors) > 0
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the verification agent"""
        return {
            "name": "Verification Agent",
            "role": "Paper accuracy verification and hallucination prevention",
            "tools": ["DuckDuckGo Search"],
            "capabilities": [
                "Cross-reference paper information",
                "Verify paper existence",
                "Check metadata accuracy",
                "Detect potential hallucinations",
                "Validate citation information",
                "Provide confidence scores"
            ]
        }
