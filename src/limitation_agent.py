# RA4U - Limitation Agent
# Specialized agent for identifying research limitations

from agno.agent import Agent
from typing import List, Dict, Any, Optional
import logging

from models import Paper, Limitation, LimitationType, AgentConfig, AnalysisResult
from utils import log_agent_activity

logger = logging.getLogger(__name__)

class LimitationAgent:
    """Limitation Agent for identifying research limitations"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the limitation agent with appropriate model and tools"""
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
            name="Limitation Agent",
            model=model,
            role="""You are a research limitation specialist. Your role is to:
            1. Identify methodological limitations in research papers
            2. Detect scope constraints and experimental limitations
            3. Categorize different types of limitations
            4. Assess the impact of limitations on research validity
            """,
            instructions=[
                "Carefully read each paper to identify research limitations",
                "Categorize limitations into: methodological, scope, data, evaluation, generalization",
                "Provide specific examples of limitations found in each paper",
                "Assess the confidence level for each limitation identified",
                "Focus on limitations that could lead to research gaps",
                "Look for both explicit and implicit limitations",
                "Consider limitations in experimental design, data collection, and analysis",
                "Evaluate the impact of limitations on research conclusions"
            ]
        )
    
    async def identify_limitations(self, papers: List[Paper], analysis_results: List[AnalysisResult]) -> List[Limitation]:
        """Identify limitations across multiple papers"""
        try:
            log_agent_activity("Limitation Agent", "Starting limitation identification", {
                "papers_count": len(papers),
                "analysis_results_count": len(analysis_results)
            })
            
            all_limitations = []
            
            for i, paper in enumerate(papers):
                # Get corresponding analysis result
                analysis_result = None
                if i < len(analysis_results):
                    analysis_result = analysis_results[i]
                
                # Identify limitations for this paper
                paper_limitations = await self._identify_paper_limitations(paper, analysis_result)
                all_limitations.extend(paper_limitations)
            
            # Categorize and rank limitations
            categorized_limitations = self._categorize_limitations(all_limitations)
            ranked_limitations = self._rank_limitations(categorized_limitations)
            
            log_agent_activity("Limitation Agent", "Limitation identification completed", {
                "total_limitations": len(all_limitations),
                "categorized_limitations": len(categorized_limitations),
                "ranked_limitations": len(ranked_limitations)
            })
            
            return ranked_limitations
            
        except Exception as e:
            logger.error(f"Limitation Agent error: {str(e)}")
            log_agent_activity("Limitation Agent", "Limitation identification failed", {
                "error": str(e)
            })
            return []
    
    async def _identify_paper_limitations(self, paper: Paper, analysis_result: Optional[AnalysisResult] = None) -> List[Limitation]:
        """Identify limitations for a single paper"""
        try:
            # Create limitation identification prompt
            limitation_prompt = self._create_limitation_prompt(paper, analysis_result)
            
            # Run the limitation agent
            response = self.agent.run(limitation_prompt, stream=False)
            
            # Parse limitation results
            limitations = self._parse_limitation_response(response.content, paper)
            
            return limitations
            
        except Exception as e:
            logger.error(f"Error identifying limitations for paper {paper.title}: {str(e)}")
            return []
    
    def _create_limitation_prompt(self, paper: Paper, analysis_result: Optional[AnalysisResult] = None) -> str:
        """Create limitation identification prompt for a specific paper"""
        prompt = f"""
        Please identify research limitations in the following academic paper:
        
        Paper Details:
        - Title: {paper.title}
        - Authors: {', '.join(paper.authors) if paper.authors else 'Unknown'}
        - Abstract: {paper.abstract}
        - Published: {paper.published}
        - Venue: {paper.venue if paper.venue else 'Not specified'}
        """
        
        if analysis_result:
            prompt += f"""
        - Methodology: {analysis_result.methodology}
        - Key Concepts: {', '.join(analysis_result.key_concepts)}
        - Contributions: {', '.join(analysis_result.contributions)}
        """
        
        prompt += """
        
        Please identify and categorize research limitations in this paper. Look for:
        
        1. METHODOLOGICAL LIMITATIONS:
           - Issues with research design, experimental setup, or methodology
           - Problems with data collection or analysis methods
           - Limitations in the approach or technique used
        
        2. SCOPE LIMITATIONS:
           - Narrow scope of the study
           - Limited applicability to broader contexts
           - Constraints on the research domain or population
        
        3. DATA LIMITATIONS:
           - Issues with data quality, quantity, or availability
           - Problems with data sources or collection methods
           - Limitations in data analysis or interpretation
        
        4. EVALUATION LIMITATIONS:
           - Problems with evaluation metrics or criteria
           - Issues with validation or testing methods
           - Limitations in performance assessment
        
        5. GENERALIZATION LIMITATIONS:
           - Issues with generalizing results to other contexts
           - Limitations in external validity
           - Problems with applying findings broadly
        
        Provide your analysis in the following format:
        - Limitation Type: [METHODOLOGICAL/SCOPE/DATA/EVALUATION/GENERALIZATION]
        - Description: [Detailed description of the limitation]
        - Confidence: [0.0-1.0 confidence score]
        - Impact Level: [LOW/MEDIUM/HIGH]
        - Evidence: [Specific evidence from the paper supporting this limitation]
        
        List each limitation separately and be specific about what you found.
        """
        
        return prompt
    
    def _parse_limitation_response(self, response_content: str, paper: Paper) -> List[Limitation]:
        """Parse limitation identification response from agent"""
        limitations = []
        
        try:
            lines = response_content.split('\n')
            current_limitation = {}
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('Limitation Type:') or line.startswith('**Limitation Type:'):
                    # Save previous limitation if exists
                    if current_limitation:
                        limitations.append(self._create_limitation_from_dict(current_limitation, paper))
                    
                    # Start new limitation
                    limitation_type = line.replace('Limitation Type:', '').replace('**Limitation Type:', '').strip().upper()
                    current_limitation = {'type': limitation_type}
                
                elif line.startswith('Description:') or line.startswith('**Description:'):
                    current_limitation['description'] = line.replace('Description:', '').replace('**Description:', '').strip()
                
                elif line.startswith('Confidence:') or line.startswith('**Confidence:'):
                    try:
                        current_limitation['confidence'] = float(line.replace('Confidence:', '').replace('**Confidence:', '').strip())
                    except ValueError:
                        current_limitation['confidence'] = 0.5
                
                elif line.startswith('Impact Level:') or line.startswith('**Impact Level:'):
                    current_limitation['impact_level'] = line.replace('Impact Level:', '').replace('**Impact Level:', '').strip().lower()
                
                elif line.startswith('Evidence:') or line.startswith('**Evidence:'):
                    current_limitation['evidence'] = line.replace('Evidence:', '').replace('**Evidence:', '').strip()
            
            # Add the last limitation if exists
            if current_limitation:
                limitations.append(self._create_limitation_from_dict(current_limitation, paper))
            
        except Exception as e:
            logger.error(f"Error parsing limitation response: {str(e)}")
        
        return limitations
    
    def _create_limitation_from_dict(self, limitation_dict: Dict[str, Any], paper: Paper) -> Limitation:
        """Create Limitation object from dictionary"""
        # Map limitation type
        type_mapping = {
            'METHODOLOGICAL': LimitationType.METHODOLOGICAL,
            'SCOPE': LimitationType.SCOPE,
            'DATA': LimitationType.DATA,
            'EVALUATION': LimitationType.EVALUATION,
            'GENERALIZATION': LimitationType.GENERALIZATION
        }
        
        limitation_type = type_mapping.get(limitation_dict.get('type', ''), LimitationType.METHODOLOGICAL)
        
        return Limitation(
            paper_title=paper.title,
            limitation_type=limitation_type,
            description=limitation_dict.get('description', ''),
            confidence=limitation_dict.get('confidence', 0.5),
            impact_level=limitation_dict.get('impact_level', 'medium')
        )
    
    def _categorize_limitations(self, limitations: List[Limitation]) -> Dict[str, List[Limitation]]:
        """Categorize limitations by type"""
        categories = {
            'methodological': [],
            'scope': [],
            'data': [],
            'evaluation': [],
            'generalization': []
        }
        
        for limitation in limitations:
            category = limitation.limitation_type.value
            if category in categories:
                categories[category].append(limitation)
        
        return categories
    
    def _rank_limitations(self, categorized_limitations: Dict[str, List[Limitation]]) -> List[Limitation]:
        """Rank limitations by importance and confidence"""
        all_limitations = []
        
        for category, limitations in categorized_limitations.items():
            # Sort by confidence and impact level
            limitations.sort(key=lambda x: (x.confidence, self._get_impact_score(x.impact_level)), reverse=True)
            all_limitations.extend(limitations)
        
        # Sort all limitations by overall importance
        all_limitations.sort(key=lambda x: (x.confidence, self._get_impact_score(x.impact_level)), reverse=True)
        
        return all_limitations
    
    def _get_impact_score(self, impact_level: str) -> float:
        """Convert impact level to numerical score"""
        impact_scores = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.9
        }
        return impact_scores.get(impact_level.lower(), 0.5)
    
    def _assess_limitation_impact(self, limitation: Limitation) -> str:
        """Assess the potential impact of a limitation on research validity"""
        if limitation.confidence >= 0.8 and limitation.impact_level == 'high':
            return "Critical - significantly affects research validity"
        elif limitation.confidence >= 0.6 and limitation.impact_level in ['high', 'medium']:
            return "Important - may affect research conclusions"
        elif limitation.confidence >= 0.4:
            return "Moderate - minor impact on research validity"
        else:
            return "Low - minimal impact on research validity"
    
    def get_limitation_summary(self, limitations: List[Limitation]) -> Dict[str, Any]:
        """Get summary statistics for limitations"""
        if not limitations:
            return {
                "total_limitations": 0,
                "by_type": {},
                "by_impact": {},
                "high_confidence": 0,
                "critical_limitations": 0
            }
        
        by_type = {}
        by_impact = {}
        high_confidence = 0
        critical_limitations = 0
        
        for limitation in limitations:
            # Count by type
            limitation_type = limitation.limitation_type.value
            by_type[limitation_type] = by_type.get(limitation_type, 0) + 1
            
            # Count by impact
            by_impact[limitation.impact_level] = by_impact.get(limitation.impact_level, 0) + 1
            
            # Count high confidence
            if limitation.confidence >= 0.7:
                high_confidence += 1
            
            # Count critical limitations
            if limitation.confidence >= 0.8 and limitation.impact_level == 'high':
                critical_limitations += 1
        
        return {
            "total_limitations": len(limitations),
            "by_type": by_type,
            "by_impact": by_impact,
            "high_confidence": high_confidence,
            "critical_limitations": critical_limitations
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the limitation agent"""
        return {
            "name": "Limitation Agent",
            "role": "Research limitation identification and categorization",
            "tools": ["Text Analysis", "Limitation Classification"],
            "capabilities": [
                "Identify methodological limitations",
                "Detect scope constraints",
                "Categorize limitation types",
                "Assess limitation impact",
                "Evaluate research validity",
                "Provide confidence scores"
            ]
        }
