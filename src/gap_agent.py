# RA4U - Gap Agent
# Specialized agent for identifying research gaps and opportunities

from agno.agent import Agent
from typing import List, Dict, Any, Optional
import logging

from models import Limitation, ResearchGap, PriorityLevel, AgentConfig
from utils import log_agent_activity

logger = logging.getLogger(__name__)

class GapAgent:
    """Gap Agent for identifying research gaps and opportunities"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the gap agent with appropriate model and tools"""
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
            name="Gap Agent",
            model=model,
            role="""You are a research gap identification expert. Your role is to:
            1. Synthesize limitations from multiple papers
            2. Identify unexplored research areas and opportunities
            3. Propose novel research directions and questions
            4. Prioritize research gaps by potential impact and feasibility
            """,
            instructions=[
                "Analyze all identified limitations to find patterns and gaps",
                "Identify research questions that haven't been addressed",
                "Propose specific research directions and methodologies",
                "Prioritize gaps by potential impact and research feasibility",
                "Provide actionable recommendations for future research",
                "Focus on creating opportunities for novel contributions",
                "Consider both theoretical and practical research gaps",
                "Evaluate the feasibility of addressing each gap"
            ]
        )
    
    async def identify_research_gaps(self, limitations: List[Limitation], research_topic: str) -> List[ResearchGap]:
        """Identify research gaps based on limitations"""
        try:
            log_agent_activity("Gap Agent", "Starting research gap identification", {
                "limitations_count": len(limitations),
                "research_topic": research_topic
            })
            
            # Group limitations by type for analysis
            grouped_limitations = self._group_limitations_by_type(limitations)
            
            # Identify gaps for each limitation type
            gap_tasks = []
            for limitation_type, type_limitations in grouped_limitations.items():
                task = self._identify_type_gaps(limitation_type, type_limitations, research_topic)
                gap_tasks.append(task)
            
            # Process all gap identification tasks
            gap_results = []
            for task in gap_tasks:
                gap_results.extend(await task)
            
            # Synthesize and prioritize gaps
            synthesized_gaps = self._synthesize_gaps(gap_results, research_topic)
            prioritized_gaps = self._prioritize_gaps(synthesized_gaps)
            
            log_agent_activity("Gap Agent", "Research gap identification completed", {
                "total_gaps": len(prioritized_gaps),
                "research_topic": research_topic
            })
            
            return prioritized_gaps
            
        except Exception as e:
            logger.error(f"Gap Agent error: {str(e)}")
            log_agent_activity("Gap Agent", "Research gap identification failed", {
                "error": str(e),
                "research_topic": research_topic
            })
            return []
    
    def _group_limitations_by_type(self, limitations: List[Limitation]) -> Dict[str, List[Limitation]]:
        """Group limitations by type for analysis"""
        grouped = {}
        for limitation in limitations:
            limitation_type = limitation.limitation_type.value
            if limitation_type not in grouped:
                grouped[limitation_type] = []
            grouped[limitation_type].append(limitation)
        return grouped
    
    async def _identify_type_gaps(self, limitation_type: str, limitations: List[Limitation], research_topic: str) -> List[ResearchGap]:
        """Identify research gaps for a specific limitation type"""
        try:
            # Create gap identification prompt
            gap_prompt = self._create_gap_prompt(limitation_type, limitations, research_topic)
            
            # Run the gap agent
            response = self.agent.run(gap_prompt, stream=False)
            
            # Parse gap results
            gaps = self._parse_gap_response(response.content, limitation_type)
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error identifying gaps for {limitation_type}: {str(e)}")
            return []
    
    def _create_gap_prompt(self, limitation_type: str, limitations: List[Limitation], research_topic: str) -> str:
        """Create gap identification prompt for a specific limitation type"""
        prompt = f"""
        Research Topic: {research_topic}
        Limitation Type: {limitation_type.upper()}
        
        The following limitations have been identified in the research area:
        """
        
        for i, limitation in enumerate(limitations, 1):
            prompt += f"""
        {i}. Paper: {limitation.paper_title}
           Description: {limitation.description}
           Impact Level: {limitation.impact_level}
           Confidence: {limitation.confidence}
        """
        
        prompt += f"""
        
        Based on these {limitation_type} limitations, please identify potential research gaps and opportunities.
        
        For each gap, provide:
        1. GAP TITLE: A clear, concise title for the research gap
        2. DESCRIPTION: Detailed description of the gap and why it's important
        3. PRIORITY: [HIGH/MEDIUM/LOW] based on potential impact and feasibility
        4. SUGGESTED METHODOLOGY: Recommended approach to address this gap
        5. POTENTIAL IMPACT: Expected impact of addressing this gap
        6. RELATED LIMITATIONS: Which limitations this gap addresses
        
        Focus on:
        - Identifying unexplored research questions
        - Proposing novel methodologies or approaches
        - Suggesting improvements to existing methods
        - Highlighting areas where current research falls short
        - Creating opportunities for significant contributions
        
        Provide your analysis in the following format:
        - Gap Title: [Title of the research gap]
        - Description: [Detailed description]
        - Priority: [HIGH/MEDIUM/LOW]
        - Suggested Methodology: [Recommended approach]
        - Potential Impact: [Expected impact level]
        - Related Limitations: [List of limitation numbers this addresses]
        
        List each gap separately and be specific about the research opportunities.
        """
        
        return prompt
    
    def _parse_gap_response(self, response_content: str, limitation_type: str) -> List[ResearchGap]:
        """Parse gap identification response from agent"""
        gaps = []
        
        try:
            lines = response_content.split('\n')
            current_gap = {}
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('Gap Title:') or line.startswith('**Gap Title:'):
                    # Save previous gap if exists
                    if current_gap:
                        gaps.append(self._create_gap_from_dict(current_gap, limitation_type))
                    
                    # Start new gap
                    current_gap = {'title': line.replace('Gap Title:', '').replace('**Gap Title:', '').strip()}
                
                elif line.startswith('Description:') or line.startswith('**Description:'):
                    current_gap['description'] = line.replace('Description:', '').replace('**Description:', '').strip()
                
                elif line.startswith('Priority:') or line.startswith('**Priority:'):
                    priority = line.replace('Priority:', '').replace('**Priority:', '').strip().upper()
                    current_gap['priority'] = priority
                
                elif line.startswith('Suggested Methodology:') or line.startswith('**Suggested Methodology:'):
                    current_gap['suggested_methodology'] = line.replace('Suggested Methodology:', '').replace('**Suggested Methodology:', '').strip()
                
                elif line.startswith('Potential Impact:') or line.startswith('**Potential Impact:'):
                    current_gap['potential_impact'] = line.replace('Potential Impact:', '').replace('**Potential Impact:', '').strip()
                
                elif line.startswith('Related Limitations:') or line.startswith('**Related Limitations:'):
                    related_text = line.replace('Related Limitations:', '').replace('**Related Limitations:', '').strip()
                    current_gap['related_limitations'] = [x.strip() for x in related_text.split(',') if x.strip()]
            
            # Add the last gap if exists
            if current_gap:
                gaps.append(self._create_gap_from_dict(current_gap, limitation_type))
            
        except Exception as e:
            logger.error(f"Error parsing gap response: {str(e)}")
        
        return gaps
    
    def _create_gap_from_dict(self, gap_dict: Dict[str, Any], limitation_type: str) -> ResearchGap:
        """Create ResearchGap object from dictionary"""
        # Map priority level
        priority_mapping = {
            'HIGH': PriorityLevel.HIGH,
            'MEDIUM': PriorityLevel.MEDIUM,
            'LOW': PriorityLevel.LOW
        }
        
        priority = priority_mapping.get(gap_dict.get('priority', 'MEDIUM'), PriorityLevel.MEDIUM)
        
        return ResearchGap(
            gap_title=gap_dict.get('title', ''),
            description=gap_dict.get('description', ''),
            priority=priority,
            related_limitations=gap_dict.get('related_limitations', []),
            suggested_methodology=gap_dict.get('suggested_methodology'),
            potential_impact=gap_dict.get('potential_impact', 'medium')
        )
    
    def _synthesize_gaps(self, gap_results: List[ResearchGap], research_topic: str) -> List[ResearchGap]:
        """Synthesize gaps from different limitation types"""
        if not gap_results:
            return []
        
        # Remove duplicate gaps based on title similarity
        unique_gaps = []
        seen_titles = set()
        
        for gap in gap_results:
            # Simple deduplication based on title similarity
            title_lower = gap.gap_title.lower()
            if not any(self._titles_similar(title_lower, seen_title) for seen_title in seen_titles):
                unique_gaps.append(gap)
                seen_titles.add(title_lower)
        
        return unique_gaps
    
    def _titles_similar(self, title1: str, title2: str, threshold: float = 0.8) -> bool:
        """Check if two titles are similar"""
        if not title1 or not title2:
            return False
        
        # Simple similarity check based on common words
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold
    
    def _prioritize_gaps(self, gaps: List[ResearchGap]) -> List[ResearchGap]:
        """Prioritize gaps by importance and feasibility"""
        if not gaps:
            return []
        
        # Sort by priority level and potential impact
        priority_scores = {
            PriorityLevel.HIGH: 3,
            PriorityLevel.MEDIUM: 2,
            PriorityLevel.LOW: 1
        }
        
        impact_scores = {
            'high': 3,
            'medium': 2,
            'low': 1
        }
        
        def gap_priority(gap):
            priority_score = priority_scores.get(gap.priority, 2)
            impact_score = impact_scores.get(gap.potential_impact.lower(), 2)
            return (priority_score, impact_score)
        
        return sorted(gaps, key=gap_priority, reverse=True)
    
    def _assess_gap_feasibility(self, gap: ResearchGap) -> str:
        """Assess the feasibility of addressing a research gap"""
        feasibility_factors = []
        
        # Check if methodology is suggested
        if gap.suggested_methodology:
            feasibility_factors.append(1.0)
        else:
            feasibility_factors.append(0.5)
        
        # Check priority level
        if gap.priority == PriorityLevel.HIGH:
            feasibility_factors.append(0.8)
        elif gap.priority == PriorityLevel.MEDIUM:
            feasibility_factors.append(0.6)
        else:
            feasibility_factors.append(0.4)
        
        # Check potential impact
        if gap.potential_impact.lower() == 'high':
            feasibility_factors.append(0.9)
        elif gap.potential_impact.lower() == 'medium':
            feasibility_factors.append(0.7)
        else:
            feasibility_factors.append(0.5)
        
        # Calculate average feasibility
        avg_feasibility = sum(feasibility_factors) / len(feasibility_factors)
        
        if avg_feasibility >= 0.8:
            return "High feasibility"
        elif avg_feasibility >= 0.6:
            return "Medium feasibility"
        else:
            return "Low feasibility"
    
    def get_gap_summary(self, gaps: List[ResearchGap]) -> Dict[str, Any]:
        """Get summary statistics for research gaps"""
        if not gaps:
            return {
                "total_gaps": 0,
                "by_priority": {},
                "by_impact": {},
                "high_priority": 0,
                "feasible_gaps": 0
            }
        
        by_priority = {}
        by_impact = {}
        high_priority = 0
        feasible_gaps = 0
        
        for gap in gaps:
            # Count by priority
            priority = gap.priority.value
            by_priority[priority] = by_priority.get(priority, 0) + 1
            
            # Count by impact
            by_impact[gap.potential_impact] = by_impact.get(gap.potential_impact, 0) + 1
            
            # Count high priority
            if gap.priority == PriorityLevel.HIGH:
                high_priority += 1
            
            # Count feasible gaps
            feasibility = self._assess_gap_feasibility(gap)
            if "High" in feasibility or "Medium" in feasibility:
                feasible_gaps += 1
        
        return {
            "total_gaps": len(gaps),
            "by_priority": by_priority,
            "by_impact": by_impact,
            "high_priority": high_priority,
            "feasible_gaps": feasible_gaps
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the gap agent"""
        return {
            "name": "Gap Agent",
            "role": "Research gap identification and opportunity analysis",
            "tools": ["Gap Analysis", "Opportunity Assessment"],
            "capabilities": [
                "Synthesize limitations into gaps",
                "Identify unexplored research areas",
                "Propose novel research directions",
                "Prioritize gaps by impact and feasibility",
                "Suggest research methodologies",
                "Evaluate research opportunities"
            ]
        }
