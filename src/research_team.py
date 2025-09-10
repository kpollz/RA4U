# RA4U - Research Team
# Team coordination and orchestration for the research workflow

from agno.team import Team
from agno.agent import Agent
from typing import List, Dict, Any, Optional
import logging
import asyncio
from datetime import datetime

from models import ResearchQuery, ResearchReport, Paper, Limitation, ResearchGap, AgentConfig, WorkflowState
from search_agent import SearchAgent
from verification_agent import VerificationAgent
from analysis_agent import AnalysisAgent
from limitation_agent import LimitationAgent
from gap_agent import GapAgent
from utils import log_agent_activity, create_research_prompt

logger = logging.getLogger(__name__)

class ResearchTeam:
    """Research Team for orchestrating the complete research workflow"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.workflow_state = None
        self.agents = self._initialize_agents()
        self.team = self._create_team()
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all specialized agents"""
        try:
            agents = {
                'search': SearchAgent(self.config),
                'verification': VerificationAgent(self.config),
                'analysis': AnalysisAgent(self.config),
                'limitation': LimitationAgent(self.config),
                'gap': GapAgent(self.config)
            }
            
            log_agent_activity("Research Team", "Agents initialized", {
                "agents": list(agents.keys())
            })
            
            return agents
            
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            raise
    
    def _create_team(self) -> Team:
        """Create the research team with all agents"""
        try:
            # Get agent instances for the team
            search_agent = self.agents['search'].agent
            verification_agent = self.agents['verification'].agent
            analysis_agent = self.agents['analysis'].agent
            limitation_agent = self.agents['limitation'].agent
            gap_agent = self.agents['gap'].agent
            
            # Create team model
            if self.config.model_provider == "OpenAI":
                from agno.models.openai import OpenAIChat
                team_model = OpenAIChat(
                    id=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    api_key=self.config.api_key
                )
            elif self.config.model_provider == "Google":
                from agno.models.google import Gemini
                team_model = Gemini(
                    id=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    api_key=self.config.api_key
                )
            else:
                raise ValueError(f"Unsupported model provider: {self.config.model_provider}")
            
            # Create the team
            team = Team(
                name="RA4U Research Team",
                mode="coordinate",
                model=team_model,
                members=[search_agent, verification_agent, analysis_agent, limitation_agent, gap_agent],
                instructions=[
                    "1. First, use the Search Agent to find relevant academic papers on the given topic",
                    "2. Then, use the Verification Agent to verify the accuracy of found papers",
                    "3. Next, use the Analysis Agent to analyze the content and relevance of verified papers",
                    "4. After that, use the Limitation Agent to identify limitations in each paper",
                    "5. Finally, use the Gap Agent to synthesize limitations and identify research gaps",
                    "6. Provide a comprehensive research report with all findings organized logically"
                ],
                show_tool_calls=True,
                markdown=True,
                debug_mode=True,
                show_members_responses=True,
            )
            
            log_agent_activity("Research Team", "Team created", {
                "team_name": "RA4U Research Team",
                "members_count": len(team.members)
            })
            
            return team
            
        except Exception as e:
            logger.error(f"Error creating team: {str(e)}")
            raise
    
    async def run_research_workflow(self, query: ResearchQuery) -> ResearchReport:
        """Run the complete research workflow"""
        try:
            # Initialize workflow state
            self.workflow_state = WorkflowState(
                query_id=f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                current_stage="initialized"
            )
            
            log_agent_activity("Research Team", "Starting research workflow", {
                "query_id": self.workflow_state.query_id,
                "topic": query.topic,
                "domain": query.domain
            })
            
            # Stage 1: Search for papers
            self.workflow_state.current_stage = "searching"
            papers = await self.agents['search'].search_papers(query)
            self.workflow_state.search_results = papers
            
            log_agent_activity("Research Team", "Search completed", {
                "papers_found": len(papers)
            })
            
            # Stage 2: Verify papers (if enabled)
            if query.include_verification and papers:
                self.workflow_state.current_stage = "verifying"
                verified_papers = await self.agents['verification'].verify_papers(papers)
                self.workflow_state.verified_papers = verified_papers
                
                log_agent_activity("Research Team", "Verification completed", {
                    "verified_papers": len(verified_papers)
                })
            else:
                self.workflow_state.verified_papers = papers
            
            # Stage 3: Analyze papers
            if self.workflow_state.verified_papers:
                self.workflow_state.current_stage = "analyzing"
                analysis_results = await self.agents['analysis'].analyze_papers(
                    self.workflow_state.verified_papers, query.topic
                )
                self.workflow_state.analysis_results = {
                    'results': analysis_results,
                    'timestamp': datetime.now()
                }
                
                log_agent_activity("Research Team", "Analysis completed", {
                    "papers_analyzed": len(analysis_results)
                })
            
            # Stage 4: Identify limitations
            if self.workflow_state.verified_papers and self.workflow_state.analysis_results:
                self.workflow_state.current_stage = "identifying_limitations"
                limitations = await self.agents['limitation'].identify_limitations(
                    self.workflow_state.verified_papers,
                    self.workflow_state.analysis_results['results']
                )
                self.workflow_state.limitations = limitations
                
                log_agent_activity("Research Team", "Limitation identification completed", {
                    "limitations_found": len(limitations)
                })
            
            # Stage 5: Identify research gaps
            if self.workflow_state.limitations:
                self.workflow_state.current_stage = "identifying_gaps"
                research_gaps = await self.agents['gap'].identify_research_gaps(
                    self.workflow_state.limitations, query.topic
                )
                self.workflow_state.research_gaps = research_gaps
                
                log_agent_activity("Research Team", "Gap identification completed", {
                    "gaps_found": len(research_gaps)
                })
            
            # Stage 6: Generate final report
            self.workflow_state.current_stage = "generating_report"
            report = self._generate_research_report(query)
            
            self.workflow_state.current_stage = "completed"
            self.workflow_state.status = "completed"
            
            log_agent_activity("Research Team", "Research workflow completed", {
                "query_id": self.workflow_state.query_id,
                "total_papers": len(self.workflow_state.verified_papers),
                "total_limitations": len(self.workflow_state.limitations),
                "total_gaps": len(self.workflow_state.research_gaps)
            })
            
            return report
            
        except Exception as e:
            logger.error(f"Research workflow error: {str(e)}")
            self.workflow_state.status = "failed"
            self.workflow_state.error_message = str(e)
            
            log_agent_activity("Research Team", "Research workflow failed", {
                "error": str(e),
                "query_id": self.workflow_state.query_id if self.workflow_state else "unknown"
            })
            
            # Return a basic report even if workflow fails
            return self._create_error_report(query, str(e))
    
    def _generate_research_report(self, query: ResearchQuery) -> ResearchReport:
        """Generate the final research report"""
        try:
            # Create summary
            summary = self._create_executive_summary()
            
            # Create recommendations
            recommendations = self._create_recommendations()
            
            # Calculate processing time
            processing_time = None
            if self.workflow_state:
                start_time = self.workflow_state.created_at
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
            
            report = ResearchReport(
                topic=query.topic,
                query=query,
                papers=self.workflow_state.verified_papers if self.workflow_state else [],
                limitations=self.workflow_state.limitations if self.workflow_state else [],
                research_gaps=self.workflow_state.research_gaps if self.workflow_state else [],
                summary=summary,
                recommendations=recommendations,
                processing_time=processing_time
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating research report: {str(e)}")
            return self._create_error_report(query, str(e))
    
    def _create_executive_summary(self) -> str:
        """Create executive summary of the research findings"""
        if not self.workflow_state:
            return "No research data available."
        
        summary_parts = []
        
        # Papers summary
        papers_count = len(self.workflow_state.verified_papers)
        summary_parts.append(f"Analysis of {papers_count} academic papers")
        
        # Limitations summary
        limitations_count = len(self.workflow_state.limitations)
        if limitations_count > 0:
            summary_parts.append(f"Identified {limitations_count} research limitations")
        
        # Gaps summary
        gaps_count = len(self.workflow_state.research_gaps)
        if gaps_count > 0:
            summary_parts.append(f"Found {gaps_count} potential research gaps")
        
        # High-priority gaps
        high_priority_gaps = [gap for gap in self.workflow_state.research_gaps if gap.priority.value == 'high']
        if high_priority_gaps:
            summary_parts.append(f"Identified {len(high_priority_gaps)} high-priority research opportunities")
        
        return ". ".join(summary_parts) + "."
    
    def _create_recommendations(self) -> List[str]:
        """Create research recommendations based on findings"""
        recommendations = []
        
        if not self.workflow_state:
            return ["No recommendations available due to insufficient data."]
        
        # Recommendations based on gaps
        high_priority_gaps = [gap for gap in self.workflow_state.research_gaps if gap.priority.value == 'high']
        for gap in high_priority_gaps[:3]:  # Top 3 high-priority gaps
            recommendations.append(f"Focus on addressing: {gap.gap_title}")
        
        # Recommendations based on limitations
        if self.workflow_state.limitations:
            methodological_limitations = [lim for lim in self.workflow_state.limitations if lim.limitation_type.value == 'methodological']
            if methodological_limitations:
                recommendations.append("Consider improving methodological approaches in future research")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Continue monitoring the research area for new developments")
            recommendations.append("Consider interdisciplinary approaches to address complex research questions")
        
        return recommendations
    
    def _create_error_report(self, query: ResearchQuery, error_message: str) -> ResearchReport:
        """Create an error report when workflow fails"""
        return ResearchReport(
            topic=query.topic,
            query=query,
            papers=[],
            limitations=[],
            research_gaps=[],
            summary=f"Research analysis failed: {error_message}",
            recommendations=["Please try again with different parameters or check your API configuration."]
        )
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        if not self.workflow_state:
            return {
                "status": "not_started",
                "current_stage": "initialized",
                "progress": 0
            }
        
        stage_progress = {
            "initialized": 0,
            "searching": 20,
            "verifying": 40,
            "analyzing": 60,
            "identifying_limitations": 80,
            "identifying_gaps": 90,
            "generating_report": 95,
            "completed": 100
        }
        
        return {
            "status": self.workflow_state.status,
            "current_stage": self.workflow_state.current_stage,
            "progress": stage_progress.get(self.workflow_state.current_stage, 0),
            "query_id": self.workflow_state.query_id,
            "papers_found": len(self.workflow_state.search_results),
            "papers_verified": len(self.workflow_state.verified_papers),
            "limitations_found": len(self.workflow_state.limitations),
            "gaps_found": len(self.workflow_state.research_gaps),
            "error_message": self.workflow_state.error_message
        }
    
    def get_team_info(self) -> Dict[str, Any]:
        """Get information about the research team"""
        return {
            "team_name": "RA4U Research Team",
            "workflow_stages": [
                "Search for relevant papers",
                "Verify paper accuracy",
                "Analyze paper content",
                "Identify research limitations",
                "Identify research gaps",
                "Generate comprehensive report"
            ],
            "agents": {
                name: agent.get_agent_info() for name, agent in self.agents.items()
            },
            "capabilities": [
                "Academic paper discovery",
                "Accuracy verification",
                "Content analysis",
                "Limitation identification",
                "Research gap analysis",
                "Comprehensive reporting"
            ]
        }
    
    def reset_workflow(self):
        """Reset the workflow state"""
        self.workflow_state = None
        log_agent_activity("Research Team", "Workflow reset", {})
