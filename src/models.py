# RA4U - Data Models
# Pydantic models for type safety and validation

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class VerificationStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    SUSPICIOUS = "suspicious"

class LimitationType(str, Enum):
    METHODOLOGICAL = "methodological"
    SCOPE = "scope"
    DATA = "data"
    EVALUATION = "evaluation"
    GENERALIZATION = "generalization"

class PriorityLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Paper(BaseModel):
    """Academic paper model"""
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    abstract: str = Field(..., description="Paper abstract")
    published: str = Field(..., description="Publication date")
    arxiv_id: str = Field(..., description="arXiv ID")
    url: str = Field(..., description="Paper URL")
    doi: Optional[str] = Field(None, description="DOI if available")
    venue: Optional[str] = Field(None, description="Publication venue")
    citation_count: int = Field(0, description="Number of citations")
    keywords: List[str] = Field(default_factory=list, description="Keywords")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Relevance score")
    verification_status: VerificationStatus = Field(VerificationStatus.PENDING, description="Verification status")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

class Limitation(BaseModel):
    """Research limitation model"""
    paper_title: str = Field(..., description="Title of the paper with limitation")
    limitation_type: LimitationType = Field(..., description="Type of limitation")
    description: str = Field(..., description="Detailed description of limitation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    impact_level: str = Field("medium", description="Impact level: low, medium, high")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

class ResearchGap(BaseModel):
    """Research gap model"""
    gap_title: str = Field(..., description="Title of the research gap")
    description: str = Field(..., description="Detailed description of the gap")
    priority: PriorityLevel = Field(..., description="Priority level")
    related_limitations: List[str] = Field(default_factory=list, description="Related limitation IDs")
    suggested_methodology: Optional[str] = Field(None, description="Suggested research methodology")
    potential_impact: str = Field("medium", description="Potential impact level")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

class ResearchQuery(BaseModel):
    """Research query model"""
    topic: str = Field(..., description="Research topic or question")
    domain: str = Field("Computer Science", description="Research domain")
    keywords: List[str] = Field(default_factory=list, description="Additional keywords")
    date_range: str = Field("Last 2 years", description="Publication date range")
    max_papers: int = Field(10, ge=1, le=50, description="Maximum papers to analyze")
    min_citations: int = Field(5, ge=0, description="Minimum citation count")
    venue_filters: List[str] = Field(default_factory=list, description="Venue filters")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")
    include_verification: bool = Field(True, description="Enable verification")

class ResearchReport(BaseModel):
    """Complete research report model"""
    topic: str = Field(..., description="Research topic")
    query: ResearchQuery = Field(..., description="Original research query")
    papers: List[Paper] = Field(default_factory=list, description="Discovered papers")
    limitations: List[Limitation] = Field(default_factory=list, description="Identified limitations")
    research_gaps: List[ResearchGap] = Field(default_factory=list, description="Research gaps")
    summary: Optional[str] = Field(None, description="Executive summary")
    recommendations: List[str] = Field(default_factory=list, description="Research recommendations")
    generated_at: datetime = Field(default_factory=datetime.now, description="Generation timestamp")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")

class AgentConfig(BaseModel):
    """Agent configuration model"""
    model_provider: str = Field("OpenAI", description="AI model provider")
    model_name: str = Field("gpt-4o-mini", description="Model name")
    max_tokens: int = Field(2048, description="Maximum tokens")
    temperature: float = Field(0.3, ge=0.0, le=2.0, description="Temperature")
    api_key: Optional[str] = Field(None, description="API key")
    timeout: int = Field(300, description="Request timeout in seconds")

class WorkflowState(BaseModel):
    """Workflow state model"""
    query_id: str = Field(..., description="Unique query ID")
    current_stage: str = Field("initialized", description="Current workflow stage")
    search_results: List[Paper] = Field(default_factory=list, description="Search results")
    verified_papers: List[Paper] = Field(default_factory=list, description="Verified papers")
    analysis_results: Dict[str, Any] = Field(default_factory=dict, description="Analysis results")
    limitations: List[Limitation] = Field(default_factory=list, description="Identified limitations")
    research_gaps: List[ResearchGap] = Field(default_factory=list, description="Research gaps")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    status: str = Field("running", description="Workflow status")
    error_message: Optional[str] = Field(None, description="Error message if any")

class VerificationResult(BaseModel):
    """Verification result model"""
    paper_id: str = Field(..., description="Paper ID")
    is_valid: bool = Field(..., description="Whether paper is valid")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Verification confidence")
    checks: Dict[str, bool] = Field(default_factory=dict, description="Individual check results")
    error_message: Optional[str] = Field(None, description="Error message if verification failed")
    verified_at: datetime = Field(default_factory=datetime.now, description="Verification timestamp")

class AnalysisResult(BaseModel):
    """Analysis result model"""
    paper_id: str = Field(..., description="Paper ID")
    key_concepts: List[str] = Field(default_factory=list, description="Key concepts")
    methodology: Optional[str] = Field(None, description="Research methodology")
    contributions: List[str] = Field(default_factory=list, description="Main contributions")
    limitations: List[str] = Field(default_factory=list, description="Identified limitations")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Relevance score")
    quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Quality score")
    analyzed_at: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
