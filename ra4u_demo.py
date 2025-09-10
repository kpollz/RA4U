# RA4U - Research Assistant 4 You Demo
# Multi-Agent System for Academic Research and Gap Analysis

import streamlit as st
import os
from typing import List, Dict, Any
from agno.agent import Agent
from agno.team import Team
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.tools.arxiv import ArxivTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from pydantic import BaseModel
import json
import re
from datetime import datetime

# Set up the Streamlit app
st.set_page_config(
    page_title="RA4U - Research Assistant 4 You",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 RA4U - Research Assistant 4 You")
st.caption("Multi-Agent AI System for Academic Research and Research Gap Analysis")

# Sidebar for API configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model_provider = st.selectbox(
        "Choose AI Model Provider",
        ["OpenAI", "Google Gemini"],
        help="Select the AI model provider for the agents"
    )
    
    # API Key input
    if model_provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    else:
        api_key = st.text_input("Google API Key", type="password", help="Enter your Google API key")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
    
    # Research parameters
    st.header("🔍 Research Parameters")
    max_papers = st.slider("Maximum Papers to Analyze", 5, 20, 10)
    similarity_threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.7)
    include_verification = st.checkbox("Enable Verification", value=True)

# Data models
class Paper(BaseModel):
    title: str
    authors: List[str]
    abstract: str
    published: str
    arxiv_id: str
    url: str
    relevance_score: float = 0.0
    verification_status: str = "pending"

class Limitation(BaseModel):
    paper_title: str
    limitation_type: str
    description: str
    confidence: float

class ResearchGap(BaseModel):
    gap_title: str
    description: str
    priority: str
    related_limitations: List[str]

class ResearchReport(BaseModel):
    topic: str
    papers: List[Paper]
    limitations: List[Limitation]
    research_gaps: List[ResearchGap]
    generated_at: datetime

# Initialize agents if API key is provided
if api_key:
    # Model configuration
    if model_provider == "OpenAI":
        model_config = OpenAIChat(id="gpt-4o-mini")
    else:
        model_config = Gemini(id="gemini-2.0-flash-exp")
        os.environ['GOOGLE_API_KEY'] = api_key
    
    # 1. Search Agent - Discovers relevant academic papers
    search_agent = Agent(
        name="Search Agent",
        model=model_config,
        role="""You are a specialized academic search agent. Your role is to:
        1. Search for relevant academic papers using arXiv and other academic databases
        2. Filter papers based on relevance to the research topic
        3. Rank papers by quality metrics (citations, venue, recency)
        4. Provide structured paper information including title, authors, abstract, and metadata
        """,
        tools=[ArxivTools()],
        instructions=[
            "When given a research topic, search for the most relevant academic papers",
            "Focus on recent papers (last 3 years) with high citation counts",
            "Prioritize papers from top-tier conferences and journals",
            "Provide detailed abstracts and author information",
            "Rank papers by relevance score (0-1 scale)"
        ]
    )
    
    # 2. Verification Agent - Ensures accuracy and prevents hallucination
    verification_agent = Agent(
        name="Verification Agent",
        model=model_config,
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
            "Focus on preventing hallucination and ensuring accuracy"
        ]
    )
    
    # 3. Analysis Agent - Processes and analyzes papers
    analysis_agent = Agent(
        name="Analysis Agent",
        model=model_config,
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
            "Provide structured analysis results for each paper"
        ]
    )
    
    # 4. Limitation Agent - Identifies research limitations
    limitation_agent = Agent(
        name="Limitation Agent",
        model=model_config,
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
            "Focus on limitations that could lead to research gaps"
        ]
    )
    
    # 5. Gap Agent - Identifies research gaps and opportunities
    gap_agent = Agent(
        name="Gap Agent",
        model=model_config,
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
            "Provide actionable recommendations for future research"
        ]
    )
    
    # Master Research Team
    research_team = Team(
        name="RA4U Research Team",
        mode="coordinate",
        model=model_config,
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
    
    # Main interface
    st.header("🔍 Research Query")
    
    # Research topic input
    research_topic = st.text_area(
        "Enter your research topic or question:",
        placeholder="e.g., 'Machine Learning in Healthcare', 'Quantum Computing Applications', 'Sustainable Energy Technologies'",
        height=100
    )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            domain = st.selectbox(
                "Research Domain",
                ["Computer Science", "Physics", "Biology", "Chemistry", "Mathematics", "Engineering", "Medicine", "Other"]
            )
            
            date_range = st.selectbox(
                "Publication Date Range",
                ["Last 2 years", "Last 5 years", "All time"]
            )
        
        with col2:
            min_citations = st.number_input("Minimum Citations", min_value=0, value=5)
            
            include_arxiv = st.checkbox("Include arXiv Papers", value=True)
    
    # Process research query
    if st.button("🚀 Start Research Analysis", type="primary"):
        if research_topic:
            with st.spinner("🔍 Conducting research analysis..."):
                # Create comprehensive research prompt
                research_prompt = f"""
                Research Topic: {research_topic}
                Domain: {domain}
                Date Range: {date_range}
                Minimum Citations: {min_citations}
                Max Papers: {max_papers}
                
                Please conduct a comprehensive research analysis following these steps:
                
                1. SEARCH: Find {max_papers} most relevant academic papers on "{research_topic}" in the {domain} domain
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
                
                # Run the research team
                try:
                    response = research_team.run(research_prompt, stream=False)
                    
                    # Display results
                    st.success("✅ Research analysis completed!")
                    
                    # Display the response
                    st.markdown("## 📊 Research Analysis Results")
                    st.markdown(response.content)
                    
                    # Additional analysis sections
                    st.markdown("---")
                    
                    # Create tabs for different sections
                    tab1, tab2, tab3, tab4 = st.tabs(["📚 Papers Found", "🔍 Analysis", "⚠️ Limitations", "🎯 Research Gaps"])
                    
                    with tab1:
                        st.markdown("### Academic Papers Discovered")
                        st.info("Papers will be listed here based on the analysis results")
                    
                    with tab2:
                        st.markdown("### Paper Analysis")
                        st.info("Detailed analysis of each paper's contributions and methodology")
                    
                    with tab3:
                        st.markdown("### Research Limitations Identified")
                        st.info("Categorized limitations found across the papers")
                    
                    with tab4:
                        st.markdown("### Research Gaps and Opportunities")
                        st.info("Identified research gaps and future opportunities")
                    
                except Exception as e:
                    st.error(f"❌ Error during research analysis: {str(e)}")
                    st.info("Please check your API key and try again.")
        else:
            st.warning("⚠️ Please enter a research topic to begin analysis.")
    
    # Display system status
    st.markdown("---")
    st.markdown("### 🔧 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Agents Active", "5", "Search, Verify, Analyze, Limitation, Gap")
    
    with col2:
        st.metric("Verification", "Enabled" if include_verification else "Disabled")
    
    with col3:
        st.metric("Model Provider", model_provider)

else:
    st.warning("⚠️ Please configure your API key in the sidebar to start using RA4U.")
    
    # Show demo information
    st.markdown("## 🎯 About RA4U Demo")
    
    st.markdown("""
    This demo showcases the RA4U (Research Assistant 4 You) multi-agent system designed to:
    
    ### 🔍 **Core Features:**
    - **Multi-Agent Architecture**: 5 specialized agents working together
    - **Academic Paper Discovery**: Search and filter relevant research papers
    - **Verification System**: Prevent hallucination and ensure accuracy
    - **Limitation Analysis**: Identify research limitations systematically
    - **Gap Identification**: Find research opportunities and future directions
    
    ### 🤖 **Agent Roles:**
    1. **Search Agent**: Discovers relevant academic papers
    2. **Verification Agent**: Ensures accuracy and prevents hallucination
    3. **Analysis Agent**: Processes and analyzes paper content
    4. **Limitation Agent**: Identifies research limitations
    5. **Gap Agent**: Synthesizes gaps and opportunities
    
    ### 🚀 **How to Use:**
    1. Enter your API key (OpenAI or Google Gemini)
    2. Specify your research topic or question
    3. Configure research parameters
    4. Click "Start Research Analysis"
    5. Review the comprehensive research report
    """)
    
    # Show example topics
    st.markdown("### 💡 Example Research Topics:")
    
    example_topics = [
        "Machine Learning in Healthcare",
        "Quantum Computing Applications",
        "Sustainable Energy Technologies",
        "Natural Language Processing Advances",
        "Computer Vision in Autonomous Vehicles",
        "Blockchain Technology in Finance",
        "Artificial Intelligence Ethics",
        "Robotics in Manufacturing"
    ]
    
    for topic in example_topics:
        st.markdown(f"- {topic}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🔬 RA4U - Research Assistant 4 You | Multi-Agent AI System for Academic Research</p>
        <p>Built with Agno Framework and Streamlit | Summer School Agentic Course - Final Project</p>
    </div>
    """,
    unsafe_allow_html=True
)
