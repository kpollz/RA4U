# RA4U - Main Application
# Streamlit UI for the Research Assistant system

import streamlit as st
import os
import asyncio
from datetime import datetime
from typing import Optional

# Import our modular components
from models import ResearchQuery, AgentConfig, ResearchReport
from research_team import ResearchTeam
from utils import setup_environment, validate_api_key, validate_research_query, create_success_response, create_error_response

# Set up the Streamlit app
st.set_page_config(
    page_title="RA4U - Research Assistant 4 You",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🔬 RA4U - Research Assistant 4 You</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Multi-Agent AI System for Academic Research and Research Gap Analysis</p>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model_provider = st.selectbox(
        "Choose AI Model Provider",
        ["OpenAI", "Google"],
        help="Select the AI model provider for the agents"
    )
    
    # API Key input
    api_key = st.text_input(
        f"{model_provider} API Key", 
        type="password", 
        help=f"Enter your {model_provider} API key"
    )
    
    # Validate API key
    if api_key:
        if not validate_api_key(api_key, model_provider):
            st.error("⚠️ Invalid API key format. Please check your key.")
        else:
            st.success("✅ API key format is valid")
    
    # Research parameters
    st.header("🔍 Research Parameters")
    max_papers = st.slider("Maximum Papers to Analyze", 5, 20, 10)
    similarity_threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.7)
    include_verification = st.checkbox("Enable Verification", value=True)
    min_citations = st.number_input("Minimum Citations", min_value=0, value=5)
    
    # Domain selection
    domain = st.selectbox(
        "Research Domain",
        ["Computer Science", "Physics", "Biology", "Chemistry", "Mathematics", "Engineering", "Medicine", "Other"]
    )
    
    # Date range
    date_range = st.selectbox(
        "Publication Date Range",
        ["Last 2 years", "Last 5 years", "All time"]
    )

# Initialize session state
if 'research_team' not in st.session_state:
    st.session_state.research_team = None
if 'research_report' not in st.session_state:
    st.session_state.research_report = None
if 'workflow_status' not in st.session_state:
    st.session_state.workflow_status = None

# Main interface
if api_key and validate_api_key(api_key, model_provider):
    # Setup environment
    setup_environment(api_key, model_provider)
    
    # Initialize research team if not already done
    if st.session_state.research_team is None:
        try:
            with st.spinner("🔧 Initializing research team..."):
                config = AgentConfig(
                    model_provider=model_provider,
                    model_name="gpt-4o-mini" if model_provider == "OpenAI" else "gemini-1.5-flash",
                    api_key=api_key
                )
                st.session_state.research_team = ResearchTeam(config)
                st.success("✅ Research team initialized successfully!")
        except Exception as e:
            st.error(f"❌ Error initializing research team: {str(e)}")
            st.stop()
    
    # Research query input
    st.header("🔍 Research Query")
    
    research_topic = st.text_area(
        "Enter your research topic or question:",
        placeholder="e.g., 'Machine Learning in Healthcare', 'Quantum Computing Applications', 'Sustainable Energy Technologies'",
        height=100,
        help="Be specific about your research area to get better results"
    )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            custom_keywords = st.text_input(
                "Additional Keywords (comma-separated)",
                help="Add specific keywords to improve search accuracy"
            )
            
        with col2:
            venue_filters = st.text_input(
                "Venue Filters (comma-separated)",
                help="Filter by specific conferences or journals"
            )
    
    # Process research query
    if st.button("🚀 Start Research Analysis", type="primary", use_container_width=True):
        if research_topic:
            # Validate query
            is_valid, error_message = validate_research_query(research_topic, max_papers, min_citations)
            
            if not is_valid:
                st.error(f"❌ {error_message}")
            else:
                try:
                    # Create research query
                    query = ResearchQuery(
                        topic=research_topic,
                        domain=domain,
                        keywords=[kw.strip() for kw in custom_keywords.split(',') if kw.strip()],
                        date_range=date_range,
                        max_papers=max_papers,
                        min_citations=min_citations,
                        venue_filters=[vf.strip() for vf in venue_filters.split(',') if vf.strip()],
                        similarity_threshold=similarity_threshold,
                        include_verification=include_verification
                    )
                    
                    # Run research workflow
                    with st.spinner("🔍 Conducting research analysis..."):
                        # Create progress container
                        progress_container = st.container()
                        status_container = st.container()
                        
                        # Run the workflow
                        report = await st.session_state.research_team.run_research_workflow(query)
                        st.session_state.research_report = report
                        
                        # Get workflow status
                        st.session_state.workflow_status = st.session_state.research_team.get_workflow_status()
                    
                    # Display results
                    st.success("✅ Research analysis completed!")
                    display_research_results(report)
                    
                except Exception as e:
                    st.error(f"❌ Error during research analysis: {str(e)}")
                    st.info("Please check your API key and try again.")
        else:
            st.warning("⚠️ Please enter a research topic to begin analysis.")
    
    # Display workflow status if available
    if st.session_state.workflow_status:
        display_workflow_status(st.session_state.workflow_status)
    
    # Display research results if available
    if st.session_state.research_report:
        display_research_results(st.session_state.research_report)

else:
    # Show demo information when no API key is provided
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

def display_workflow_status(status: dict):
    """Display current workflow status"""
    st.markdown("### 📊 Workflow Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Status", status['status'].title())
    
    with col2:
        st.metric("Progress", f"{status['progress']}%")
    
    with col3:
        st.metric("Papers Found", status['papers_found'])
    
    with col4:
        st.metric("Gaps Identified", status['gaps_found'])
    
    # Progress bar
    st.progress(status['progress'] / 100)
    
    if status['error_message']:
        st.error(f"Error: {status['error_message']}")

def display_research_results(report: ResearchReport):
    """Display comprehensive research results"""
    st.markdown("## 📊 Research Analysis Results")
    
    # Executive summary
    if report.summary:
        st.markdown("### 📋 Executive Summary")
        st.info(report.summary)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Papers Found", 
        "🔍 Analysis", 
        "⚠️ Limitations", 
        "🎯 Research Gaps",
        "💡 Recommendations"
    ])
    
    with tab1:
        display_papers_section(report.papers)
    
    with tab2:
        display_analysis_section(report)
    
    with tab3:
        display_limitations_section(report.limitations)
    
    with tab4:
        display_gaps_section(report.research_gaps)
    
    with tab5:
        display_recommendations_section(report.recommendations)
    
    # Processing information
    if report.processing_time:
        st.markdown("---")
        st.markdown(f"**Processing Time:** {report.processing_time:.2f} seconds")
        st.markdown(f"**Generated At:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")

def display_papers_section(papers):
    """Display discovered papers"""
    if not papers:
        st.info("No papers found.")
        return
    
    st.markdown(f"### Found {len(papers)} Academic Papers")
    
    for i, paper in enumerate(papers, 1):
        with st.expander(f"{i}. {paper.title}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Authors:** {', '.join(paper.authors) if paper.authors else 'Unknown'}")
                st.markdown(f"**Published:** {paper.published}")
                st.markdown(f"**Venue:** {paper.venue if paper.venue else 'Not specified'}")
                st.markdown(f"**Citations:** {paper.citation_count}")
                st.markdown(f"**Relevance Score:** {paper.relevance_score:.2f}")
            
            with col2:
                st.markdown(f"**Status:** {paper.verification_status.value}")
                if paper.doi:
                    st.markdown(f"**DOI:** {paper.doi}")
                if paper.url:
                    st.markdown(f"[View Paper]({paper.url})")
            
            st.markdown("**Abstract:**")
            st.markdown(paper.abstract)

def display_analysis_section(report: ResearchReport):
    """Display analysis results"""
    st.markdown("### Paper Analysis")
    
    if not report.papers:
        st.info("No papers to analyze.")
        return
    
    # Analysis summary
    total_papers = len(report.papers)
    avg_relevance = sum(paper.relevance_score for paper in report.papers) / total_papers if total_papers > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Papers", total_papers)
    
    with col2:
        st.metric("Average Relevance", f"{avg_relevance:.2f}")
    
    with col3:
        st.metric("Verified Papers", len([p for p in report.papers if p.verification_status.value == 'verified']))
    
    # Paper analysis details
    for i, paper in enumerate(report.papers, 1):
        with st.expander(f"Analysis: {paper.title}"):
            st.markdown(f"**Relevance Score:** {paper.relevance_score:.2f}")
            st.markdown(f"**Verification Status:** {paper.verification_status.value}")
            if paper.keywords:
                st.markdown(f"**Keywords:** {', '.join(paper.keywords)}")

def display_limitations_section(limitations):
    """Display identified limitations"""
    if not limitations:
        st.info("No limitations identified.")
        return
    
    st.markdown(f"### Identified {len(limitations)} Research Limitations")
    
    # Group by type
    by_type = {}
    for limitation in limitations:
        limitation_type = limitation.limitation_type.value
        if limitation_type not in by_type:
            by_type[limitation_type] = []
        by_type[limitation_type].append(limitation)
    
    for limitation_type, type_limitations in by_type.items():
        st.markdown(f"#### {limitation_type.title()} Limitations ({len(type_limitations)})")
        
        for limitation in type_limitations:
            with st.expander(f"{limitation.paper_title} - {limitation.description[:100]}..."):
                st.markdown(f"**Type:** {limitation.limitation_type.value}")
                st.markdown(f"**Description:** {limitation.description}")
                st.markdown(f"**Confidence:** {limitation.confidence:.2f}")
                st.markdown(f"**Impact Level:** {limitation.impact_level}")

def display_gaps_section(research_gaps):
    """Display identified research gaps"""
    if not research_gaps:
        st.info("No research gaps identified.")
        return
    
    st.markdown(f"### Identified {len(research_gaps)} Research Gaps")
    
    # Group by priority
    by_priority = {}
    for gap in research_gaps:
        priority = gap.priority.value
        if priority not in by_priority:
            by_priority[priority] = []
        by_priority[priority].append(gap)
    
    for priority in ['high', 'medium', 'low']:
        if priority in by_priority:
            gaps = by_priority[priority]
            st.markdown(f"#### {priority.title()} Priority Gaps ({len(gaps)})")
            
            for gap in gaps:
                with st.expander(f"{gap.gap_title}"):
                    st.markdown(f"**Priority:** {gap.priority.value}")
                    st.markdown(f"**Description:** {gap.description}")
                    st.markdown(f"**Potential Impact:** {gap.potential_impact}")
                    if gap.suggested_methodology:
                        st.markdown(f"**Suggested Methodology:** {gap.suggested_methodology}")

def display_recommendations_section(recommendations):
    """Display research recommendations"""
    if not recommendations:
        st.info("No recommendations available.")
        return
    
    st.markdown(f"### Research Recommendations ({len(recommendations)})")
    
    for i, recommendation in enumerate(recommendations, 1):
        st.markdown(f"{i}. {recommendation}")

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
