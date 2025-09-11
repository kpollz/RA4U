# RA4U - Research Assistant 4 You Demo
# Multi-Agent System for Academic Research and Gap Analysis

import streamlit as st
from dotenv import load_dotenv
from research_agent import run_research

load_dotenv()

# Set up the Streamlit app
st.set_page_config(
    page_title="RA4U - Research Assistant 4 You",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 RA4U - Research Assistant 4 You")
st.caption("Multi-Agent AI System for Academic Research and Research Gap Analysis")

tab1, tab2 = st.tabs(["🎯 About RA4U Demo", "🔍 Research Query"])
                    
with tab1:
    # Show demo information
    # st.markdown("## 🎯 About RA4U Demo")

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

with tab2:

    # Main interface
    # st.header("🔍 Research Query")

    # Research topic input
    research_topic = st.text_area(
        "Enter your research topic or question:",
        placeholder="e.g., 'Machine Learning in Healthcare', 'Quantum Computing Applications', 'Sustainable Energy Technologies'",
        height=100
    )

    # Process research query
    if st.button("🚀 Start Research Analysis", type="primary"):
        if research_topic:
            with st.spinner("🔍 Conducting research analysis..."):
            
                # Run the research team
                try:
                    response = run_research(research_topic)
                    
                    # Display results
                    st.success("✅ Research analysis completed!")
                    
                    # Display the response
                    st.markdown("## 📊 Research Analysis Results")
                    st.markdown(response)
                    
                except Exception as e:
                    st.error(f"❌ Error during research analysis: {str(e)}")
                    st.info("Please check your API key and try again.")
        else:
            st.warning("⚠️ Please enter a research topic to begin analysis.")

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
