import streamlit as st
from dotenv import load_dotenv
from ra4u_agents.ochestrator import run_research
import time
import re
import sys

load_dotenv()

class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []
        self.colors = ['blue', 'green', 'orange', 'red', 'violet']
        self.color_index = 0

    def write(self, data):
        if not isinstance(data, str):
            data = str(data)
            
    def flush(self):
        if self.buffer:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []
            
    def write(self, data):
        # Filter out ANSI escape codes
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)
        
        # Check for agent names and color them
        agent_patterns = [
            "Article Crawler",
            "Article Reader",
            "Technical Writer"
        ]
        
        for agent in agent_patterns:
            if agent in cleaned_data:
                self.color_index = agent_patterns.index(agent)
                cleaned_data = cleaned_data.replace(
                    agent, 
                    f":{self.colors[self.color_index]}[{agent}]"
                )
        
        # Color the workflow markers
        workflow_markers = [
            ("Entering new CrewAgentExecutor chain", "starting"),
            ("Finished chain.", "completed")
        ]
        
        for marker, status in workflow_markers:
            if marker in cleaned_data:
                cleaned_data = cleaned_data.replace(
                    marker,
                    f":{self.colors[self.color_index]}[{marker}]"
                )
                if status == "starting":
                    st.toast(f"🤖 {agent_patterns[self.color_index]} is working...")
                elif status == "completed":
                    st.toast(f"✅ {agent_patterns[self.color_index]} completed!", icon="✅")

        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.flush()

# Set up the Streamlit app
st.set_page_config(
    page_title="RA4U - Research Assistant 4 You",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 RA4U - Research Assistant 4 You")
st.caption("Multi-Agent AI System for Academic Research and Research Gap Analysis")

# Create tabs
tab1, tab2 = st.tabs(["🎯 About RA4U Demo", "🔍 Research Query"])

with tab1:
    st.markdown("""
    This demo showcases the RA4U (Research Assistant 4 You) multi-agent system designed to:

    ### 🔍 **Core Features:**
    - **Multi-Agent Architecture**: 4 specialized agents working together
    - **Academic Paper Discovery**: Search and filter relevant research papers
    - **Verification System**: Prevent hallucination and ensure accuracy
    - **Limitation Analysis**: Identify research limitations systematically
    - **Gap Identification**: Find research opportunities and future directions

    ### 🤖 **Agent Roles:**
    1. **Crawler**: Discovers relevant academic papers, ensures accuracy and prevents hallucination.
    2. **Reader**: Read and analyzes the limitations of topic via the discovered papers.
    3. **Writer**: Writes a comprehensive report.
    4. **Orchestrator**: Ochestrates the workflow among all agents.

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
    # Research topic input
    research_topic = st.text_area(
        "Enter your research topic or question:",
        placeholder="e.g., 'Machine Learning in Healthcare', 'Quantum Computing Applications'",
        height=100
    )

    # Process research query
    if st.button("🚀 Start Research Analysis", type="primary"):
        if research_topic:
            with st.status("🤖 **Research Agents at Work...**", state="running", expanded=True) as status:

                with st.container(height=600, border=False):
                    sys.stdout = StreamToExpander(st)
                    response = run_research(research_topic)
                    sys.stdout = sys.__stdout__  # Reset stdout
                
            status.update(label="✅ Research Analysis Complete!", 
                        state="complete", 
                        expanded=False)
            
            st.markdown("## 📊 Research Analysis Results")
            st.markdown(response)
        else:
            st.warning("⚠️ Please enter a research topic to begin analysis.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🔬 RA4U - Research Assistant 4 You | Multi-Agent AI System for Academic Research</p>
        <p>Built with CrewAI Framework and Streamlit | Summer School Agentic Course - Final Project</p>
    </div>
    """,
    unsafe_allow_html=True
)