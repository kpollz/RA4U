from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from utils import LinkUpSearchTool
from ra4u_agents.writer import create_writer_agent
from typing import Callable, Optional

# Load environment variables (for non-LinkUp settings)
load_dotenv()

LLM_CLIENT = LLM(model="gemini/gemini-2.0-flash")
LINKUP_SEARCH_TOOL = LinkUpSearchTool()


def init_agents(callback: Optional[Callable] = None):
    """Initialize agents with progress tracking"""
    if callback:
        callback("Web Searcher", "initializing", "Initializing Web Search Agent...")
    
    web_searcher = Agent(
        role="Web Searcher",
        goal="Find the 10 most relevant scientific articles on the web (conferences, top scientific research journals,...), along with source links (urls).",
        backstory="An expert at formulating search queries and retrieving relevant information. Passes the results to the 'Research Analyst' only.",
        verbose=True,
        allow_delegation=True,
        tools=[LINKUP_SEARCH_TOOL],
        llm=LLM_CLIENT,
    )

    if callback:
        callback("Research Analyst", "initializing", "Initializing Research Analysis Agent...")
    
    research_analyst = Agent(
        role="Research Analyst",
        goal="Analyze and synthesize raw information about current limitation into structured insights, along with source links (urls) as citations.",
        backstory="An expert at analyzing information, identifying patterns, and extracting key insights. If required, can delagate the task of fact checking/verification to 'Web Searcher' only. Passes the final results to the 'Technical Writer' only.",
        verbose=True,
        allow_delegation=True,
        llm=LLM_CLIENT,
    )

    if callback:
        callback("Technical Writer", "initializing", "Initializing Technical Writer Agent...")
    
    technical_writer = create_writer_agent(llm=LLM_CLIENT)
    return web_searcher, research_analyst, technical_writer


def create_research_crew(query: str, web_searcher=None, research_analyst=None, technical_writer=None, callback: Optional[Callable] = None) -> Crew:
    """Create and configure the research crew with all agents and tasks"""

    # Define tasks with progress tracking
    search_task = Task(
        description=f"Search for comprehensive information about: {query}.",
        agent=web_searcher,
        expected_output="Detailed raw search results including sources (urls).",
        tools=[LINKUP_SEARCH_TOOL],
        on_start=lambda: callback("Web Searcher", "started", f"Starting web search for: {query}") if callback else None,
        on_end=lambda: callback("Web Searcher", "completed", "Web search completed") if callback else None,
    )

    analysis_task = Task(
        description="Analyze the raw search results, identify key limitation, verify facts and prepare a structured analysis.",
        agent=research_analyst,
        expected_output="A structured analysis of the limitation with verified facts and key insights, along with source links",
        context=[search_task],
        on_start=lambda: callback("Research Analyst", "started", "Starting research analysis") if callback else None,
        on_end=lambda: callback("Research Analyst", "completed", "Research analysis completed") if callback else None,
    )

    writing_task = Task(
        description="Create a comprehensive, well-organized response based on the research analysis.",
        agent=technical_writer,
        expected_output="A clear, comprehensive response that directly answers the query with proper citations/source links (urls).",
        context=[analysis_task],
        on_start=lambda: callback("Technical Writer", "started", "Starting technical writing") if callback else None,
        on_end=lambda: callback("Technical Writer", "completed", "Technical writing completed") if callback else None,
    )

    # Create the crew
    crew = Crew(
        agents=[web_searcher, research_analyst, technical_writer],
        tasks=[search_task, analysis_task, writing_task],
        verbose=True,
        process=Process.sequential
    )

    return crew


def run_research(query: str, callback: Optional[Callable] = None):
    """Run the research process and return results with progress tracking"""
    try:
        if callback:
            callback("System", "started", "Initializing research process...")
        
        web_searcher, research_analyst, technical_writer = init_agents(callback)
        crew = create_research_crew(query, web_searcher, research_analyst, technical_writer, callback)
        
        if callback:
            callback("System", "progress", "Starting crew tasks...")
        
        result = crew.kickoff()
        
        if callback:
            callback("System", "completed", "Research process completed successfully!")
        
        return result.raw
    except Exception as e:
        if callback:
            callback("System", "error", f"Error occurred: {str(e)}")
        return f"Error: {str(e)}"