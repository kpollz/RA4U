from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from utils import LinkUpSearchTool

# Load environment variables (for non-LinkUp settings)
load_dotenv()

LLM_CLIENT = LLM(model="gemini/gemini-2.0-flash")
LINKUP_SEARCH_TOOL = LinkUpSearchTool()


def init_agents():
    web_searcher = Agent(
        role="Web Searcher",
        goal="Find the 10 most relevant scientific articles on the web (conferences, top scientific research journals,...), along with source links (urls).",
        backstory="An expert at formulating search queries and retrieving relevant information. Passes the results to the 'Research Analyst' only.",
        verbose=True,
        allow_delegation=True,
        tools=[LINKUP_SEARCH_TOOL],
        llm=LLM_CLIENT,
    )

    # Define the research analyst
    research_analyst = Agent(
        role="Research Analyst",
        goal="Analyze and synthesize raw information about current limitation into structured insights, along with source links (urls) as citations.",
        backstory="An expert at analyzing information, identifying patterns, and extracting key insights. If required, can delagate the task of fact checking/verification to 'Web Searcher' only. Passes the final results to the 'Technical Writer' only.",
        verbose=True,
        allow_delegation=True,
        llm=LLM_CLIENT,
    )

    # Define the technical writer
    technical_writer = Agent(
        role="Technical Writer",
        goal="Create well-structured, clear, and comprehensive responses in markdown format, with citations/source links (urls). The structure should include an introduction, limitation, research gap and future research.",
        backstory="An expert at communicating complex information in an accessible way.",
        verbose=True,
        allow_delegation=False,
        llm=LLM_CLIENT,
    )
    return web_searcher, research_analyst, technical_writer


def create_research_crew(query: str, web_searcher=None, research_analyst=None, technical_writer=None) -> Crew:
    """Create and configure the research crew with all agents and tasks"""

    # Define tasks
    search_task = Task(
        description=f"Search for comprehensive information about: {query}.",
        agent=web_searcher,
        expected_output="Detailed raw search results including sources (urls).",
        tools=[LINKUP_SEARCH_TOOL]
    )

    analysis_task = Task(
        description="Analyze the raw search results, identify key limitation, verify facts and prepare a structured analysis.",
        agent=research_analyst,
        expected_output="A structured analysis of the limitation with verified facts and key insights, along with source links",
        context=[search_task]
    )

    writing_task = Task(
        description="Create a comprehensive, well-organized response based on the research analysis.",
        agent=technical_writer,
        expected_output="A clear, comprehensive response that directly answers the query with proper citations/source links (urls).",
        context=[analysis_task]
    )

    # Create the crew
    crew = Crew(
        agents=[web_searcher, research_analyst, technical_writer],
        tasks=[search_task, analysis_task, writing_task],
        verbose=True,
        process=Process.sequential
    )

    return crew


def run_research(query: str):
    """Run the research process and return results"""
    try:
        web_searcher, research_analyst, technical_writer = init_agents()
        crew = create_research_crew(query, web_searcher, research_analyst, technical_writer)
        result = crew.kickoff()
        return result.raw
    except Exception as e:
        return f"Error: {str(e)}"
