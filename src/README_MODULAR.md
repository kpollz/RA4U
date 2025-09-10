# RA4U - Modular Architecture

## 📁 Project Structure

```
RA4U/
├── app.py                    # Main Streamlit UI application
├── models.py                 # Pydantic data models and schemas
├── utils.py                  # Utility functions and helpers
├── search_agent.py          # Search Agent for paper discovery
├── verification_agent.py    # Verification Agent for accuracy checking
├── analysis_agent.py        # Analysis Agent for content analysis
├── limitation_agent.py      # Limitation Agent for identifying limitations
├── gap_agent.py             # Gap Agent for research gap identification
├── research_team.py         # Team coordination and workflow orchestration
├── requirements.txt         # Python dependencies
├── README.md               # Main project documentation
├── README_DEMO.md          # Demo-specific documentation
├── README_MODULAR.md       # This file - modular architecture guide
└── docs/                   # Documentation folder
    ├── High-level_Design.md
    └── Summer School Agentic Course - Final Project.txt
```

## 🏗️ Modular Architecture Overview

The RA4U system has been refactored into a clean, modular architecture where each component has a specific responsibility:

### Core Components

#### 1. **`models.py`** - Data Layer
- **Purpose**: Defines all Pydantic models for type safety and validation
- **Contains**: Paper, ResearchQuery, ResearchReport, Limitation, ResearchGap, etc.
- **Benefits**: Centralized data structures, automatic validation, IDE support

#### 2. **`utils.py`** - Utility Layer
- **Purpose**: Helper functions and common utilities
- **Contains**: Text processing, validation, formatting, configuration management
- **Benefits**: Reusable code, consistent behavior, easy maintenance

#### 3. **`app.py`** - UI Layer
- **Purpose**: Streamlit user interface and user interaction
- **Contains**: UI components, user input handling, result display
- **Benefits**: Clean separation of UI and business logic

### Agent Components

#### 4. **`search_agent.py`** - Search Agent
- **Purpose**: Discovers relevant academic papers
- **Responsibilities**:
  - Search arXiv and other academic databases
  - Filter papers by relevance and quality
  - Rank papers by importance metrics
  - Extract paper metadata

#### 5. **`verification_agent.py`** - Verification Agent
- **Purpose**: Ensures accuracy and prevents hallucination
- **Responsibilities**:
  - Cross-reference paper information
  - Verify paper existence and metadata
  - Detect potential hallucinations
  - Provide confidence scores

#### 6. **`analysis_agent.py`** - Analysis Agent
- **Purpose**: Processes and analyzes paper content
- **Responsibilities**:
  - Extract key concepts and methodologies
  - Identify research contributions
  - Calculate relevance and quality scores
  - Analyze paper structure

#### 7. **`limitation_agent.py`** - Limitation Agent
- **Purpose**: Identifies research limitations
- **Responsibilities**:
  - Find methodological limitations
  - Detect scope constraints
  - Categorize limitation types
  - Assess limitation impact

#### 8. **`gap_agent.py`** - Gap Agent
- **Purpose**: Identifies research gaps and opportunities
- **Responsibilities**:
  - Synthesize limitations into gaps
  - Identify unexplored research areas
  - Propose novel research directions
  - Prioritize gaps by importance

### Coordination Layer

#### 9. **`research_team.py`** - Team Orchestration
- **Purpose**: Coordinates the complete research workflow
- **Responsibilities**:
  - Manage workflow state and progress
  - Orchestrate agent interactions
  - Generate comprehensive reports
  - Handle error recovery

## 🔄 Workflow Architecture

```
User Input (app.py)
    ↓
Research Query (models.py)
    ↓
Research Team (research_team.py)
    ↓
┌─────────────────────────────────────────┐
│  Agent Workflow (Sequential)           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Search  │→│ Verify  │→│ Analyze │   │
│  └─────────┘ └─────────┘ └─────────┘   │
│  ┌─────────┐ ┌─────────┐               │
│  │Limitation│→│  Gap   │               │
│  └─────────┘ └─────────┘               │
└─────────────────────────────────────────┘
    ↓
Research Report (models.py)
    ↓
UI Display (app.py)
```

## 🚀 Usage Instructions

### Running the Application

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the main application:**
   ```bash
   streamlit run app.py
   ```

3. **Access the application:**
   - Open your browser to `http://localhost:8501`
   - Configure your API key
   - Enter research topic and start analysis

### Using Individual Components

#### Using a Single Agent

```python
from models import AgentConfig, ResearchQuery
from search_agent import SearchAgent

# Configure agent
config = AgentConfig(
    model_provider="OpenAI",
    model_name="gpt-4o-mini",
    api_key="your-api-key"
)

# Create search agent
search_agent = SearchAgent(config)

# Create research query
query = ResearchQuery(
    topic="Machine Learning in Healthcare",
    domain="Computer Science",
    max_papers=10
)

# Search for papers
papers = await search_agent.search_papers(query)
```

#### Using the Research Team

```python
from research_team import ResearchTeam
from models import AgentConfig, ResearchQuery

# Configure team
config = AgentConfig(
    model_provider="OpenAI",
    model_name="gpt-4o-mini",
    api_key="your-api-key"
)

# Create research team
team = ResearchTeam(config)

# Create research query
query = ResearchQuery(
    topic="Quantum Computing Applications",
    domain="Physics",
    max_papers=15
)

# Run complete workflow
report = await team.run_research_workflow(query)
```

## 🛠️ Development Guidelines

### Adding New Agents

1. **Create agent file** (e.g., `new_agent.py`):
   ```python
   from agno.agent import Agent
   from models import AgentConfig
   
   class NewAgent:
       def __init__(self, config: AgentConfig):
           self.config = config
           self.agent = self._create_agent()
       
       def _create_agent(self) -> Agent:
           # Agent configuration
           pass
       
       async def process_data(self, data):
           # Agent logic
           pass
   ```

2. **Update research_team.py**:
   ```python
   from new_agent import NewAgent
   
   class ResearchTeam:
       def _initialize_agents(self):
           agents = {
               # ... existing agents
               'new_agent': NewAgent(self.config)
           }
   ```

3. **Update workflow** in `research_team.py`:
   ```python
   async def run_research_workflow(self, query):
       # ... existing workflow
       
       # Add new stage
       if self.workflow_state.verified_papers:
           self.workflow_state.current_stage = "new_processing"
           new_results = await self.agents['new_agent'].process_data(data)
   ```

### Modifying Data Models

1. **Update models.py**:
   ```python
   class NewModel(BaseModel):
       field1: str
       field2: int
   ```

2. **Update related agents** to use the new model
3. **Update UI components** in `app.py` to display new data

### Adding New Utilities

1. **Add to utils.py**:
   ```python
   def new_utility_function(param1: str, param2: int) -> str:
       """New utility function description"""
       # Implementation
       pass
   ```

2. **Import and use** in relevant agents or UI components

## 📊 Benefits of Modular Architecture

### 1. **Maintainability**
- Each component has a single responsibility
- Easy to locate and fix issues
- Clear separation of concerns

### 2. **Scalability**
- Easy to add new agents or features
- Components can be developed independently
- Parallel development possible

### 3. **Testability**
- Each component can be tested in isolation
- Mock dependencies easily
- Unit tests for individual agents

### 4. **Reusability**
- Agents can be used independently
- Common utilities shared across components
- Easy to create different UI interfaces

### 5. **Flexibility**
- Easy to swap implementations
- Configure different model providers
- Customize workflow stages

## 🔧 Configuration

### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Google Gemini
export GOOGLE_API_KEY="your-google-key"
```

### Agent Configuration

```python
config = AgentConfig(
    model_provider="OpenAI",  # or "Google"
    model_name="gpt-4o-mini",  # or "gemini-1.5-flash"
    max_tokens=2048,
    temperature=0.3,
    api_key="your-api-key"
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **API Key Issues**: Verify API key format and permissions
3. **Agent Errors**: Check individual agent logs
4. **UI Issues**: Verify Streamlit installation and configuration

### Debug Mode

Enable debug mode in agents:
```python
# In agent configuration
debug_mode=True,
show_tool_calls=True,
show_members_responses=True
```

### Logging

Check logs for detailed error information:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

*This modular architecture provides a clean, maintainable, and scalable foundation for the RA4U Research Assistant system.*
