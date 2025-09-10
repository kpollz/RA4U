# RA4U - Project Structure

## 📁 Complete File Structure

```
RA4U/
├── app.py                           # 🎯 Main Streamlit UI application
├── models.py                        # 📊 Pydantic data models and schemas
├── utils.py                         # 🔧 Utility functions and helpers
├── search_agent.py                  # 🔍 Search Agent for paper discovery
├── verification_agent.py            # ✅ Verification Agent for accuracy checking
├── analysis_agent.py                # 📈 Analysis Agent for content analysis
├── limitation_agent.py              # ⚠️ Limitation Agent for identifying limitations
├── gap_agent.py                     # 🎯 Gap Agent for research gap identification
├── research_team.py                 # 🤝 Team coordination and workflow orchestration
├── requirements.txt                 # 📦 Python dependencies
├── README.md                        # 📖 Main project documentation
├── README_DEMO.md                   # 🚀 Demo-specific documentation
├── README_MODULAR.md                # 🏗️ Modular architecture guide
├── PROJECT_STRUCTURE.md             # 📁 This file - project structure
├── ra4u_demo.py                     # 🗑️ Legacy demo file (can be removed)
└── docs/                            # 📚 Documentation folder
    ├── High-level_Design.md         # 🏗️ High-level system design
    ├── Low-level_Design.md          # 🔧 Low-level technical specifications
    └── Summer School Agentic Course - Final Project.txt  # 📋 Project requirements
```

## 🎯 File Purposes

### Core Application Files

| File | Purpose | Key Components |
|------|---------|----------------|
| `app.py` | Main UI application | Streamlit interface, user interaction, result display |
| `models.py` | Data layer | Pydantic models, type definitions, validation |
| `utils.py` | Utility layer | Helper functions, text processing, configuration |

### Agent Files

| File | Purpose | Key Components |
|------|---------|----------------|
| `search_agent.py` | Paper discovery | arXiv search, filtering, ranking |
| `verification_agent.py` | Accuracy checking | Cross-reference validation, hallucination detection |
| `analysis_agent.py` | Content analysis | Concept extraction, contribution analysis |
| `limitation_agent.py` | Limitation identification | Methodological constraints, scope limitations |
| `gap_agent.py` | Gap identification | Research opportunities, future directions |

### Coordination Files

| File | Purpose | Key Components |
|------|---------|----------------|
| `research_team.py` | Workflow orchestration | Agent coordination, state management, report generation |

### Documentation Files

| File | Purpose | Content |
|------|---------|---------|
| `README.md` | Main documentation | Project overview, features, usage |
| `README_DEMO.md` | Demo guide | Quick start, examples, troubleshooting |
| `README_MODULAR.md` | Architecture guide | Modular design, development guidelines |
| `PROJECT_STRUCTURE.md` | This file | File structure, purposes, organization |

## 🔄 Data Flow

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

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Application
```bash
streamlit run app.py
```

### 3. Access Interface
- Open browser to `http://localhost:8501`
- Configure API key
- Enter research topic
- Start analysis

## 🛠️ Development Workflow

### Adding New Features

1. **Data Models**: Update `models.py`
2. **Utilities**: Add to `utils.py`
3. **New Agent**: Create new agent file
4. **Team Integration**: Update `research_team.py`
5. **UI Updates**: Modify `app.py`

### Testing Individual Components

```python
# Test individual agent
from search_agent import SearchAgent
from models import AgentConfig, ResearchQuery

config = AgentConfig(model_provider="OpenAI", api_key="your-key")
agent = SearchAgent(config)
query = ResearchQuery(topic="Test Topic")
papers = await agent.search_papers(query)
```

### Debugging

```python
# Enable debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Check workflow status
status = research_team.get_workflow_status()
print(status)
```

## 📊 Architecture Benefits

### ✅ **Modularity**
- Each file has a single responsibility
- Easy to locate and modify specific functionality
- Clear separation of concerns

### ✅ **Maintainability**
- Isolated components reduce complexity
- Easy to debug and fix issues
- Consistent code organization

### ✅ **Scalability**
- Easy to add new agents or features
- Components can be developed independently
- Parallel development possible

### ✅ **Testability**
- Each component can be tested in isolation
- Mock dependencies easily
- Unit tests for individual agents

### ✅ **Reusability**
- Agents can be used independently
- Common utilities shared across components
- Easy to create different interfaces

## 🔧 Configuration

### Environment Setup
```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Google Gemini
export GOOGLE_API_KEY="your-google-key"
```

### Model Configuration
```python
config = AgentConfig(
    model_provider="OpenAI",  # or "Google"
    model_name="gpt-4o-mini",  # or "gemini-1.5-flash"
    max_tokens=2048,
    temperature=0.3,
    api_key="your-api-key"
)
```

## 📝 File Naming Conventions

- **Snake case** for all Python files
- **Descriptive names** that indicate purpose
- **Agent files** end with `_agent.py`
- **Documentation files** start with `README_` or end with `.md`

## 🗑️ Cleanup

### Files to Remove
- `ra4u_demo.py` - Legacy demo file (replaced by modular structure)

### Files to Keep
- All other files are part of the modular architecture
- Each serves a specific purpose in the system

---

*This modular structure provides a clean, maintainable, and scalable foundation for the RA4U Research Assistant system.*
