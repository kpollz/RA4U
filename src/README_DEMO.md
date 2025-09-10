# RA4U Demo - Research Assistant 4 You

## 🚀 Quick Start

This demo showcases the RA4U multi-agent research system using the Agno framework and Streamlit.

### Prerequisites

- Python 3.8 or higher
- OpenAI API key OR Google Gemini API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd RA4U
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the demo:**
   ```bash
   streamlit run ra4u_demo.py
   ```

4. **Open your browser:**
   Navigate to `http://localhost:8501`

## 🔧 Configuration

### API Keys

You'll need one of the following API keys:

- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Google Gemini API Key**: Get from [Google AI Studio](https://aistudio.google.com/apikey)

### Model Selection

The demo supports both OpenAI and Google Gemini models:
- **OpenAI**: GPT-4o-mini (recommended for cost-effectiveness)
- **Google Gemini**: Gemini-1.5-flash (good performance)

## 🎯 Features Demonstrated

### Multi-Agent Architecture

The demo implements 5 specialized agents:

1. **🔍 Search Agent**
   - Discovers relevant academic papers using arXiv
   - Filters by relevance, citations, and recency
   - Ranks papers by quality metrics

2. **✅ Verification Agent**
   - Cross-references paper information
   - Prevents hallucination and ensures accuracy
   - Validates publication details

3. **📊 Analysis Agent**
   - Extracts key concepts and methodologies
   - Analyzes paper contributions
   - Calculates relevance scores

4. **⚠️ Limitation Agent**
   - Identifies research limitations
   - Categorizes limitation types
   - Assesses impact on research validity

5. **🎯 Gap Agent**
   - Synthesizes limitations across papers
   - Identifies research opportunities
   - Proposes future research directions

### User Interface

- **Research Query Input**: Natural language topic specification
- **Advanced Options**: Domain selection, date range, citation filters
- **Real-time Progress**: Live updates during analysis
- **Structured Results**: Organized tabs for different analysis aspects

## 📝 Usage Examples

### Example 1: Machine Learning Research
```
Topic: "Deep Learning in Medical Image Analysis"
Domain: Computer Science
Date Range: Last 2 years
Min Citations: 10
```

### Example 2: Quantum Computing
```
Topic: "Quantum Machine Learning Algorithms"
Domain: Physics
Date Range: Last 5 years
Min Citations: 5
```

### Example 3: Sustainable Technology
```
Topic: "Renewable Energy Storage Solutions"
Domain: Engineering
Date Range: All time
Min Citations: 20
```

## 🔍 How It Works

### Research Workflow

1. **Input Processing**: User specifies research topic and parameters
2. **Paper Discovery**: Search Agent finds relevant academic papers
3. **Verification**: Verification Agent ensures accuracy and prevents hallucination
4. **Analysis**: Analysis Agent processes paper content and contributions
5. **Limitation Identification**: Limitation Agent finds research constraints
6. **Gap Analysis**: Gap Agent synthesizes findings and identifies opportunities
7. **Report Generation**: Comprehensive research report with structured results

### Prompt Engineering

The system uses sophisticated prompt engineering to:
- Guide each agent's specialized role
- Ensure consistent output formatting
- Maintain focus on research objectives
- Prevent hallucination and ensure accuracy

## 🛠️ Technical Implementation

### Framework Stack
- **Agno**: Multi-agent orchestration framework
- **Streamlit**: Interactive web interface
- **Pydantic**: Data validation and modeling
- **arXiv API**: Academic paper discovery

### Agent Communication
- **Coordination Mode**: Agents work together under team coordination
- **Tool Integration**: Each agent has specialized tools
- **Response Streaming**: Real-time progress updates
- **Error Handling**: Robust error management

## 📊 Expected Outputs

### Research Report Structure
1. **Executive Summary**: Overview of findings
2. **Related Works**: List of discovered papers with metadata
3. **Analysis Results**: Content analysis and contributions
4. **Limitations**: Categorized research constraints
5. **Research Gaps**: Identified opportunities and future directions
6. **Recommendations**: Actionable next steps for researchers

### Quality Metrics
- **Accuracy**: Verification prevents hallucination
- **Relevance**: Papers ranked by topic similarity
- **Completeness**: Comprehensive coverage of research area
- **Actionability**: Clear recommendations for future work

## 🔧 Customization

### Adding New Agents
```python
new_agent = Agent(
    name="Custom Agent",
    model=model_config,
    role="Your custom role description",
    tools=[YourCustomTools()],
    instructions=["Your custom instructions"]
)
```

### Modifying Prompts
Edit the agent instructions in `ra4u_demo.py` to customize behavior:
```python
instructions=[
    "Your custom instruction 1",
    "Your custom instruction 2",
    # Add more instructions as needed
]
```

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your API key is valid and has sufficient credits
2. **Import Error**: Install all requirements with `pip install -r requirements.txt`
3. **No Results**: Try broader search terms or adjust parameters
4. **Slow Response**: Reduce max_papers or use a faster model

### Debug Mode
Enable debug mode in the team configuration to see detailed agent interactions:
```python
debug_mode=True,
show_members_responses=True,
```

## 📈 Performance Tips

1. **Model Selection**: Use GPT-4o-mini for cost-effectiveness
2. **Paper Limits**: Start with 5-10 papers for faster results
3. **Topic Specificity**: More specific topics yield better results
4. **Parameter Tuning**: Adjust similarity thresholds based on needs

## 🔮 Future Enhancements

- **Vector Database Integration**: For semantic similarity search
- **Citation Network Analysis**: Graph-based paper relationships
- **Real-time Collaboration**: Multiple users working together
- **Export Features**: PDF and Word report generation
- **Advanced Filtering**: More sophisticated paper selection criteria

---

*This demo is part of the Summer School Agentic Course - Final Project*
