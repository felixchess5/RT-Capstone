# Assignment Grading System with Parallel Processing

An intelligent academic assignment grading system that uses LLM-based evaluation with parallel processing for efficient analysis of student submissions.

## 🌟 Features

- **Parallel Processing**: Asynchronous orchestrator executes all grading nodes simultaneously
- **LLM-Based Evaluation**: Uses Groq LLM for intelligent content analysis
- **Multi-Criteria Grading**: Evaluates assignments on 4 key metrics (0-10 scale)
- **Plagiarism Detection**: Automated plagiarism checking with detailed reports
- **Source Relevance**: Compares assignments against reference material
- **Grammar Analysis**: Writing quality assessment independent of content accuracy
- **Automated Summarization**: Generates concise assignment summaries
- **CSV Export**: Batch processing results exported to structured format
- **LangSmith Integration**: Comprehensive tracing and monitoring of all LLM operations

## 📊 Grading Criteria

| Criterion | Scale | Description |
|-----------|-------|-------------|
| **Factual Accuracy** | 0-10 | Content accuracy compared to source material |
| **Relevance to Source** | 0-10 | How well assignment relates to reference material |
| **Coherence** | 0-10 | Logical structure and flow of writing |
| **Grammar** | 1-10 | Writing quality, spelling, grammar (minimum score: 1) |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Groq API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/felixchess5/RT-Project-2.git
   cd RT-Project-2
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment setup**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env file and add your API keys
   # Required:
   # GROQ_API_KEY=your_actual_api_key_here
   
   # Optional (for LangSmith tracing):
   # LANGCHAIN_TRACING_V2=true
   # LANGCHAIN_API_KEY=your_langsmith_api_key
   # LANGCHAIN_PROJECT=Assignment Grader
   ```

5. **Configure paths** (Optional)
   - Edit `paths.py` to customize file locations
   - Default folders will be created automatically

### Usage

1. **Place assignment files**
   - Add student assignment `.txt` files to the `Assignments/` folder
   - Use this format for each file:
     ```
     Name: John Doe
     Date: 2025-08-25
     Class: 10
     Subject: English
     
     [Assignment content here...]
     ```

2. **Run the grading system**
   ```bash
   python main.py
   ```

3. **View results**
   - Check the generated CSV file in `output/summary.csv` for all results
   - Individual plagiarism reports in `plagiarism_reports/` folder
   - Graph visualization saved as PNG

## 🏗️ Architecture

### System Overview

```
┌─────────────────────┐
│   Main Pipeline     │
│                     │
│  ┌───────────────┐  │
│  │ Orchestrator  │──┼─── Executes nodes in parallel
│  │   (Async)     │  │
│  └───────────────┘  │
│         │            │
│  ┌─────────────────┐ │
│  │ Parallel Nodes  │ │
│  │ ┌─────┐ ┌─────┐ │ │
│  │ │Gram.││Plag.│ │ │
│  │ └─────┘ └─────┘ │ │
│  │ ┌─────┐ ┌─────┐ │ │
│  │ │Rel. ││Grad.│ │ │
│  │ └─────┘ └─────┘ │ │
│  │    ┌─────┐      │ │
│  │    │Summ.│      │ │
│  │    └─────┘      │ │
│  └─────────────────┘ │
└─────────────────────┘
```

### File Structure

```
RT-Project-2/
├── main.py              # Application entry point
├── nodes.py             # Processing nodes (grammar, plagiarism, etc.)
├── orchestrator_node    # Parallel execution orchestrator
├── llms.py              # LLM configuration and setup
├── prompts.py           # Centralized prompt templates
├── utils.py             # Utility functions and graph visualization
├── paths.py             # File path configurations
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
├── .gitignore          # Git ignore patterns
├── Assignments/        # Student assignment files (.txt)
├── plagiarism_reports/ # Generated plagiarism reports (.json)
├── output/             # Generated CSV and other outputs
└── README.md          # This documentation
```

### Code Flow

1. **Initialization** (`main.py`)
   - Load environment variables and configuration
   - Build LangGraph workflow with orchestrator node
   - Generate graph visualization

2. **File Processing** (`process_assignments()`)
   - Scan assignments folder for `.txt` files
   - Extract metadata from file headers
   - Process each file through the orchestrator

3. **Parallel Execution** (`orchestrator_node()`)
   - Create async tasks for all processing nodes
   - Execute simultaneously using `asyncio.gather()`
   - Merge results back into state dictionary

4. **Individual Node Processing**
   - **Grammar Check**: Count grammatical errors using LLM
   - **Plagiarism Check**: Analyze content originality, save reports
   - **Source Check**: Compare assignment to reference material
   - **Grading**: Multi-criteria evaluation (0-10 scale)
   - **Summarization**: Generate 2-3 sentence summaries

5. **Export Results** (`export_summary()`)
   - Aggregate all processing results
   - Generate CSV file with structured data
   - Include metadata and all evaluation scores

## 📊 LangSmith Integration

The system includes comprehensive LangSmith tracing for monitoring and debugging:

### Benefits
- **Full Pipeline Visibility**: Track every LLM call and node execution
- **Performance Monitoring**: Analyze latency and token usage across all operations
- **Error Debugging**: Detailed traces for troubleshooting failed operations
- **Cost Optimization**: Monitor API usage and optimize prompts
- **Quality Assurance**: Compare different prompt versions and model outputs

### Traced Operations
- All grading criteria evaluations (Grammar, Plagiarism, Relevance, Grading, Summary)
- Individual student assignment processing
- Parallel orchestrator execution
- Complete pipeline runs with metadata

Enable tracing by setting `LANGCHAIN_TRACING_V2=true` in your `.env` file.

## 🔧 Configuration

### Environment Variables

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API key:
   ```env
   GROQ_API_KEY=your_actual_groq_api_key_here
   ```

### Path Configuration

Edit `paths.py` to customize:
```python
ASSIGNMENTS_FOLDER = "Assignments"
PLAGIARISM_REPORTS_FOLDER = "plagiarism_reports"  
SUMMARY_CSV_PATH = "output/summary.csv"
GRAPH_OUTPUT_PATH = "graph.png"
```

### LLM Configuration

Modify `llms.py` to adjust:
```python
def create_groq_llm(model="llama-3.1-8b-instant", temperature=0.7):
    # Customize model and parameters
```

## 🚀 Future Improvements

### Short Term Enhancements

- [ ] **Batch Processing**: File-level parallelization for multiple assignments
- [ ] **Enhanced Error Handling**: Retry mechanisms and graceful degradation
- [ ] **Output Formats**: JSON, HTML, and PDF export options
- [ ] **Progress Tracking**: Real-time processing status and progress bars

### Medium Term Goals

- [ ] **Multi-LLM Support**: Integration with OpenAI, Anthropic, and local models
- [ ] **Web Interface**: Flask/FastAPI dashboard with file upload
- [ ] **Advanced Plagiarism**: External database integration and citation validation
- [ ] **Performance Optimization**: Caching and distributed processing

### Long Term Vision

- [ ] **Machine Learning**: Historical pattern analysis and personalized feedback
- [ ] **Enterprise Features**: Multi-tenant support and LMS integration
- [ ] **Research Tools**: Citation analysis and knowledge graphs
- [ ] **Cross-linguistic Support**: Multi-language assignment processing

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for educators and students**