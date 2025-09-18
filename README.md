# Capstone - Intelligent Assignment Grading System

An advanced academic assignment grading system with subject-specific processing, multi-language support, and intelligent orchestration for comprehensive student evaluation.

## 🌟 Core Features

- **🎯 Subject-Specific Processing**: Specialized analyzers for Math, Spanish, English, Science, and History with comprehensive analysis
- **🤖 Intelligent Orchestration**: Automatic subject detection and routing to appropriate processors
- **📐 Mathematical Analysis**: Equation solving, symbolic computation, and step-by-step verification
- **🇪🇸 Spanish Language Assessment**: Grammar analysis, vocabulary evaluation, and cultural understanding
- **🔬 Scientific Analysis**: Lab reports, experimental design, scientific method evaluation, and formula identification
- **📚 Historical Assessment**: Chronological analysis, source evaluation, contextual understanding, and argument development
- **🌍 Multi-Language Support**: 14+ languages with automatic detection and localized prompts
- **📄 Multi-Format Processing**: PDF (text & scanned), DOCX, DOC, MD, TXT, and image formats
- **🔍 OCR Integration**: Free Tesseract OCR for scanned documents with preprocessing
- **⚡ Agentic Workflow**: LangGraph-powered intelligent processing pipeline
- **📊 Subject-Specific Outputs**: Organized CSV and JSON files by academic subject
- **🔧 MCP Integration**: 30+ tools for external system integration
- **📈 Comprehensive Analytics**: Detailed grading with specialized criteria per subject

## 📊 Subject-Specific Grading

### 📐 Mathematics Assignments
| Criterion | Scale | Description |
|-----------|-------|-------------|
| **Mathematical Accuracy** | 0-10 | Correctness of solutions and calculations |
| **Problem Solving Approach** | 0-10 | Method and strategy used to solve problems |
| **Notation Clarity** | 0-10 | Proper use of mathematical notation and formatting |
| **Step-by-Step Work** | 0-10 | Clear demonstration of solution process |

### 🇪🇸 Spanish Assignments
| Criterion | Scale | Description |
|-----------|-------|-------------|
| **Grammar Accuracy** | 0-10 | Correct use of Spanish grammar rules |
| **Vocabulary Usage** | 0-10 | Appropriateness and variety of vocabulary |
| **Fluency & Communication** | 0-10 | Natural flow and expression in Spanish |
| **Cultural Understanding** | 0-10 | Knowledge of Hispanic culture and context |

### 🔬 Science Assignments
| Criterion | Scale | Description |
|-----------|-------|-------------|
| **Scientific Accuracy** | 0-10 | Correctness of facts, formulas, and concepts |
| **Hypothesis Quality** | 0-10 | Clear, testable hypothesis formulation |
| **Data Analysis** | 0-10 | Proper data presentation and interpretation |
| **Experimental Design** | 0-10 | Quality of experimental methodology |
| **Conclusion Validity** | 0-10 | Evidence-based conclusions and reasoning |

### 📚 History Assignments
| Criterion | Scale | Description |
|-----------|-------|-------------|
| **Historical Accuracy** | 0-10 | Correctness of facts, dates, and events |
| **Chronological Understanding** | 0-10 | Proper sequence and timing awareness |
| **Source Analysis** | 0-10 | Effective use and evaluation of sources |
| **Contextual Awareness** | 0-10 | Understanding of historical context |
| **Argument Development** | 0-10 | Well-structured historical arguments |

### 📝 General Assignments
| Criterion | Scale | Description |
|-----------|-------|-------------|
| **Factual Accuracy** | 0-10 | Content accuracy compared to source material |
| **Relevance to Source** | 0-10 | How well assignment relates to reference material |
| **Coherence** | 0-10 | Logical structure and flow of writing |
| **Grammar** | 1-10 | Writing quality, spelling, grammar (minimum score: 1) |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Groq API key (required)
- Gemini API key (optional, for redundancy)
- Tesseract OCR (for scanned documents)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/felixchess5/RT-Capstone.git
   cd RT-Capstone
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt

   # Install additional dependencies for specialized processing
   pip install sympy spacy langdetect

   # Optional: Install Spanish language model for enhanced Spanish processing
   python -m spacy download es_core_news_sm
   ```

4. **Install Tesseract OCR** (for scanned documents)
   ```bash
   # macOS
   brew install tesseract

   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr

   # Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   ```

5. **Environment setup**
   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Edit .env file and add your API keys
   # Required:
   GROQ_API_KEY=your_groq_api_key_here

   # Optional (for LLM redundancy):
   GEMINI_API_KEY=your_gemini_api_key_here

   # Optional (for LangSmith tracing):
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_PROJECT=Assignment Grader
   ```

6. **Configure paths** (Optional)
   - Edit `paths.py` to customize file locations
   - Default folders will be created automatically

### Usage

1. **Place assignment files**
   - Add assignment files to the `Assignments/` folder
   - **Supported formats**: PDF, DOCX, DOC, MD, TXT, PNG, JPEG, TIFF, BMP
   - Use this format for text files:
     ```
     Name: John Doe
     Date: 2025-08-25
     Class: Algebra II
     Subject: Mathematics

     Solve for x: 2x + 5 = 13
     Step 1: Subtract 5 from both sides
     2x = 8
     Step 2: Divide by 2
     x = 4
     ```

2. **Run the grading system**
   ```bash
   # Enhanced agentic workflow (recommended)
   python src/main_agentic.py

   # Alternative: MCP server mode
   python src/main_agentic.py mcp

   # Run tests
   python tests/test_specialized_processors.py
   ```

3. **View results**
   - **General summary**: `output/summary.csv`
   - **Subject-specific files**:
     - `output/math_assignments.csv` - Mathematics assignments with specialized fields
     - `output/spanish_assignments.csv` - Spanish assignments with language metrics
     - `output/english_assignments.csv` - English assignments with writing analysis
     - `output/science_assignments.csv` - Science assignments with experimental analysis
     - `output/history_assignments.csv` - History assignments with chronological analysis
   - **Detailed data**: JSON files for complete assignment information
   - **Reports**: Individual plagiarism reports in `plagiarism_reports/` folder
   - **Export summary**: `output/export_summary.txt` with processing statistics

## 🏗️ System Architecture

### Intelligent Processing Pipeline

```
📁 Assignment Files → 🎯 Subject Classification → 🔬 Specialized Processing → 📊 Subject-Specific Outputs
      ↓                        ↓                         ↓                        ↓
  Multi-Format         Automatic Detection      Math/Spanish/English         Organized CSV/JSON
   Processing           & Confidence            Specialized Analysis          Files by Subject
      ↓                        ↓                         ↓                        ↓
  OCR for Scanned      Intelligent Routing      Advanced Grading           Export Summary &
    Documents           to Processors           Criteria per Subject        Statistics Report
```

### Core Components

#### 🎯 Assignment Orchestrator (`src/core/assignment_orchestrator.py`)
- **Subject Classification**: Automatically detects Math, Spanish, English, Science, History
- **Complexity Assessment**: Elementary, Middle School, High School, College levels
- **Intelligent Routing**: Directs to appropriate specialized processors
- **Processing Optimization**: Selects best methodology per subject type

#### 📐 Math Processor (`src/processors/math_processor.py`)
- **Equation Solving**: Symbolic computation using SymPy
- **Problem Type Detection**: Algebra, Calculus, Geometry, Statistics, etc.
- **Step-by-Step Analysis**: Evaluates solution methodology and presentation
- **Mathematical Notation**: Assesses proper formatting and notation usage

#### 🇪🇸 Spanish Processor (`src/processors/spanish_processor.py`)
- **Grammar Analysis**: Spanish-specific grammar rule checking
- **Vocabulary Assessment**: Beginner/Intermediate/Advanced level detection
- **Cultural References**: Identifies and evaluates Hispanic cultural knowledge
- **Fluency Scoring**: Comprehensive language proficiency assessment

#### 🔬 Science Processor (`science_processor.py`)
- **Subject Classification**: Physics, Chemistry, Biology, Earth Science identification
- **Scientific Method Analysis**: Hypothesis, procedure, observations, conclusions evaluation
- **Formula Recognition**: Mathematical and chemical equation identification
- **Experimental Design**: Variables, controls, and methodology assessment
- **Safety Evaluation**: Laboratory safety considerations and protocols

#### 📚 History Processor (`history_processor.py`)
- **Period Classification**: Ancient, Medieval, Modern, Contemporary identification
- **Chronological Analysis**: Timeline accuracy and sequence evaluation
- **Source Evaluation**: Primary and secondary source analysis
- **Historical Context**: Understanding of time period and circumstances
- **Argument Assessment**: Historical reasoning and evidence usage

#### 📊 Subject Output Manager (`subject_output_manager.py`)
- **Automatic Classification**: Routes results to appropriate output files
- **Specialized Fields**: Subject-specific CSV columns and data extraction
- **Multiple Formats**: CSV for analysis, JSON for detailed data
- **Export Statistics**: Comprehensive reporting and summaries

#### ⚡ Agentic Workflow (`agentic_workflow.py`)
- **LangGraph Integration**: State-based workflow orchestration
- **Adaptive Processing**: Dynamic routing based on classification results
- **Error Recovery**: Robust handling of processing failures
- **Quality Validation**: Multi-stage validation and verification

### Data Flow

```
🎯 Assignment Classification
        ↓
┌─────────────────────────────────────────┐
│     Subject Detection & Routing         │
├─────────────────────────────────────────┤
│  📐 Math  🇪🇸 Spanish  📝 English  🔬 Science  📚 History  │
│   ↓         ↓         ↓         ↓          ↓         │
│ Equation  Grammar   Literature Scientific Historical │
│ Solving   Analysis  Analysis   Method     Context    │
│   ↓         ↓         ↓         ↓          ↓         │
│ Step-by-  Vocabulary Writing   Lab       Chronology  │
│ Step      Assessment Quality   Reports   Analysis    │
│ Analysis    ↓         ↓         ↓          ↓         │
│   ↓       Cultural  Citation  Formula   Source      │
│ Math      References Quality  Recognition Evaluation │
│ Notation    ↓         ↓         ↓          ↓         │
│   ↓       Fluency   Thesis   Data       Argument    │
│ Problem   Scoring   Strength  Analysis   Structure   │
│ Types       ↓         ↓         ↓          ↓         │
│   ↓         ↓         ↓         ↓          ↓         │
└─────────────────────────────────────────┘
        ↓
📊 Subject-Specific Output Files
   ↓
📋 Export Summary & Statistics
```

## 📁 Project Structure

```
RT-Capstone/
├── 🔧 Core System
│   ├── main_agentic.py           # Enhanced agentic workflow entry point
│   ├── agentic_workflow.py       # LangGraph-based intelligent workflow
│   ├── assignment_orchestrator.py # Subject classification & routing
│   ├── llms.py                   # Multi-LLM configuration (Groq + Gemini)
│   └── paths.py                  # Comprehensive path configurations
│
├── 🎯 Specialized Processors
│   ├── math_processor.py         # Mathematical analysis & equation solving
│   ├── spanish_processor.py      # Spanish language assessment
│   ├── science_processor.py      # Scientific analysis & experimental design
│   ├── history_processor.py      # Historical analysis & chronological assessment
│   └── subject_output_manager.py # Subject-specific file generation
│
├── 🌍 Multi-Language & OCR
│   ├── language_support.py       # 14+ language support system
│   ├── ocr_processor.py          # Tesseract OCR integration
│   └── file_processor.py         # Multi-format file processing
│
├── 🔧 Integration & Tools
│   ├── mcp_server.py             # 30+ MCP tools for external integration
│   ├── prompts.py                # Localized prompt templates
│   └── utils.py                  # Utilities & visualization
│
├── 📁 Data Directories
│   ├── Assignments/              # Input files (PDF, DOCX, TXT, images)
│   ├── output/                   # Subject-specific CSV & JSON files
│   └── plagiarism_reports/       # Detailed analysis reports
│
├── 🧪 Testing & Demo
│   ├── test_specialized_processors.py # Comprehensive test suite
│   ├── test_subject_outputs.py    # Output system testing
│   ├── test_new_subjects.py       # Science & History processor tests
│   └── demo_subject_outputs.py    # Quick demonstration
│
└── 📋 Configuration
    ├── requirements.txt          # Python dependencies
    ├── .env.example             # Environment variables template
    ├── .gitignore              # Git ignore patterns
    └── README.md               # This documentation
```

## 🔄 Processing Workflow

### Enhanced Agentic Processing Flow

1. **Initialization** (`main_agentic.py`)
   - Load environment variables and multi-LLM configuration
   - Initialize specialized processors and orchestrator
   - Build LangGraph workflow with intelligent routing

2. **File Processing** (`file_processor.py`)
   - **Multi-format support**: PDF, DOCX, DOC, MD, TXT, images
   - **OCR processing**: Automatic detection and processing of scanned documents
   - **Language detection**: Automatic language identification for 14+ languages
   - **Metadata extraction**: Parse assignment headers and classify content

3. **Intelligent Classification** (`assignment_orchestrator.py`)
   - **Subject detection**: Automatic classification (Math, Spanish, English, etc.)
   - **Complexity assessment**: Grade level and difficulty analysis
   - **Confidence scoring**: Reliability of classification decisions
   - **Processing route selection**: Choose optimal processor for content type

4. **Specialized Processing**
   - **📐 Math Assignments**: Equation solving, step-by-step analysis, notation assessment
   - **🇪🇸 Spanish Assignments**: Grammar checking, vocabulary analysis, cultural evaluation
   - **🔬 Science Assignments**: Scientific method evaluation, formula recognition, experimental design
   - **📚 History Assignments**: Chronological analysis, source evaluation, historical context assessment
   - **📝 General Assignments**: Standard grading criteria with multi-language support
   - **Fallback processing**: Graceful degradation if specialized processing fails

5. **Parallel Analysis** (Agentic Workflow)
   - **Grammar Analysis**: Multi-language grammar checking with localized prompts
   - **Plagiarism Detection**: Content originality analysis with detailed reports
   - **Relevance Assessment**: Source material comparison and alignment evaluation
   - **Specialized Grading**: Subject-specific criteria and advanced scoring
   - **Summary Generation**: Intelligent summarization with language awareness

6. **Subject-Specific Export** (`subject_output_manager.py`)
   - **Automatic classification**: Route results to appropriate subject files
   - **Specialized CSV files**: Math, Spanish, English, Science, History with subject-specific columns
   - **Detailed JSON exports**: Complete assignment data with full analysis
   - **Export statistics**: Summary reports with processing metrics and averages

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

## 📋 Feature Roadmap

For a comprehensive list of planned features and enhancements, see our detailed [Feature List](FEATURES.md). This document tracks all current capabilities and future development plans organized by category:

- **Core Functionality**: Web interfaces, analytics, OCR/ICR, additional subjects
- **AI/ML Infrastructure**: Multiple LLM providers, failback systems, model monitoring
- **Security & Compliance**: OWASP guidelines, EVAL toolkit, audit logging
- **System Reliability**: Health checks, circuit breakers, graceful degradation
- **Performance & Scalability**: Caching, microservices, database optimization

### Quick Feature Highlights

#### Immediate Priorities
- [ ] **Multi-LLM Support**: Integration with OpenAI, Anthropic, and local models
- [ ] **Failback Systems**: Automatic switching when LLM services are down
- [ ] **Web Interface**: Gradio/FastAPI dashboard with file upload
- [ ] **Enhanced Testing**: Comprehensive test suite and quality assurance

#### Coming Soon
- [ ] **MCP Extensions**: PDF, Word, Markdown support with edge case handling
- [ ] **Advanced Analytics**: Student performance tracking and institutional reporting
- [ ] **Security Hardening**: OWASP compliance and security guardrails
- [ ] **Human-in-the-Loop**: Teacher review and feedback integration

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