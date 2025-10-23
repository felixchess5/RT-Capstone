#!/usr/bin/env python3
"""
Intelligent Assignment Grading System
Hugging Face Spaces Deployment Entry Point

An advanced academic assignment grading system with enterprise-grade security,
comprehensive testing, and intelligent multi-subject processing capabilities.
"""

import os
import sys
import gradio as gr
import logging
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment variables and configuration for Hugging Face Spaces."""
    # Set default environment variables for HF Spaces
    os.environ.setdefault('PYTHONPATH', 'src')

    # Create necessary directories
    directories = ['Assignments', 'output', 'plagiarism_reports', 'temp']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

    logger.info("Environment setup completed for Hugging Face Spaces")

def check_api_keys():
    """Check if API keys are configured."""
    api_keys = {
        'GROQ_API_KEY': os.getenv('GROQ_API_KEY'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY')
    }

    available = {k: v is not None and v.strip() != '' for k, v in api_keys.items()}
    return available, any(available.values())

def process_assignment(file, subject="auto", groq_key="", openai_key="", anthropic_key="", google_key=""):
    """Process assignment with comprehensive AI analysis."""
    if file is None:
        return "Please upload a file.", "", ""

    try:
        # Set API keys if provided
        if groq_key.strip():
            os.environ['GROQ_API_KEY'] = groq_key.strip()
        if openai_key.strip():
            os.environ['OPENAI_API_KEY'] = openai_key.strip()
        if anthropic_key.strip():
            os.environ['ANTHROPIC_API_KEY'] = anthropic_key.strip()
        if google_key.strip():
            os.environ['GOOGLE_API_KEY'] = google_key.strip()

        # Check API availability
        api_status, has_api = check_api_keys()

        if not has_api:
            return "⚠️ API key required", "**Please provide at least one API key to enable AI analysis.**", ""

        # Process file with full AI system
        try:
            import tempfile
            import shutil
            from pathlib import Path

            # Create temp directory for processing
            temp_dir = tempfile.mkdtemp()
            temp_file = None

            if hasattr(file, 'name'):
                # Copy file to temp location with proper handling
                file_name = Path(file.name).name
                temp_file = os.path.join(temp_dir, file_name)

                try:
                    # Copy file safely
                    shutil.copy2(file.name, temp_file)
                except Exception as e:
                    logger.warning(f"File copy failed, trying alternative method: {e}")
                    # Alternative copy method for problematic files
                    with open(file.name, 'rb') as src, open(temp_file, 'wb') as dst:
                        dst.write(src.read())

                # Use full AI workflow
                try:
                    from workflows.agentic_workflow import run_agentic_workflow
                    from support.file_processor import FileProcessor
                    from core.paths import ASSIGNMENTS_FOLDER, OUTPUT_FOLDER

                    # Initialize processors
                    file_processor = FileProcessor()

                    # Process file content
                    processing_result = file_processor.process_file(temp_file)

                    if processing_result.get('success'):
                        content = processing_result.get('content', '')
                        metadata = processing_result.get('metadata', {})

                        # Run AI workflow
                        workflow_result = run_agentic_workflow(
                            content=content,
                            file_path=temp_file,
                            output_path=OUTPUT_FOLDER,
                            metadata=metadata
                        )

                        if workflow_result.get('success'):
                            analysis_data = workflow_result.get('analysis', {})

                            # Format comprehensive results
                            analysis = f"""
# 📋 **Comprehensive Assignment Analysis**

## 📁 **File Information**
- **File**: {file_name}
- **Type**: {metadata.get('file_type', 'Unknown').upper()}
- **Language**: {metadata.get('language', 'Unknown')}
- **Subject**: {analysis_data.get('subject', subject)}

## 🎯 **Analysis Results**

### Overall Score: {analysis_data.get('overall_score', 'N/A')}/100

### 📊 **Detailed Scores:**
{format_scores(analysis_data.get('scores', {}))}

### 💬 **Feedback:**
{analysis_data.get('feedback', 'No feedback available')}

### 🔍 **Detailed Analysis:**
{analysis_data.get('detailed_analysis', 'No detailed analysis available')}

### ✅ **Strengths:**
{format_list(analysis_data.get('strengths', []))}

### 🎯 **Areas for Improvement:**
{format_list(analysis_data.get('improvements', []))}

### 📚 **Recommendations:**
{format_list(analysis_data.get('recommendations', []))}

---
*Analysis powered by enterprise-grade AI with security protection*
                            """

                            return "✅ Comprehensive analysis complete", analysis, ""
                        else:
                            error_msg = workflow_result.get('error', 'Workflow failed')
                            return f"❌ AI analysis failed: {error_msg}", "", error_msg
                    else:
                        error_msg = processing_result.get('error', 'File processing failed')
                        return f"❌ File processing failed: {error_msg}", "", error_msg

                except ImportError as e:
                    logger.warning(f"AI modules not available: {e}")
                    # Fallback to basic analysis with API
                    return run_basic_ai_analysis(temp_file, api_status, subject)

            else:
                return "❌ Invalid file", "Could not process the uploaded file.", "Invalid file format"

        except Exception as e:
            logger.error(f"Processing error: {e}")
            return f"❌ Processing failed: {str(e)}", "", str(e)

        finally:
            # Cleanup
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        logger.error(f"General error: {e}")
        return f"❌ Error: {str(e)}", "", str(e)

def format_scores(scores):
    """Format scores dictionary into readable text."""
    if not scores:
        return "No scores available"

    formatted = []
    for category, score in scores.items():
        formatted.append(f"- **{category.replace('_', ' ').title()}**: {score}/10")
    return "\n".join(formatted)

def format_list(items):
    """Format list into readable bullet points."""
    if not items:
        return "None identified"

    if isinstance(items, str):
        return items

    return "\n".join([f"- {item}" for item in items])

def run_basic_ai_analysis(file_path, api_status, subject):
    """Fallback basic AI analysis when full system not available."""
    try:
        # Read file content with proper encoding handling
        content = ""
        file_type = Path(file_path).suffix.lower()

        if file_type == '.pdf':
            # Handle PDF files
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    content = ""
                    for page in reader.pages[:3]:  # First 3 pages
                        content += page.extract_text() + "\n"
                    content = content[:2000]  # Limit content
            except Exception as e:
                try:
                    import pdfplumber
                    with pdfplumber.open(file_path) as pdf:
                        content = ""
                        for page in pdf.pages[:3]:  # First 3 pages
                            content += page.extract_text() + "\n"
                        content = content[:2000]
                except Exception as e2:
                    content = f"[PDF file detected but text extraction failed: {str(e)}, {str(e2)}]"

        elif file_type in ['.docx', '.doc']:
            # Handle Word documents
            try:
                from docx import Document
                doc = Document(file_path)
                content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                content = content[:2000]
            except Exception as e:
                content = f"[Word document detected but text extraction failed: {str(e)}]"

        else:
            # Handle text files with encoding detection
            try:
                # Try UTF-8 first
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()[:2000]
            except UnicodeDecodeError:
                try:
                    # Try with chardet
                    import chardet
                    with open(file_path, 'rb') as f:
                        raw_data = f.read()
                        encoding = chardet.detect(raw_data)['encoding']

                    with open(file_path, 'r', encoding=encoding or 'latin-1') as f:
                        content = f.read()[:2000]
                except Exception as e:
                    content = f"[Text file detected but encoding failed: {str(e)}]"

        # Validate content
        if not content or len(content.strip()) < 10:
            return "⚠️ Insufficient content", "Could not extract sufficient text content from the file for analysis.", ""

        # Clean content for AI analysis
        content_clean = content.replace('\x00', '').replace('\ufffd', '').strip()
        if len(content_clean) < 10:
            return "⚠️ Content extraction failed", "File content could not be properly extracted for analysis.", ""

        # Basic AI prompt
        prompt = f"""Analyze this {subject} assignment and provide:
1. Overall quality score (1-10)
2. Key strengths (2-3 specific points)
3. Areas for improvement (2-3 specific points)
4. Specific actionable feedback

Assignment content:
{content_clean[:1500]}"""

        # Use available API for basic analysis
        if api_status.get('GROQ_API_KEY'):
            try:
                import groq
                client = groq.Groq()
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.7
                )
                ai_feedback = response.choices[0].message.content

                analysis = f"""
# 🤖 **AI Assignment Analysis**

## 📁 **File Information**
- **Subject**: {subject}
- **Content Length**: {len(content)} characters
- **Analysis Model**: Groq Llama 3.1

## 🔍 **AI Feedback:**
{ai_feedback}

---
*Basic AI analysis - upgrade to full system for comprehensive grading*
                """
                return "✅ AI analysis complete", analysis, ""

            except Exception as e:
                return f"❌ AI analysis failed: {str(e)}", "", str(e)

        return "⚠️ No compatible AI service available", "Please configure a supported API key.", ""

    except Exception as e:
        return f"❌ Basic analysis failed: {str(e)}", "", str(e)

def create_interface():
    """Create and configure the Gradio interface for the application."""

    with gr.Blocks(
        title="🎓 Intelligent Assignment Grading System",
        theme=gr.themes.Soft()
    ) as interface:

        gr.Markdown("""
        # 🎓 Intelligent Assignment Grading System

        **Advanced Academic Assignment Grading with Enterprise Security**

        Upload assignments in multiple formats and get comprehensive, subject-specific grading
        with intelligent analysis across Math, Spanish, Science, History, and more.

        ✨ **Features:**
        - 🔒 Enterprise-grade security protection
        - 📐 Advanced mathematical analysis with symbolic computation
        - 🇪🇸 Spanish language assessment with cultural understanding
        - 🔬 Scientific analysis with experimental design evaluation
        - 📚 Historical assessment with source analysis
        - 🌍 Multi-language support (14+ languages)
        - 🧪 Comprehensive testing framework
        """)

        with gr.Tab("Single Assignment"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="📁 Upload Assignment",
                        file_types=[".txt", ".pdf", ".docx", ".png", ".jpg", ".jpeg"]
                    )
                    subject_input = gr.Dropdown(
                        choices=["auto", "mathematics", "spanish", "science", "history", "english"],
                        value="auto",
                        label="🎯 Subject (auto-detect or manual)"
                    )

                    with gr.Accordion("🔑 API Configuration (Optional for AI Analysis)", open=False):
                        gr.Markdown("**Enter at least one API key to enable full AI-powered grading:**")
                        groq_key_input = gr.Textbox(
                            label="Groq API Key",
                            type="password",
                            placeholder="gsk_...",
                            info="Fast and affordable AI models"
                        )
                        openai_key_input = gr.Textbox(
                            label="OpenAI API Key",
                            type="password",
                            placeholder="sk-...",
                            info="GPT models for advanced analysis"
                        )
                        anthropic_key_input = gr.Textbox(
                            label="Anthropic API Key",
                            type="password",
                            placeholder="sk-ant-...",
                            info="Claude models for detailed feedback"
                        )
                        google_key_input = gr.Textbox(
                            label="Google API Key",
                            type="password",
                            placeholder="AI...",
                            info="Gemini models for comprehensive analysis"
                        )

                    process_btn = gr.Button("🚀 Analyze Assignment", variant="primary")

                with gr.Column():
                    status_output = gr.Textbox(label="📊 Status", interactive=False)
                    result_output = gr.Markdown(label="📋 Analysis Results")
                    error_output = gr.Textbox(label="⚠️ Errors", visible=False)

        with gr.Tab("Batch Processing"):
            gr.Markdown("**Batch processing coming soon!** Upload multiple files for analysis.")

        with gr.Tab("System Status"):
            with gr.Column():
                gr.Markdown("""
                **System Information:**
                - ✅ Basic file upload: Active
                - ⚠️ AI Analysis: Requires API configuration
                - ✅ Security: Enterprise-grade protection enabled
                - ✅ Multi-format support: PDF, DOCX, Images

                **🔑 How to Enable Full AI Analysis:**

                1. **Get API Keys** (choose one or more):
                   - **Groq**: Fast & affordable → [groq.com](https://groq.com) → Create account → API Keys
                   - **OpenAI**: Advanced GPT models → [platform.openai.com](https://platform.openai.com) → API Keys
                   - **Anthropic**: Claude models → [console.anthropic.com](https://console.anthropic.com) → API Keys
                   - **Google**: Gemini models → [ai.google.dev](https://ai.google.dev) → Get API Key

                2. **Enter Keys**: Use the "API Configuration" section in the Single Assignment tab

                3. **Upload Assignment**: Full AI analysis will automatically activate

                **💡 Recommendations:**
                - **Groq**: Best for speed and cost-effectiveness
                - **OpenAI**: Best for comprehensive analysis
                - **Anthropic**: Best for detailed educational feedback
                """)

        # Event handlers
        process_btn.click(
            fn=process_assignment,
            inputs=[
                file_input,
                subject_input,
                groq_key_input,
                openai_key_input,
                anthropic_key_input,
                google_key_input
            ],
            outputs=[status_output, result_output, error_output]
        )

        # Show errors only when there are errors
        status_output.change(
            fn=lambda status: gr.update(visible="❌" in status),
            inputs=[status_output],
            outputs=[error_output]
        )

    return interface

def main():
    """Main entry point for the Hugging Face Spaces deployment."""
    logger.info("🚀 Starting Intelligent Assignment Grading System")

    # Setup environment
    setup_environment()

    # Create and launch interface
    try:
        interface = create_interface()

        # Launch with HF Spaces configuration
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,  # Not needed in HF Spaces
            show_error=True,
            quiet=False,
            favicon_path=None,
            ssl_verify=False
        )

    except Exception as e:
        logger.error(f"Failed to launch application: {e}")

        # Emergency fallback
        def emergency_interface():
            return f"""
            🚨 Application Error

            The Assignment Grading system encountered an initialization error:
            {str(e)}

            Please try refreshing the page or contact support.
            """

        fallback = gr.Interface(
            fn=emergency_interface,
            inputs=[],
            outputs=gr.Textbox(),
            title="Assignment Grading System - Error",
            description="System initialization error"
        )

        fallback.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )

if __name__ == "__main__":
    main()