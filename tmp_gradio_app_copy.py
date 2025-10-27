#!/usr/bin/env python3
"""
Gradio Web Interface for Assignment Grading System

This module provides a comprehensive web interface using Gradio for the
RT-Capstone assignment grading system with multi-LLM support and
subject-specific processing.
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import httpx
import mimetypes

import gradio as gr
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from core.llms import llm_manager
from core.paths import ASSIGNMENTS_FOLDER, OUTPUT_FOLDER
from core.subject_output_manager import create_subject_output_manager


class GradioAssignmentGrader:
    """Main Gradio interface for assignment grading."""

    def __init__(self):
        """Initialize the Gradio interface."""
        self.output_manager = create_subject_output_manager(OUTPUT_FOLDER)
        self.temp_dir = tempfile.mkdtemp()
        self.backend_url = os.getenv("BACKEND_URL", "").strip()
        self.max_upload_mb = int(os.getenv("DEMO_MAX_UPLOAD_MB", "20"))

    def process_single_file(
        self, file_path: str, requirements: Dict[str, bool]
    ) -> Tuple[str, str, str, str]:
        """
        Process a single assignment file.

        Args:
            file_path: Path to the uploaded file
            requirements: Dictionary of processing requirements

        Returns:
            Tuple of (status_message, results_json, download_link, error_message)
        """
        if not file_path:
            return "❌ No file uploaded", "", None, "No file provided"

        try:
            # Read and process the file
            print(f"📄 Processing file: {file_path}")

            # Extract content from file
            content_result = self.file_processor.extract_text_content(file_path)
            if not content_result.success:
                error_msg = content_result.error or "Unknown file processing error"
                return f"❌ File processing failed: {error_msg}", "", None, error_msg

            content = content_result.content
            if not content.strip():
                return (
                    "❌ No content extracted from file",
                    "",
                    None,
                    "Empty file content",
                )

            # Process with agentic workflow
            async def run_processing():
                try:
                    # Extract basic metadata from filename or use defaults
                    filename = os.path.basename(file_path)
                    base_name = os.path.splitext(filename)[0]

                    # Create metadata for the workflow
                    metadata = {
                        "name": base_name,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "class": "Unknown",
                        "subject": "General",
                    }

                    # Try to extract metadata from content if present
                    try:
                        lines = content.split("\n")[:10]  # Check first 10 lines
                        for line in lines:
                            if "name:" in line.lower():
                                metadata["name"] = line.split(":", 1)[1].strip()
                            elif "date:" in line.lower():
                                metadata["date"] = line.split(":", 1)[1].strip()
                            elif "class:" in line.lower():
                                metadata["class"] = line.split(":", 1)[1].strip()
                            elif "subject:" in line.lower():
                                metadata["subject"] = line.split(":", 1)[1].strip()
                    except:
                        pass  # Use defaults if extraction fails

                    # Process assignment with agentic workflow
                    result = await run_agentic_workflow(content, metadata, "")
                    return result

                except Exception as e:
                    print(f"❌ Processing error: {e}")
                    return {"error": str(e)}

            # Run async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(run_processing())
            finally:
                loop.close()

            if "error" in result:
                error_msg = result["error"]
                return f"❌ Processing failed: {error_msg}", "", "", error_msg

            # Format results
            results_summary = self._format_results(result)
            results_json = json.dumps(result, indent=2, default=str)

            # Create downloadable files
            download_path = self._create_download_files(
                result, os.path.basename(file_path)
            )

            return (
                f"✅ Processing completed successfully!",
                results_summary,
                download_path if download_path else None,
                "",
            )

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"❌ {error_msg}")
            return f"❌ {error_msg}", "", None, error_msg

    def process_multiple_files(
        self, files: List[str], requirements: Dict[str, bool]
    ) -> Tuple[str, str, str, str]:
        """
        Process multiple assignment files.

        Args:
            files: List of file paths
            requirements: Dictionary of processing requirements

        Returns:
            Tuple of (status_message, summary_table, download_link, error_message)
        """
        if not files:
            return "❌ No files uploaded", "", None, "No files provided"

        results = []
        errors = []

        try:
            print(f"📚 Processing {len(files)} files...")

            for i, file_path in enumerate(files):
                try:
                    print(
                        f"📄 Processing file {i+1}/{len(files)}: {os.path.basename(file_path)}"
                    )

                    # Process individual file
                    status, result_summary, _, error = self.process_single_file(
                        file_path, requirements
                    )

                    if error:
                        errors.append(f"{os.path.basename(file_path)}: {error}")
                    else:
                        results.append(
                            {
                                "filename": os.path.basename(file_path),
                                "status": (
                                    "✅ Success" if "✅" in status else "❌ Failed"
                                ),
                                "summary": (
                                    result_summary[:300] + "..."
                                    if len(result_summary) > 300
                                    else result_summary
                                ),
                            }
                        )

                except Exception as e:
                    error_msg = f"{os.path.basename(file_path)}: {str(e)}"
                    errors.append(error_msg)
                    print(f"❌ Error processing {file_path}: {e}")

            # Create summary
            success_count = len(results)
            total_count = len(files)
            error_count = len(errors)

            status_message = f"📊 Processed {total_count} files: {success_count} successful, {error_count} failed"

            # Create summary table
            if results:
                df = pd.DataFrame(results)
                summary_table = df.to_html(index=False, classes="gradio-table")
            else:
                summary_table = "No successful results to display"

            # Create error summary
            error_summary = "\n".join(errors) if errors else ""

            # Create batch download
            download_path = self._create_batch_download(results)

            return (
                status_message,
                summary_table,
                download_path if download_path else None,
                error_summary,
            )

        except Exception as e:
            error_msg = f"Batch processing error: {str(e)}"
            print(f"❌ {error_msg}")
            return f"❌ {error_msg}", "", None, error_msg

    # Backend-driven implementations (UI-only env)
    def process_single_file_v2(
        self, file_path: str, requirements: Dict[str, bool]
    ) -> Tuple[str, str, Optional[str], str]:
        if not file_path:
            return "? No file uploaded", "", None, "No file provided"
        try:
            # size guard
            try:
                if os.path.getsize(file_path) > self.max_upload_mb * 1024 * 1024:
                    return (
                        f"? File too large (> {self.max_upload_mb} MB)",
                        "",
                        None,
                        "File exceeds upload size limit",
                    )
            except Exception:
                pass

            if not self.backend_url:
                return (
                    "? Backend URL not configured",
                    "",
                    None,
                    "Set BACKEND_URL for demo UI",
                )

            mime, _ = mimetypes.guess_type(file_path)
            mime = mime or "application/octet-stream"
            data = {"requirements": json.dumps(requirements or {})}
            with open(file_path, "rb") as fh:
                files = {"file": (os.path.basename(file_path), fh, mime)}
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(
                        f"{self.backend_url.rstrip('/')}/process_file",
                        files=files,
                        data=data,
                    )
            if resp.status_code != 200:
                return (
                    f"? Processing failed: {resp.status_code}",
                    "",
                    None,
                    resp.text,
                )
            result = resp.json()
            if isinstance(result, dict) and "error" in result:
                return f"? Processing failed: {result['error']}", "", None, result["error"]

            summary = self._format_results(result)
            download_path = self._create_download_files(
                result, os.path.basename(file_path)
            )
            return (
                f"? Processing completed successfully!",
                summary,
                download_path if download_path else None,
                "",
            )
        except Exception as e:
            return f"? Backend error: {e}", "", None, str(e)

    def process_multiple_files_v2(
        self, files: List[str], requirements: Dict[str, bool]
    ) -> Tuple[str, str, Optional[str], str]:
        if not files:
            return "? No files uploaded", "", None, "No files provided"
        results = []
        errors = []
        for p in files:
            status, result_summary, _, error = self.process_single_file_v2(p, requirements)
            if error:
                errors.append(f"{os.path.basename(p)}: {error}")
            else:
                results.append(
                    {
                        "filename": os.path.basename(p),
                        "status": ("? Success" if "?" in status else "? Failed"),
                        "summary": (
                            result_summary[:300] + "..."
                            if len(result_summary) > 300
                            else result_summary
                        ),
                    }
                )
        # summary text/html
        if results:
            df = pd.DataFrame(results)
            summary_table = df.to_html(index=False, classes="gradio-table")
        else:
            summary_table = "No successful results to display"
        error_summary = "\n".join(errors) if errors else ""
        status_message = f"?? Processed {len(files)} files: {len(results)} successful, {len(errors)} failed"
        # optional CSV download
        download_path = self._create_batch_download(results)
        return status_message, summary_table, download_path, error_summary

    def _format_results(self, result: Dict[str, Any]) -> str:
        """Format processing results for display."""
        try:
            output = []

            # Basic info
            if "student_name" in result:
                output.append(f"**Student:** {result['student_name']}")

            if "overall_score" in result:
                score = result["overall_score"]
                if isinstance(score, (int, float)):
                    output.append(f"**Overall Score:** {score:.2f}/10")

                    # Add letter grade
                    if score >= 9:
                        letter = "A+"
                    elif score >= 8:
                        letter = "A"
                    elif score >= 7:
                        letter = "B"
                    elif score >= 6:
                        letter = "C"
                    elif score >= 5:
                        letter = "D"
                    else:
                        letter = "F"
                    output.append(f"**Letter Grade:** {letter}")

            # Subject classification
            if "assignment_classification" in result:
                classification = result["assignment_classification"]
                if "subject" in classification:
                    output.append(f"**Subject:** {classification['subject'].title()}")
                if "complexity" in classification:
                    output.append(
                        f"**Complexity:** {classification['complexity'].title()}"
                    )

            # Summary
            if "summary_result" in result and "summary" in result["summary_result"]:
                summary = result["summary_result"]["summary"]
                output.append(f"**Summary:**\n{summary}")
            elif "summary" in result:
                # Fallback for direct summary field
                output.append(f"**Summary:**\n{result['summary']}")

            # Grammar errors
            if "grammar_result" in result:
                grammar = result["grammar_result"]
                if "error_count" in grammar:
                    output.append(f"**Grammar Errors:** {grammar['error_count']}")

            # Specialized feedback
            if "specialized_feedback" in result and result["specialized_feedback"]:
                output.append("**Key Feedback:**")
                for feedback in result["specialized_feedback"][:8]:  # Show first 8
                    output.append(f"• {feedback}")

                if len(result["specialized_feedback"]) > 8:
                    output.append(
                        f"• ... and {len(result['specialized_feedback']) - 8} more items"
                    )

            # Add processing metadata
            if "processing_metadata" in result:
                metadata = result["processing_metadata"]
                if "completed_steps" in metadata:
                    output.append(
                        f"**Processing Steps Completed:** {len(metadata['completed_steps'])}"
                    )
                if "errors" in metadata and metadata["errors"]:
                    output.append(f"**Processing Errors:** {len(metadata['errors'])}")

            # Add detailed scores if available
            if "initial_grade" in result and isinstance(result["initial_grade"], dict):
                scores = result["initial_grade"]
                output.append("**Detailed Scores:**")
                for category, score in scores.items():
                    if isinstance(score, (int, float)):
                        output.append(f"• {category.title()}: {score:.2f}/10")

            return (
                "\n\n".join(output)
                if output
                else "Processing completed - see JSON for details"
            )

        except Exception as e:
            return f"Error formatting results: {str(e)}"

    def _create_download_files(self, result: Dict[str, Any], filename: str) -> str:
        """Create downloadable files for results."""
        try:
            base_name = os.path.splitext(filename)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create download directory
            download_dir = os.path.join(
                self.temp_dir, f"results_{base_name}_{timestamp}"
            )
            os.makedirs(download_dir, exist_ok=True)

            # Save JSON results
            json_path = os.path.join(download_dir, f"{base_name}_results.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)

            # Save summary
            summary_path = os.path.join(download_dir, f"{base_name}_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(self._format_results(result))

            # Create ZIP file
            zip_path = os.path.join(
                self.temp_dir, f"{base_name}_results_{timestamp}.zip"
            )
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(download_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, download_dir)
                        zipf.write(file_path, arcname)

            return zip_path

        except Exception as e:
            print(f"❌ Error creating download files: {e}")
            return None

    def _create_batch_download(self, results: List[Dict[str, Any]]) -> str:
        """Create batch download for multiple results."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create batch summary CSV
            if results:
                df = pd.DataFrame(results)
                csv_path = os.path.join(self.temp_dir, f"batch_results_{timestamp}.csv")
                df.to_csv(csv_path, index=False)
                return csv_path

            return None

        except Exception as e:
            print(f"❌ Error creating batch download: {e}")
            return None

    def get_system_status(self) -> str:
        """Get current system status with detailed LLM information."""
        try:
            status_parts = []
            # Backend status
            if hasattr(self, "backend_url") and self.backend_url:
                try:
                    with httpx.Client(timeout=5.0) as client:
                        r = client.get(f"{self.backend_url.rstrip('/')}/status")
                    if r.status_code == 200:
                        status_parts.append("?? Backend: reachable")
                    else:
                        status_parts.append(f"? Backend: HTTP {r.status_code}")
                except Exception as e:
                    status_parts.append(f"? Backend: unreachable ({e})")

            # Detailed LLM status
            if llm_manager:
                health = llm_manager.get_health_status()
                healthy_providers = sum(
                    1
                    for provider_status in health.values()
                    if provider_status.get("is_healthy", False)
                )
                total_providers = len(health)
                status_parts.append(
                    f"🔗 LLM Providers: {healthy_providers}/{total_providers} healthy"
                )

                # Show individual provider status
                for provider_name, provider_health in health.items():
                    status_icon = (
                        "✅" if provider_health.get("is_healthy", False) else "❌"
                    )

                    # Safe calculation of success rate
                    success_rate = 0
                    total_requests = provider_health.get("total_requests", 0)
                    successful_requests = provider_health.get("successful_requests", 0)

                    if isinstance(total_requests, (int, float)) and total_requests > 0:
                        if isinstance(successful_requests, (int, float)):
                            success_rate = (successful_requests / total_requests) * 100

                    # Safe formatting of average time
                    avg_time = provider_health.get("average_response_time", 0)
                    if not isinstance(avg_time, (int, float)):
                        avg_time = 0

                    status_parts.append(
                        f"  {status_icon} {provider_name.title()}: {success_rate:.1f}% success, {avg_time:.2f}s avg"
                    )

                # Show priority order
                try:
                    priority_order = llm_manager.get_priority_order()
                    if priority_order:
                        status_parts.append(
                            f"📋 Priority Order: {' → '.join(priority_order)}"
                        )
                    else:
                        status_parts.append("📋 Priority Order: Not configured")
                except Exception as e:
                    status_parts.append(
                        f"📋 Priority Order: Error getting order ({str(e)})"
                    )

                # Show available models
                try:
                    available_providers = list(llm_manager.providers.keys())
                    if available_providers:
                        status_parts.append(
                            f"🤖 Available Models: {', '.join(available_providers)}"
                        )
                    else:
                        status_parts.append("🤖 Available Models: None initialized")
                except Exception as e:
                    status_parts.append(
                        f"🤖 Available Models: Error getting providers ({str(e)})"
                    )

            else:
                status_parts.append("❌ LLM Manager: Not initialized")

            # File processor status
            status_parts.append("📄 File Processor: Ready")
            status_parts.append("🔧 Workflow Engine: Ready")

            return "\n".join(status_parts)

        except Exception as e:
            return f"❌ Error getting system status: {str(e)}"


def create_interface():
    """Create and configure the Gradio interface."""

    grader = GradioAssignmentGrader()

    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .gradio-table {
        border-collapse: collapse;
        width: 100%;
    }
    .gradio-table th, .gradio-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    .gradio-table th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    """

    with gr.Blocks(css=custom_css, title="RT-Capstone Assignment Grader") as interface:

        # Header
        gr.Markdown(
            """
        # 🎓 RT-Capstone Assignment Grading System

        **Advanced AI-powered assignment grading with multi-LLM support and subject-specific analysis**

        Upload assignments in various formats (PDF, DOCX, TXT, MD) and get comprehensive grading with:
        • Subject-specific analysis (Math, Spanish, Science, History, English)
        • Grammar and plagiarism detection
        • Detailed feedback and scoring
        • Multi-language support
        """
        )

        # System status
        with gr.Row():
            status_display = gr.Textbox(
                label="🔧 System Status",
                value=grader.get_system_status(),
                interactive=False,
                lines=8,
                max_lines=15,
                show_copy_button=True,
            )
            status_refresh = gr.Button("🔄 Refresh Status", size="sm")

        # Main interface tabs
        with gr.Tabs():

            # Single file processing tab
            with gr.Tab("📄 Single Assignment"):

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload Assignment")
                        single_file = gr.File(
                            label="Upload Assignment File",
                            file_types=[".pdf", ".docx", ".doc", ".txt", ".md"],
                            type="filepath",
                        )

                        gr.Markdown("### Processing Options")
                        grammar_check = gr.Checkbox(
                            label="Grammar Analysis", value=True
                        )
                        plagiarism_check = gr.Checkbox(
                            label="Plagiarism Detection", value=True
                        )
                        relevance_check = gr.Checkbox(
                            label="Source Relevance", value=True
                        )
                        grading_check = gr.Checkbox(
                            label="Automated Grading", value=True
                        )
                        summary_check = gr.Checkbox(
                            label="Generate Summary", value=True
                        )
                        specialized_check = gr.Checkbox(
                            label="Subject-Specific Analysis", value=True
                        )

                        process_single_btn = gr.Button(
                            "🚀 Process Assignment", variant="primary"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Results")
                        single_status = gr.Textbox(label="Status", interactive=False)
                        single_results = gr.Textbox(
                            label="Summary",
                            lines=20,
                            max_lines=50,
                            interactive=False,
                            show_copy_button=True,
                        )
                        single_json = gr.JSON(label="Detailed Results")
                        single_download = gr.File(label="Download Results")
                        single_errors = gr.Textbox(
                            label="Errors", interactive=False, visible=False
                        )

            # Batch processing tab
            with gr.Tab("📚 Batch Processing"):

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload Multiple Assignments")
                        batch_files = gr.File(
                            label="Upload Assignment Files",
                            file_count="multiple",
                            file_types=[".pdf", ".docx", ".doc", ".txt", ".md"],
                            type="filepath",
                        )

                        gr.Markdown("### Batch Processing Options")
                        batch_grammar = gr.Checkbox(
                            label="Grammar Analysis", value=True
                        )
                        batch_plagiarism = gr.Checkbox(
                            label="Plagiarism Detection", value=True
                        )
                        batch_relevance = gr.Checkbox(
                            label="Source Relevance", value=True
                        )
                        batch_grading = gr.Checkbox(
                            label="Automated Grading", value=True
                        )
                        batch_summary = gr.Checkbox(
                            label="Generate Summary", value=True
                        )
                        batch_specialized = gr.Checkbox(
                            label="Subject-Specific Analysis", value=True
                        )

                        process_batch_btn = gr.Button(
                            "🚀 Process All Assignments", variant="primary"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Batch Results")
                        batch_status = gr.Textbox(
                            label="Batch Status", interactive=False
                        )
                        batch_summary_table = gr.HTML(label="Results Summary")
                        batch_download = gr.File(label="Download Batch Results")
                        batch_errors = gr.Textbox(
                            label="Errors", lines=5, interactive=False, visible=False
                        )

            # Help and documentation tab
            with gr.Tab("❓ Help & Documentation"):
                gr.Markdown(
                    """
                ## How to Use the Assignment Grading System

                ### Supported File Formats
                - **PDF**: Text and scanned documents (with OCR)
                - **Microsoft Word**: DOCX and DOC files
                - **Text**: Plain text files (.txt)
                - **Markdown**: Markdown formatted files (.md)

                ### Processing Features

                #### 📝 Grammar Analysis
                - Detects and counts grammatical errors
                - Multi-language support (14+ languages)
                - Provides error severity scoring

                #### 🔍 Plagiarism Detection
                - Checks for potential plagiarism patterns
                - Generates detailed reports
                - Confidence scoring for detection

                #### 📊 Source Relevance
                - Analyzes how well content relates to source material
                - Factual consistency checking
                - Content alignment scoring

                #### 🎯 Automated Grading
                - Subject-specific grading criteria
                - 10-point scale with letter grades
                - Detailed rubric-based assessment

                #### 📋 Summary Generation
                - AI-powered content summaries
                - Subject-aware context
                - Multi-language summary support

                #### 🔬 Subject-Specific Analysis
                - **Mathematics**: Equation solving, step-by-step analysis
                - **Spanish**: Grammar, cultural references, fluency
                - **Science**: Scientific method, formulas, hypothesis evaluation
                - **History**: Chronological analysis, source evaluation
                - **English**: Literary analysis, writing quality

                ### System Architecture
                - **Multi-LLM Support**: 5 providers with automatic failover
                - **Circuit Breakers**: Reliability and health monitoring
                - **Specialized Processors**: Domain-specific analysis engines
                - **Agentic Workflow**: 11-node intelligent processing pipeline

                ### Tips for Best Results
                1. **Clear Text**: Ensure documents have clear, readable text
                2. **Proper Format**: Use standard academic formatting
                3. **Complete Content**: Include all assignment components
                4. **Multiple Subjects**: System automatically detects subject type
                5. **Batch Processing**: Process multiple assignments efficiently

                ### Troubleshooting
                - **File Upload Issues**: Check file format and size
                - **Processing Errors**: Verify content is readable
                - **Slow Processing**: System handles rate limits automatically
                - **Missing Results**: Check error messages for details
                """
                )

        # Event handlers
        def create_requirements_dict(
            grammar, plagiarism, relevance, grading, summary, specialized
        ):
            return {
                "grammar": grammar,
                "plagiarism": plagiarism,
                "relevance": relevance,
                "grading": grading,
                "summary": summary,
                "specialized": specialized,
            }

        # Single file processing
        process_single_btn.click(
            fn=lambda file, g, p, r, gr, s, sp: grader.process_single_file_v2(
                file, create_requirements_dict(g, p, r, gr, s, sp)
            ),
            inputs=[
                single_file,
                grammar_check,
                plagiarism_check,
                relevance_check,
                grading_check,
                summary_check,
                specialized_check,
            ],
            outputs=[single_status, single_results, single_download, single_errors],
        )

        # Batch processing
        process_batch_btn.click(
            fn=lambda files, g, p, r, gr, s, sp: grader.process_multiple_files_v2(
                files, create_requirements_dict(g, p, r, gr, s, sp)
            ),
            inputs=[
                batch_files,
                batch_grammar,
                batch_plagiarism,
                batch_relevance,
                batch_grading,
                batch_summary,
                batch_specialized,
            ],
            outputs=[batch_status, batch_summary_table, batch_download, batch_errors],
        )

        # Status refresh
        status_refresh.click(fn=grader.get_system_status, outputs=[status_display])

        # Show/hide error displays based on content
        single_status.change(
            fn=lambda status: gr.update(visible="?" in str(status or "")),
            inputs=[single_status],
            outputs=[single_errors],
        )

        batch_status.change(
            fn=lambda status: gr.update(visible="failed" in str(status or "").lower()),
            inputs=[batch_status],
            outputs=[batch_errors],
        )
    return interface


def main():
    """Main entry point for the Gradio application."""

    print("🚀 Starting RT-Capstone Assignment Grading System")
    print("📊 Initializing Gradio interface...")

    try:
        # Create interface
        interface = create_interface()

        # Optional basic auth
        demo_user = os.getenv("DEMO_USER", "").strip()
        demo_pass = os.getenv("DEMO_PASS", "").strip()
        auth = (demo_user, demo_pass) if demo_user and demo_pass else None

        inbrowser = os.getenv("DEMO_INBROWSER", "false").lower() in ("1", "true", "yes")

        # Resolve host/port from env with safe fallbacks
        server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1").strip() or "127.0.0.1"
        port_env = (os.getenv("GRADIO_SERVER_PORT", "7860") or "").strip().lower()
        share = os.getenv("GRADIO_SHARE", "false").lower() in ("1", "true", "yes")

        # Determine desired port: None lets Gradio pick a free port
        requested_port = None if port_env in ("", "0", "auto", "random") else int(port_env)

        # Launch the interface. Avoid passing server_port when we want auto.
        def do_launch(chosen_port: int | None):
            kwargs = dict(
                server_name=server_name,
                share=share,
                debug=False,
                show_error=True,
                inbrowser=inbrowser,
                auth=auth,
            )
            if chosen_port is not None:
                kwargs["server_port"] = chosen_port
            return interface.launch(**kwargs)

        try:
            do_launch(requested_port)
        except Exception as launch_err:
            msg = str(launch_err)
            if "Cannot find empty port" in msg or "port" in msg.lower():
                print("! Preferred port unavailable; retrying on a free port...")
                # Probe an open port explicitly to avoid server_port=0 issues
                import socket

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((server_name, 0))
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    free_port = s.getsockname()[1]
                do_launch(free_port)
            else:
                raise

    except Exception as e:
        print(f"❌ Failed to start Gradio interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


