"""
MCP Server for Assignment Grading Tools
Provides tools for grammar checking, plagiarism detection, relevance analysis, grading, and summarization.
"""
import asyncio
import json
import os
import re
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    raise ImportError("MCP is not available. Please install with: pip install 'mcp[cli]'")

try:
    import language_tool_python
    LANGUAGE_TOOL_AVAILABLE = True
except ImportError:
    LANGUAGE_TOOL_AVAILABLE = False

from llms import groq_llm
from paths import PLAGIARISM_REPORTS_FOLDER
from prompts import PLAGIARISM_CHECK, GRAMMAR_CHECK, RELEVANCE_CHECK, GRADING_PROMPT, SUMMARY_PROMPT
from file_processor import file_processor, FileRejectionReason

load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("assignment-grader")


# @mcp.tool()
# def grammar_check_llm(text: str) -> Dict[str, Any]:
#     """
#     Check grammatical errors in assignment text using LLM (legacy implementation).
#     
#     Args:
#         text: The assignment text to check for grammar errors
#         
#     Returns:
#         Dictionary with grammar error count and status
#     """
#     if groq_llm is None:
#         return {
#             "error": "LLM not available. Cannot check grammar.",
#             "grammar_errors": -1,
#             "status": "error"
#         }
#     
#     try:
#         prompt = GRAMMAR_CHECK.format(text=text)
#         response = groq_llm.invoke(prompt)
#         raw = response.content if hasattr(response, "content") else str(response).strip()
# 
#         match = re.search(r"\d+", raw)
#         if match:
#             error_count = int(match.group())
#             return {
#                 "grammar_errors": error_count,
#                 "raw_response": raw,
#                 "status": "success"
#             }
#         else:
#             return {
#                 "error": f"Could not parse grammar error count from response: {raw}",
#                 "grammar_errors": -1,
#                 "raw_response": raw,
#                 "status": "parse_error"
#             }
#     except Exception as e:
#         return {
#             "error": f"Grammar check failed: {str(e)}",
#             "grammar_errors": -1,
#             "status": "error"
#         }


@mcp.tool()
def grammar_check(text: str) -> Dict[str, Any]:
    """
    Check grammatical errors in assignment text using LanguageTool.
    
    Args:
        text: The assignment text to check for grammar errors
        
    Returns:
        Dictionary with grammar error details and status
    """
    if not LANGUAGE_TOOL_AVAILABLE:
        return {
            "error": "LanguageTool not available. Please install with: pip install language-tool-python",
            "grammar_errors": -1,
            "status": "error"
        }
    
    try:
        # Initialize LanguageTool for English
        tool = language_tool_python.LanguageTool('en-US')
        
        # Check the text for errors
        matches = tool.check(text)
        
        # Process errors into a structured format
        errors = []
        for match in matches:
            error_info = {
                "message": match.message,
                "context": match.context,
                "offset": match.offset,
                "length": match.errorLength,
                "rule_id": match.ruleId,
                "category": match.category,
                "suggestions": match.replacements[:3] if match.replacements else []
            }
            errors.append(error_info)
        
        # Close the tool to free resources
        tool.close()
        
        return {
            "grammar_errors": len(errors),
            "error_count": len(errors),
            "error_list": [f"{error['message']} (Suggestion: {', '.join(error['suggestions'][:2]) if error['suggestions'] else 'No suggestions'})" for error in errors],
            "error_details": errors,
            "status": "success",
            "tool": "LanguageTool"
        }
        
    except Exception as e:
        return {
            "error": f"LanguageTool grammar check failed: {str(e)}",
            "grammar_errors": -1,
            "status": "error"
        }


@mcp.tool()
def plagiarism_check(text: str, student_name: str) -> Dict[str, Any]:
    """
    Check for plagiarism in assignment text and save detailed report.
    
    Args:
        text: The assignment text to check for plagiarism
        student_name: Name of the student (used for report filename)
        
    Returns:
        Dictionary with plagiarism analysis results and report file path
    """
    if groq_llm is None:
        return {
            "error": "LLM not available",
            "report_file": None,
            "status": "error"
        }
    
    try:
        prompt = PLAGIARISM_CHECK.replace("{text}", text)
        response = groq_llm.invoke(prompt)
        result = response.content if hasattr(response, "content") else str(response).strip()

        # Ensure plagiarism reports directory exists
        os.makedirs(PLAGIARISM_REPORTS_FOLDER, exist_ok=True)
        
        file_path = os.path.join(PLAGIARISM_REPORTS_FOLDER, f"{student_name}_report.json")
        with open(file_path, "w") as f:
            f.write(result)

        return {
            "report_file": file_path,
            "analysis": result,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Plagiarism check failed: {str(e)}",
            "report_file": None,
            "status": "error"
        }


@mcp.tool()
def relevance_check(text: str, source: str) -> Dict[str, Any]:
    """
    Check content relevance against source material.
    
    Args:
        text: The assignment text to analyze
        source: The source material to compare against
        
    Returns:
        Dictionary with relevance analysis results
    """
    if groq_llm is None:
        return {
            "error": "LLM not available",
            "relevance_analysis": None,
            "status": "error"
        }
    
    try:
        prompt = RELEVANCE_CHECK.format(text=text, source=source)
        response = groq_llm.invoke(prompt)
        analysis = response.content if hasattr(response, "content") else str(response).strip()
        
        return {
            "relevance_analysis": analysis,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Relevance check failed: {str(e)}",
            "relevance_analysis": None,
            "status": "error"
        }


@mcp.tool()
def grade_assignment(assignment_text: str, source_text: str) -> Dict[str, Any]:
    """
    Grade assignment using LLM-based evaluation on multiple criteria (0-10 scale).
    
    Args:
        assignment_text: The student assignment to grade
        source_text: The reference/source material
        
    Returns:
        Dictionary with detailed grading scores for each criterion
    """
    if groq_llm is None:
        return {
            "error": "LLM not available",
            "grades": {
                "factuality": 0.0,
                "relevance": 0.0,
                "coherence": 0.0,
                "grammar": 1.0  # Grammar minimum score is 1
            },
            "status": "error"
        }
    
    try:
        prompt = GRADING_PROMPT.format(answer=assignment_text, source=source_text)
        response = groq_llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response).strip()

        try:
            scores = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: try to extract scores with regex
            matches = re.findall(r'"(\w+)":\s*([0-9.]+)', raw)
            scores = {k: float(v) for k, v in matches}

        grades = {
            "factuality": round(scores.get("factuality", 0), 2),
            "relevance": round(scores.get("relevance", 0), 2),
            "coherence": round(scores.get("coherence", 0), 2),
            "grammar": round(max(scores.get("grammar", 1), 1), 2)  # Ensure grammar is never below 1
        }
        
        return {
            "grades": grades,
            "raw_response": raw,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Grading failed: {str(e)}",
            "grades": {
                "factuality": 0.0,
                "relevance": 0.0,
                "coherence": 0.0,
                "grammar": 1.0
            },
            "raw_response": raw if 'raw' in locals() else 'No response',
            "status": "error"
        }


@mcp.tool()
def summarize_assignment(text: str) -> Dict[str, Any]:
    """
    Generate a concise summary of assignment text.
    
    Args:
        text: The assignment text to summarize
        
    Returns:
        Dictionary with assignment summary
    """
    if groq_llm is None:
        return {
            "error": "LLM not available for summarization",
            "summary": None,
            "status": "error"
        }
    
    try:
        prompt = SUMMARY_PROMPT.format(text=text)
        response = groq_llm.invoke(prompt)
        summary = response.content if hasattr(response, "content") else str(response).strip()
        
        return {
            "summary": summary,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Summarization failed: {str(e)}",
            "summary": None,
            "status": "error"
        }


@mcp.tool()
async def process_assignment_parallel(
    assignment_text: str,
    source_text: str,
    student_name: str
) -> Dict[str, Any]:
    """
    Process an assignment through all analysis tools in parallel for maximum efficiency.

    Args:
        assignment_text: The student assignment content
        source_text: The reference/source material
        student_name: Name of the student for report generation

    Returns:
        Dictionary with results from all analysis tools
    """
    try:
        # Create async wrappers for sync functions
        async def async_grammar():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: grammar_check(assignment_text))

        async def async_plagiarism():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: plagiarism_check(assignment_text, student_name))

        async def async_relevance():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: relevance_check(assignment_text, source_text))

        async def async_grading():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: grade_assignment(assignment_text, source_text))

        async def async_summary():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: summarize_assignment(assignment_text))

        # Execute all tools in parallel
        results = await asyncio.gather(
            async_grammar(),
            async_plagiarism(),
            async_relevance(),
            async_grading(),
            async_summary(),
            return_exceptions=True
        )

        # Process results
        grammar_result, plagiarism_result, relevance_result, grading_result, summary_result = results

        return {
            "student_name": student_name,
            "grammar_check": grammar_result if not isinstance(grammar_result, Exception) else {"error": str(grammar_result), "status": "error"},
            "plagiarism_check": plagiarism_result if not isinstance(plagiarism_result, Exception) else {"error": str(plagiarism_result), "status": "error"},
            "relevance_check": relevance_result if not isinstance(relevance_result, Exception) else {"error": str(relevance_result), "status": "error"},
            "grading": grading_result if not isinstance(grading_result, Exception) else {"error": str(grading_result), "status": "error"},
            "summary": summary_result if not isinstance(summary_result, Exception) else {"error": str(summary_result), "status": "error"},
            "processing_status": "completed"
        }

    except Exception as e:
        return {
            "error": f"Parallel processing failed: {str(e)}",
            "processing_status": "failed"
        }


@mcp.tool()
async def process_assignment_agentic(
    assignment_text: str,
    source_text: str,
    student_name: str,
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Process an assignment using the advanced agentic AI workflow.

    Args:
        assignment_text: The student assignment content
        source_text: The reference/source material
        student_name: Name of the student
        metadata: Optional metadata dictionary

    Returns:
        Dictionary with comprehensive results from agentic workflow
    """
    try:
        # Import agentic workflow
        from agentic_workflow import run_agentic_workflow

        # Prepare metadata
        if metadata is None:
            metadata = {
                "name": student_name,
                "date": "Unknown",
                "class": "Unknown",
                "subject": "Unknown"
            }

        # Run the agentic workflow
        result = await run_agentic_workflow(assignment_text, metadata, source_text)

        return {
            "workflow_type": "agentic",
            "student_name": student_name,
            "processing_status": "completed",
            "results": result
        }

    except ImportError:
        return {
            "error": "Agentic workflow not available",
            "processing_status": "failed",
            "fallback_suggestion": "Use process_assignment_parallel instead"
        }
    except Exception as e:
        return {
            "error": f"Agentic workflow failed: {str(e)}",
            "processing_status": "failed"
        }


@mcp.tool()
def process_file_content(file_path: str) -> Dict[str, Any]:
    """
    Process various file formats (PDF, DOCX, DOC, MD, TXT) and extract content.

    Args:
        file_path: Path to the file to process

    Returns:
        Dictionary with extracted content or rejection details
    """
    try:
        result = file_processor.extract_text_content(file_path)

        return {
            "success": result.success,
            "content": result.content,
            "metadata": result.metadata,
            "error": result.error,
            "rejection_reason": result.rejection_reason.value if result.rejection_reason else None,
            "file_path": file_path,
            "processing_method": "file_processor"
        }

    except Exception as e:
        return {
            "success": False,
            "content": "",
            "metadata": {},
            "error": f"File processing failed: {str(e)}",
            "rejection_reason": "processing_error",
            "file_path": file_path,
            "processing_method": "file_processor"
        }


@mcp.tool()
def validate_file_format(file_path: str) -> Dict[str, Any]:
    """
    Validate file format and check if it's supported for processing.

    Args:
        file_path: Path to the file to validate

    Returns:
        Dictionary with validation results and format information
    """
    try:
        # Detect file format
        file_format, detection_info = file_processor.detect_file_format(file_path)

        # Validate file
        rejection_reason = file_processor.validate_file(file_path)

        result = {
            "file_path": file_path,
            "detected_format": file_format.value,
            "detection_info": detection_info,
            "is_valid": rejection_reason is None,
            "supported": file_format.value != "unknown"
        }

        if rejection_reason:
            result.update({
                "rejection_reason": rejection_reason.value,
                "rejection_message": file_processor.get_rejection_message(rejection_reason, file_path)
            })

        return result

    except Exception as e:
        return {
            "file_path": file_path,
            "detected_format": "unknown",
            "detection_info": f"Error during detection: {str(e)}",
            "is_valid": False,
            "supported": False,
            "rejection_reason": "validation_error",
            "rejection_message": f"File validation failed: {str(e)}"
        }


@mcp.tool()
async def process_assignment_from_file(
    file_path: str,
    source_text: str,
    workflow_type: str = "agentic"
) -> Dict[str, Any]:
    """
    Process an assignment file through the complete grading workflow with robust error handling.

    Args:
        file_path: Path to the assignment file
        source_text: Reference source material
        workflow_type: Type of workflow to use ("agentic", "parallel", "traditional")

    Returns:
        Dictionary with processing results or detailed rejection information
    """
    try:
        # Step 1: Process the file to extract content
        file_result = file_processor.extract_text_content(file_path)

        if not file_result.success:
            # File was rejected - return detailed rejection info
            return {
                "processing_status": "rejected",
                "file_path": file_path,
                "rejection_reason": file_result.rejection_reason.value if file_result.rejection_reason else "unknown",
                "rejection_message": file_processor.get_rejection_message(
                    file_result.rejection_reason, file_path
                ) if file_result.rejection_reason else file_result.error,
                "error": file_result.error,
                "file_metadata": file_result.metadata,
                "workflow_type": workflow_type
            }

        # Step 2: Extract student metadata from content
        from utils import extract_metadata_from_content

        try:
            metadata = extract_metadata_from_content(file_path, file_result.content)
        except:
            # Fallback metadata
            import os
            metadata = {
                "name": os.path.splitext(os.path.basename(file_path))[0],
                "date": "Unknown",
                "class": "Unknown",
                "subject": "Unknown"
            }

        # Step 3: Process through appropriate workflow
        if workflow_type == "agentic":
            try:
                from agentic_workflow import run_agentic_workflow
                workflow_result = await run_agentic_workflow(file_result.content, metadata, source_text)

                return {
                    "processing_status": "completed",
                    "workflow_type": "agentic",
                    "file_path": file_path,
                    "file_metadata": file_result.metadata,
                    "student_metadata": metadata,
                    "results": workflow_result
                }

            except ImportError:
                # Fallback to parallel processing
                workflow_type = "parallel"
            except Exception as e:
                return {
                    "processing_status": "failed",
                    "workflow_type": "agentic",
                    "file_path": file_path,
                    "error": f"Agentic workflow failed: {str(e)}",
                    "file_metadata": file_result.metadata,
                    "student_metadata": metadata
                }

        if workflow_type == "parallel":
            try:
                parallel_result = await process_assignment_parallel(
                    file_result.content, source_text, metadata["name"]
                )

                return {
                    "processing_status": "completed",
                    "workflow_type": "parallel",
                    "file_path": file_path,
                    "file_metadata": file_result.metadata,
                    "student_metadata": metadata,
                    "results": parallel_result
                }

            except Exception as e:
                return {
                    "processing_status": "failed",
                    "workflow_type": "parallel",
                    "file_path": file_path,
                    "error": f"Parallel processing failed: {str(e)}",
                    "file_metadata": file_result.metadata,
                    "student_metadata": metadata
                }

        # Fallback: basic processing
        return {
            "processing_status": "completed",
            "workflow_type": "basic",
            "file_path": file_path,
            "file_metadata": file_result.metadata,
            "student_metadata": metadata,
            "content": file_result.content,
            "message": "Basic content extraction completed - advanced workflows unavailable"
        }

    except Exception as e:
        return {
            "processing_status": "error",
            "file_path": file_path,
            "error": f"Unexpected error during file processing: {str(e)}",
            "workflow_type": workflow_type
        }


@mcp.tool()
def get_supported_file_formats() -> Dict[str, Any]:
    """
    Get information about supported file formats and processing capabilities.

    Returns:
        Dictionary with supported formats and their descriptions
    """
    from file_processor import get_supported_formats

    return {
        "supported_formats": get_supported_formats(),
        "max_file_size_mb": file_processor.MAX_FILE_SIZE // (1024 * 1024),
        "min_content_length": file_processor.MIN_CONTENT_LENGTH,
        "processing_capabilities": {
            "pdf": "Multiple extraction methods (pdfplumber, PyPDF2)",
            "docx": "Full document processing including tables",
            "doc": "Legacy format support via mammoth",
            "markdown": "Markdown to plain text conversion",
            "text": "Multi-encoding support with auto-detection"
        },
        "rejection_reasons": [reason.value for reason in FileRejectionReason]
    }


@mcp.tool()
async def batch_process_files(
    file_paths: List[str],
    source_text: str,
    workflow_type: str = "agentic",
    include_rejections: bool = True
) -> Dict[str, Any]:
    """
    Process multiple assignment files in batch with comprehensive error handling.

    Args:
        file_paths: List of file paths to process
        source_text: Reference source material
        workflow_type: Type of workflow to use
        include_rejections: Whether to include rejected files in output

    Returns:
        Dictionary with batch processing results
    """
    results = {
        "processed_files": [],
        "rejected_files": [],
        "failed_files": [],
        "summary": {
            "total_files": len(file_paths),
            "successful": 0,
            "rejected": 0,
            "failed": 0
        }
    }

    for file_path in file_paths:
        try:
            result = await process_assignment_from_file(file_path, source_text, workflow_type)

            if result["processing_status"] == "completed":
                results["processed_files"].append(result)
                results["summary"]["successful"] += 1
            elif result["processing_status"] == "rejected":
                if include_rejections:
                    results["rejected_files"].append(result)
                results["summary"]["rejected"] += 1
            else:
                results["failed_files"].append(result)
                results["summary"]["failed"] += 1

        except Exception as e:
            error_result = {
                "file_path": file_path,
                "processing_status": "error",
                "error": f"Batch processing error: {str(e)}"
            }
            results["failed_files"].append(error_result)
            results["summary"]["failed"] += 1

    # Add processing statistics
    total = results["summary"]["total_files"]
    if total > 0:
        results["summary"]["success_rate"] = results["summary"]["successful"] / total
        results["summary"]["rejection_rate"] = results["summary"]["rejected"] / total
        results["summary"]["failure_rate"] = results["summary"]["failed"] / total

    return results


@mcp.resource("assignment://metadata/{file_path}")
def get_assignment_metadata(file_path: str) -> str:
    """
    Extract metadata from assignment file header.
    
    Expected format:
    Name: John Doe
    Date: 2025-08-25
    Class: 10
    Subject: English
    """
    try:
        with open(file_path, "r") as f:
            lines = [line.strip() for line in f.readlines()[:10]]

        meta = {"name": "Unknown", "date": "Unknown", "class": "Unknown", "subject": "Unknown"}
        
        for line in lines:
            if line.lower().startswith("name:"):
                meta["name"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("date:"):
                meta["date"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("class:"):
                meta["class"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("subject:"):
                meta["subject"] = line.split(":", 1)[1].strip()

        return json.dumps(meta, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to extract metadata: {str(e)}"})


@mcp.prompt()
def assignment_analysis_prompt(assignment_type: str = "general") -> str:
    """
    Generate prompts for different types of assignment analysis.
    
    Args:
        assignment_type: Type of analysis (grammar, plagiarism, relevance, grading, summary)
    """
    prompts = {
        "grammar": "Analyze the following assignment for grammatical errors, spelling mistakes, and writing quality issues. Provide a detailed breakdown.",
        "plagiarism": "Examine this assignment for potential plagiarism. Look for copied content, unusual phrasing, or inconsistent writing styles.",
        "relevance": "Evaluate how well this assignment addresses the given source material and topic requirements.",
        "grading": "Grade this assignment comprehensively on factual accuracy, relevance, coherence, and grammar using a 0-10 scale.",
        "summary": "Create a concise summary that captures the main points and quality of this assignment.",
        "general": "Perform a comprehensive analysis of this assignment including grammar, originality, relevance, and overall quality."
    }
    
    return prompts.get(assignment_type, prompts["general"])


if __name__ == "__main__":
    # For development testing
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        mcp.run(transport="stdio")
    else:
        print("Assignment Grading MCP Server - Enhanced File Processing")
        print("Usage: python mcp_server.py dev  # Run in development mode")
        print("")
        print("Core Analysis Tools:")
        print("- grammar_check: Check grammatical errors")
        print("- plagiarism_check: Detect potential plagiarism")
        print("- relevance_check: Analyze content relevance")
        print("- grade_assignment: Comprehensive grading")
        print("- summarize_assignment: Generate summary")
        print("")
        print("Workflow Processing:")
        print("- process_assignment_parallel: Run all tools in parallel")
        print("- process_assignment_agentic: Use advanced agentic AI workflow")
        print("")
        print("File Processing Tools:")
        print("- process_file_content: Extract content from PDF, DOCX, DOC, MD, TXT")
        print("- validate_file_format: Check file format and validate")
        print("- process_assignment_from_file: Complete file-to-grade workflow")
        print("- batch_process_files: Process multiple files with error handling")
        print("- get_supported_file_formats: Get format information")
        print("")
        print("Supported Formats: PDF, DOCX, DOC, MD, TXT")
        print("Max File Size: 50MB | Robust error handling & rejection tracking")