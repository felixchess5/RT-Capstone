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
from ocr_processor import ocr_processor, OCRMethod, ImageProcessingMethod
from language_support import language_manager, detect_text_language, get_supported_languages
from math_processor import create_math_processor, MathProblemType
from spanish_processor import create_spanish_processor, SpanishAssignmentType
from science_processor import create_science_processor, ScienceSubject, ScienceAssignmentType
from history_processor import create_history_processor, HistoryPeriod, HistoryAssignmentType
from assignment_orchestrator import create_assignment_orchestrator, SubjectType, AssignmentComplexity
from subject_output_manager import create_subject_output_manager, OutputSubject

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


@mcp.tool()
def extract_text_from_scanned_pdf(file_path: str, enhanced: bool = True) -> Dict[str, Any]:
    """
    Extract text from scanned/image-based PDF files using OCR.

    Args:
        file_path: Path to the scanned PDF file
        enhanced: Whether to use enhanced OCR methods for better accuracy

    Returns:
        Dictionary with extracted text, confidence, and metadata
    """
    try:
        from ocr_processor import extract_text_from_scanned_pdf

        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": "File does not exist",
                "file_path": file_path
            }

        # Check if file is actually a PDF
        if not file_path.lower().endswith('.pdf'):
            return {
                "success": False,
                "error": "File is not a PDF",
                "file_path": file_path
            }

        result = extract_text_from_scanned_pdf(file_path, enhanced)

        return {
            "success": result.success,
            "text": result.text if result.success else "",
            "confidence": result.confidence,
            "error": result.error if not result.success else "",
            "metadata": result.metadata,
            "file_path": file_path,
            "enhanced_mode": enhanced
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"OCR processing failed: {str(e)}",
            "file_path": file_path
        }


@mcp.tool()
def extract_text_from_image(image_path: str,
                           enhanced: bool = True,
                           ocr_method: str = "tesseract_enhanced",
                           preprocessing: str = "adaptive_threshold") -> Dict[str, Any]:
    """
    Extract text from image files using OCR.

    Args:
        image_path: Path to the image file (PNG, JPEG, TIFF, BMP)
        enhanced: Whether to try multiple OCR methods for best results
        ocr_method: OCR method to use ("tesseract" or "tesseract_enhanced")
        preprocessing: Image preprocessing method

    Returns:
        Dictionary with extracted text, confidence, and metadata
    """
    try:
        from ocr_processor import extract_text_from_image_file, OCRMethod, ImageProcessingMethod

        if not os.path.exists(image_path):
            return {
                "success": False,
                "error": "Image file does not exist",
                "file_path": image_path
            }

        # Check if file is an image
        valid_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
        if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
            return {
                "success": False,
                "error": "File is not a supported image format",
                "file_path": image_path,
                "supported_formats": valid_extensions
            }

        if enhanced:
            result = extract_text_from_image_file(image_path, enhanced=True)
        else:
            # Use specific OCR method and preprocessing
            ocr_method_enum = OCRMethod.TESSERACT_ENHANCED if ocr_method == "tesseract_enhanced" else OCRMethod.TESSERACT
            preprocessing_enum = getattr(ImageProcessingMethod, preprocessing.upper(), ImageProcessingMethod.ADAPTIVE_THRESHOLD)

            result = ocr_processor.extract_text_from_image(image_path, ocr_method_enum, preprocessing_enum)

        return {
            "success": result.success,
            "text": result.text if result.success else "",
            "confidence": result.confidence,
            "error": result.error if not result.success else "",
            "metadata": result.metadata,
            "file_path": image_path,
            "enhanced_mode": enhanced
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Image OCR processing failed: {str(e)}",
            "file_path": image_path
        }


@mcp.tool()
def check_if_pdf_is_scanned(pdf_path: str) -> Dict[str, Any]:
    """
    Check if a PDF file is scanned/image-based or contains searchable text.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with scan status and analysis details
    """
    try:
        from ocr_processor import is_scanned_pdf

        if not os.path.exists(pdf_path):
            return {
                "error": "PDF file does not exist",
                "file_path": pdf_path
            }

        if not pdf_path.lower().endswith('.pdf'):
            return {
                "error": "File is not a PDF",
                "file_path": pdf_path
            }

        is_scanned = is_scanned_pdf(pdf_path)

        return {
            "file_path": pdf_path,
            "is_scanned": is_scanned,
            "requires_ocr": is_scanned,
            "recommendation": "Use OCR extraction" if is_scanned else "Use standard text extraction"
        }

    except Exception as e:
        return {
            "error": f"PDF analysis failed: {str(e)}",
            "file_path": pdf_path
        }


@mcp.tool()
def get_ocr_capabilities() -> Dict[str, Any]:
    """
    Get information about available OCR capabilities and dependencies.

    Returns:
        Dictionary with OCR system status and capabilities
    """
    try:
        capabilities = {
            "tesseract_available": ocr_processor.tesseract_available,
            "pdf2image_available": ocr_processor.pdf2image_available,
            "opencv_available": True,  # Always available if OCR processor loads
            "supported_image_formats": ["PNG", "JPEG", "TIFF", "BMP"],
            "supported_pdf_types": ["Scanned PDFs", "Image-based PDFs"],
            "ocr_methods": ["tesseract", "tesseract_enhanced"],
            "preprocessing_methods": [
                "none", "grayscale", "threshold", "adaptive_threshold",
                "denoise", "morphological"
            ],
            "features": {
                "multi_method_enhancement": True,
                "confidence_scoring": True,
                "automatic_preprocessing": True,
                "batch_processing": True,
                "scanned_pdf_detection": True
            }
        }

        if ocr_processor.tesseract_available:
            try:
                import pytesseract
                capabilities["tesseract_version"] = str(pytesseract.get_tesseract_version())
            except:
                capabilities["tesseract_version"] = "Unknown"

        return capabilities

    except Exception as e:
        return {
            "error": f"Failed to get OCR capabilities: {str(e)}",
            "available": False
        }


@mcp.tool()
def detect_language(text: str) -> Dict[str, Any]:
    """
    Detect the language of the provided text.

    Args:
        text: Text to analyze for language detection

    Returns:
        Dictionary with detected language, confidence, and metadata
    """
    try:
        if not text or len(text.strip()) < 3:
            return {
                "error": "Text too short for reliable language detection",
                "text_length": len(text.strip())
            }

        result = detect_text_language(text)

        return {
            "primary_language": result.primary_language,
            "language_name": language_manager.get_language_config(result.primary_language).name,
            "confidence": result.confidence,
            "is_supported": result.is_supported,
            "fallback_language": result.fallback_language,
            "all_detected": result.all_detected,
            "text_sample": text[:100] + "..." if len(text) > 100 else text
        }

    except Exception as e:
        return {
            "error": f"Language detection failed: {str(e)}",
            "text_length": len(text) if text else 0
        }


@mcp.tool()
def get_supported_languages_info() -> Dict[str, Any]:
    """
    Get information about all supported languages for assignment grading.

    Returns:
        Dictionary with supported languages and their capabilities
    """
    try:
        languages = get_supported_languages()

        return {
            "total_languages": len(languages),
            "languages": languages,
            "features": {
                "grammar_checking": "Available for all supported languages",
                "plagiarism_detection": "Language-aware prompts available",
                "relevance_analysis": "Localized evaluation criteria",
                "content_grading": "Language-specific grading rubrics",
                "ocr_support": "Multi-language OCR with Tesseract",
                "automatic_detection": "Automatic language detection from content"
            },
            "default_language": "en",
            "fallback_behavior": "Unsupported languages fallback to closest supported language"
        }

    except Exception as e:
        return {
            "error": f"Failed to get language information: {str(e)}",
            "available": False
        }


@mcp.tool()
def grade_assignment_multilingual(assignment_text: str,
                                source_text: str,
                                language_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Grade assignment with automatic language detection and localized evaluation.

    Args:
        assignment_text: The student assignment text
        source_text: Source material for comparison
        language_hint: Optional language hint (e.g., 'en', 'es', 'fr')

    Returns:
        Dictionary with grading results and language information
    """
    try:
        from llms import invoke_with_fallback, groq_llm, gemini_llm
        from language_support import get_localized_prompt

        # Detect language or use hint
        if language_hint and language_hint in language_manager.supported_languages:
            detected_language = language_hint
            lang_confidence = 1.0
        else:
            lang_result = detect_text_language(assignment_text)
            detected_language = lang_result.fallback_language
            lang_confidence = lang_result.confidence

        language_name = language_manager.get_language_config(detected_language).name

        # Use localized grading prompt
        prompt = get_localized_prompt("grading_prompt", detected_language,
                                    answer=assignment_text, source=source_text)

        response = invoke_with_fallback(prompt, groq_llm, gemini_llm)
        raw_response = response.content if hasattr(response, "content") else str(response)

        # Parse grading results
        try:
            import json
            scores = json.loads(raw_response.strip())
        except json.JSONDecodeError:
            # Fallback parsing
            import re
            numbers = re.findall(r'\d+\.?\d*', raw_response)
            if len(numbers) >= 4:
                scores = {
                    "factuality": float(numbers[0]),
                    "relevance": float(numbers[1]),
                    "coherence": float(numbers[2]),
                    "grammar": float(numbers[3])
                }
            else:
                scores = {"factuality": 5.0, "relevance": 5.0, "coherence": 5.0, "grammar": 5.0}

        # Calculate overall score
        overall_score = (scores["factuality"] + scores["relevance"] +
                        scores["coherence"] + scores["grammar"]) / 4

        # Determine letter grade
        if overall_score >= 9: letter_grade = "A"
        elif overall_score >= 8: letter_grade = "B"
        elif overall_score >= 7: letter_grade = "C"
        elif overall_score >= 6: letter_grade = "D"
        else: letter_grade = "F"

        return {
            "individual_scores": scores,
            "overall_score": round(overall_score, 2),
            "letter_grade": letter_grade,
            "language_info": {
                "detected_language": detected_language,
                "language_name": language_name,
                "confidence": lang_confidence,
                "language_hint_used": language_hint is not None
            },
            "evaluation_method": "multilingual_grading",
            "raw_response": raw_response[:200] + "..." if len(raw_response) > 200 else raw_response
        }

    except Exception as e:
        return {
            "error": f"Multilingual grading failed: {str(e)}",
            "assignment_length": len(assignment_text) if assignment_text else 0
        }


@mcp.tool()
def grammar_check_multilingual(text: str, language_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Check grammar with automatic language detection and language-appropriate analysis.

    Args:
        text: Text to check for grammar errors
        language_hint: Optional language hint (e.g., 'en', 'es', 'fr')

    Returns:
        Dictionary with grammar analysis and language information
    """
    try:
        from llms import invoke_with_fallback, groq_llm, gemini_llm
        from language_support import get_localized_prompt

        # Detect language or use hint
        if language_hint and language_hint in language_manager.supported_languages:
            detected_language = language_hint
            lang_confidence = 1.0
        else:
            lang_result = detect_text_language(text)
            detected_language = lang_result.fallback_language
            lang_confidence = lang_result.confidence

        language_name = language_manager.get_language_config(detected_language).name

        # Use localized grammar check prompt
        prompt = get_localized_prompt("grammar_check", detected_language, text=text)

        response = invoke_with_fallback(prompt, groq_llm, gemini_llm)
        raw_response = response.content if hasattr(response, "content") else str(response)

        # Extract error count
        import re
        error_match = re.search(r'\d+', raw_response)
        error_count = int(error_match.group()) if error_match else 0

        return {
            "error_count": error_count,
            "language_info": {
                "detected_language": detected_language,
                "language_name": language_name,
                "confidence": lang_confidence,
                "language_hint_used": language_hint is not None
            },
            "analysis": raw_response,
            "quality_impact": min(error_count * 0.1, 1.0),
            "evaluation_method": "multilingual_grammar_check"
        }

    except Exception as e:
        return {
            "error": f"Multilingual grammar check failed: {str(e)}",
            "text_length": len(text) if text else 0
        }


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
        print("- process_file_content: Extract content from PDF, DOCX, DOC, MD, TXT, Images")
        print("- validate_file_format: Check file format and validate")
        print("- process_assignment_from_file: Complete file-to-grade workflow")
        print("- batch_process_files: Process multiple files with error handling")
        print("- get_supported_file_formats: Get format information")
        print("")
        print("OCR & Scanned Document Tools:")
        print("- extract_text_from_scanned_pdf: OCR for scanned PDF files")
        print("- extract_text_from_image: OCR for image files (PNG, JPEG, TIFF, BMP)")
        print("- check_if_pdf_is_scanned: Detect if PDF needs OCR processing")
        print("- get_ocr_capabilities: Check OCR system status and features")
        print("")
        print("Multi-Language Support Tools:")
        print("- detect_language: Automatic language detection from text")
        print("- get_supported_languages_info: List of supported languages and features")
        print("- grade_assignment_multilingual: Language-aware assignment grading")
        print("- grammar_check_multilingual: Multi-language grammar checking")
        print("")
        print("Specialized Subject Processing Tools:")
        print("- analyze_math_assignment: Complete mathematical analysis and grading")
        print("- solve_equation: Individual equation solving with step-by-step solutions")
        print("- analyze_spanish_assignment: Comprehensive Spanish language assessment")
        print("- check_spanish_grammar: Targeted Spanish grammar checking")
        print("- classify_assignment_intelligent: Automatic subject classification")
        print("- process_assignment_intelligent: Intelligent routing to specialized processors")
        print("- get_available_subject_processors: List all specialized capabilities")
        print("")
        print("Subject-Specific Output Tools:")
        print("- export_subject_specific_results: Export all assignments to subject-specific files")
        print("- export_math_assignments: Export only mathematics assignments")
        print("- export_spanish_assignments: Export only Spanish assignments")
        print("- export_english_assignments: Export only English assignments")
        print("- get_subject_classification_info: Get subject classification without full processing")
        print("")
        print("Supported Subjects: Mathematics, Spanish, English, Science, History, General")
        print("Math Features: Equation solving, Symbolic computation, Step-by-step analysis, Problem type detection")
        print("Spanish Features: Grammar checking, Vocabulary analysis, Cultural references, Fluency assessment")
        print("Output Formats: Subject-specific CSV and JSON files with specialized fields")
        print("")
        print("Supported Languages: English, Spanish, French, German, Italian, Portuguese, Dutch, Russian, Chinese, Japanese, Korean, Arabic, Hindi")
        print("Supported Formats: PDF (text & scanned), DOCX, DOC, MD, TXT, PNG, JPEG, TIFF, BMP")
        print("OCR Features: Free Tesseract OCR, Multi-language support, Enhanced preprocessing, Confidence scoring")
        print("Language Features: Auto-detection, Localized prompts, Multi-language OCR, Fallback support")
        print("Max File Size: 50MB | Robust error handling & rejection tracking")


# ==================== SPECIALIZED SUBJECT PROCESSORS ====================

# Initialize specialized processors
math_processor = create_math_processor()
spanish_processor = create_spanish_processor()
orchestrator = create_assignment_orchestrator()


@mcp.tool()
def analyze_math_assignment(assignment_text: str) -> Dict[str, Any]:
    """
    Analyze mathematical assignment with equation solving and specialized math grading.

    Args:
        assignment_text: The math assignment text to analyze

    Returns:
        Dictionary with mathematical analysis, solutions, and grading
    """
    try:
        analysis = math_processor.analyze_math_assignment(assignment_text)
        grading = math_processor.grade_math_assignment(assignment_text)

        return {
            "analysis": {
                "problem_types": analysis["problem_types"],
                "equations_found": analysis["equations_found"],
                "completeness_score": analysis["completeness_score"],
                "step_by_step_present": analysis["step_by_step_present"],
                "mathematical_notation": analysis["mathematical_notation"]
            },
            "solutions": analysis["solutions"],
            "grading": {
                "mathematical_accuracy": grading["mathematical_accuracy"],
                "problem_solving_approach": grading["problem_solving_approach"],
                "notation_clarity": grading["notation_clarity"],
                "step_by_step_work": grading["step_by_step_work"],
                "overall_score": grading["overall_score"]
            },
            "feedback": grading["feedback"],
            "subject": "mathematics",
            "processing_type": "specialized_math"
        }
    except Exception as e:
        return {
            "error": f"Math analysis failed: {str(e)}",
            "subject": "mathematics",
            "processing_type": "error"
        }


@mcp.tool()
def solve_equation(equation: str) -> Dict[str, Any]:
    """
    Solve a mathematical equation using symbolic computation.

    Args:
        equation: Mathematical equation to solve

    Returns:
        Dictionary with solution steps and result
    """
    try:
        solution = math_processor.solve_equation(equation)

        return {
            "problem": solution.problem,
            "solution": str(solution.solution),
            "steps": solution.steps,
            "problem_type": solution.problem_type.value,
            "confidence": solution.confidence,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Equation solving failed: {str(e)}",
            "problem": equation,
            "status": "error"
        }


@mcp.tool()
def identify_math_problem_type(text: str) -> Dict[str, Any]:
    """
    Identify the type of mathematical problem from text.

    Args:
        text: Text containing mathematical problem

    Returns:
        Dictionary with identified problem type and confidence
    """
    try:
        problem_type = math_processor.identify_problem_type(text)
        equations = math_processor.extract_equations(text)

        return {
            "problem_type": problem_type.value,
            "equations_found": equations,
            "equation_count": len(equations),
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Problem type identification failed: {str(e)}",
            "status": "error"
        }


@mcp.tool()
def analyze_spanish_assignment(assignment_text: str, source_text: str = None) -> Dict[str, Any]:
    """
    Analyze Spanish language assignment with grammar, vocabulary, and cultural assessment.

    Args:
        assignment_text: The Spanish assignment text to analyze
        source_text: Optional source text for comparison

    Returns:
        Dictionary with Spanish language analysis and grading
    """
    try:
        analysis = spanish_processor.analyze_spanish_assignment(assignment_text)
        grading = spanish_processor.grade_spanish_assignment(assignment_text, source_text)

        return {
            "analysis": {
                "assignment_type": analysis.assignment_type.value,
                "vocabulary_level": analysis.vocabulary_level,
                "fluency_score": analysis.fluency_score,
                "complexity_score": analysis.complexity_score,
                "grammar_errors_count": len(analysis.grammar_errors),
                "cultural_references_count": len(analysis.cultural_references),
                "comprehension_questions_count": len(analysis.comprehension_questions)
            },
            "grammar_errors": analysis.grammar_errors,
            "verb_conjugations": analysis.verb_conjugations,
            "cultural_references": analysis.cultural_references,
            "comprehension_questions": analysis.comprehension_questions,
            "grading": {
                "grammar_accuracy": grading["grammar_accuracy"],
                "vocabulary_usage": grading["vocabulary_usage"],
                "fluency_communication": grading["fluency_communication"],
                "cultural_understanding": grading["cultural_understanding"],
                "overall_score": grading["overall_score"]
            },
            "feedback": grading["feedback"],
            "subject": "spanish",
            "processing_type": "specialized_spanish"
        }
    except Exception as e:
        return {
            "error": f"Spanish analysis failed: {str(e)}",
            "subject": "spanish",
            "processing_type": "error"
        }


@mcp.tool()
def check_spanish_grammar(text: str) -> Dict[str, Any]:
    """
    Check Spanish grammar and provide specific linguistic feedback.

    Args:
        text: Spanish text to check

    Returns:
        Dictionary with grammar errors and suggestions
    """
    try:
        errors = spanish_processor.check_grammar(text)
        vocab_level = spanish_processor.analyze_vocabulary_level(text)
        fluency = spanish_processor.calculate_fluency_score(text)

        return {
            "grammar_errors": errors,
            "error_count": len(errors),
            "vocabulary_level": vocab_level,
            "fluency_score": fluency,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Spanish grammar check failed: {str(e)}",
            "status": "error"
        }


@mcp.tool()
def analyze_spanish_vocabulary(text: str) -> Dict[str, Any]:
    """
    Analyze Spanish vocabulary usage and level.

    Args:
        text: Spanish text to analyze

    Returns:
        Dictionary with vocabulary analysis
    """
    try:
        vocab_level = spanish_processor.analyze_vocabulary_level(text)
        conjugations = spanish_processor.analyze_verb_conjugations(text)
        cultural_refs = spanish_processor.extract_cultural_references(text)

        return {
            "vocabulary_level": vocab_level,
            "verb_conjugations": conjugations,
            "cultural_references": cultural_refs,
            "conjugation_variety": len(conjugations),
            "cultural_depth": len(cultural_refs),
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Spanish vocabulary analysis failed: {str(e)}",
            "status": "error"
        }


@mcp.tool()
def classify_assignment_intelligent(assignment_text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Intelligently classify assignment by subject, complexity, and determine appropriate processing approach.

    Args:
        assignment_text: The assignment text to classify
        metadata: Optional metadata with hints (subject, class, etc.)

    Returns:
        Dictionary with classification results and processing recommendations
    """
    try:
        if metadata is None:
            metadata = {}

        classification = orchestrator.classify_assignment(assignment_text, metadata)

        return {
            "classification": {
                "subject": classification.subject.value,
                "complexity": classification.complexity.value,
                "specific_type": classification.specific_type,
                "confidence": classification.confidence,
                "language": classification.language,
                "processing_approach": classification.processing_approach
            },
            "tools_needed": classification.tools_needed,
            "recommended_processor": classification.subject.value,
            "processing_suggestions": {
                "use_math_processor": classification.subject == SubjectType.MATHEMATICS,
                "use_spanish_processor": classification.subject == SubjectType.SPANISH,
                "use_general_processor": classification.subject in [SubjectType.ENGLISH, SubjectType.GENERAL]
            },
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Assignment classification failed: {str(e)}",
            "status": "error"
        }


@mcp.tool()
async def process_assignment_intelligent(assignment_text: str, source_text: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Intelligently process assignment using the most appropriate specialized processor.

    Args:
        assignment_text: The assignment text to process
        source_text: Optional source text for comparison
        metadata: Optional metadata with hints

    Returns:
        Dictionary with comprehensive processing results
    """
    try:
        if metadata is None:
            metadata = {}

        result = await orchestrator.process_assignment(assignment_text, source_text, metadata)

        return {
            "classification": result["classification"],
            "processing_results": result["processing_results"],
            "overall_score": result["overall_score"],
            "specialized_feedback": result["specialized_feedback"],
            "recommended_next_steps": result["recommended_next_steps"],
            "processor_used": result["classification"]["subject"],
            "processing_approach": result["classification"]["processing_approach"],
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Intelligent processing failed: {str(e)}",
            "status": "error"
        }


@mcp.tool()
def get_available_subject_processors() -> Dict[str, Any]:
    """
    Get information about available specialized subject processors and their capabilities.

    Returns:
        Dictionary with processor information and capabilities
    """
    try:
        processors_info = orchestrator.get_available_processors()

        return {
            "available_subjects": processors_info["subjects"],
            "math_capabilities": {
                "problem_types": processors_info["math_problem_types"],
                "features": [
                    "Equation solving", "Symbolic computation", "Calculus operations",
                    "Step-by-step solutions", "Problem type detection", "Mathematical notation analysis"
                ]
            },
            "spanish_capabilities": {
                "assignment_types": processors_info["spanish_assignment_types"],
                "features": [
                    "Grammar checking", "Vocabulary analysis", "Conjugation verification",
                    "Cultural reference detection", "Fluency assessment", "Reading comprehension analysis"
                ]
            },
            "complexity_levels": processors_info["complexity_levels"],
            "available_tools": processors_info["available_tools"],
            "orchestrator_features": [
                "Automatic subject detection", "Complexity assessment", "Tool recommendation",
                "Processing approach optimization", "Multi-language support"
            ],
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Failed to get processor information: {str(e)}",
            "status": "error"
        }


# ==================== SUBJECT-SPECIFIC OUTPUT TOOLS ====================

# Initialize output manager
output_manager = create_subject_output_manager()


@mcp.tool()
def export_subject_specific_results(assignments_data: List[Dict[str, Any]], output_folder: str = "./output") -> Dict[str, Any]:
    """
    Export assignment results to subject-specific CSV and JSON files.

    Args:
        assignments_data: List of assignment result dictionaries
        output_folder: Optional output folder path (defaults to "./output")

    Returns:
        Dictionary with export results and file paths
    """
    try:
        # Create output manager for specified folder
        subject_manager = create_subject_output_manager(output_folder)

        # Export all subjects
        export_results = subject_manager.export_all_subjects(assignments_data)

        # Count assignments by subject
        subject_counts = {}
        for assignment in assignments_data:
            subject = subject_manager.determine_subject(assignment)
            subject_counts[subject.value] = subject_counts.get(subject.value, 0) + 1

        return {
            "export_results": export_results,
            "subject_counts": subject_counts,
            "total_assignments": len(assignments_data),
            "total_files_created": sum(len(files) for files in export_results.values()),
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Subject-specific export failed: {str(e)}",
            "status": "error"
        }


@mcp.tool()
def export_math_assignments(assignments_data: List[Dict[str, Any]], output_folder: str = "./output") -> Dict[str, Any]:
    """
    Export only mathematics assignments to CSV and JSON files.

    Args:
        assignments_data: List of assignment result dictionaries
        output_folder: Optional output folder path

    Returns:
        Dictionary with math assignment export results
    """
    try:
        subject_manager = create_subject_output_manager(output_folder)

        csv_path = subject_manager.export_subject_csv(assignments_data, OutputSubject.MATHEMATICS)
        json_path = subject_manager.export_subject_json(assignments_data, OutputSubject.MATHEMATICS)

        # Count math assignments
        math_assignments = [
            assignment for assignment in assignments_data
            if subject_manager.determine_subject(assignment) == OutputSubject.MATHEMATICS
        ]

        return {
            "csv_file": csv_path,
            "json_file": json_path,
            "math_assignments_count": len(math_assignments),
            "files_created": [csv_path, json_path] if csv_path and json_path else [],
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Math assignments export failed: {str(e)}",
            "status": "error"
        }


@mcp.tool()
def export_spanish_assignments(assignments_data: List[Dict[str, Any]], output_folder: str = "./output") -> Dict[str, Any]:
    """
    Export only Spanish assignments to CSV and JSON files.

    Args:
        assignments_data: List of assignment result dictionaries
        output_folder: Optional output folder path

    Returns:
        Dictionary with Spanish assignment export results
    """
    try:
        subject_manager = create_subject_output_manager(output_folder)

        csv_path = subject_manager.export_subject_csv(assignments_data, OutputSubject.SPANISH)
        json_path = subject_manager.export_subject_json(assignments_data, OutputSubject.SPANISH)

        # Count Spanish assignments
        spanish_assignments = [
            assignment for assignment in assignments_data
            if subject_manager.determine_subject(assignment) == OutputSubject.SPANISH
        ]

        return {
            "csv_file": csv_path,
            "json_file": json_path,
            "spanish_assignments_count": len(spanish_assignments),
            "files_created": [csv_path, json_path] if csv_path and json_path else [],
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Spanish assignments export failed: {str(e)}",
            "status": "error"
        }


@mcp.tool()
def export_english_assignments(assignments_data: List[Dict[str, Any]], output_folder: str = "./output") -> Dict[str, Any]:
    """
    Export only English assignments to CSV and JSON files.

    Args:
        assignments_data: List of assignment result dictionaries
        output_folder: Optional output folder path

    Returns:
        Dictionary with English assignment export results
    """
    try:
        subject_manager = create_subject_output_manager(output_folder)

        csv_path = subject_manager.export_subject_csv(assignments_data, OutputSubject.ENGLISH)
        json_path = subject_manager.export_subject_json(assignments_data, OutputSubject.ENGLISH)

        # Count English assignments
        english_assignments = [
            assignment for assignment in assignments_data
            if subject_manager.determine_subject(assignment) == OutputSubject.ENGLISH
        ]

        return {
            "csv_file": csv_path,
            "json_file": json_path,
            "english_assignments_count": len(english_assignments),
            "files_created": [csv_path, json_path] if csv_path and json_path else [],
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"English assignments export failed: {str(e)}",
            "status": "error"
        }


@mcp.tool()
def get_subject_classification_info(assignment_text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get subject classification information for an assignment without full processing.

    Args:
        assignment_text: The assignment text to classify
        metadata: Optional metadata with hints

    Returns:
        Dictionary with classification information
    """
    try:
        if metadata is None:
            metadata = {}

        subject_manager = create_subject_output_manager()
        orchestrator = create_assignment_orchestrator()

        # Classify the assignment
        classification = orchestrator.classify_assignment(assignment_text, metadata)

        # Determine output subject
        mock_assignment = {
            "assignment_classification": {
                "subject": classification.subject.value
            }
        }
        output_subject = subject_manager.determine_subject(mock_assignment)

        return {
            "classification": {
                "subject": classification.subject.value,
                "complexity": classification.complexity.value,
                "specific_type": classification.specific_type,
                "confidence": classification.confidence,
                "language": classification.language
            },
            "output_subject": output_subject.value,
            "recommended_files": {
                "csv": f"{output_subject.value}_assignments.csv",
                "json": f"{output_subject.value}_assignments.json"
            },
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Subject classification failed: {str(e)}",
            "status": "error"
        }


# ============================================================================
# SCIENCE ASSIGNMENT TOOLS
# ============================================================================

@mcp.tool()
def analyze_science_assignment(assignment_text: str) -> Dict[str, Any]:
    """
    Analyze a science assignment for scientific method, vocabulary, and subject area.

    Args:
        assignment_text: The science assignment text to analyze

    Returns:
        Dictionary with comprehensive science assignment analysis
    """
    try:
        science_processor = create_science_processor()
        analysis = science_processor.analyze_science_assignment(assignment_text)

        return {
            "subject_area": analysis.subject_area.value,
            "assignment_type": analysis.assignment_type.value,
            "scientific_method_elements": analysis.scientific_method_elements,
            "units_and_measurements": analysis.units_and_measurements,
            "formulas_identified": analysis.formulas_identified,
            "data_tables_present": analysis.data_tables_present,
            "graphs_charts_present": analysis.graphs_charts_present,
            "hypothesis_present": analysis.hypothesis_present,
            "conclusion_present": analysis.conclusion_present,
            "scientific_vocabulary_score": analysis.scientific_vocabulary_score,
            "experimental_variables": analysis.experimental_variables,
            "safety_considerations": analysis.safety_considerations,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Science analysis failed: {str(e)}",
            "status": "error"
        }


@mcp.tool()
async def grade_science_assignment(assignment_text: str, source_text: str = None) -> Dict[str, Any]:
    """
    Grade a science assignment with specialized science criteria.

    Args:
        assignment_text: The science assignment text to grade
        source_text: Optional source/reference material

    Returns:
        Dictionary with science-specific grading results
    """
    try:
        science_processor = create_science_processor()
        grading_result = await science_processor.grade_science_assignment(assignment_text, source_text)

        return {
            "grading_result": grading_result,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Science grading failed: {str(e)}",
            "status": "error"
        }


@mcp.tool()
def identify_science_subject(assignment_text: str) -> Dict[str, Any]:
    """
    Identify the specific science subject area (physics, chemistry, biology, etc.).

    Args:
        assignment_text: The assignment text to analyze

    Returns:
        Dictionary with science subject identification
    """
    try:
        science_processor = create_science_processor()
        subject_area = science_processor.identify_science_subject(assignment_text)

        return {
            "science_subject": subject_area.value,
            "available_subjects": [subject.value for subject in ScienceSubject],
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Science subject identification failed: {str(e)}",
            "status": "error"
        }


@mcp.tool()
def extract_scientific_formulas(assignment_text: str) -> Dict[str, Any]:
    """
    Extract scientific formulas and equations from assignment text.

    Args:
        assignment_text: The assignment text to analyze

    Returns:
        Dictionary with identified formulas and equations
    """
    try:
        science_processor = create_science_processor()
        formulas = science_processor.identify_formulas(assignment_text)
        units = science_processor.extract_units_and_measurements(assignment_text)

        return {
            "formulas_identified": formulas,
            "units_and_measurements": units,
            "formula_count": len(formulas),
            "unit_count": len(units),
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Formula extraction failed: {str(e)}",
            "status": "error"
        }


@mcp.tool()
def check_scientific_method(assignment_text: str) -> Dict[str, Any]:
    """
    Check for presence of scientific method elements in the assignment.

    Args:
        assignment_text: The assignment text to check

    Returns:
        Dictionary with scientific method element analysis
    """
    try:
        science_processor = create_science_processor()
        method_elements = science_processor.analyze_scientific_method(assignment_text)
        variables = science_processor.identify_experimental_variables(assignment_text)

        elements_present = sum(method_elements.values())
        total_elements = len(method_elements)

        return {
            "scientific_method_elements": method_elements,
            "experimental_variables": variables,
            "elements_present": elements_present,
            "total_elements": total_elements,
            "completeness_percentage": (elements_present / total_elements) * 100 if total_elements > 0 else 0,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Scientific method check failed: {str(e)}",
            "status": "error"
        }


@mcp.tool()
def export_science_assignments(assignments_data: List[Dict[str, Any]], output_folder: str = "./output") -> Dict[str, Any]:
    """
    Export only Science assignments to CSV and JSON files.

    Args:
        assignments_data: List of assignment result dictionaries
        output_folder: Optional output folder path

    Returns:
        Dictionary with Science assignment export results
    """
    try:
        subject_manager = create_subject_output_manager(output_folder)

        csv_path = subject_manager.export_subject_csv(assignments_data, OutputSubject.SCIENCE)
        json_path = subject_manager.export_subject_json(assignments_data, OutputSubject.SCIENCE)

        # Count Science assignments
        science_assignments = [
            assignment for assignment in assignments_data
            if subject_manager.determine_subject(assignment) == OutputSubject.SCIENCE
        ]

        return {
            "csv_file": csv_path,
            "json_file": json_path,
            "science_assignments_count": len(science_assignments),
            "files_created": [csv_path, json_path] if csv_path and json_path else [],
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Science assignments export failed: {str(e)}",
            "status": "error"
        }


# ============================================================================
# HISTORY ASSIGNMENT TOOLS
# ============================================================================

@mcp.tool()
def analyze_history_assignment(assignment_text: str) -> Dict[str, Any]:
    """
    Analyze a history assignment for chronology, context, and historical accuracy.

    Args:
        assignment_text: The history assignment text to analyze

    Returns:
        Dictionary with comprehensive history assignment analysis
    """
    try:
        history_processor = create_history_processor()
        analysis = history_processor.analyze_history_assignment(assignment_text)

        return {
            "historical_period": analysis.period.value,
            "assignment_type": analysis.assignment_type.value,
            "region_focus": analysis.region_focus.value,
            "dates_identified": analysis.dates_identified,
            "historical_figures": analysis.historical_figures,
            "events_mentioned": analysis.events_mentioned,
            "sources_cited": analysis.sources_cited,
            "chronological_accuracy": analysis.chronological_accuracy,
            "historical_context_score": analysis.historical_context_score,
            "argument_structure_score": analysis.argument_structure_score,
            "evidence_usage_score": analysis.evidence_usage_score,
            "bias_awareness_score": analysis.bias_awareness_score,
            "historical_vocabulary_score": analysis.historical_vocabulary_score,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"History analysis failed: {str(e)}",
            "status": "error"
        }


@mcp.tool()
async def grade_history_assignment(assignment_text: str, source_text: str = None) -> Dict[str, Any]:
    """
    Grade a history assignment with specialized history criteria.

    Args:
        assignment_text: The history assignment text to grade
        source_text: Optional source/reference material

    Returns:
        Dictionary with history-specific grading results
    """
    try:
        history_processor = create_history_processor()
        grading_result = await history_processor.grade_history_assignment(assignment_text, source_text)

        return {
            "grading_result": grading_result,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"History grading failed: {str(e)}",
            "status": "error"
        }


@mcp.tool()
def identify_historical_period(assignment_text: str) -> Dict[str, Any]:
    """
    Identify the historical time period focus of the assignment.

    Args:
        assignment_text: The assignment text to analyze

    Returns:
        Dictionary with historical period identification
    """
    try:
        history_processor = create_history_processor()
        period = history_processor.identify_historical_period(assignment_text)
        region = history_processor.identify_region_focus(assignment_text)

        return {
            "historical_period": period.value,
            "region_focus": region.value,
            "available_periods": [period.value for period in HistoryPeriod],
            "available_regions": [region.value for region in RegionFocus],
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Historical period identification failed: {str(e)}",
            "status": "error"
        }


@mcp.tool()
def extract_historical_elements(assignment_text: str) -> Dict[str, Any]:
    """
    Extract historical dates, figures, and events from assignment text.

    Args:
        assignment_text: The assignment text to analyze

    Returns:
        Dictionary with extracted historical elements
    """
    try:
        history_processor = create_history_processor()
        dates = history_processor.extract_dates(assignment_text)
        figures = history_processor.extract_historical_figures(assignment_text)
        events = history_processor.extract_historical_events(assignment_text)
        sources = history_processor.extract_sources(assignment_text)

        return {
            "dates_identified": dates,
            "historical_figures": figures,
            "events_mentioned": events,
            "sources_cited": sources,
            "dates_count": len(dates),
            "figures_count": len(figures),
            "events_count": len(events),
            "sources_count": len(sources),
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Historical element extraction failed: {str(e)}",
            "status": "error"
        }


@mcp.tool()
def check_historical_accuracy(assignment_text: str) -> Dict[str, Any]:
    """
    Check historical accuracy and context awareness in the assignment.

    Args:
        assignment_text: The assignment text to check

    Returns:
        Dictionary with historical accuracy assessment
    """
    try:
        history_processor = create_history_processor()
        dates = history_processor.extract_dates(assignment_text)

        chronological_accuracy = history_processor.assess_chronological_accuracy(assignment_text, dates)
        context_score = history_processor.assess_historical_context(assignment_text,
                                                                  history_processor.identify_historical_period(assignment_text))
        bias_awareness = history_processor.assess_bias_awareness(assignment_text)
        vocabulary_score = history_processor.assess_historical_vocabulary(assignment_text)

        return {
            "chronological_accuracy": chronological_accuracy,
            "historical_context_score": context_score,
            "bias_awareness_score": bias_awareness,
            "historical_vocabulary_score": vocabulary_score,
            "overall_accuracy_score": (chronological_accuracy + context_score + bias_awareness + vocabulary_score) / 4,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Historical accuracy check failed: {str(e)}",
            "status": "error"
        }


@mcp.tool()
def export_history_assignments(assignments_data: List[Dict[str, Any]], output_folder: str = "./output") -> Dict[str, Any]:
    """
    Export only History assignments to CSV and JSON files.

    Args:
        assignments_data: List of assignment result dictionaries
        output_folder: Optional output folder path

    Returns:
        Dictionary with History assignment export results
    """
    try:
        subject_manager = create_subject_output_manager(output_folder)

        csv_path = subject_manager.export_subject_csv(assignments_data, OutputSubject.HISTORY)
        json_path = subject_manager.export_subject_json(assignments_data, OutputSubject.HISTORY)

        # Count History assignments
        history_assignments = [
            assignment for assignment in assignments_data
            if subject_manager.determine_subject(assignment) == OutputSubject.HISTORY
        ]

        return {
            "csv_file": csv_path,
            "json_file": json_path,
            "history_assignments_count": len(history_assignments),
            "files_created": [csv_path, json_path] if csv_path and json_path else [],
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"History assignments export failed: {str(e)}",
            "status": "error"
        }