"""
Agentic AI Workflow using LangGraph with proper nodes and edges.
This refactors the async running functionality into a full-fledged workflow orchestration system.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from enum import Enum
from typing import Dict, List, Literal, Optional, TypedDict

try:
    from langsmith import traceable

    LANGSMITH_AVAILABLE = True
except ImportError:

    def traceable(func=None, **kwargs):
        def decorator(f):
            return f

        return decorator(func) if func else decorator

    LANGSMITH_AVAILABLE = False

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:  # Make LangGraph optional for demo/local runs
    MemorySaver = None  # type: ignore[assignment]
    END = "__end__"
    # Placeholder so annotations don't break when using postponed evaluation
    class StateGraph:  # type: ignore[no-redef]
        ...

    LANGGRAPH_AVAILABLE = False

from core.assignment_orchestrator import SubjectType, create_assignment_orchestrator
from core.llms import gemini_llm, groq_llm, invoke_with_fallback
from support.language_support import detect_text_language, get_localized_prompt
from support.prompts import (
    GRADING_PROMPT,
    GRAMMAR_CHECK,
    PLAGIARISM_CHECK,
    RELEVANCE_CHECK,
    SUMMARY_PROMPT,
)


class WorkflowState(TypedDict):
    """State definition for the agentic workflow."""

    # Input data
    content: str
    metadata: Dict
    source_text: str

    # Processing flags
    requires_grammar_check: bool
    requires_plagiarism_check: bool
    requires_relevance_check: bool
    requires_grading: bool
    requires_summary: bool

    # Processing results
    grammar_result: Optional[Dict]
    plagiarism_result: Optional[Dict]
    relevance_result: Optional[Dict]
    grading_result: Optional[Dict]
    summary_result: Optional[Dict]

    # Subject-specific processing
    assignment_classification: Optional[Dict]
    specialized_processing_result: Optional[Dict]
    requires_specialized_processing: bool

    # Workflow control
    current_step: str
    completed_steps: List[str]
    errors: List[str]
    retry_count: int
    quality_score: float

    # Final outputs
    final_results: Optional[Dict]


class WorkflowStep(Enum):
    """Enumeration of workflow steps."""

    INITIALIZE = "initialize"
    QUALITY_CHECK = "quality_check"
    SUBJECT_CLASSIFICATION = "subject_classification"
    SPECIALIZED_PROCESSING = "specialized_processing"
    GRAMMAR_ANALYSIS = "grammar_analysis"
    PLAGIARISM_DETECTION = "plagiarism_detection"
    RELEVANCE_ANALYSIS = "relevance_analysis"
    CONTENT_GRADING = "content_grading"
    SUMMARY_GENERATION = "summary_generation"
    QUALITY_VALIDATION = "quality_validation"
    RESULTS_AGGREGATION = "results_aggregation"
    ERROR_RECOVERY = "error_recovery"
    FINALIZE = "finalize"


@traceable(name="workflow_initializer")
def initialize_workflow(state: WorkflowState) -> WorkflowState:
    """Initialize the workflow state and determine processing requirements."""
    print(
        f"🚀 Initializing workflow for student: {state['metadata'].get('name', 'Unknown')}"
    )

    # Set processing requirements based on content analysis
    content_length = len(state["content"])

    state["requires_grammar_check"] = True
    state["requires_plagiarism_check"] = (
        content_length > 100
    )  # Skip plagiarism for very short texts
    state["requires_relevance_check"] = bool(state.get("source_text"))
    state["requires_grading"] = True
    state["requires_summary"] = content_length > 200
    state["requires_specialized_processing"] = True  # Always try specialized processing

    # Initialize workflow control
    state["current_step"] = WorkflowStep.QUALITY_CHECK.value
    state["completed_steps"] = [WorkflowStep.INITIALIZE.value]
    state["errors"] = []
    state["retry_count"] = 0
    state["quality_score"] = 0.0

    # Initialize result containers
    state["grammar_result"] = None
    state["plagiarism_result"] = None
    state["relevance_result"] = None
    state["grading_result"] = None
    state["summary_result"] = None
    state["final_results"] = None

    # Initialize subject-specific containers
    state["assignment_classification"] = None
    state["specialized_processing_result"] = None

    print(
        f"📋 Processing requirements: Grammar={state['requires_grammar_check']}, "
        f"Plagiarism={state['requires_plagiarism_check']}, "
        f"Relevance={state['requires_relevance_check']}, "
        f"Grading={state['requires_grading']}, "
        f"Summary={state['requires_summary']}, "
        f"Specialized={state['requires_specialized_processing']}"
    )

    return state


@traceable(name="quality_check_agent")
def quality_check_agent(state: WorkflowState) -> WorkflowState:
    """Assess content quality and adjust processing requirements."""
    print("🔍 Performing initial quality assessment...")

    content = state["content"]
    content_length = len(content)
    word_count = len(content.split())

    # Basic quality metrics
    has_structure = any(
        marker in content.lower()
        for marker in ["introduction", "conclusion", "paragraph"]
    )
    has_citations = any(marker in content for marker in ["[", "]", "(", ")"])

    # Calculate quality score
    quality_score = 0.0
    if word_count >= 50:
        quality_score += 0.3
    if has_structure:
        quality_score += 0.3
    if has_citations:
        quality_score += 0.2
    if content_length >= 500:
        quality_score += 0.2

    state["quality_score"] = quality_score
    state["current_step"] = WorkflowStep.SUBJECT_CLASSIFICATION.value
    state["completed_steps"].append(WorkflowStep.QUALITY_CHECK.value)

    print(
        f"📊 Quality score: {quality_score:.2f} | Words: {word_count} | Length: {content_length}"
    )

    return state


@traceable(name="subject_classification_agent")
def subject_classification_agent(state: WorkflowState) -> WorkflowState:
    """Classify assignment by subject and determine specialized processing approach."""
    if not state["requires_specialized_processing"]:
        state["current_step"] = WorkflowStep.GRAMMAR_ANALYSIS.value
        return state

    print("🎯 Classifying assignment subject and determining processing approach...")

    try:
        # Initialize orchestrator
        orchestrator = create_assignment_orchestrator()

        # Classify the assignment
        classification = orchestrator.classify_assignment(
            state["content"], state["metadata"]
        )

        state["assignment_classification"] = {
            "subject": classification.subject.value,
            "complexity": classification.complexity.value,
            "specific_type": classification.specific_type,
            "confidence": classification.confidence,
            "language": classification.language,
            "tools_needed": classification.tools_needed,
            "processing_approach": classification.processing_approach,
        }

        print(
            f"   Subject: {classification.subject.value} ({classification.specific_type})"
        )
        print(f"   Complexity: {classification.complexity.value}")
        print(f"   Confidence: {classification.confidence:.2f}")
        print(f"   Processing approach: {classification.processing_approach}")

        # Determine if we should use specialized processing
        if (
            classification.subject in [SubjectType.MATHEMATICS, SubjectType.SPANISH]
            and classification.confidence > 0.3
        ):
            state["current_step"] = WorkflowStep.SPECIALIZED_PROCESSING.value
        else:
            state["current_step"] = WorkflowStep.GRAMMAR_ANALYSIS.value

    except Exception as e:
        print(f"❌ Subject classification failed: {e}")
        state["assignment_classification"] = {
            "subject": "unknown",
            "error": str(e),
            "status": "error",
        }
        state["errors"].append(f"Subject classification: {str(e)}")
        state["current_step"] = WorkflowStep.GRAMMAR_ANALYSIS.value

    state["completed_steps"].append(WorkflowStep.SUBJECT_CLASSIFICATION.value)
    return state


@traceable(name="specialized_processing_agent")
async def specialized_processing_agent(state: WorkflowState) -> WorkflowState:
    """Process assignment using subject-specific specialized processors."""
    print("🔬 Processing with specialized subject processor...")

    try:
        # Initialize orchestrator
        orchestrator = create_assignment_orchestrator()

        # Process with specialized processor
        result = await orchestrator.process_assignment(
            state["content"], state.get("source_text"), state["metadata"]
        )

        # Convert any analysis objects to dicts for serialization
        if "processing_results" in result and hasattr(
            result["processing_results"], "to_dict"
        ):
            result["processing_results"] = result["processing_results"].to_dict()

        state["specialized_processing_result"] = result

        print(f"   Processor used: {result['classification']['subject']}")
        print(f"   Overall score: {result['overall_score']:.2f}")
        print(f"   Specialized feedback items: {len(result['specialized_feedback'])}")

        # Use specialized grading result as primary grading if available
        if (
            result["processing_results"]
            and "overall_score" in result["processing_results"]
        ):
            state["grading_result"] = {
                "overall_score": result["overall_score"],
                "specialized_scores": result["processing_results"],
                "feedback": result["specialized_feedback"],
                "processor_used": result["classification"]["subject"],
                "status": "success_specialized",
            }
            # Skip regular grading since we have specialized results
            state["requires_grading"] = False

    except Exception as e:
        print(f"❌ Specialized processing failed: {e}")
        state["specialized_processing_result"] = {"error": str(e), "status": "error"}
        state["errors"].append(f"Specialized processing: {str(e)}")

    state["current_step"] = WorkflowStep.GRAMMAR_ANALYSIS.value
    state["completed_steps"].append(WorkflowStep.SPECIALIZED_PROCESSING.value)
    return state


@traceable(name="grammar_analysis_agent")
def grammar_analysis_agent(state: WorkflowState) -> WorkflowState:
    """Analyze grammar and language quality."""
    if not state["requires_grammar_check"]:
        state["current_step"] = WorkflowStep.PLAGIARISM_DETECTION.value
        return state

    print("📝 Analyzing grammar and language quality...")

    try:
        # Detect language first
        lang_result = detect_text_language(state["content"])
        detected_language = lang_result.fallback_language

        print(
            f"   Detected language: {detected_language} (confidence: {lang_result.confidence:.2f})"
        )

        # Use localized grammar check prompt
        prompt = get_localized_prompt(
            "grammar_check", detected_language, text=state["content"]
        )
        response = invoke_with_fallback(prompt, groq_llm, gemini_llm)
        raw_response = (
            response.content if hasattr(response, "content") else str(response).strip()
        )

        # Enhanced grammar analysis
        import re

        error_match = re.search(r"\d+", raw_response)
        error_count = int(error_match.group()) if error_match else 0

        state["grammar_result"] = {
            "error_count": error_count,
            "raw_analysis": raw_response,
            "quality_impact": min(error_count * 0.1, 1.0),  # Impact on quality score
            "detected_language": detected_language,
            "language_confidence": lang_result.confidence,
            "status": "success",
        }

        print(f"✅ Grammar analysis complete: {error_count} errors detected")

    except Exception as e:
        print(f"❌ Grammar analysis failed: {e}")
        state["grammar_result"] = {
            "error_count": -1,
            "error": str(e),
            "status": "error",
        }
        state["errors"].append(f"Grammar analysis: {str(e)}")

    state["current_step"] = WorkflowStep.PLAGIARISM_DETECTION.value
    state["completed_steps"].append(WorkflowStep.GRAMMAR_ANALYSIS.value)
    return state


@traceable(name="plagiarism_detection_agent")
def plagiarism_detection_agent(state: WorkflowState) -> WorkflowState:
    """Detect potential plagiarism and originality issues."""
    if not state["requires_plagiarism_check"]:
        state["current_step"] = WorkflowStep.RELEVANCE_ANALYSIS.value
        return state

    print("🕵️ Detecting plagiarism and analyzing originality...")

    try:
        # Use the language detected from grammar analysis if available
        detected_language = state.get("grammar_result", {}).get(
            "detected_language", "en"
        )

        # Use localized plagiarism check prompt
        prompt = get_localized_prompt(
            "plagiarism_check", detected_language, text=state["content"]
        )
        response = invoke_with_fallback(prompt, groq_llm, gemini_llm)
        analysis = (
            response.content if hasattr(response, "content") else str(response).strip()
        )

        # Save detailed report
        import os

        from core.paths import PLAGIARISM_REPORTS_FOLDER

        student_name = state["metadata"]["name"]
        os.makedirs(PLAGIARISM_REPORTS_FOLDER, exist_ok=True)
        report_path = os.path.join(
            PLAGIARISM_REPORTS_FOLDER, f"{student_name}_workflow_report.json"
        )

        with open(report_path, "w") as f:
            import time

            report_data = {
                "student": student_name,
                "analysis": analysis,
                "timestamp": str(time.time()),
                "workflow_version": "agentic_v1",
            }
            json.dump(report_data, f, indent=2)

        state["plagiarism_result"] = {
            "report_file": report_path,
            "analysis": analysis,
            "originality_score": state[
                "quality_score"
            ],  # Could be enhanced with more analysis
            "status": "success",
        }

        print(f"✅ Plagiarism analysis complete, report saved: {report_path}")

    except Exception as e:
        print(f"❌ Plagiarism detection failed: {e}")
        state["plagiarism_result"] = {
            "report_file": None,
            "error": str(e),
            "status": "error",
        }
        state["errors"].append(f"Plagiarism detection: {str(e)}")

    state["current_step"] = WorkflowStep.RELEVANCE_ANALYSIS.value
    state["completed_steps"].append(WorkflowStep.PLAGIARISM_DETECTION.value)
    return state


@traceable(name="relevance_analysis_agent")
def relevance_analysis_agent(state: WorkflowState) -> WorkflowState:
    """Analyze content relevance to source material."""
    if not state["requires_relevance_check"]:
        state["current_step"] = WorkflowStep.CONTENT_GRADING.value
        return state

    print("🎯 Analyzing content relevance to source material...")

    try:
        # Use the language detected from grammar analysis if available
        detected_language = state.get("grammar_result", {}).get(
            "detected_language", "en"
        )

        # Use localized relevance check prompt
        prompt = get_localized_prompt(
            "relevance_check",
            detected_language,
            text=state["content"],
            source=state["source_text"],
        )
        response = invoke_with_fallback(prompt, groq_llm, gemini_llm)
        analysis = (
            response.content if hasattr(response, "content") else str(response).strip()
        )

        # Extract relevance score if possible
        import re

        score_match = re.search(r"(\d+(?:\.\d+)?)/10|(\d+(?:\.\d+)?)%", analysis)
        relevance_score = 0.0
        if score_match:
            if score_match.group(1):  # X/10 format
                relevance_score = float(score_match.group(1)) / 10
            elif score_match.group(2):  # X% format
                relevance_score = float(score_match.group(2)) / 100

        state["relevance_result"] = {
            "analysis": analysis,
            "relevance_score": relevance_score,
            "source_alignment": relevance_score > 0.7,
            "status": "success",
        }

        print(f"✅ Relevance analysis complete: {relevance_score:.2f} alignment score")

    except Exception as e:
        print(f"❌ Relevance analysis failed: {e}")
        state["relevance_result"] = {
            "analysis": None,
            "error": str(e),
            "status": "error",
        }
        state["errors"].append(f"Relevance analysis: {str(e)}")

    state["current_step"] = WorkflowStep.CONTENT_GRADING.value
    state["completed_steps"].append(WorkflowStep.RELEVANCE_ANALYSIS.value)
    return state


@traceable(name="content_grading_agent")
def content_grading_agent(state: WorkflowState) -> WorkflowState:
    """Grade content on multiple criteria."""
    if not state["requires_grading"]:
        state["current_step"] = WorkflowStep.SUMMARY_GENERATION.value
        return state

    print("📊 Grading content on multiple criteria...")

    try:
        # Use the language detected from grammar analysis if available
        detected_language = state.get("grammar_result", {}).get(
            "detected_language", "en"
        )

        # Use localized grading prompt
        prompt = get_localized_prompt(
            "grading_prompt",
            detected_language,
            answer=state["content"],
            source=state.get("source_text", ""),
        )
        response = invoke_with_fallback(prompt, groq_llm, gemini_llm)
        raw_response = (
            response.content if hasattr(response, "content") else str(response).strip()
        )

        # Parse grading results
        try:
            scores = json.loads(raw_response)
        except json.JSONDecodeError:
            # Fallback parsing
            import re

            matches = re.findall(r'"(\w+)":\s*([0-9.]+)', raw_response)
            scores = {k: float(v) for k, v in matches}

        # Ensure all required scores are present
        final_grades = {
            "factuality": round(scores.get("factuality", 0), 2),
            "relevance": round(scores.get("relevance", 0), 2),
            "coherence": round(scores.get("coherence", 0), 2),
            "grammar": round(max(scores.get("grammar", 1), 1), 2),
        }

        # Calculate overall score
        overall_score = sum(final_grades.values()) / len(final_grades)

        state["grading_result"] = {
            "individual_scores": final_grades,
            "overall_score": round(overall_score, 2),
            "raw_response": raw_response,
            "grade_letter": get_letter_grade(overall_score),
            "status": "success",
        }

        print(
            f"✅ Grading complete: Overall {overall_score:.2f}/10 ({get_letter_grade(overall_score)})"
        )

    except Exception as e:
        print(f"❌ Content grading failed: {e}")
        state["grading_result"] = {
            "individual_scores": {
                "factuality": 0,
                "relevance": 0,
                "coherence": 0,
                "grammar": 1,
            },
            "overall_score": 0.25,
            "error": str(e),
            "status": "error",
        }
        state["errors"].append(f"Content grading: {str(e)}")

    state["current_step"] = WorkflowStep.SUMMARY_GENERATION.value
    state["completed_steps"].append(WorkflowStep.CONTENT_GRADING.value)
    return state


@traceable(name="summary_generation_agent")
def summary_generation_agent(state: WorkflowState) -> WorkflowState:
    """Generate comprehensive summary of the assignment."""
    if not state["requires_summary"]:
        state["current_step"] = WorkflowStep.QUALITY_VALIDATION.value
        return state

    print("📄 Generating comprehensive assignment summary...")

    try:
        # Use the language detected from grammar analysis if available
        detected_language = state.get("grammar_result", {}).get(
            "detected_language", "en"
        )

        # Get the subject from assignment classification
        subject = "assignment"  # default fallback
        if (
            "assignment_classification" in state
            and "subject" in state["assignment_classification"]
        ):
            subject = state["assignment_classification"]["subject"]

        # Use localized summary prompt with dynamic subject
        prompt = get_localized_prompt(
            "summary_prompt", detected_language, text=state["content"], subject=subject
        )
        response = invoke_with_fallback(prompt, groq_llm, gemini_llm)
        summary = (
            response.content if hasattr(response, "content") else str(response).strip()
        )

        # Clean up unwanted preambles that LLM might add despite our instructions
        import re

        # Use regex to match and remove various preamble patterns
        preamble_patterns = [
            r"^Here\s+is\s+a\s+2-3\s+sentence\s+summary\s+of\s+the\s+English\s+assignment[^:]*:",
            r"^Here's\s+a\s+2-3\s+sentence\s+summary\s+of\s+the\s+English\s+assignment[^:]*:",
            r"^Here\s+is\s+a\s+summary\s+of\s+the\s+English\s+assignment[^:]*:",
            r"^Here's\s+a\s+summary\s+of\s+the\s+English\s+assignment[^:]*:",
            r"^Here\s+is\s+a\s+2-3\s+sentence\s+summary\s+of\s+the\s+[^:]*assignment[^:]*:",
            r"^Here's\s+a\s+2-3\s+sentence\s+summary\s+of\s+the\s+[^:]*assignment[^:]*:",
            r"^Summary\s+of\s+the\s+English\s+assignment[^:]*:",
        ]

        for pattern in preamble_patterns:
            match = re.search(pattern, summary, re.IGNORECASE)
            if match:
                # Remove the matched preamble and any following newlines/spaces
                summary = summary[match.end() :].lstrip("\n ").strip()
                break

        # Enhanced summary with metadata
        import time

        state["summary_result"] = {
            "summary": summary,
            "word_count": len(state["content"].split()),
            "character_count": len(state["content"]),
            "generated_at": str(time.time()),
            "status": "success",
        }

        print(f"✅ Summary generated: {len(summary)} characters")

    except Exception as e:
        print(f"❌ Summary generation failed: {e}")
        state["summary_result"] = {"summary": None, "error": str(e), "status": "error"}
        state["errors"].append(f"Summary generation: {str(e)}")

    state["current_step"] = WorkflowStep.QUALITY_VALIDATION.value
    state["completed_steps"].append(WorkflowStep.SUMMARY_GENERATION.value)
    return state


@traceable(name="quality_validation_agent")
def quality_validation_agent(state: WorkflowState) -> WorkflowState:
    """Validate the quality of all processing results."""
    print("🔍 Validating processing quality and completeness...")

    validation_score = 0.0
    validation_issues = []

    # Check grammar analysis
    if state["requires_grammar_check"] and state["grammar_result"]:
        if state["grammar_result"]["status"] == "success":
            validation_score += 0.2
        else:
            validation_issues.append("Grammar analysis failed")

    # Check plagiarism detection
    if state["requires_plagiarism_check"] and state["plagiarism_result"]:
        if state["plagiarism_result"]["status"] == "success":
            validation_score += 0.2
        else:
            validation_issues.append("Plagiarism detection failed")

    # Check relevance analysis
    if state["requires_relevance_check"] and state["relevance_result"]:
        if state["relevance_result"]["status"] == "success":
            validation_score += 0.2
        else:
            validation_issues.append("Relevance analysis failed")

    # Check grading
    if state["requires_grading"] and state["grading_result"]:
        if state["grading_result"]["status"] == "success":
            validation_score += 0.2
        else:
            validation_issues.append("Content grading failed")

    # Check summary
    if state["requires_summary"] and state["summary_result"]:
        if state["summary_result"]["status"] == "success":
            validation_score += 0.2
        else:
            validation_issues.append("Summary generation failed")

    # Determine next step based on validation
    if validation_score >= 0.8 and len(validation_issues) == 0:
        state["current_step"] = WorkflowStep.RESULTS_AGGREGATION.value
        print(f"✅ Quality validation passed: {validation_score:.2f}")
    elif state["retry_count"] < 2 and len(validation_issues) > 0:
        state["current_step"] = WorkflowStep.ERROR_RECOVERY.value
        print(f"⚠️ Quality issues detected, attempting recovery: {validation_issues}")
    else:
        state["current_step"] = WorkflowStep.RESULTS_AGGREGATION.value
        print(f"🔧 Proceeding with available results: {validation_score:.2f}")

    state["completed_steps"].append(WorkflowStep.QUALITY_VALIDATION.value)
    return state


@traceable(name="error_recovery_agent")
def error_recovery_agent(state: WorkflowState) -> WorkflowState:
    """Attempt to recover from processing errors."""
    print("🔧 Attempting error recovery...")

    state["retry_count"] += 1

    # Simple recovery strategy: retry failed components
    if state["grammar_result"] and state["grammar_result"]["status"] == "error":
        print("🔄 Retrying grammar analysis...")
        state = grammar_analysis_agent(state)

    if state["plagiarism_result"] and state["plagiarism_result"]["status"] == "error":
        print("🔄 Retrying plagiarism detection...")
        state = plagiarism_detection_agent(state)

    # After recovery, go to results aggregation
    state["current_step"] = WorkflowStep.RESULTS_AGGREGATION.value
    state["completed_steps"].append(WorkflowStep.ERROR_RECOVERY.value)

    return state


@traceable(name="results_aggregation_agent")
def results_aggregation_agent(state: WorkflowState) -> WorkflowState:
    """Aggregate all processing results into final output."""
    print("📋 Aggregating all processing results...")

    # Collect all results
    final_results = {
        "student_name": state["metadata"]["name"],
        "date_of_submission": state["metadata"]["date"],
        "class": state["metadata"]["class"],
        "subject": state["metadata"]["subject"],
        "workflow_version": "agentic_v1",
        "processing_metadata": {
            "completed_steps": state["completed_steps"],
            "retry_count": state["retry_count"],
            "quality_score": state["quality_score"],
            "errors": state["errors"],
        },
    }

    # Add grammar results
    if state["grammar_result"]:
        if state["grammar_result"]["status"] == "success":
            final_results["grammar_errors"] = state["grammar_result"]["error_count"]
        else:
            final_results["grammar_errors"] = "Analysis failed"
    else:
        final_results["grammar_errors"] = "Not analyzed"

    # Add plagiarism results
    if state["plagiarism_result"]:
        if state["plagiarism_result"]["status"] == "success":
            final_results["plagiarism_file"] = state["plagiarism_result"]["report_file"]
        else:
            final_results["plagiarism_file"] = "Analysis failed"
    else:
        final_results["plagiarism_file"] = "Not analyzed"

    # Add relevance results
    if state["relevance_result"]:
        if state["relevance_result"]["status"] == "success":
            final_results["content_relevance"] = state["relevance_result"]["analysis"]
        else:
            final_results["content_relevance"] = "Analysis failed"
    else:
        final_results["content_relevance"] = "Not analyzed"

    # Add grading results
    if state["grading_result"]:
        if state["grading_result"]["status"] == "success":
            final_results["initial_grade"] = state["grading_result"][
                "individual_scores"
            ]
            final_results["overall_score"] = state["grading_result"]["overall_score"]
            final_results["letter_grade"] = state["grading_result"]["grade_letter"]
        else:
            final_results["initial_grade"] = "Grading failed"
    else:
        final_results["initial_grade"] = "Not graded"

    # Add summary results
    if state["summary_result"]:
        if state["summary_result"]["status"] == "success":
            final_results["summary"] = state["summary_result"]["summary"]
        else:
            final_results["summary"] = "Summary generation failed"
    else:
        final_results["summary"] = "No summary generated"

    # Add subject classification results
    if state["assignment_classification"]:
        final_results["assignment_classification"] = state["assignment_classification"]
    else:
        final_results["assignment_classification"] = "Not classified"

    # Add specialized processing results
    if state["specialized_processing_result"]:
        final_results["specialized_processing"] = state["specialized_processing_result"]

        # If we have specialized grading, override general grading
        if (
            state["grading_result"]
            and state["grading_result"].get("status") == "success_specialized"
        ):
            final_results["overall_score"] = state["grading_result"]["overall_score"]
            final_results["specialized_grades"] = state["grading_result"][
                "specialized_scores"
            ]
            final_results["specialized_feedback"] = state["grading_result"]["feedback"]
            final_results["processor_used"] = state["grading_result"]["processor_used"]
    else:
        final_results["specialized_processing"] = "Not processed"

    state["final_results"] = final_results
    state["current_step"] = WorkflowStep.FINALIZE.value
    state["completed_steps"].append(WorkflowStep.RESULTS_AGGREGATION.value)

    print("✅ Results aggregation complete")
    return state


@traceable(name="workflow_finalizer")
def finalize_workflow(state: WorkflowState) -> WorkflowState:
    """Finalize the workflow and prepare outputs."""
    print("🏁 Finalizing workflow execution...")

    state["completed_steps"].append(WorkflowStep.FINALIZE.value)

    # Generate execution summary
    total_steps = len(state["completed_steps"])
    error_count = len(state["errors"])
    success_rate = (total_steps - error_count) / total_steps if total_steps > 0 else 0

    print(f"📊 Workflow Summary:")
    print(f"   Steps completed: {total_steps}")
    print(f"   Errors encountered: {error_count}")
    print(f"   Success rate: {success_rate:.2%}")
    print(f"   Retry attempts: {state['retry_count']}")

    return state


def get_letter_grade(score: float) -> str:
    """Convert numerical score to letter grade."""
    if score >= 9.0:
        return "A+"
    elif score >= 8.5:
        return "A"
    elif score >= 8.0:
        return "A-"
    elif score >= 7.5:
        return "B+"
    elif score >= 7.0:
        return "B"
    elif score >= 6.5:
        return "B-"
    elif score >= 6.0:
        return "C+"
    elif score >= 5.5:
        return "C"
    elif score >= 5.0:
        return "C-"
    elif score >= 4.0:
        return "D"
    else:
        return "F"


def route_workflow(
    state: WorkflowState,
) -> Literal[
    "subject_classification",
    "specialized_processing",
    "grammar_analysis",
    "plagiarism_detection",
    "relevance_analysis",
    "content_grading",
    "summary_generation",
    "quality_validation",
    "error_recovery",
    "results_aggregation",
    "finalize",
    "__end__",
]:
    """Route workflow based on current step."""
    current_step = state["current_step"]

    if current_step == WorkflowStep.SUBJECT_CLASSIFICATION.value:
        return "subject_classification"
    elif current_step == WorkflowStep.SPECIALIZED_PROCESSING.value:
        return "specialized_processing"
    elif current_step == WorkflowStep.GRAMMAR_ANALYSIS.value:
        return "grammar_analysis"
    elif current_step == WorkflowStep.PLAGIARISM_DETECTION.value:
        return "plagiarism_detection"
    elif current_step == WorkflowStep.RELEVANCE_ANALYSIS.value:
        return "relevance_analysis"
    elif current_step == WorkflowStep.CONTENT_GRADING.value:
        return "content_grading"
    elif current_step == WorkflowStep.SUMMARY_GENERATION.value:
        return "summary_generation"
    elif current_step == WorkflowStep.QUALITY_VALIDATION.value:
        return "quality_validation"
    elif current_step == WorkflowStep.ERROR_RECOVERY.value:
        return "error_recovery"
    elif current_step == WorkflowStep.RESULTS_AGGREGATION.value:
        return "results_aggregation"
    elif current_step == WorkflowStep.FINALIZE.value:
        return "finalize"
    else:
        return "__end__"


def build_agentic_workflow() -> StateGraph:
    """Build the comprehensive agentic workflow graph.

    If LangGraph is not installed, return a lightweight sequential fallback
    that executes the same agent functions without graph features.
    """
    print("🏗️ Building agentic AI workflow...")

    if not LANGGRAPH_AVAILABLE:
        print("⚠️ LangGraph not installed; using sequential fallback runner.")

        async def _maybe_call(func, state):
            result = func(state)
            if inspect.isawaitable(result):
                return await result
            return result

        class _SequentialWorkflow:
            async def ainvoke(self, initial_state: WorkflowState, config=None):
                state = initialize_workflow(initial_state)
                # Simple loop guided by route_workflow decisions
                while True:
                    step = route_workflow(state)
                    if step in ("__end__", "finalize"):
                        if step == "finalize":
                            state = finalize_workflow(state)
                        break

                    step_map = {
                        "quality_check": quality_check_agent,
                        "subject_classification": subject_classification_agent,
                        "specialized_processing": specialized_processing_agent,
                        "grammar_analysis": grammar_analysis_agent,
                        "plagiarism_detection": plagiarism_detection_agent,
                        "relevance_analysis": relevance_analysis_agent,
                        "content_grading": content_grading_agent,
                        "summary_generation": summary_generation_agent,
                        "quality_validation": quality_validation_agent,
                        "error_recovery": error_recovery_agent,
                        "results_aggregation": results_aggregation_agent,
                    }

                    func = step_map.get(step)
                    if func is None:
                        # Unknown step; attempt to finalize and exit
                        state = finalize_workflow(state)
                        break
                    state = await _maybe_call(func, state)

                return state

        print("✅ Fallback runner ready.")
        return _SequentialWorkflow()  # type: ignore[return-value]

    # LangGraph path
    from langgraph.graph import StateGraph as _StateGraph  # local alias for clarity

    # Create the state graph
    workflow = _StateGraph(WorkflowState)

    # Add all agent nodes
    workflow.add_node("initialize", initialize_workflow)
    workflow.add_node("quality_check", quality_check_agent)
    workflow.add_node("subject_classification", subject_classification_agent)
    workflow.add_node("specialized_processing", specialized_processing_agent)
    workflow.add_node("grammar_analysis", grammar_analysis_agent)
    workflow.add_node("plagiarism_detection", plagiarism_detection_agent)
    workflow.add_node("relevance_analysis", relevance_analysis_agent)
    workflow.add_node("content_grading", content_grading_agent)
    workflow.add_node("summary_generation", summary_generation_agent)
    workflow.add_node("quality_validation", quality_validation_agent)
    workflow.add_node("error_recovery", error_recovery_agent)
    workflow.add_node("results_aggregation", results_aggregation_agent)
    workflow.add_node("finalize", finalize_workflow)

    # Set entry point
    workflow.set_entry_point("initialize")

    # Add edges
    workflow.add_edge("initialize", "quality_check")
    workflow.add_conditional_edges("quality_check", route_workflow)
    workflow.add_conditional_edges("subject_classification", route_workflow)
    workflow.add_conditional_edges("specialized_processing", route_workflow)
    workflow.add_conditional_edges("grammar_analysis", route_workflow)
    workflow.add_conditional_edges("plagiarism_detection", route_workflow)
    workflow.add_conditional_edges("relevance_analysis", route_workflow)
    workflow.add_conditional_edges("content_grading", route_workflow)
    workflow.add_conditional_edges("summary_generation", route_workflow)
    workflow.add_conditional_edges("quality_validation", route_workflow)
    workflow.add_conditional_edges("error_recovery", route_workflow)
    workflow.add_conditional_edges("results_aggregation", route_workflow)
    workflow.add_edge("finalize", END)

    # Add memory for state persistence
    memory = MemorySaver()

    # Compile the workflow
    compiled_workflow = workflow.compile(checkpointer=memory)

    print("✅ Agentic workflow built successfully!")
    return compiled_workflow


# Backward-compatible alias for tests/legacy imports
def create_workflow(*args, **kwargs) -> StateGraph:
    """Compatibility wrapper returning the compiled workflow.

    Some tests import `create_workflow`; forward to `build_agentic_workflow`.
    Any provided args are ignored for compatibility.
    """
    return build_agentic_workflow()


@traceable(name="run_agentic_workflow")
async def run_agentic_workflow(
    content: str, metadata: Dict, source_text: str = ""
) -> Dict:
    """Run the complete agentic workflow on a single assignment."""
    print(f"🚀 Starting agentic workflow for: {metadata.get('name', 'Unknown')}")

    # Build the workflow
    workflow = build_agentic_workflow()

    # Initialize state
    initial_state = WorkflowState(
        content=content,
        metadata=metadata,
        source_text=source_text,
        requires_grammar_check=False,  # Will be set by initializer
        requires_plagiarism_check=False,
        requires_relevance_check=False,
        requires_grading=False,
        requires_summary=False,
        grammar_result=None,
        plagiarism_result=None,
        relevance_result=None,
        grading_result=None,
        summary_result=None,
        current_step="",
        completed_steps=[],
        errors=[],
        retry_count=0,
        quality_score=0.0,
        final_results=None,
    )

    try:
        # Execute the workflow
        config = {
            "configurable": {
                "thread_id": f"assignment_{metadata.get('name', 'unknown')}"
            }
        }
        final_state = await workflow.ainvoke(initial_state, config=config)

        # Return the final results
        if final_state["final_results"]:
            return final_state["final_results"]
        else:
            # Fallback format
            return {
                "Student Name": metadata["name"],
                "Date of Submission": metadata["date"],
                "Class": metadata["class"],
                "Subject": metadata["subject"],
                "Summary": "Workflow incomplete",
                "Grammar Errors": "N/A",
                "Plagiarism File": "N/A",
                "Content Relevance": "N/A",
                "Initial Grade": "N/A",
            }

    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        return {
            "Student Name": metadata["name"],
            "Date of Submission": metadata["date"],
            "Class": metadata["class"],
            "Subject": metadata["subject"],
            "Summary": f"Workflow error: {str(e)}",
            "Grammar Errors": "N/A",
            "Plagiarism File": "N/A",
            "Content Relevance": "N/A",
            "Initial Grade": "N/A",
        }


if __name__ == "__main__":
    # Test the workflow
    import asyncio

    test_content = """
    Name: Test Student
    Date: 2025-01-15
    Class: 10
    Subject: English

    The Renaissance was a period of great cultural and artistic flourishing in Europe.
    It marked the transition from medieval to modern times, bringing significant changes
    in art, science, and philosophy. This era produced great artists like Leonardo da Vinci
    and Michelangelo, whose works continue to inspire us today.
    """

    test_metadata = {
        "name": "Test Student",
        "date": "2025-01-15",
        "class": "10",
        "subject": "English",
    }

    test_source = "The Renaissance was a cultural movement that began in Italy..."

    async def test_workflow():
        result = await run_agentic_workflow(test_content, test_metadata, test_source)
        print("\n" + "=" * 50)
        print("WORKFLOW TEST RESULTS")
        print("=" * 50)
        for key, value in result.items():
            print(f"{key}: {value}")

    asyncio.run(test_workflow())
