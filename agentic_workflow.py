"""
Agentic AI Workflow using LangGraph with proper nodes and edges.
This refactors the async running functionality into a full-fledged workflow orchestration system.
"""
import asyncio
import json
from typing import Dict, List, Literal, Optional, TypedDict
from enum import Enum

try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    def traceable(func=None, **kwargs):
        def decorator(f):
            return f
        return decorator(func) if func else decorator
    LANGSMITH_AVAILABLE = False

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from llms import groq_llm
from prompts import GRAMMAR_CHECK, PLAGIARISM_CHECK, RELEVANCE_CHECK, GRADING_PROMPT, SUMMARY_PROMPT


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
    print(f"🚀 Initializing workflow for student: {state['metadata'].get('name', 'Unknown')}")

    # Set processing requirements based on content analysis
    content_length = len(state["content"])

    state["requires_grammar_check"] = True
    state["requires_plagiarism_check"] = content_length > 100  # Skip plagiarism for very short texts
    state["requires_relevance_check"] = bool(state.get("source_text"))
    state["requires_grading"] = True
    state["requires_summary"] = content_length > 200

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

    print(f"📋 Processing requirements: Grammar={state['requires_grammar_check']}, "
          f"Plagiarism={state['requires_plagiarism_check']}, "
          f"Relevance={state['requires_relevance_check']}, "
          f"Grading={state['requires_grading']}, "
          f"Summary={state['requires_summary']}")

    return state


@traceable(name="quality_check_agent")
def quality_check_agent(state: WorkflowState) -> WorkflowState:
    """Assess content quality and adjust processing requirements."""
    print("🔍 Performing initial quality assessment...")

    content = state["content"]
    content_length = len(content)
    word_count = len(content.split())

    # Basic quality metrics
    has_structure = any(marker in content.lower() for marker in ['introduction', 'conclusion', 'paragraph'])
    has_citations = any(marker in content for marker in ['[', ']', '(', ')'])

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
    state["current_step"] = WorkflowStep.GRAMMAR_ANALYSIS.value
    state["completed_steps"].append(WorkflowStep.QUALITY_CHECK.value)

    print(f"📊 Quality score: {quality_score:.2f} | Words: {word_count} | Length: {content_length}")

    return state


@traceable(name="grammar_analysis_agent")
def grammar_analysis_agent(state: WorkflowState) -> WorkflowState:
    """Analyze grammar and language quality."""
    if not state["requires_grammar_check"]:
        state["current_step"] = WorkflowStep.PLAGIARISM_DETECTION.value
        return state

    print("📝 Analyzing grammar and language quality...")

    try:
        if groq_llm is None:
            raise Exception("LLM not available for grammar analysis")

        prompt = GRAMMAR_CHECK.format(text=state["content"])
        response = groq_llm.invoke(prompt)
        raw_response = response.content if hasattr(response, "content") else str(response).strip()

        # Enhanced grammar analysis
        import re
        error_match = re.search(r"\d+", raw_response)
        error_count = int(error_match.group()) if error_match else 0

        state["grammar_result"] = {
            "error_count": error_count,
            "raw_analysis": raw_response,
            "quality_impact": min(error_count * 0.1, 1.0),  # Impact on quality score
            "status": "success"
        }

        print(f"✅ Grammar analysis complete: {error_count} errors detected")

    except Exception as e:
        print(f"❌ Grammar analysis failed: {e}")
        state["grammar_result"] = {
            "error_count": -1,
            "error": str(e),
            "status": "error"
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
        if groq_llm is None:
            raise Exception("LLM not available for plagiarism detection")

        prompt = PLAGIARISM_CHECK.replace("{text}", state["content"])
        response = groq_llm.invoke(prompt)
        analysis = response.content if hasattr(response, "content") else str(response).strip()

        # Save detailed report
        from paths import PLAGIARISM_REPORTS_FOLDER
        import os
        student_name = state["metadata"]["name"]
        os.makedirs(PLAGIARISM_REPORTS_FOLDER, exist_ok=True)
        report_path = os.path.join(PLAGIARISM_REPORTS_FOLDER, f"{student_name}_workflow_report.json")

        with open(report_path, "w") as f:
            report_data = {
                "student": student_name,
                "analysis": analysis,
                "timestamp": str(asyncio.get_event_loop().time()),
                "workflow_version": "agentic_v1"
            }
            json.dump(report_data, f, indent=2)

        state["plagiarism_result"] = {
            "report_file": report_path,
            "analysis": analysis,
            "originality_score": state["quality_score"],  # Could be enhanced with more analysis
            "status": "success"
        }

        print(f"✅ Plagiarism analysis complete, report saved: {report_path}")

    except Exception as e:
        print(f"❌ Plagiarism detection failed: {e}")
        state["plagiarism_result"] = {
            "report_file": None,
            "error": str(e),
            "status": "error"
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
        if groq_llm is None:
            raise Exception("LLM not available for relevance analysis")

        prompt = RELEVANCE_CHECK.format(text=state["content"], source=state["source_text"])
        response = groq_llm.invoke(prompt)
        analysis = response.content if hasattr(response, "content") else str(response).strip()

        # Extract relevance score if possible
        import re
        score_match = re.search(r'(\d+(?:\.\d+)?)/10|(\d+(?:\.\d+)?)%', analysis)
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
            "status": "success"
        }

        print(f"✅ Relevance analysis complete: {relevance_score:.2f} alignment score")

    except Exception as e:
        print(f"❌ Relevance analysis failed: {e}")
        state["relevance_result"] = {
            "analysis": None,
            "error": str(e),
            "status": "error"
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
        if groq_llm is None:
            raise Exception("LLM not available for grading")

        prompt = GRADING_PROMPT.format(answer=state["content"], source=state.get("source_text", ""))
        response = groq_llm.invoke(prompt)
        raw_response = response.content if hasattr(response, "content") else str(response).strip()

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
            "grammar": round(max(scores.get("grammar", 1), 1), 2)
        }

        # Calculate overall score
        overall_score = sum(final_grades.values()) / len(final_grades)

        state["grading_result"] = {
            "individual_scores": final_grades,
            "overall_score": round(overall_score, 2),
            "raw_response": raw_response,
            "grade_letter": get_letter_grade(overall_score),
            "status": "success"
        }

        print(f"✅ Grading complete: Overall {overall_score:.2f}/10 ({get_letter_grade(overall_score)})")

    except Exception as e:
        print(f"❌ Content grading failed: {e}")
        state["grading_result"] = {
            "individual_scores": {"factuality": 0, "relevance": 0, "coherence": 0, "grammar": 1},
            "overall_score": 0.25,
            "error": str(e),
            "status": "error"
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
        if groq_llm is None:
            raise Exception("LLM not available for summarization")

        prompt = SUMMARY_PROMPT.format(text=state["content"])
        response = groq_llm.invoke(prompt)
        summary = response.content if hasattr(response, "content") else str(response).strip()

        # Enhanced summary with metadata
        state["summary_result"] = {
            "summary": summary,
            "word_count": len(state["content"].split()),
            "character_count": len(state["content"]),
            "generated_at": str(asyncio.get_event_loop().time()),
            "status": "success"
        }

        print(f"✅ Summary generated: {len(summary)} characters")

    except Exception as e:
        print(f"❌ Summary generation failed: {e}")
        state["summary_result"] = {
            "summary": None,
            "error": str(e),
            "status": "error"
        }
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
            "errors": state["errors"]
        }
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
            final_results["initial_grade"] = state["grading_result"]["individual_scores"]
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


def route_workflow(state: WorkflowState) -> Literal["grammar_analysis", "plagiarism_detection", "relevance_analysis", "content_grading", "summary_generation", "quality_validation", "error_recovery", "results_aggregation", "finalize", "__end__"]:
    """Route workflow based on current step."""
    current_step = state["current_step"]

    if current_step == WorkflowStep.GRAMMAR_ANALYSIS.value:
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
    """Build the comprehensive agentic workflow graph."""
    print("🏗️ Building agentic AI workflow...")

    # Create the state graph
    workflow = StateGraph(WorkflowState)

    # Add all agent nodes
    workflow.add_node("initialize", initialize_workflow)
    workflow.add_node("quality_check", quality_check_agent)
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


@traceable(name="run_agentic_workflow")
async def run_agentic_workflow(content: str, metadata: Dict, source_text: str = "") -> Dict:
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
        final_results=None
    )

    try:
        # Execute the workflow
        config = {"configurable": {"thread_id": f"assignment_{metadata.get('name', 'unknown')}"}}
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
                "Initial Grade": "N/A"
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
            "Initial Grade": "N/A"
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
        "subject": "English"
    }

    test_source = "The Renaissance was a cultural movement that began in Italy..."

    async def test_workflow():
        result = await run_agentic_workflow(test_content, test_metadata, test_source)
        print("\n" + "="*50)
        print("WORKFLOW TEST RESULTS")
        print("="*50)
        for key, value in result.items():
            print(f"{key}: {value}")

    asyncio.run(test_workflow())