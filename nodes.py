import asyncio
import csv
import json
import os
import re
from typing import Dict, List

try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    # Fallback decorator if LangSmith is not available
    def traceable(func=None, **kwargs):
        def decorator(f):
            return f
        return decorator(func) if func else decorator
    LANGSMITH_AVAILABLE = False

from llms import groq_llm
from paths import PLAGIARISM_REPORTS_FOLDER
from prompts import PLAGIARISM_CHECK, GRAMMAR_CHECK, RELEVANCE_CHECK, GRADING_PROMPT, SUMMARY_PROMPT


@traceable(name="grammar_check_node")
def grammar_check_node(state: Dict) -> Dict:
    """Node for checking grammatical errors in assignment content."""
    content = state["content"]
    state["grammar_errors"] = grammar_check_fn(content)
    return state


@traceable(name="plagiarism_check_node")
def plagiarism_check_node(state: Dict) -> Dict:
    """Node for checking plagiarism in assignment content."""
    content = state["content"]
    student_name = state["metadata"]["name"]
    state["plagiarism_file"] = plagiarism_check_fn(content, student_name)
    return state


@traceable(name="source_check_node")
def source_check_node(state: Dict) -> Dict:
    """Node for checking content relevance against source material."""
    content = state["content"]
    source = state["source_text"]
    state["relevance"] = relevance_check(content, source)
    return state


@traceable(name="initial_grading_node")
def initial_grading_node(state: Dict) -> Dict:
    """Node for initial grading of assignment content."""
    content = state["content"]
    source = state["source_text"]
    ragas_input = [{
        "question": "Evaluate this assignment",
        "ground_truth": source,
        "answer": content,
        "context": [source],
        "retrieved_contexts": [source]
    }]
    state["grade"] = grading_node(ragas_input)
    return state


@traceable(name="summary_node")
def summary_node(state: Dict) -> Dict:
    """Node for generating assignment summary."""
    content = state["content"]
    state["summary"] = summarize(content)
    return state


@traceable(name="orchestrator_node")
async def orchestrator_node(state: Dict) -> Dict:
    """Orchestrator node that executes all processing nodes in parallel."""
    print("Starting parallel execution of all nodes...")
    
    # Create async wrappers for sync functions
    async def async_grammar_check():
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: grammar_check_node(state.copy()))
    
    async def async_plagiarism_check():
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: plagiarism_check_node(state.copy()))
    
    async def async_source_check():
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: source_check_node(state.copy()))
    
    async def async_initial_grading():
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: initial_grading_node(state.copy()))
    
    async def async_summary():
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: summary_node(state.copy()))
    
    # Execute all nodes in parallel
    try:
        results = await asyncio.gather(
            async_grammar_check(),
            async_plagiarism_check(),
            async_source_check(),
            async_initial_grading(),
            async_summary(),
            return_exceptions=True
        )
        
        # Merge results back into the state
        for result in results:
            if isinstance(result, dict):
                state.update(result)
            elif isinstance(result, Exception):
                print(f"[ERROR] Node execution failed: {result}")
        
        print("Parallel execution completed.")
        return state
        
    except Exception as e:
        print(f"[ERROR] Orchestrator failed: {e}")
        return state


@traceable(name="grammar_check_fn")
def grammar_check_fn(text: str) -> int:
    """Check grammatical errors in text using LLM."""
    if groq_llm is None:
        print("[ERROR] LLM not available. Cannot check grammar.")
        return -1
        
    print("Checking Grammar...")
    prompt = GRAMMAR_CHECK.format(text=text)
    response = groq_llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response).strip()

    match = re.search(r"\d+", raw)
    if match:
        return int(match.group())
    else:
        print(f"[WARN] Could not parse grammar error count from response: {raw}")
        return -1


@traceable(name="plagiarism_check_fn")
def plagiarism_check_fn(text: str, student_name: str) -> str:
    """Check for plagiarism using LLM and save report."""
    if groq_llm is None:
        return "Error: LLM not available"
        
    print("Checking for plagiarism...")

    prompt = PLAGIARISM_CHECK.replace("{text}", text)

    try:
        response = groq_llm.invoke(prompt)
        result = response.content if hasattr(response, "content") else str(response).strip()

        file_path = os.path.join(PLAGIARISM_REPORTS_FOLDER, f"{student_name}_report.json")
        with open(file_path, "w") as f:
            f.write(result)

        return file_path
    except Exception as e:
        print(f"[ERROR] Groq-based plagiarism check failed for {student_name}: {e}")
        return f"Error generating report: {str(e)}"


@traceable(name="relevance_check")
def relevance_check(text: str, source: str) -> str:
    """Check content relevance against source material."""
    if groq_llm is None:
        return "Error: LLM not available"
        
    print("Checking for relevance...")
    prompt = RELEVANCE_CHECK.format(text=text, source=source)
    response = groq_llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response).strip()


@traceable(name="grading_node")
def grading_node(ragas_input: List[Dict]) -> Dict:
    """Grade assignment using LLM-based evaluation (0-10 scale)."""
    if groq_llm is None:
        return {
            "factuality": 0.0,
            "relevance": 0.0,
            "coherence": 0.0,
            "grammar": 1.0  # Grammar minimum score is 1
        }
        
    print("Grading...")

    item = ragas_input[0]
    answer = item["answer"]
    source = item["ground_truth"]

    prompt = GRADING_PROMPT.format(answer=answer, source=source)

    try:
        response = groq_llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response).strip()

        try:
            scores = json.loads(raw)
        except json.JSONDecodeError:
            matches = re.findall(r'"(\w+)":\s*([0-9.]+)', raw)
            scores = {k: float(v) for k, v in matches}

        return {
            "factuality": round(scores.get("factuality", 0), 2),
            "relevance": round(scores.get("relevance", 0), 2),
            "coherence": round(scores.get("coherence", 0), 2),
            "grammar": round(max(scores.get("grammar", 1), 1), 2)  # Ensure grammar is never below 1
        }

    except Exception as e:
        print(f"[ERROR] Grading failed: {e}")
        print(f"[DEBUG] Raw response: {raw if 'raw' in locals() else 'No response'}")
        return {
            "factuality": 0.0,
            "relevance": 0.0,
            "coherence": 0.0,
            "grammar": 1.0  # Grammar minimum score is 1
        }


@traceable(name="summarize")
def summarize(text: str) -> str:
    """Generate summary of assignment text."""
    if groq_llm is None:
        return "Error: LLM not available for summarization"
        
    print("Generating summary...")
    prompt = SUMMARY_PROMPT.format(text=text)
    response = groq_llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response).strip()


def export_summary(assignments: List[Dict], output_path: str) -> str:
    """Export assignment summaries to CSV file."""
    print("Exporting summary...")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Student Name", "Date of Submission", "Class", "Subject", "Summary",
            "Grammar Errors", "Plagiarism File", "Content Relevance", "Initial Grade"
        ])
        writer.writeheader()
        for a in assignments:
            writer.writerow(a)
    return output_path