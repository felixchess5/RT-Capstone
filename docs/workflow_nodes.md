# Agentic Workflow Nodes

This document explains every node in the LangGraph-based agentic workflow, what each node does, and how data flows between them.

Workflow implementation: `src/workflows/agentic_workflow.py`

Key graph construction (node names):
- `initialize`
- `quality_check`
- `subject_classification`
- `specialized_processing`
- `grammar_analysis`
- `plagiarism_detection`
- `relevance_analysis`
- `content_grading`
- `summary_generation`
- `quality_validation`
- `error_recovery`
- `results_aggregation`
- `finalize` → `END` (LangGraph terminal)

See also:
- Graph builders: `visualize_graph.py`, `simple_graph_viz.py`
- Quick visual: run `python visualize_graph.py` to generate `workflow_graph.png`

## Shared State (WorkflowState)
The nodes operate on a shared, evolving dictionary-like state with keys including:
- Inputs: `content`, `metadata`, `source_text`
- Required flags: `requires_grammar_check`, `requires_plagiarism_check`, `requires_relevance_check`, `requires_grading`, `requires_summary`
- Results: `grammar_result`, `plagiarism_result`, `relevance_result`, `grading_result`, `summary_result`, `specialized_processing_result`
- Tracking: `current_step`, `completed_steps`, `errors`

All nodes should only place JSON-serializable values into state (for msgpack checkpointing).

---

## initialize
- File: `src/workflows/agentic_workflow.py:97`
- Purpose: Bootstraps the workflow state, normalizes flags, prepares tracking lists.
- Inputs: raw `content`, `metadata`, optional `source_text`.
- Outputs: ensures required keys exist; sets `current_step`.
- On Success: edge to `quality_check`.
- On Error: records error; may route to `error_recovery` depending on logic.

## quality_check
- File: `src/workflows/agentic_workflow.py:147`
- Purpose: Basic content sanity checks (size, emptiness, simple heuristics).
- Inputs: `content`, `metadata`.
- Outputs: updates tracking; may set early warnings.
- Next: `subject_classification`.

## subject_classification
- File: `src/workflows/agentic_workflow.py:185`
- Purpose: Classifies the assignment (subject + complexity) using orchestrator.
- Inputs: `content`, `metadata`.
- Outputs: `assignment_classification` in state.
- Next: `specialized_processing` (preferred) or directly to `grammar_analysis` when specialized processing is disabled or not applicable.

## specialized_processing
- File: `src/workflows/agentic_workflow.py:243` (async)
- Purpose: Runs subject-specific processors (Math, Spanish, Science, History).
- Inputs: `content`, `source_text`, classification-guided context.
- Outputs: `specialized_processing_result`; may set `grading_result` directly when specialized scores exist; can disable regular grading.
- Notes: Ensures results are dict-serializable; complex objects converted via `to_dict()`.
- Next: `grammar_analysis`.

## grammar_analysis
- File: `src/workflows/agentic_workflow.py:294`
- Purpose: Performs grammar/language analysis with localization.
- Inputs: `content`, language detection.
- Outputs: `grammar_result` (counts, scores, issues).
- Skip: If `requires_grammar_check` is false → go to `plagiarism_detection`.
- Next: `plagiarism_detection`.

## plagiarism_detection
- File: `src/workflows/agentic_workflow.py:352`
- Purpose: Detects potential plagiarism patterns and produces a report/score.
- Inputs: `content` (and optionally `source_text`).
- Outputs: `plagiarism_result`.
- Skip: If `requires_plagiarism_check` is false → go to `relevance_analysis`.
- Next: `relevance_analysis`.

## relevance_analysis
- File: `src/workflows/agentic_workflow.py:423`
- Purpose: Measures content relevance/consistency with the source material.
- Inputs: `content`, `source_text`.
- Outputs: `relevance_result` (alignment metrics, highlights).
- Skip: If `requires_relevance_check` is false → go to `content_grading`.
- Next: `content_grading`.

## content_grading
- File: `src/workflows/agentic_workflow.py:484`
- Purpose: Grades the assignment with multiple criteria; can leverage specialized results.
- Inputs: prior analysis results and/or `specialized_processing_result`.
- Outputs: `grading_result` (category scores, overall score).
- Skip: If `requires_grading` is false → go to `summary_generation`.
- Next: `summary_generation`.

## summary_generation
- File: `src/workflows/agentic_workflow.py:564`
- Purpose: Generates a concise summary tailored to detected subject/language.
- Inputs: prior results, `content`, language metadata.
- Outputs: `summary_result` (summary text and key points).
- Skip: If `requires_summary` is false → go to `quality_validation`.
- Next: `quality_validation`.

## quality_validation
- File: `src/workflows/agentic_workflow.py:640`
- Purpose: Validates result completeness/quality; decides whether to proceed or recover.
- Inputs: all accumulated results and `errors`.
- Outputs: updates validation flags, may append warnings.
- Next: `results_aggregation` or `error_recovery` on failure thresholds.

## error_recovery
- File: `src/workflows/agentic_workflow.py:698`
- Purpose: Attempts to recover from prior failures (retries, simplified paths).
- Inputs: `errors`, partial results.
- Outputs: updated `errors`, possibly partial fallbacks.
- Next: back to `results_aggregation` when recovery is done.

## results_aggregation
- File: `src/workflows/agentic_workflow.py:721`
- Purpose: Assembles all intermediate results into a single response schema.
- Inputs: analysis outputs, grading, summary, specialized data, metadata.
- Outputs: consolidated `result` dict used by the UI/backend; updates tracking.
- Next: `finalize`.

## finalize
- File: `src/workflows/agentic_workflow.py:822`
- Purpose: Last actionable step; finalize artifacts and mark completion.
- Inputs: aggregated state.
- Outputs: marks step as completed; prepares final payload.
- Next: `END` (LangGraph terminal node) — engine halts execution.

## END (terminal)
- Import: `from langgraph.graph import END`
- Purpose: Terminal sentinel; not a user-implemented node.
- Behavior: No logic; indicates the graph should stop.

---

## Typical Control Flow
1. initialize → quality_check → subject_classification
2. specialized_processing (optional) → grammar_analysis → plagiarism_detection → relevance_analysis
3. content_grading → summary_generation → quality_validation
4. results_aggregation → finalize → END
5. On validation failures: quality_validation → error_recovery → results_aggregation

## Notes and Best Practices
- Ensure node outputs are JSON-serializable (avoid raw class instances in state; convert via `.to_dict()`).
- Keep `completed_steps`, `current_step`, and `errors` updated to improve UI status and recovery.
- Specialized processors may supply grading directly; set `requires_grading = False` when appropriate.
- Use environment configuration (via `config/llm_config.yaml`) to control provider priorities and timeouts.

