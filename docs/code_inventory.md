# Code Inventory and Module Guide

This document lists the Python modules in this repository, explains what they do, the main classes/functions they expose, expected I/O at a high level, notable third-party libraries used (and why), and known limitations.

If you are looking for the agentic workflow node descriptions, see docs/workflow_nodes.md. For the MCP tool surface, see docs/mcp_tools.md.

---

## Top-Level Entrypoints

- launch_gradio.py
  - Purpose: Thin launcher for the Gradio UI. Adds src/ to sys.path and calls gradio_app.main().
  - Key functions: none (delegates to src/gradio_app.py:main).
  - I/O: Console output only.
  - Libraries: os, sys.

- visualize_graph.py, simple_graph_viz.py
  - Purpose: Render workflow graphs to PNGs. visualize_graph introspects the LangGraph JSON; simple_graph_viz uses a curated main path.
  - Key functions: visualize_workflow_graph, print_workflow_structure, get_current_graph_json, visualize_simple_workflow.
  - I/O: Reads workflow structure from code; writes workflow_graph.png and simple_workflow.png.
  - Libraries: matplotlib, networkx.

- run_tests.py
  - Purpose: Local test runner for pytest.
  - I/O: Console.

---

## Web UI (Frontend)

- src/gradio_app.py
  - Purpose: Gradio interface for single/batch processing; talks to the backend over HTTP.
  - Key class: GradioAssignmentGrader (process_single_file_v2, process_multiple_files_v2, download creation).
  - I/O: Reads uploaded files; posts to BACKEND_URL; writes ZIP/JSON to temp/output.
  - Libraries: gradio, httpx, pandas, mimetypes, json, tempfile, zipfile.
  - Notes: Includes JSON viewers for single and batch results; robust port/share handling via env.

---

## Backend (FastAPI)

- src/server/main.py
  - Purpose: FastAPI service exposing /status and /process_file. Extracts text, runs the agentic workflow, returns structured results.
  - Key functions: status, process_file.
  - I/O: Multipart uploads (files) and JSON responses.
  - Libraries: fastapi, starlette (CORS).

---

## Core (Orchestration, LLMs, Paths)

- src/core/assignment_orchestrator.py
  - Purpose: Intelligent routing: classify subject/complexity and dispatch to specialized processors (Math, Spanish, Science, History).
  - Key classes: AssignmentOrchestrator, AssignmentClassification, enums SubjectType, AssignmentComplexity.
  - Key functions: classify_assignment, process_assignment, generate_recommendations.
  - I/O: Input text + metadata; returns classification, specialized processing results, feedback, overall scores.

- src/core/llms.py
  - Purpose: Multi-provider LLM manager with failover, health and circuit breakers; optional security wrappers.
  - Key classes: MultiLLMManager, ProviderHealth, CircuitBreaker, LLMResponse, GeminiWrapper.
  - Key functions: invoke_with_fallback, get_priority_order, get_health_status.
  - I/O: Accepts prompts; returns standardized content; reads config/llm_config.yaml and environment.
  - Libraries: langchain providers, dotenv, pyyaml.

- src/core/paths.py
  - Purpose: Central paths and filenames (assignments, output, summary, per-subject outputs).

- src/core/subject_output_manager.py
  - Purpose: Subject-specific output routing and CSV/JSON writing.
  - Key classes: OutputSubject, SubjectOutput, SubjectOutputManager.
  - Key functions: create_subject_output_manager, determine_subject, extract_specialized_data.

---

## Specialized Processors

- src/processors/math_processor.py
  - Purpose: Analyze math assignments (accuracy, approach, notation, steps) and identify problem type.
  - Key: MathProcessor, MathProblemType; functions grade_math_assignment, identify_problem_type.
  - I/O: Input text; returns dict with scores, problem type, analysis details, feedback.

- src/processors/spanish_processor.py
  - Purpose: Spanish grammar/vocabulary/fluency/culture analysis and grading.
  - Key: SpanishProcessor, SpanishAssignmentType, SpanishAnalysis (with to_dict).
  - I/O: Input text; returns JSON-serializable dict (analysis to_dict), feedback, overall score.
  - Note: Falls back when spaCy model not installed.

- src/processors/science_processor.py
  - Purpose: Science analysis (scientific method, vocab, variables, formulas) and grading.
  - Key: ScienceProcessor + enums.
  - I/O: Returns analysis dicts, grading scores, feedback; can call LLM for feedback.

- src/processors/history_processor.py
  - Purpose: History analysis (period, region focus, dates/figures/sources) and grading.
  - Key: HistoryProcessor + enums.
  - I/O: Returns analysis dicts, grading scores, feedback; can call LLM for feedback.

---

## Support Utilities

- src/support/file_processor.py
  - Purpose: Validate formats/sizes; extract text from PDF/DOCX/MD/TXT/images; collect metadata.
  - Key: FileProcessor, FileRejectionReason.
  - I/O: Reads files; returns success/content/metadata or rejection reasons.
  - Libraries: pdfplumber, PyPDF2, python-docx, mammoth, pdf2image/pillow, chardet, python-magic.

- src/support/ocr_processor.py
  - Purpose: OCR for scanned PDFs/images with preprocessing options.
  - Key: OCRMethod, ImageProcessingMethod, ocr_processor.
  - Libraries: pytesseract, opencv-python, pillow.

- src/support/language_support.py
  - Purpose: Language detection and localized prompt selection.
  - Key: detect_text_language, get_supported_languages, get_localized_prompt.

- src/support/prompts.py
  - Purpose: Prompt templates (GRAMMAR_CHECK, PLAGIARISM_CHECK, RELEVANCE_CHECK, SUMMARY_PROMPT, GRADING_PROMPT).

---

## Agentic Workflow

- src/workflows/agentic_workflow.py
  - Purpose: LangGraph-based stateful pipeline for end-to-end processing.
  - Nodes: initialize, quality_check, subject_classification, specialized_processing, grammar_analysis, plagiarism_detection, relevance_analysis, content_grading, summary_generation, quality_validation, error_recovery, results_aggregation, finalize -> END.
  - Key: build_agentic_workflow, run_agentic_workflow, per-node agents.
  - I/O: State dict per node; returns consolidated JSON result.
  - Libraries: langgraph, optional langsmith.
  - Limitation: State must remain msgpack-serializable for checkpointing.

---

## MCP Server

- src/mcp/mcp_server.py
  - Purpose: Exposes capabilities as MCP tools (grammar, plagiarism, relevance, grading, summarization; file processing; OCR; orchestrator; subject exports; batch).
  - Server: FastMCP("assignment-grader") with many @mcp.tool functions.
  - Representative tools: grammar_check, plagiarism_check, relevance_check, grade_assignment, summarize_assignment, process_assignment_parallel, process_assignment_agentic, process_assignment_from_file, batch_process_files, detect_language, analyze_* and export_* per subject.
  - I/O: JSON inputs/outputs; some tools write reports/CSVs.
  - Libraries: mcp[cli], dotenv, core/support/processors, LangChain providers.

---

## Tests

- tests/**
  - Purpose: Unit tests for orchestrator, file processor, math processor, security, etc.
  - Libraries: pytest, pytest-asyncio, unittest.mock.

---

## Cross-Cutting Notes

- Environments: Use separate venvs for demo UI (Gradio) and backend (FastAPI + LangChain + spaCy). See README for setup scripts.
- Serialization: Convert custom classes to dicts before placing in workflow state (to keep msgpack-compatible).
- Providers and keys: LLM providers configured via config/llm_config.yaml and environment variables.
- Windows console: Prefer ASCII logs or run with python -X utf8 if Unicode is needed.

