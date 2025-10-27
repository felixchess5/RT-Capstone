# MCP Tools Reference

This document describes every MCP tool exposed by `src/mcp/mcp_server.py`, including purpose, parameters, return shapes, and prerequisites. Use this as a quick reference when invoking tools via an MCP client.

Notes
- LLM-backed tools require `GROQ_API_KEY` in the environment.
- File/ocr tools rely on support modules under `src/support/` and respect project limits (size, formats).
- All tools return JSON-serializable dicts with a `status` key (`success`, `error`, or other), plus tool-specific fields.

## Core Analysis Tools

- grammar_check(text: str) -> Dict[str, Any]
  - Purpose: Grammar analysis via LanguageTool (local library if installed).
  - Returns: `{ status, language, matches: [...], counts: {...} }` (shape depends on LanguageTool availability).

- plagiarism_check(text: str, student_name: str) -> Dict[str, Any]
  - Purpose: LLM-based plagiarism heuristics and saves a report under `plagiarism_reports/`.
  - Returns: `{ report_file, analysis, status }`.
  - Requires: `GROQ_API_KEY`.

- relevance_check(text: str, source: str) -> Dict[str, Any]
  - Purpose: LLM-based relevance/consistency of text against source.
  - Returns: `{ relevance_analysis, status }`.
  - Requires: `GROQ_API_KEY`.

- grade_assignment(assignment_text: str, source_text: str) -> Dict[str, Any]
  - Purpose: LLM-based grading on multiple 0–10 criteria (enforces grammar >= 1).
  - Returns: `{ grades: {factuality, relevance, coherence, grammar}, raw_response, status }`.
  - Requires: `GROQ_API_KEY`.

- summarize_assignment(text: str) -> Dict[str, Any]
  - Purpose: LLM-generated concise summary.
  - Returns: `{ summary, status }`.
  - Requires: `GROQ_API_KEY`.

## Parallel / Agentic Wrappers

- process_assignment_parallel(assignment_text: str, source_text: str, student_name: str) -> Dict[str, Any]
  - Purpose: Runs grammar, plagiarism, relevance, grading, and summary concurrently.
  - Returns: `{ grammar, plagiarism, relevance, grading, summary, status }` (aggregated fields).

- process_assignment_agentic(assignment_text: str, source_text: str, student_name: str, metadata: Optional[Dict] = None) -> Dict[str, Any]
  - Purpose: Uses the LangGraph agentic workflow to process the assignment.
  - Returns: Workflow result payload or `{ error, status }` on failure.

## File Processing & Batch

- process_file_content(file_path: str) -> Dict[str, Any]
  - Purpose: Extract text and metadata from a single file using `support.file_processor`.
  - Returns: `{ success, content, metadata, rejection_reason? }`.

- validate_file_format(file_path: str) -> Dict[str, Any]
  - Purpose: Validate file against supported types and size limits.
  - Returns: `{ valid: bool, reason? }`.

- process_assignment_from_file(file_path: str, source_text: str, workflow_type: str = "agentic") -> Dict[str, Any]
  - Purpose: Full file processing pipeline (extract → (agentic|parallel|basic)).
  - Returns: `{ processing_status, workflow_type, file_metadata, student_metadata, results? | content? | error? }`.

- get_supported_file_formats() -> Dict[str, Any]
  - Purpose: Report supported formats, size limits, and rejection reasons.
  - Returns: `{ supported_formats, max_file_size_mb, min_content_length, processing_capabilities, rejection_reasons }`.

- batch_process_files(file_paths: List[str], source_text: str, workflow_type: str = "agentic", include_rejections: bool = True) -> Dict[str, Any]
  - Purpose: Batch wrapper that calls `process_assignment_from_file` safely per file.
  - Returns: `{ processed_files, rejected_files, failed_files, summary }`.

## OCR / Scanned Content

- extract_text_from_scanned_pdf(file_path: str, enhanced: bool = True) -> Dict[str, Any]
  - Purpose: Extract text from scanned PDFs using OCR pipeline.
  - Returns: `{ content, metadata, status }`.

- extract_text_from_image(image_path: str, enhanced: bool = True, ocr_method: str = "tesseract_enhanced", preprocessing: str = "adaptive_threshold") -> Dict[str, Any]
  - Purpose: OCR for images with selectable preprocessing and OCR backend.
  - Returns: `{ content, metadata, status }`.

- check_if_pdf_is_scanned(pdf_path: str) -> Dict[str, Any]
  - Purpose: Heuristic to detect scanned PDFs.
  - Returns: `{ is_scanned: bool, details }`.

- get_ocr_capabilities() -> Dict[str, Any]
  - Purpose: Report which OCR methods/preprocessing options are available.
  - Returns: `{ methods, preprocessing, notes }`.

## Language Utilities

- detect_language(text: str) -> Dict[str, Any]
  - Purpose: Detect language and confidence using `support.language_support`.
  - Returns: `{ language, confidence, details }`.

- get_supported_languages_info() -> Dict[str, Any]
  - Purpose: List languages supported by the system.
  - Returns: `{ languages, default }`.

- grade_assignment_multilingual(assignment_text: str, source_text: str, language_hint: Optional[str] = None) -> Dict[str, Any]
  - Purpose: LLM grading for non-English texts (uses hint if provided).
  - Returns: same shape as `grade_assignment`.

- grammar_check_multilingual(text: str, language_hint: Optional[str] = None) -> Dict[str, Any]
  - Purpose: Grammar check with explicit language override.
  - Returns: same shape as `grammar_check`.

## Subject: Mathematics

- analyze_math_assignment(assignment_text: str) -> Dict[str, Any]
  - Purpose: Math-specific analysis and scoring.
  - Returns: `{ analysis, scores, overall_score, feedback }`.

- solve_equation(equation: str) -> Dict[str, Any]
  - Purpose: Attempt to solve a math equation and provide steps if available.
  - Returns: `{ solution, steps?, status }`.

- identify_math_problem_type(text: str) -> Dict[str, Any]
  - Purpose: Classify math problem type (algebra, calculus, etc.).
  - Returns: `{ problem_type, confidence }`.

## Subject: Spanish

- analyze_spanish_assignment(assignment_text: str, source_text: str = None) -> Dict[str, Any]
  - Purpose: Full Spanish analysis (grammar, vocabulary, fluency, culture).
  - Returns: `{ analysis: {...}, overall_score, feedback }`.

- check_spanish_grammar(text: str) -> Dict[str, Any]
  - Purpose: Spanish grammar rules and error detection.
  - Returns: `{ errors, counts, status }`.

- analyze_spanish_vocabulary(text: str) -> Dict[str, Any]
  - Purpose: Vocabulary level estimation and diversity metrics.
  - Returns: `{ level, diversity, terms }`.

## Orchestrator / Intelligent Routing

- classify_assignment_intelligent(assignment_text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]
  - Purpose: Use orchestrator to classify subject, complexity, tools, and approach.
  - Returns: `{ classification: {...}, confidence }`.

- process_assignment_intelligent(assignment_text: str, source_text: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]
  - Purpose: Orchestrated processing with subject-specific routing.
  - Returns: `{ results, classification, specialized_feedback, overall_score }`.

- get_available_subject_processors() -> Dict[str, Any]
  - Purpose: Report which subject processors are available.
  - Returns: `{ subjects: [...], modules }`.

- export_subject_specific_results(assignments_data: List[Dict[str, Any]], output_folder: str = "./output") -> Dict[str, Any]
  - Purpose: Write subject-organized CSV/JSON outputs using `SubjectOutputManager`.
  - Returns: `{ written_files: [...], counts, status }`.

- export_math_assignments(assignments_data: List[Dict[str, Any]], output_folder: str = "./output") -> Dict[str, Any]
- export_spanish_assignments(assignments_data: List[Dict[str, Any]], output_folder: str = "./output") -> Dict[str, Any]
- export_english_assignments(assignments_data: List[Dict[str, Any]], output_folder: str = "./output") -> Dict[str, Any]
  - Purpose: Subject-specific export helpers.
  - Returns: `{ written_files, status }`.

## Subject: Science

- analyze_science_assignment(assignment_text: str) -> Dict[str, Any]
- grade_science_assignment(assignment_text: str, source_text: str = None) -> Dict[str, Any]
- identify_science_subject(assignment_text: str) -> Dict[str, Any]
- extract_scientific_formulas(assignment_text: str) -> Dict[str, Any]
- check_scientific_method(assignment_text: str) -> Dict[str, Any]
- export_science_assignments(assignments_data: List[Dict[str, Any]], output_folder: str = "./output") -> Dict[str, Any]
  - Purpose: Science-focused analyses, grading, extraction, and export.
  - Returns: Shapes similar to other subject processors.

## Subject: History

- analyze_history_assignment(assignment_text: str) -> Dict[str, Any]
- grade_history_assignment(assignment_text: str, source_text: str = None) -> Dict[str, Any]
- identify_historical_period(assignment_text: str) -> Dict[str, Any]
- extract_historical_elements(assignment_text: str) -> Dict[str, Any]
- check_historical_accuracy(assignment_text: str) -> Dict[str, Any]
- export_history_assignments(assignments_data: List[Dict[str, Any]], output_folder: str = "./output") -> Dict[str, Any]
  - Purpose: History-focused analyses, grading, extraction, and export.

## Usage Tips

- Authentication
  - Set `GROQ_API_KEY` for LLM-backed tools.
  - Some tools gracefully degrade (return `error` in response) when the LLM is not available.

- File limits
  - See `get_supported_file_formats()` for max size and accepted types.
  - OCR methods and preprocessing are reported by `get_ocr_capabilities()`.

- Serialization
  - All returned values are JSON-serializable; tool internals convert custom objects using `.to_dict()` where needed.

- Choosing workflows
  - `process_assignment_parallel`: fastest aggregate of core tools.
  - `process_assignment_agentic`: most comprehensive results via LangGraph.
  - In batches, prefer `batch_process_files()` with `workflow_type="agentic"` (default).

