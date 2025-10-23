# RT-Capstone — Intelligent Assignment Grading System

Subtitle: Agentic AI workflow, subject‑specific processors, MCP tools, and Gradio UI

---

## Problem & Goals

- Automate grading across Math, Spanish, Science, History, and English
- Provide grammar, plagiarism, relevance checks + concise summaries
- Produce subject‑specific CSV/JSON outputs for downstream analysis
- Offer interactive UI (Gradio), CLI, and MCP tool server

---

## Capabilities At A Glance

- Agentic workflow (LangGraph) with 11+ nodes and error recovery
- Subject classification + specialized processors (Math, Spanish, Science, History)
- Multi‑format input: TXT, PDF, DOCX/DOC, MD; OCR for images/scans
- Multi‑LLM with fallback; optional LangSmith tracing
- Exports: unified summary.csv + per‑subject CSV/JSON

---

## Architecture Overview

```
           +-------------------+         +------------------------------+
           |   Gradio UI       |         |     CLI / Auto / Demo        |
           |  (launch_gradio)  |         |   (src/main_agentic.py)      |
           +----------+--------+         +---------------+--------------+
                      |                                  |
                      v                                  v
                 +----+----------------------------------+----+
                 |     Agentic Workflow (LangGraph)           |
                 |  (src/workflows/agentic_workflow.py)       |
                 +----+--------------+--------------+---------+
                      |              |              |
                      v              v              v
             +--------+--+   +------+-----+   +-----+---------+
             | Assignment |   | Grammar     |   | Plagiarism   |
             | Orchestrator|  | /Relevance  |   | /Summary     |
             | (core/...)  |  |  Agents     |   |   Agents     |
             +------+-----+   +------------+   +---------------+
                    |
     +--------------+-----------------------------+
     v                                            v
 +---+--------+  +----------------+  +------------+---------+
 | Math Proc  |  | Spanish Proc   |  | Science/History Procs|
 | processors/|  | processors/... |  | processors/...        |
 +------------+  +----------------+  +-----------------------+
     |
     v
 +---+----------------------------------+
 | Subject Output Manager (core/...)     |
 | -> output/*.csv + *.json              |
 +---------------------------------------+
```

---

## Agentic Flow (High Level)

1) Initialize & quality check
2) Subject classification (Orchestrator)
3) Specialized processing (if applicable)
4) Grammar → Plagiarism → Relevance → Grading → Summary
5) Quality validation → Error recovery (if needed)
6) Result aggregation → Finalize

```
[Init] -> [Quality] -> [Classify] -> [Specialized?] -> [Grammar]
                                       | no         -> [Plagiarism] -> [Relevance]
                                       | yes        -> [Grading] -> [Summary]
                                          \
                                           -> [Validate] -> [Recover?] -> [Aggregate] -> [Finalize]
```

---

## Specialized Processors

- Mathematics: problem type detection, sympy‑backed analysis, notation/steps scoring
- Spanish: spaCy‑powered language insights, grammar rules, cultural signals
- Science: scientific method, formulas, variables, safety considerations
- History: period/region, chronology, sources and context, argument evaluation

---

## Inputs & Extraction

- Multi‑format: .txt, .pdf, .docx/.doc, .md (+ images via OCR)
- Robust file detection + rejection reasons (size, corruption, empty)
- Metadata extraction (Name/Date/Class/Subject headers)

---

## Outputs & Exports

- Unified: `output/summary.csv` (enhanced fields, status/rejection tracking)
- Subject CSV/JSON:
  - `output/math_assignments.csv`, `output/math_assignments.json`
  - `output/spanish_assignments.csv`, `output/spanish_assignments.json`
  - `output/science_assignments.csv`, `output/science_assignments.json`
  - `output/history_assignments.csv`, `output/history_assignments.json`
  - `output/english_assignments.csv`, `output/english_assignments.json`
- Plagiarism reports: `plagiarism_reports/{student}_workflow_report.json`

---

## Run Modes (For Demo)

- UI: `python launch_gradio.py` → http://localhost:7860
- Agentic batch: `python src/main_agentic.py agentic`
- Auto (best available): `python src/main_agentic.py`
- Graph demo: `python src/main_agentic.py demo` → `graph.png`
- MCP server: `python src/main_agentic.py mcp`

---

## Security & Reliability (Brief)

- Secure LLM wrapper: prompt isolation, response filtering
- Input validation, sanitization, rate limiting, threat logging
- Multi‑LLM fallback + optional tracing (LangSmith)

---

## Links & Pointers

- README quickstart and features
- Key files: workflows, orchestrator, processors, support utilities, UI, MCP
- Outputs: `output/` and `plagiarism_reports/`

