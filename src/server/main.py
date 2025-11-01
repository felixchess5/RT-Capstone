"""
FastAPI backend service exposing core processing endpoints.

This service runs with the core requirements (spaCy, processors, LLMs) and
is intended to be called by the Gradio demo UI running in a separate env.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.llms import llm_manager
from support.file_processor import FileProcessor
from workflows.agentic_workflow import run_agentic_workflow

app = FastAPI(title="Intelligent-Assignment-Grading-System Backend", version="1.0.0")

# CORS settings: allow UI origin via env BACKEND_CORS_ORIGINS (comma-separated)
origins = [o.strip() for o in os.getenv("BACKEND_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file_processor = FileProcessor()


@app.get("/status")
def status() -> Dict[str, Any]:
    """Return basic system status for UI."""
    info: Dict[str, Any] = {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}
    try:
        if llm_manager:
            info["providers"] = list(llm_manager.providers.keys())
    except Exception:
        info["providers"] = []
    return info


@app.post("/process_file")
async def process_file(
    file: UploadFile = File(...),
    requirements: Optional[str] = Form(None),
) -> JSONResponse:
    """Process an uploaded file and return analysis results.

    requirements: optional JSON-encoded dict of processing toggles (unused here but accepted).
    """
    try:
        # Read file content to a temp path as our processor expects a path
        import shutil
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "")[1] or ".bin") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Extract text
        fp_result = file_processor.extract_text_content(tmp_path)
        if isinstance(fp_result, str):
            extracted_text = fp_result
        else:
            if not fp_result.success:
                return JSONResponse(status_code=400, content={"error": fp_result.error or "file processing failed"})
            extracted_text = fp_result.content

        if not extracted_text.strip():
            return JSONResponse(status_code=400, content={"error": "empty file content"})

        # Metadata heuristic
        metadata = {
            "name": os.path.splitext(os.path.basename(file.filename or "assignment"))[0],
            "date": datetime.now().strftime("%Y-%m-%d"),
            "class": "Unknown",
            "subject": "General",
        }
        try:
            head = extracted_text.splitlines()[:10]
            for line in head:
                low = line.lower()
                if "name:" in low:
                    metadata["name"] = line.split(":", 1)[1].strip()
                elif "date:" in low:
                    metadata["date"] = line.split(":", 1)[1].strip()
                elif "class:" in low:
                    metadata["class"] = line.split(":", 1)[1].strip()
                elif "subject:" in low:
                    metadata["subject"] = line.split(":", 1)[1].strip()
        except Exception:
            pass

        # Parse requirements if provided (reserved for future flags)
        req_dict: Dict[str, Any] = {}
        if requirements:
            try:
                parsed = json.loads(requirements)
                if isinstance(parsed, dict):
                    req_dict = parsed
                else:
                    # If client sent a bare true/false or list/string, ignore
                    req_dict = {}
            except Exception:
                req_dict = {}

        # Run agentic workflow with requested feature toggles
        result = await run_agentic_workflow(extracted_text, metadata, "", req_dict)

        if isinstance(result, dict) and "error" in result:
            return JSONResponse(status_code=500, content=result)

        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
