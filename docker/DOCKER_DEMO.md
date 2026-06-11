# Docker Demo Quick Start

This project includes a Docker Compose setup for the backend API and the Gradio demo UI.

## Services

- backend: FastAPI service on port 8000
- demo: Gradio UI on port 7860

## Prerequisites

- Docker Desktop (or Docker Engine + Compose)
- At least one LLM API key configured in a Docker env file

## Configure Environment Variables

1. Copy `docker/.env.example` to `docker/.env`.
2. Set at least one provider key (for example `GROQ_API_KEY`).
3. Keep `docker/.env` out of git.

## Start Demo

From the repository root:

docker compose --env-file docker/.env -f docker/docker-compose.yml up --build

## Start Demo (Build Directly From GitHub)

Use this when you want Docker to fetch source directly from Git instead of local files:

docker compose --env-file docker/.env -f docker/docker-compose.git.yml up --build

## Portable Mode (Any Empty Folder)

Use this when a user creates a brand new folder and wants to run directly from GitHub with only two files in that folder.

See also: `docker/standalone/README.md`

1. Create a folder (example: `assignment-demo`).
2. Copy these files into it:
  - `docker/standalone/docker-compose.yml`
  - `docker/standalone/.env.example` (rename to `.env`)
3. Edit `.env` and add at least one provider API key.
4. In that folder, run:

docker compose up --build

This works because the compose file builds from GitHub URLs and uses only Docker named volumes.

Optional direct download (PowerShell):

Invoke-WebRequest -Uri "https://raw.githubusercontent.com/felixchess5/Intelligent-Assignment-Grading-System/main/docker/standalone/docker-compose.yml" -OutFile "docker-compose.yml"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/felixchess5/Intelligent-Assignment-Grading-System/main/docker/standalone/.env.example" -OutFile ".env"

Notes:

- This mode builds from `https://github.com/felixchess5/Intelligent-Assignment-Grading-System.git#main`.
- Outputs are stored in Docker named volumes (not local folders).
- You can still pass API keys from your shell environment the same way.

Open:

- UI: http://localhost:7860
- Backend status: http://localhost:8000/status

## Stop Demo

docker compose -f docker/docker-compose.yml down

## Common Commands

Rebuild from scratch:

docker compose --env-file docker/.env -f docker/docker-compose.yml build --no-cache

Start in background:

docker compose --env-file docker/.env -f docker/docker-compose.yml up -d --build

Git-based background start:

docker compose --env-file docker/.env -f docker/docker-compose.git.yml up -d --build

View logs:

docker compose --env-file docker/.env -f docker/docker-compose.yml logs -f

Stop and remove containers, networks, and anonymous volumes:

docker compose --env-file docker/.env -f docker/docker-compose.yml down -v

For Git-based mode:

docker compose --env-file docker/.env -f docker/docker-compose.git.yml down -v

For portable mode (run from the standalone folder):

docker compose down -v

## Notes

- Backend output folders are mounted to the host:
  - output/
  - plagiarism_reports/
- Demo output folder is mounted to the host:
  - demo_output/
- The demo container calls backend via internal URL http://backend:8000.
