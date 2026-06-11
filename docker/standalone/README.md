# Standalone Docker Quickstart (No Clone Required)

Use this mode when running from any folder without cloning the repository.

## Files Needed In Your New Folder

- docker-compose.yml
- .env

You can copy them from this repository's standalone templates:

- docker/standalone/docker-compose.yml
- docker/standalone/.env.example (rename to .env)

## One-Time Setup

1. Create an empty folder.
2. Place `docker-compose.yml` and `.env` in that folder.
3. Edit `.env` and set at least one API key.

## Run

docker compose up --build

## Stop

docker compose down -v

## Optional: Download Files Directly With PowerShell

Replace `<branch>` with `main` (or your desired branch/tag):

Invoke-WebRequest -Uri "https://raw.githubusercontent.com/felixchess5/Intelligent-Assignment-Grading-System/<branch>/docker/standalone/docker-compose.yml" -OutFile "docker-compose.yml"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/felixchess5/Intelligent-Assignment-Grading-System/<branch>/docker/standalone/.env.example" -OutFile ".env"

Then edit `.env` and run `docker compose up --build`.
