#!/bin/bash
# setup.sh
# AutoEval-C — One-command project setup script.
# Installs all dependencies and prepares the environment.

set -e  # Exit immediately on any error

echo "════════════════════════════════════════════════════"
echo "   AutoEval-C — Setup Script"
echo "════════════════════════════════════════════════════"

# ── STEP 1 — CHECK PYTHON VERSION ─────────────────────────────────
echo ""
echo "► Checking Python version..."
python_version=$(python3 --version 2>&1)
echo "  Found: $python_version"

required="3.10"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"; then
    echo "  ERROR: Python 3.10 or higher is required."
    exit 1
fi
echo "  Python version OK. ✅"

# ── STEP 2 — CREATE VIRTUAL ENVIRONMENT ───────────────────────────
echo ""
echo "► Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Virtual environment created. ✅"
else
    echo "  Virtual environment already exists — skipping."
fi

# ── STEP 3 — ACTIVATE VIRTUAL ENVIRONMENT ─────────────────────────
echo ""
echo "► Activating virtual environment..."
source venv/bin/activate
echo "  Activated. ✅"

# ── STEP 4 — UPGRADE PIP ──────────────────────────────────────────
echo ""
echo "► Upgrading pip..."
pip install --upgrade pip --quiet
echo "  pip upgraded. ✅"

# ── STEP 5 — INSTALL DEPENDENCIES ─────────────────────────────────
echo ""
echo "► Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo "  Dependencies installed. ✅"

# ── STEP 6 — CREATE .env IF NOT EXISTS ────────────────────────────
echo ""
echo "► Checking .env file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  .env created from .env.example."
    echo "  ACTION REQUIRED: Add your GROQ_API_KEY to .env before running."
else
    echo "  .env already exists — skipping."
fi

# ── STEP 7 — CREATE REQUIRED DIRECTORIES ──────────────────────────
echo ""
echo "► Creating required directories..."
mkdir -p data/inputs/students
mkdir -p data/outputs
mkdir -p data/chroma_storage
mkdir -p data/templates
mkdir -p logs
mkdir -p frontend
echo "  Directories ready. ✅"

# ── STEP 8 — CREATE LOG FILE IF NOT EXISTS ────────────────────────
echo ""
echo "► Checking log file..."
if [ ! -f "logs/engine.log" ]; then
    touch logs/engine.log
    echo "  logs/engine.log created. ✅"
else
    echo "  logs/engine.log already exists — skipping."
fi

# ── DONE ──────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "   AutoEval-C Setup Complete ✅"
echo "════════════════════════════════════════════════════"
echo ""
echo "  Next steps:"
echo "  1. Add your GROQ_API_KEY to .env"
echo "  2. Activate venv:  source venv/bin/activate"
echo "  3. Start server:   python3 -m backend.main"
echo "  4. Open Swagger:   http://127.0.0.1:8000/docs"
echo ""
```
