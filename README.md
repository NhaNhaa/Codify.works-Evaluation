# AutoEval-C / Codify.works

AI-powered C code evaluation system that replaces time-consuming manual grading.
Built with FastAPI, ChromaDB, and constrained AI agents.

No scoring — qualitative feedback only.

---

## What It Does

A teacher uploads 4 files for a C programming assignment. The system:
1. Extracts or generates micro skills from the assignment
2. Evaluates each student's code against those skills
3. Produces detailed PASS/FAIL feedback per skill with line-level references

```
Teacher uploads files → AI extracts skills → Students submit code → AI evaluates → Feedback report
```

---

## Tech Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| API Framework    | FastAPI + Swagger (Python)          |
| AI Models        | Groq / Cerebras / SambaNova (free)  |
| Vector Database  | ChromaDB (local, persistent)        |
| Embeddings       | sentence-transformers (local, free) |
| Server           | Uvicorn                             |

---

## Prerequisites

- **Python 3.10+**
- **Docker Desktop** (for containerized deployment)
- **Git**
- **Free API key** from at least one provider:
  - [Groq](https://console.groq.com) (recommended — 7 model slots)
  - [Cerebras](https://cloud.cerebras.ai)
  - [SambaNova](https://cloud.sambanova.ai)
  - [OpenRouter](https://openrouter.ai)

---

## Quick Start (Local Development)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Codify.works.git
cd Codify.works
```

### 2. Create virtual environment

```bash
python -m venv venv
```

**Activate it:**

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your API key(s):

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
AGENT1_MODEL=moonshotai/kimi-k2-instruct
AGENT2_MODEL=moonshotai/kimi-k2-instruct
AGENT3_MODEL=moonshotai/kimi-k2-instruct
CHROMA_DB_PATH=data/chroma_storage
```

> **Minimum requirement:** Only `GROQ_API_KEY` is needed to start.
> Other provider keys are optional backup slots.

### 5. Start the server

```bash
python -m backend.main
```

### 6. Open Swagger UI

Go to: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

You should see all 11 API endpoints.

---

## Testing the Full Pipeline

### Using Swagger UI (Recommended)

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) and follow these steps:

**Step 1 — Upload assignment files:**
```
POST /upload
  - assignment_id: lab_01
  - instructions: Upload/instructions.md
  - starter_code: Upload/starter_code.c
  - teacher_code: Upload/teacher_correction_code.c
```

**Step 2 — Extract micro skills:**
```
POST /extract-skills
  Body: { "assignment_id": "lab_01", "force_regenerate": true }
```
⏱️ Takes 15-30 seconds (AI processing)

**Step 3 — Verify skills were extracted:**
```
GET /skills/lab_01
```
Should return 3-4 skills with ranks and weights summing to 10.

**Step 4 — Upload student submission:**
```
POST /upload-student
  - assignment_id: lab_01
  - student_id: student_01
  - student_code: Upload/student_01.c
```

**Step 5 — Evaluate student:**
```
POST /evaluate
  Body: { "assignment_id": "lab_01", "student_id": "student_01" }
```
⏱️ Takes 30-90 seconds (AI evaluation + verification + feedback)

**Step 6 — View results:**
```
GET /results/student_01
```
Returns JSON report + Markdown feedback.

### Using curl

```bash
# Step 1: Upload files
curl -X POST http://localhost:8000/upload \
  -F "assignment_id=lab_01" \
  -F "instructions=@Upload/instructions.md" \
  -F "starter_code=@Upload/starter_code.c" \
  -F "teacher_code=@Upload/teacher_correction_code.c"

# Step 2: Extract skills
curl -X POST http://localhost:8000/extract-skills \
  -H "Content-Type: application/json" \
  -d '{"assignment_id": "lab_01", "force_regenerate": true}'

# Step 3: Upload student
curl -X POST http://localhost:8000/upload-student \
  -F "assignment_id=lab_01" \
  -F "student_id=student_01" \
  -F "student_code=@Upload/student_01.c"

# Step 4: Evaluate
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"assignment_id": "lab_01", "student_id": "student_01"}'

# Step 5: Get results
curl http://localhost:8000/results/student_01
```

### Cleanup

```bash
# Delete student results only
curl -X DELETE http://localhost:8000/results/student_01

# Delete skills (to re-extract)
curl -X DELETE http://localhost:8000/skills/lab_01

# Delete everything for an assignment
curl -X DELETE http://localhost:8000/assignment/lab_01
```

---

## Docker Deployment

### Build the container

```bash
docker build -f docker/Dockerfile -t codify-evaluation-c .
```

### Run the container

```bash
docker run -p 8000:8000 --env-file .env codify-evaluation
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) to verify.

---

## Integration with Codify Backend (NestJS)

This service is designed to run alongside the Codify Backend as a Docker microservice.

### Folder Layout

```
ParentFolder/
├── Codify-Backend/          ← NestJS backend
│   └── docker-compose.yaml  ← includes codify_evaluation service
├── Codify.works/            ← This repository
│   ├── docker/
│   │   └── Dockerfile
│   └── .env
```

### Running Both Together

```bash
cd Codify-Backend
docker-compose up --build
```

| Service            | URL                          |
|--------------------|------------------------------|
| NestJS Backend     | http://localhost:4000/api     |
| AutoEval-C Swagger | http://localhost:8000/docs    |
| AutoEval-C Health  | http://localhost:8000/health  |

The NestJS backend calls AutoEval-C internally via `http://codify_evaluation:8000`.
No external port needed in production — remove `ports: - "8000:8000"` from docker-compose.

See [AUTOEVAL_INTEGRATION.md](AUTOEVAL_INTEGRATION.md) for the full API contract.

---

## API Endpoints (11 Total)

| Method | Endpoint                         | Description                        | Auth |
|--------|----------------------------------|------------------------------------|------|
| POST   | /upload                          | Upload 3 assignment files          | No   |
| POST   | /upload-student                  | Upload student .c file             | No   |
| POST   | /extract-skills                  | Extract micro skills (Phase 1)     | No   |
| POST   | /evaluate                        | Evaluate student (Phase 2+3)       | No   |
| GET    | /results/{student_id}            | Get feedback report                | No   |
| GET    | /skills/{assignment_id}          | View stored skills                 | No   |
| GET    | /health                          | Health check                       | No   |
| DELETE | /results/{student_id}            | Delete output files                | No   |
| DELETE | /student/{student_id}            | Delete student + outputs           | No   |
| DELETE | /skills/{assignment_id}          | Delete skills from ChromaDB        | No   |
| DELETE | /assignment/{assignment_id}      | Delete everything                  | No   |

---

## Expected Results

### Test Case: Array Left Shift (with micro skills in instructions.md)

| Skill                                    | Expected | System Output | Match |
|------------------------------------------|----------|---------------|-------|
| Shift elements using arr[i-1]            | PASS     | PASS          | ✅    |
| Use temporary variable                   | FAIL     | FAIL          | ✅    |
| Access array elements using arr[i]       | PASS     | PASS          | ✅    |
| Use scanf to read 5 integers             | PASS     | PASS          | ✅    |

4/4 verdicts match teacher expectations. ✅

---

## Free API Rotation Strategy

11 free API slots for unlimited debugging. No credit card required.

| Slot | Provider   | Model                              | Limit            |
|------|------------|------------------------------------|------------------|
| 1    | Groq       | moonshotai/kimi-k2-instruct        | 60 RPM           |
| 2    | Groq       | qwen/qwen3-32b                     | 60 RPM           |
| 3    | Groq       | llama-3.3-70b-versatile            | 30 RPM           |
| 4    | Groq       | openai/gpt-oss-120b               | 30 RPM           |
| 5    | Groq       | openai/gpt-oss-20b                | 30 RPM           |
| 6    | Groq       | llama-4-scout-17b-16e-instruct     | 30 RPM           |
| 7    | Groq       | llama-3.1-8b-instant              | 30 RPM / 14.4K RPD |
| 8    | Cerebras   | qwen-3-235b-a22b-instruct-2507    | ~1M tokens/day   |
| 9    | SambaNova  | Meta-Llama-3.3-70B-Instruct       | 20 RPM           |
| 10   | StepFun    | step-3.5-flash                     | Own free pool    |
| 11   | OpenRouter | arcee-ai/trinity-large-preview:free | 50 RPD shared   |

**To switch models:** Change `AGENT1_MODEL`, `AGENT2_MODEL`, `AGENT3_MODEL` in `.env`.
Groq slots 1-7 use the same API key — only the model name changes.

### Test all slots

```bash
python -m backend.tests.test_llm_quota
```

---

## Project Structure

```
Codify.works
├─ .dockerignore
├─ backend
│  ├─ agents
│  │  ├─ agent1_extractor.py
│  │  ├─ agent1_validators
│  │  │  ├─ dedup_ranking.py
│  │  │  ├─ validation.py
│  │  │  └─ __init__.py
│  │  ├─ agent2_evaluator.py
│  │  ├─ agent3_feedback.py
│  │  └─ __init__.py
│  ├─ api.py
│  ├─ config
│  │  ├─ config.py
│  │  └─ constants.py
│  ├─ main.py
│  ├─ rag
│  │  ├─ chroma_client.py
│  │  ├─ embedder.py
│  │  ├─ rag_pipeline.py
│  │  └─ __init__.py
│  ├─ tests
│  │  ├─ test_01_constants.py
│  │  ├─ test_02_config.py
│  │  ├─ test_03_logger.py
│  │  ├─ test_04_security.py
│  │  ├─ test_05_formatter.py
│  │  ├─ test_06_embedder.py
│  │  ├─ test_07_chroma.py
│  │  ├─ test_08_rag_pipeline.py
│  │  ├─ test_09_agent1_extractor.py
│  │  ├─ test_10_agent2.py
│  │  ├─ test_11_agent3.py
│  │  ├─ test_12_api.py
│  │  ├─ test_13_skill_parser.py
│  │  ├─ test_14_agent1_validator.py
│  │  ├─ test_15_llm_client.py
│  │  ├─ test_16_main.py
│  │  ├─ test_live_llm_quota.py
│  │  ├─ test_llm_full_run_usage.py
│  │  └─ test_llm_quota.py
│  ├─ utils
│  │  ├─ formatter.py
│  │  ├─ llm_client.py
│  │  ├─ logger.py
│  │  ├─ security.py
│  │  └─ skill_parser.py
│  └─ __init__.py
├─ data
│  └─ inputs
│     └─ lab_01
│        ├─ instructions.md
│        ├─ starter_code.c
│        ├─ students
│        │  └─ student_01.c
│        └─ teacher_correction_code.c
├─ docker
│  └─ Dockerfile
├─ frontend
├─ README.md
├─ requirements.txt
├─ setup.sh
└─ Upload
   ├─ instructions.md
   ├─ starter_code.c
   ├─ student_01.c
   └─ teacher_correction_code.c

```

---

## Run Tests

```bash
# Run all unit tests
python -m pytest backend/tests/ -v

# Run single test file
python -m pytest backend/tests/test_12_api.py -v

# Check API quota across all 11 slots
python -m backend.tests.test_llm_quota
```

---

## Key Design Decisions

- **No scoring** — qualitative feedback only (teacher's request)
- **No teacher escalation** — system is fully autonomous
- **Agents constrained to RAG** — cannot hallucinate criteria
- **Recommended fix always from teacher code** — zero hallucination
- **Pure Python ranking** — deterministic, no LLM involvement
- **Weight sum always = 10** — validated before ChromaDB write

---

## License

CADT Capstone Project — Cambodia Academy of Digital Technology