"""
backend/config/config.py
Handles environment variables and model configurations.
Ensures zero hardcoding of sensitive or environment-specific data.
Supports 11-slot free API rotation strategy for debugging/prototyping
and Fireworks Batch API for production.
"""

import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# ── PROVIDER SWITCH ────────────────────────────────────────────────
# "groq"        → Slots 1-7, free tier, 7 independent model buckets
# "cerebras"    → Slot 8, free tier, ~1M tokens/day volume tank
# "sambanova"   → Slot 9, free tier, 20 RPM emergency buffer
# "stepfun"     → Slot 10, free tier, highest SWE-Bench 74.4%
# "openrouter"  → Slot 11, free tier, 50 RPD shared (last resort)
# "fireworks"   → Production, paid, batch API, 500+ students
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# ── API KEYS ───────────────────────────────────────────────────────
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
CEREBRAS_API_KEY   = os.getenv("CEREBRAS_API_KEY")
SAMBANOVA_API_KEY  = os.getenv("SAMBANOVA_API_KEY")
STEPFUN_API_KEY    = os.getenv("STEPFUN_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
FIREWORKS_API_KEY  = os.getenv("FIREWORKS_API_KEY")

# ── PROVIDER CONFIGURATIONS ────────────────────────────────────────
# All providers are OpenAI-compatible — same client, different base_url.
# Slots 1-7: Same Groq key, different models (each model = own RPM bucket).
# Slots 8-11: Different provider, change LLM_PROVIDER in .env.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key":  GROQ_API_KEY,
        "mode":     "realtime"
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key":  CEREBRAS_API_KEY,
        "mode":     "realtime"
    },
    "sambanova": {
        "base_url": "https://api.sambanova.ai/v1",
        "api_key":  SAMBANOVA_API_KEY,
        "mode":     "realtime"
    },
    "stepfun": {
        "base_url": "https://api.stepfun.ai/v1",
        "api_key":  STEPFUN_API_KEY,
        "mode":     "realtime"
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key":  OPENROUTER_API_KEY,
        "mode":     "realtime"
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key":  FIREWORKS_API_KEY,
        "mode":     "batch"
    }
}

# ── PER-AGENT MODEL SLOTS ──────────────────────────────────────────
# Each agent has its own model slot — change one line to swap models.
# For Groq rotation: only change these 3 lines, keep LLM_PROVIDER=groq.
# For non-Groq slots: change LLM_PROVIDER AND these 3 lines.
AGENT_MODELS = {
    "agent1_skill_extractor": os.getenv("AGENT1_MODEL", "moonshotai/kimi-k2-instruct"),
    "agent2_evaluator":       os.getenv("AGENT2_MODEL", "moonshotai/kimi-k2-instruct"),
    "agent3_feedback_writer": os.getenv("AGENT3_MODEL", "moonshotai/kimi-k2-instruct")
}

# ── EMBEDDING MODEL ────────────────────────────────────────────────
# Local sentence-transformers model — zero API cost
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ── CHROMADB CONFIGURATION ─────────────────────────────────────────
CHROMA_PERSIST_DIRECTORY     = os.getenv("CHROMA_DB_PATH", "data/chroma_storage")
CHROMA_SKILLS_COLLECTION     = "micro_skills"
CHROMA_REFERENCES_COLLECTION = "teacher_references"

# ── SYSTEM PATHS ───────────────────────────────────────────────────
DATA_INPUTS_PATH  = os.path.join("data", "inputs")
DATA_OUTPUTS_PATH = os.path.join("data", "outputs")
LOG_FILE_PATH     = os.path.join("logs", "engine.log")

# ── HELPER FUNCTIONS ───────────────────────────────────────────────
def get_provider_config() -> dict:
    """Returns active provider configuration based on LLM_PROVIDER setting."""
    if LLM_PROVIDER not in PROVIDERS:
        raise ValueError(
            f"Invalid LLM_PROVIDER '{LLM_PROVIDER}'. "
            f"Must be one of: {list(PROVIDERS.keys())}"
        )
    return PROVIDERS[LLM_PROVIDER]

def get_model(agent_key: str) -> str:
    """Returns model name for a specific agent."""
    if agent_key not in AGENT_MODELS:
        raise ValueError(
            f"Invalid agent key '{agent_key}'. "
            f"Must be one of: {list(AGENT_MODELS.keys())}"
        )
    return AGENT_MODELS[agent_key]