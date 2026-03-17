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


def _get_env_stripped(key: str, default: str | None = None) -> str | None:
    """
    Returns an environment variable with surrounding whitespace removed.
    Empty strings are normalized to None unless a default is provided.
    """
    value = os.getenv(key, default)

    if value is None:
        return None

    cleaned = value.strip()
    if cleaned == "":
        return None if default is None else default

    return cleaned


# ── PROVIDER SWITCH ────────────────────────────────────────────────
# "groq"        → Slots 1-7, free tier, 7 independent model buckets
# "cerebras"    → Slot 8, free tier, ~1M tokens/day volume tank
# "sambanova"   → Slot 9, free tier, 20 RPM emergency buffer
# "stepfun"     → Slot 10, free tier, highest SWE-Bench 74.4%
# "openrouter"  → Slot 11, free tier, 50 RPD shared (last resort)
# "fireworks"   → Production, paid, batch API, 500+ students
LLM_PROVIDER = _get_env_stripped("LLM_PROVIDER", "groq")
if LLM_PROVIDER is None:
    LLM_PROVIDER = "groq"
LLM_PROVIDER = LLM_PROVIDER.lower()

# ── API KEYS ───────────────────────────────────────────────────────
GROQ_API_KEY       = _get_env_stripped("GROQ_API_KEY")
CEREBRAS_API_KEY   = _get_env_stripped("CEREBRAS_API_KEY")
SAMBANOVA_API_KEY  = _get_env_stripped("SAMBANOVA_API_KEY")
STEPFUN_API_KEY    = _get_env_stripped("STEPFUN_API_KEY")
OPENROUTER_API_KEY = _get_env_stripped("OPENROUTER_API_KEY")
FIREWORKS_API_KEY  = _get_env_stripped("FIREWORKS_API_KEY")

# ── PROVIDER CONFIGURATIONS ────────────────────────────────────────
# All providers are OpenAI-compatible — same client, different base_url.
# Slots 1-7: Same Groq key, different models (each model = own RPM bucket).
# Slots 8-11: Different provider, change LLM_PROVIDER in .env.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": GROQ_API_KEY,
        "mode": "realtime",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key": CEREBRAS_API_KEY,
        "mode": "realtime",
    },
    "sambanova": {
        "base_url": "https://api.sambanova.ai/v1",
        "api_key": SAMBANOVA_API_KEY,
        "mode": "realtime",
    },
    "stepfun": {
        "base_url": "https://api.stepfun.ai/v1",
        "api_key": STEPFUN_API_KEY,
        "mode": "realtime",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_API_KEY,
        "mode": "realtime",
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key": FIREWORKS_API_KEY,
        "mode": "batch",
    },
}

# ── PER-AGENT MODEL SLOTS ──────────────────────────────────────────
# Each agent has its own model slot — change one line to swap models.
# For Groq rotation: only change these 3 lines, keep LLM_PROVIDER=groq.
# For non-Groq slots: change LLM_PROVIDER AND these 3 lines.
AGENT_MODELS = {
    "agent1_skill_extractor": _get_env_stripped("AGENT1_MODEL", "moonshotai/kimi-k2-instruct"),
    "agent2_evaluator": _get_env_stripped("AGENT2_MODEL", "moonshotai/kimi-k2-instruct"),
    "agent3_feedback_writer": _get_env_stripped("AGENT3_MODEL", "moonshotai/kimi-k2-instruct"),
}

# ── EMBEDDING MODEL ────────────────────────────────────────────────
# Local sentence-transformers model — zero API cost
EMBEDDING_MODEL_NAME = _get_env_stripped("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# ── CHROMADB CONFIGURATION ─────────────────────────────────────────
CHROMA_PERSIST_DIRECTORY = _get_env_stripped("CHROMA_DB_PATH", "data/chroma_storage")
CHROMA_SKILLS_COLLECTION = _get_env_stripped("CHROMA_SKILLS_COLLECTION", "micro_skills")
CHROMA_REFERENCES_COLLECTION = _get_env_stripped("CHROMA_REFERENCES_COLLECTION", "teacher_references")

# ── SYSTEM PATHS ───────────────────────────────────────────────────
DATA_INPUTS_PATH = _get_env_stripped("DATA_INPUTS_PATH", os.path.join("data", "inputs"))
DATA_OUTPUTS_PATH = _get_env_stripped("DATA_OUTPUTS_PATH", os.path.join("data", "outputs"))
LOG_FILE_PATH = _get_env_stripped("LOG_FILE_PATH", os.path.join("logs", "engine.log"))


# ── HELPER FUNCTIONS ───────────────────────────────────────────────
def get_provider_config() -> dict:
    """
    Returns active provider configuration based on LLM_PROVIDER setting.
    Raises clear errors for invalid provider names or missing API keys.
    """
    if LLM_PROVIDER not in PROVIDERS:
        raise ValueError(
            f"Invalid LLM_PROVIDER '{LLM_PROVIDER}'. "
            f"Must be one of: {list(PROVIDERS.keys())}"
        )

    provider_config = PROVIDERS[LLM_PROVIDER]

    if not provider_config["api_key"]:
        raise ValueError(
            f"Missing API key for provider '{LLM_PROVIDER}'. "
            f"Set the correct key in your .env file."
        )

    return provider_config


def get_model(agent_key: str) -> str:
    """
    Returns model name for a specific agent.
    Raises clear errors for invalid agent keys or empty model values.
    """
    if agent_key not in AGENT_MODELS:
        raise ValueError(
            f"Invalid agent key '{agent_key}'. "
            f"Must be one of: {list(AGENT_MODELS.keys())}"
        )

    model_name = AGENT_MODELS[agent_key]

    if not model_name:
        raise ValueError(
            f"Model for agent '{agent_key}' is missing. "
            f"Set it in your environment or keep the default."
        )

    return model_name