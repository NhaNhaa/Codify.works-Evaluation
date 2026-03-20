"""
backend/config/config.py

Handles environment variables and model configurations.
Ensures zero hardcoding of sensitive or environment-specific data.
Supports the verified free‑tier providers from the final .env.
"""

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_LLM_PROVIDER = "groq"
DEFAULT_AGENT_MODEL = "moonshotai/kimi-k2-instruct"
DEFAULT_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHROMA_DB_PATH = os.path.join("data", "chroma_storage")
DEFAULT_CHROMA_SKILLS_COLLECTION = "micro_skills"
DEFAULT_CHROMA_REFERENCES_COLLECTION = "teacher_references"
DEFAULT_DATA_INPUTS_PATH = os.path.join("data", "inputs")
DEFAULT_DATA_OUTPUTS_PATH = os.path.join("data", "outputs")
DEFAULT_LOG_FILE_PATH = os.path.join("logs", "engine.log")

DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8000
DEFAULT_API_RELOAD = True
DEFAULT_API_LOG_LEVEL = "info"

# LLM inference defaults
DEFAULT_LLM_TEMPERATURE = 0.1
DEFAULT_LLM_MAX_TOKENS = 4096
DEFAULT_LLM_RESPONSE_FORMAT = "json_object"

PROVIDER_MODE_REALTIME = "realtime"
PROVIDER_MODE_BATCH = "batch"

AGENT1_MODEL_KEY = "agent1_skill_extractor"
AGENT2_MODEL_KEY = "agent2_evaluator"
AGENT3_MODEL_KEY = "agent3_feedback_writer"

VALID_API_LOG_LEVELS = {"critical", "error", "warning", "info", "debug", "trace"}


def _get_env_stripped(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key, default)
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else (None if default is None else default)


def _get_normalized_path(key: str, default: str) -> str:
    value = _get_env_stripped(key, default)
    return os.path.normpath(value if value is not None else default)


def _get_env_int(key: str, default: int, minimum: int | None = None) -> int:
    value = _get_env_stripped(key)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    if minimum is not None and parsed < minimum:
        return default
    return parsed


def _get_env_float(key: str, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    value = _get_env_stripped(key)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    if minimum is not None and parsed < minimum:
        return default
    if maximum is not None and parsed > maximum:
        return default
    return parsed


def _get_env_bool(key: str, default: bool) -> bool:
    value = _get_env_stripped(key)
    if value is None:
        return default
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    return default


def _get_env_log_level(key: str, default: str) -> str:
    value = _get_env_stripped(key, default)
    if value is None:
        return default
    normalized = value.lower()
    return normalized if normalized in VALID_API_LOG_LEVELS else default


# ── PROVIDER SWITCH ────────────────────────────────────────────────
LLM_PROVIDER = _get_env_stripped("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
if LLM_PROVIDER is None:
    LLM_PROVIDER = DEFAULT_LLM_PROVIDER
LLM_PROVIDER = LLM_PROVIDER.lower()

# ── API KEYS (only those used in the final .env) ──────────────────
GROQ_API_KEY = _get_env_stripped("GROQ_API_KEY")
CEREBRAS_API_KEY = _get_env_stripped("CEREBRAS_API_KEY")
MISTRAL_API_KEY = _get_env_stripped("MISTRAL_API_KEY")
ZHIPU_API_KEY = _get_env_stripped("ZHIPU_API_KEY")
CLOUDFLARE_API_KEY = _get_env_stripped("CLOUDFLARE_API_KEY")

# Optional base URL for Cloudflare (needs account + gateway)
CLOUDFLARE_BASE_URL = _get_env_stripped("CLOUDFLARE_BASE_URL")

# ── PROVIDER CONFIGURATIONS (verified free‑tier providers) ────────
PROVIDERS: dict[str, dict[str, str | None]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": GROQ_API_KEY,
        "mode": PROVIDER_MODE_REALTIME,
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key": CEREBRAS_API_KEY,
        "mode": PROVIDER_MODE_REALTIME,
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "api_key": MISTRAL_API_KEY,
        "mode": PROVIDER_MODE_REALTIME,
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": ZHIPU_API_KEY,
        "mode": PROVIDER_MODE_REALTIME,
    },
    "cloudflare": {
        "base_url": CLOUDFLARE_BASE_URL or "https://gateway.ai.cloudflare.com/v1/your-account/your-gateway/openai",
        "api_key": CLOUDFLARE_API_KEY,
        "mode": PROVIDER_MODE_REALTIME,
    },
}

# ── PER-AGENT MODEL SLOTS ──────────────────────────────────────────
AGENT_MODELS = {
    AGENT1_MODEL_KEY: _get_env_stripped("AGENT1_MODEL", DEFAULT_AGENT_MODEL),
    AGENT2_MODEL_KEY: _get_env_stripped("AGENT2_MODEL", DEFAULT_AGENT_MODEL),
    AGENT3_MODEL_KEY: _get_env_stripped("AGENT3_MODEL", DEFAULT_AGENT_MODEL),
}

# ── EMBEDDING MODEL ────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = _get_env_stripped("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL_NAME)

# ── CHROMADB CONFIGURATION ─────────────────────────────────────────
CHROMA_PERSIST_DIRECTORY = _get_normalized_path("CHROMA_DB_PATH", DEFAULT_CHROMA_DB_PATH)
CHROMA_SKILLS_COLLECTION = _get_env_stripped("CHROMA_SKILLS_COLLECTION", DEFAULT_CHROMA_SKILLS_COLLECTION)
CHROMA_REFERENCES_COLLECTION = _get_env_stripped("CHROMA_REFERENCES_COLLECTION", DEFAULT_CHROMA_REFERENCES_COLLECTION)

# ── SYSTEM PATHS ───────────────────────────────────────────────────
DATA_INPUTS_PATH = _get_normalized_path("DATA_INPUTS_PATH", DEFAULT_DATA_INPUTS_PATH)
DATA_OUTPUTS_PATH = _get_normalized_path("DATA_OUTPUTS_PATH", DEFAULT_DATA_OUTPUTS_PATH)
LOG_FILE_PATH = _get_normalized_path("LOG_FILE_PATH", DEFAULT_LOG_FILE_PATH)

# ── API SERVER CONFIGURATION ───────────────────────────────────────
API_HOST = _get_env_stripped("API_HOST", DEFAULT_API_HOST) or DEFAULT_API_HOST
API_PORT = _get_env_int("API_PORT", DEFAULT_API_PORT, minimum=1)
API_RELOAD = _get_env_bool("API_RELOAD", DEFAULT_API_RELOAD)
API_LOG_LEVEL = _get_env_log_level("API_LOG_LEVEL", DEFAULT_API_LOG_LEVEL)

# ── LLM INFERENCE SETTINGS ─────────────────────────────────────────
LLM_TEMPERATURE = _get_env_float("LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE, minimum=0.0, maximum=2.0)
LLM_MAX_TOKENS = _get_env_int("LLM_MAX_TOKENS", DEFAULT_LLM_MAX_TOKENS, minimum=1)
LLM_RESPONSE_FORMAT = _get_env_stripped("LLM_RESPONSE_FORMAT", DEFAULT_LLM_RESPONSE_FORMAT)

# ── RATE LIMIT HANDLING ────────────────────────────────────────────
LLM_RETRY_DELAY_SECONDS = _get_env_int("LLM_RETRY_DELAY_SECONDS", 20, minimum=1)
LLM_MAX_RETRIES = _get_env_int("LLM_MAX_RETRIES", 3, minimum=1)


# ── HELPER FUNCTIONS ───────────────────────────────────────────────
def get_provider_config() -> dict[str, str]:
    if LLM_PROVIDER not in PROVIDERS:
        raise ValueError(
            f"Invalid LLM_PROVIDER '{LLM_PROVIDER}'. "
            f"Must be one of: {list(PROVIDERS.keys())}"
        )
    provider_config = PROVIDERS[LLM_PROVIDER]
    base_url = provider_config.get("base_url")
    api_key = provider_config.get("api_key")
    mode = provider_config.get("mode")

    if not base_url:
        raise ValueError(f"Missing base_url for provider '{LLM_PROVIDER}'. Check PROVIDERS configuration.")
    if not mode:
        raise ValueError(f"Missing mode for provider '{LLM_PROVIDER}'. Check PROVIDERS configuration.")
    if not api_key:
        raise ValueError(f"Missing API key for provider '{LLM_PROVIDER}'. Set the correct key in .env.")

    return {"base_url": base_url, "api_key": api_key, "mode": mode}


def get_model(agent_key: str) -> str:
    if agent_key not in AGENT_MODELS:
        raise ValueError(f"Invalid agent key '{agent_key}'. Must be one of: {list(AGENT_MODELS.keys())}")
    model_name = AGENT_MODELS[agent_key]
    if not model_name:
        raise ValueError(f"Model for agent '{agent_key}' is missing. Set it in .env.")
    return model_name


def get_all_agent_models() -> dict[str, str]:
    return {agent_key: get_model(agent_key) for agent_key in AGENT_MODELS}


def get_system_paths() -> dict[str, str]:
    return {
        "data_inputs_path": DATA_INPUTS_PATH,
        "data_outputs_path": DATA_OUTPUTS_PATH,
        "log_file_path": LOG_FILE_PATH,
        "chroma_persist_directory": CHROMA_PERSIST_DIRECTORY,
    }


def get_api_server_config() -> dict[str, str | int | bool]:
    return {
        "host": API_HOST,
        "port": API_PORT,
        "reload": API_RELOAD,
        "log_level": API_LOG_LEVEL,
    }


def get_llm_settings() -> dict[str, float | int | str | None]:
    """Returns LLM inference settings loaded from environment."""
    return {
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "response_format": LLM_RESPONSE_FORMAT,
    }


def get_rate_limit_settings() -> dict[str, int]:
    """Returns rate limit retry settings."""
    return {
        "retry_delay": LLM_RETRY_DELAY_SECONDS,
        "max_retries": LLM_MAX_RETRIES,
    }