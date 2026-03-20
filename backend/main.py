"""
backend/main.py
Entry point for AutoEval-C.
Starts the FastAPI application via Uvicorn.
"""

import uvicorn

from backend.config.config import (
    API_HOST,
    API_LOG_LEVEL,
    API_PORT,
    API_RELOAD,
)
from backend.utils.logger import engine_logger


def main() -> None:
    """
    Starts the AutoEval-C FastAPI server.
    """
    engine_logger.info("MAIN: Starting AutoEval-C server...")
    engine_logger.info(
        f"MAIN: Swagger UI available at http://127.0.0.1:{API_PORT}/docs"
    )

    uvicorn.run(
        "backend.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
        log_level=API_LOG_LEVEL,
    )


if __name__ == "__main__":
    main()