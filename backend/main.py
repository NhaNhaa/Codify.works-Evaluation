"""
backend/main.py
Entry point for AutoEval-C.
Starts the FastAPI application via Uvicorn.
"""

import uvicorn
from backend.utils.logger import engine_logger


def main():
    """
    Starts the AutoEval-C FastAPI server.
    """
    engine_logger.info("MAIN: Starting AutoEval-C server...")
    engine_logger.info("MAIN: Swagger UI available at http://127.0.0.1:8000/docs")

    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()