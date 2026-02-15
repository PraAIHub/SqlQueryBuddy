"""Main entry point for SQL Query Buddy"""
import logging
from src.config import settings

# Configure root logger from LOG_LEVEL env var before any other imports
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.app import main

if __name__ == "__main__":
    main()
