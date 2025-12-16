import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent

# Явная загрузка .env из корня проекта
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)


API_KEY: str | None = os.getenv("API_KEY")
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
TELEGRAM_BOT_TOKEN: str | None = os.getenv("TELEGRAM_BOT_TOKEN")


