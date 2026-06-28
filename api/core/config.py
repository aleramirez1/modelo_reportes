from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_URL = "mysql+pymysql://root:root@localhost:3306/recolecta_api?charset=utf8mb4"


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url
    return DEFAULT_DATABASE_URL


APP_TITLE = "Recolecta Inference API"
APP_DESCRIPTION = (
    "API para ejecutar inferencias no supervisadas sobre reportes y consultar el historial de inferencias."
)
APP_VERSION = "1.0.0"
