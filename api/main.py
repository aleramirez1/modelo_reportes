from __future__ import annotations

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from api.core.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from api.db.session import Base, engine
from api.models import inference as inference_model  # noqa: F401
from api.routers.inference import router as inference_router


app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    summary="API para inferencia no supervisada y consulta de historial de reportes.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "Salud",
            "description": "Verificacion basica del estado de la API.",
        },
        {
            "name": "Inferencias",
            "description": "Operacion de inferencia y consulta del historial almacenado.",
        },
    ],
    contact={
        "name": "Recolecta API",
    },
    license_info={
        "name": "Uso interno",
    },
    servers=[
        {"url": "http://127.0.0.1:8000", "description": "Servidor local"},
    ],
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        summary=app.summary,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
        servers=app.servers,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


@app.get(
    "/health",
    summary="Health check",
    description="Retorna el estado operativo de la API.",
    tags=["Salud"],
)
def health_check() -> dict:
    return {"status": "ok"}


app.include_router(inference_router)
