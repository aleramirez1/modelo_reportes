from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.db.session import get_db
from api.repositories.inference_repository import count_inferences, create_inference, list_inferences
from api.schemas.inference import InferenceCreate, InferenceListResponse, InferenceRead
from api.services.inference_service import InferenceService


router = APIRouter(tags=["Inferencias"])
service = InferenceService()


@router.post(
    "/infer",
    response_model=InferenceRead,
    summary="Realizar una inferencia",
    description="Recibe un reporte en texto libre, ejecuta la inferencia del modelo no supervisado y guarda el resultado en la base de datos.",
    response_description="Inferencia generada y persistida correctamente.",
    responses={
        200: {
            "description": "Inferencia creada.",
        },
        400: {
            "description": "Solicitud invalida o error durante la inferencia.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "No fue posible procesar el reporte enviado."
                    }
                }
            },
        },
    },
)
def infer(request: InferenceCreate, db: Session = Depends(get_db)):
    try:
        result = service.infer(request)
        created = create_inference(db, service.to_model(result))
        return created
    except Exception as exc:  # pragma: no cover - defensive API layer
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/inferences",
    response_model=InferenceListResponse,
    summary="Consultar inferencias realizadas",
    description="Devuelve el historial de inferencias almacenadas, ordenadas de la mas reciente a la mas antigua.",
    response_description="Listado paginado de inferencias.",
)
def get_inferences(
    limit: int = Query(default=20, ge=1, le=100, description="Cantidad maxima de registros por pagina."),
    offset: int = Query(default=0, ge=0, description="Indice inicial para paginacion."),
    db: Session = Depends(get_db),
):
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    items = list_inferences(db, limit=limit, offset=offset)
    total = count_inferences(db)
    return InferenceListResponse(items=items, total=total, limit=limit, offset=offset)
