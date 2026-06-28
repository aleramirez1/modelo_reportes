from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InferenceCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reporte": "hay una persona sospechosa afuera de la escuela tomando fotos",
                "tiempo": 15,
            }
        }
    )

    reporte: str = Field(..., min_length=1, description="Texto completo del reporte")
    tiempo: int | None = Field(default=None, ge=0, description="Minutos desde el ultimo reporte")


class InferenceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    reporte: str
    tiempo_desde_ultimo_reporte_min: int | None
    cluster: int
    distancia_centroide: float
    anomalia_if: int
    score_anomalia_if: float
    nivel_riesgo: str
    nivel_riesgo_final: str
    caracteristicas_extraidas: dict[str, Any]
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "reporte": "hey pendejo menso",
                "tiempo_desde_ultimo_reporte_min": 320,
                "cluster": 1,
                "distancia_centroide": 4.613001492866945,
                "anomalia_if": 1,
                "score_anomalia_if": -0.07428016772051949,
                "nivel_riesgo": "bajo",
                "nivel_riesgo_final": "medio",
                "caracteristicas_extraidas": {
                    "groserias": 2.0,
                    "palabras_repetidas": 0.0,
                    "caracteres_especiales": 0.0,
                    "caracteres_especiales_repitidos": 0.0,
                    "cantidad_numeros": 0.0,
                    "palabras": 3.0,
                    "tiempo_desde_ultimo_reporte_min": 320.0,
                    "contiene_url": 0.0,
                    "porcentaje_groserias": 66.66666666666666,
                    "palabras_claves": 0.0
                },
                "created_at": "2026-06-28T16:30:52"
            }
        },
    )


class InferenceListResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "items": [
                    {
                        "id": 1,
                        "reporte": "hey pendejo menso",
                        "tiempo_desde_ultimo_reporte_min": 320,
                        "cluster": 1,
                        "distancia_centroide": 4.613001492866945,
                        "anomalia_if": 1,
                        "score_anomalia_if": -0.07428016772051949,
                        "nivel_riesgo": "bajo",
                        "nivel_riesgo_final": "medio",
                        "caracteristicas_extraidas": {
                            "groserias": 2.0,
                            "palabras_repetidas": 0.0,
                            "caracteres_especiales": 0.0,
                            "caracteres_especiales_repitidos": 0.0,
                            "cantidad_numeros": 0.0,
                            "palabras": 3.0,
                            "tiempo_desde_ultimo_reporte_min": 320.0,
                            "contiene_url": 0.0,
                            "porcentaje_groserias": 66.66666666666666,
                            "palabras_claves": 0.0
                        },
                        "created_at": "2026-06-28T16:30:52"
                    }
                ],
                "total": 1,
                "limit": 20,
                "offset": 0
            }
        },
    )

    items: list[InferenceRead]
    total: int
    limit: int
    offset: int
