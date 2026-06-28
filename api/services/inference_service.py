from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aplicar_modelo import (
    agregar_porcentaje_groserias,
    cargar_artefactos,
    preparar_df_desde_reporte,
    transformar_entrada,
)
from api.models.inference import Inference
from api.schemas.inference import InferenceCreate


@dataclass
class InferenceEngine:
    base_dir: Path = Path(__file__).resolve().parents[2]

    def __post_init__(self):
        self.scaler, self.features, self.weights, self.kmeans, self.isolation, self.riesgo_meta = cargar_artefactos(
            self.base_dir
        )

    def infer(self, reporte: str, tiempo: int | None = None) -> dict:
        df = preparar_df_desde_reporte(reporte, tiempo)
        df = agregar_porcentaje_groserias(df)
        df, X_modelo = transformar_entrada(df, self.features, self.scaler, self.weights)

        cluster = int(self.kmeans.predict(X_modelo)[0])
        distancia = float(self.kmeans.transform(X_modelo).min(axis=1)[0])
        anomalia_if = int(self.isolation.predict(X_modelo)[0] == -1)
        score_if = float(self.isolation.decision_function(X_modelo)[0])

        p90 = float(self.riesgo_meta["p90_distancia"])
        p97 = float(self.riesgo_meta["p97_distancia"])
        nivel_riesgo = "alto" if distancia >= p97 else "medio" if distancia >= p90 else "bajo"

        nivel_riesgo_final = nivel_riesgo
        if anomalia_if == 1 and distancia >= p97:
            nivel_riesgo_final = "alto"
        elif anomalia_if == 1 or distancia >= p90:
            nivel_riesgo_final = "medio"

        return {
            "reporte": reporte,
            "tiempo_desde_ultimo_reporte_min": tiempo,
            "caracteristicas_extraidas": df.iloc[0].to_dict(),
            "cluster": cluster,
            "distancia_centroide": distancia,
            "anomalia_if": anomalia_if,
            "score_anomalia_if": score_if,
            "nivel_riesgo": nivel_riesgo,
            "nivel_riesgo_final": nivel_riesgo_final,
        }


class InferenceService:
    def __init__(self):
        self.engine = InferenceEngine()

    def infer(self, request: InferenceCreate) -> dict:
        return self.engine.infer(request.reporte, request.tiempo)

    @staticmethod
    def to_model(payload: dict) -> Inference:
        return Inference(
            reporte=payload["reporte"],
            tiempo_desde_ultimo_reporte_min=payload.get("tiempo_desde_ultimo_reporte_min"),
            cluster=payload["cluster"],
            distancia_centroide=payload["distancia_centroide"],
            anomalia_if=payload["anomalia_if"],
            score_anomalia_if=payload["score_anomalia_if"],
            nivel_riesgo=payload["nivel_riesgo"],
            nivel_riesgo_final=payload["nivel_riesgo_final"],
            caracteristicas_extraidas=payload["caracteristicas_extraidas"],
        )
