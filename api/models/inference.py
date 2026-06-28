from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from api.db.session import Base


class Inference(Base):
    __tablename__ = "inferencias"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    reporte: Mapped[str] = mapped_column(Text, nullable=False)
    tiempo_desde_ultimo_reporte_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cluster: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    distancia_centroide: Mapped[float] = mapped_column(Float, nullable=False)
    anomalia_if: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    score_anomalia_if: Mapped[float] = mapped_column(Float, nullable=False)
    nivel_riesgo: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    nivel_riesgo_final: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    caracteristicas_extraidas: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
