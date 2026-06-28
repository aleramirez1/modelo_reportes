from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.models.inference import Inference


def create_inference(db: Session, inference: Inference) -> Inference:
    db.add(inference)
    db.commit()
    db.refresh(inference)
    return inference


def list_inferences(db: Session, limit: int, offset: int) -> list[Inference]:
    stmt = select(Inference).order_by(Inference.created_at.desc(), Inference.id.desc()).limit(limit).offset(offset)
    return list(db.execute(stmt).scalars().all())


def count_inferences(db: Session) -> int:
    return int(db.execute(select(func.count()).select_from(Inference)).scalar_one())
