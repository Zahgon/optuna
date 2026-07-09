
from __future__ import annotations

import enum

import numpy as np
from alembic import op
import sqlalchemy as sa
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import orm

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base


revision = "v3.0.0.d"
down_revision = "v3.0.0.c"
branch_labels = None
depends_on = None


BaseModel = declarative_base()
RDB_MAX_FLOAT = np.finfo(np.float32).max
RDB_MIN_FLOAT = np.finfo(np.float32).min


FLOAT_PRECISION = 53


class TrialValueModel(BaseModel):
    class TrialValueType(enum.Enum):
        FINITE = 1
        INF_POS = 2
        INF_NEG = 3

    __tablename__ = "trial_values"
    trial_value_id = sa.Column(sa.Integer, primary_key=True)
    value = sa.Column(sa.Float(precision=FLOAT_PRECISION), nullable=True)
    value_type = sa.Column(sa.Enum(TrialValueType), nullable=False)

    @classmethod
    def value_to_stored_repr(
        cls,
        value: float,
    ) -> tuple[float | None, TrialValueType]:
        if value == float("inf"):
            return None, cls.TrialValueType.INF_POS
        elif value == float("-inf"):
            return None, cls.TrialValueType.INF_NEG
        else:
            return value, cls.TrialValueType.FINITE

    @classmethod
    def stored_repr_to_value(cls, value: float | None, float_type: TrialValueType) -> float:
        if float_type == cls.TrialValueType.INF_POS:
            assert value is None
            return float("inf")
        elif float_type == cls.TrialValueType.INF_NEG:
            assert value is None
            return float("-inf")
        else:
            assert float_type == cls.TrialValueType.FINITE
            assert value is not None
            return value




