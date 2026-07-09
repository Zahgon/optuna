
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


revision = "v3.0.0.c"
down_revision = "v3.0.0.b"
branch_labels = None
depends_on = None


BaseModel = declarative_base()
RDB_MAX_FLOAT = np.finfo(np.float32).max
RDB_MIN_FLOAT = np.finfo(np.float32).min


FLOAT_PRECISION = 53


class IntermediateValueModel(BaseModel):
    class TrialIntermediateValueType(enum.Enum):
        FINITE = 1
        INF_POS = 2
        INF_NEG = 3
        NAN = 4

    __tablename__ = "trial_intermediate_values"
    trial_intermediate_value_id = sa.Column(sa.Integer, primary_key=True)
    intermediate_value = sa.Column(sa.Float(precision=FLOAT_PRECISION), nullable=True)
    intermediate_value_type = sa.Column(sa.Enum(TrialIntermediateValueType), nullable=False)

    @classmethod
    def intermediate_value_to_stored_repr(
        cls,
        value: float,
    ) -> tuple[float | None, TrialIntermediateValueType]:
        if np.isnan(value):
            return None, cls.TrialIntermediateValueType.NAN
        elif value == float("inf"):
            return None, cls.TrialIntermediateValueType.INF_POS
        elif value == float("-inf"):
            return None, cls.TrialIntermediateValueType.INF_NEG
        else:
            return value, cls.TrialIntermediateValueType.FINITE




