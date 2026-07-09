
from alembic import op
import sqlalchemy as sa
from typing import Any

from sqlalchemy import Column
from sqlalchemy import Enum
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import UniqueConstraint
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import orm

from optuna.study import StudyDirection

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base


revision = "v2.4.0.a"
down_revision = "v1.3.0.a"
branch_labels = None
depends_on = None

BaseModel = declarative_base()


class StudyModel(BaseModel):
    __tablename__ = "studies"
    study_id = Column(Integer, primary_key=True)
    direction = sa.Column(sa.Enum(StudyDirection))


class StudyDirectionModel(BaseModel):
    __tablename__ = "study_directions"
    __table_args__: Any = (UniqueConstraint("study_id", "objective"),)
    study_direction_id = Column(Integer, primary_key=True)
    direction = Column(Enum(StudyDirection), nullable=False)
    study_id = Column(Integer, ForeignKey("studies.study_id"), nullable=False)
    objective = Column(Integer, nullable=False)


class TrialModel(BaseModel):
    __tablename__ = "trials"
    trial_id = Column(Integer, primary_key=True)
    number = Column(Integer)
    study_id = Column(Integer, ForeignKey("studies.study_id"))
    value = sa.Column(sa.Float)


class TrialValueModel(BaseModel):
    __tablename__ = "trial_values"
    __table_args__: Any = (UniqueConstraint("trial_id", "objective"),)
    trial_value_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey("trials.trial_id"), nullable=False)
    objective = Column(Integer, nullable=False)
    value = Column(Float, nullable=False)
    step = sa.Column(sa.Integer)


class TrialIntermediateValueModel(BaseModel):
    __tablename__ = "trial_intermediate_values"
    __table_args__: Any = (UniqueConstraint("trial_id", "step"),)
    trial_intermediate_value_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey("trials.trial_id"), nullable=False)
    step = Column(Integer, nullable=False)
    intermediate_value = Column(Float, nullable=False)




def downgrade():
    pass
