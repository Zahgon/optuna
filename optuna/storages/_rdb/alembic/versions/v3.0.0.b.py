
import enum

from alembic import op
from sqlalchemy import and_
from sqlalchemy import Column
from sqlalchemy import Enum
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy.orm import Session

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base


revision = "v3.0.0.b"
down_revision = "v3.0.0.a"
branch_labels = None
depends_on = None

BaseModel = declarative_base()
FLOAT_PRECISION = 53


class TrialState(enum.Enum):
    RUNNING = 0
    COMPLETE = 1
    PRUNED = 2
    FAIL = 3
    WAITING = 4


class TrialModel(BaseModel):
    __tablename__ = "trials"
    trial_id = Column(Integer, primary_key=True)
    number = Column(Integer)
    state = Column(Enum(TrialState), nullable=False)


class TrialValueModel(BaseModel):
    __tablename__ = "trial_values"
    trial_value_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey("trials.trial_id"), nullable=False)
    value = Column(Float, nullable=False)




