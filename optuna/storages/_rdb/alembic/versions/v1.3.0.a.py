
import json

from alembic import op
import sqlalchemy as sa

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import orm

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

revision = "v1.3.0.a"
down_revision = "v1.2.0.a"
branch_labels = None
depends_on = None

MAX_INDEXED_STRING_LENGTH = 512
MAX_STRING_LENGTH = 2048
BaseModel = declarative_base()


class TrialModel(BaseModel):
    __tablename__ = "trials"
    trial_id = sa.Column(sa.Integer, primary_key=True)
    number = sa.Column(sa.Integer)


class TrialSystemAttributeModel(BaseModel):
    __tablename__ = "trial_system_attributes"
    trial_system_attribute_id = sa.Column(sa.Integer, primary_key=True)
    trial_id = sa.Column(sa.Integer, sa.ForeignKey("trials.trial_id"))
    key = sa.Column(sa.String(MAX_INDEXED_STRING_LENGTH))
    value_json = sa.Column(sa.String(MAX_STRING_LENGTH))




