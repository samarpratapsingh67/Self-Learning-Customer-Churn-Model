import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool

load_dotenv()

DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql+psycopg2://churn_user:churn_password_123@localhost:5432/churn_prediction'
)

# SQLAlchemy Engine Configuration
connect_args = {}
if DATABASE_URL.startswith('postgresql'):
    connect_args['connect_timeout'] = int(os.getenv('DB_CONNECT_TIMEOUT', 5))

engine = create_engine(
    DATABASE_URL,
    pool_size=int(os.getenv('DB_POOL_SIZE', 10)),
    max_overflow=int(os.getenv('DB_MAX_OVERFLOW', 20)),
    pool_pre_ping=True,
    connect_args=connect_args,
    echo=False  # Set to True for SQL debugging
)

# Session Factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative Base for Models
Base = declarative_base()


def get_db():
    """
    Dependency for getting database session.
    Usage in Flask routes:
        from src.database.db_config import get_db
        db = next(get_db())
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def drop_db():
    """Drop all tables (WARNING: Use only in development/testing)."""
    Base.metadata.drop_all(bind=engine)
