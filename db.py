# db.py
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./memory_chat.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def init_db():
    with engine.begin() as conn:
        # Create table if it doesn't exist
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding TEXT
            );
        """))

        # Add 'source' column if not already there
        result = conn.execute(text("PRAGMA table_info(notes);")).fetchall()
        columns = [row[1] for row in result]  # row[1] is column name

        if "source" not in columns:
            conn.execute(text("""
                ALTER TABLE notes ADD COLUMN source TEXT;
            """))
