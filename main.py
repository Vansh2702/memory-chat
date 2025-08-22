# main.py
from fastapi import FastAPI, Form, Query
from db import init_db, SessionLocal
from sqlalchemy import text
from typing import Optional

app = FastAPI()
init_db()

@app.post("/upload")
def upload_note(content: str = Form(...)):
    with SessionLocal() as session:
        session.execute(
            text("INSERT INTO notes (content) VALUES (:content)"),
            {"content": content}
        )
        session.commit()
    return {"status": "note stored"}

@app.get("/search")
def search_notes(q: Optional[str] = Query(None)):
    with SessionLocal() as session:
        if q:
            result = session.execute(
                text("SELECT id, content FROM notes WHERE content LIKE :q"),
                {"q": f"%{q}%"}
            ).fetchall()
        else:
            result = session.execute(
                text("SELECT id, content FROM notes ORDER BY id DESC LIMIT 10")
            ).fetchall()

        return [{"id": row[0], "content": row[1]} for row in result]