# main.py
from fastapi import FastAPI, Form, Query, Body
from db import init_db, SessionLocal
from sqlalchemy import text
from typing import Optional
from sentence_transformers import SentenceTransformer
import json
from sentence_transformers.util import cos_sim
from pydantic import BaseModel


app = FastAPI()
init_db()

model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')  # or use another model

class SearchQuery(BaseModel):
    query: str

@app.post("/upload")
def upload_note(content: str = Form(...)):
    embedding = model.encode(content).tolist()
    emb_json = json.dumps(embedding)

    with SessionLocal() as session:
        session.execute(
            text("INSERT INTO notes (content, embedding) VALUES (:content, :embedding)"),
            {"content": content, "embedding": emb_json}
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
    
@app.post("/semantic-search")
def semantic_search(input: SearchQuery):
    query_emb = model.encode(input.query)

    with SessionLocal() as session:
        result = session.execute(text("SELECT id, content, embedding FROM notes")).fetchall()

    scored = []
    for row in result:
        emb = json.loads(row[2])
        sim = cos_sim([query_emb], [emb])[0][0].item()
        scored.append({"id": row[0], "content": row[1], "score": sim})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:5]
