from fastapi import APIRouter, Form, Query, UploadFile, File
from sqlalchemy import text as sql_text
from typing import Optional
import json

from db import SessionLocal
from schemas.search import SearchQuery
from services.embedding_service import encode
from services.search_utils import semantic_search
from services.file_utils import extract_text_from_file, chunk_text

router = APIRouter()

@router.post("/upload")
def upload_note(content: str = Form(...)):
    embedding = encode(content)
    emb_json = json.dumps(embedding)

    with SessionLocal() as session:
        session.execute(
            sql_text("INSERT INTO notes (content, embedding) VALUES (:content, :embedding)"),
            {"content": content, "embedding": emb_json}
        )
        session.commit()
    return {"status": "note stored"}


@router.get("/search")
def search_notes(q: Optional[str] = Query(None)):
    with SessionLocal() as session:
        if q:
            result = session.execute(
                sql_text("SELECT id, content FROM notes WHERE content LIKE :q"),
                {"q": f"%{q}%"}
            ).fetchall()
        else:
            result = session.execute(
                sql_text("SELECT id, content FROM notes ORDER BY id DESC LIMIT 10")
            ).fetchall()

    return [{"id": row[0], "content": row[1]} for row in result]


@router.post("/semantic-search")
def semantic_search_endpoint(input: SearchQuery):
    return semantic_search(input.query)


@router.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_text = await extract_text_from_file(file)
    except ValueError as e:
        return {"error": str(e)}

    chunks = chunk_text(file_text, chunk_size=600)

    for chunk in chunks:
        embedding = encode(chunk)
        with SessionLocal() as session:
            session.execute(
                sql_text("INSERT INTO notes (content, embedding, source) VALUES (:content, :embedding, :source)"),
                {"content": chunk, "embedding": json.dumps(embedding), "source": file.filename}
            )
            session.commit()

    return {
        "status": "file processed",
        "file": file.filename,
        "chunks_stored": len(chunks)
    }
