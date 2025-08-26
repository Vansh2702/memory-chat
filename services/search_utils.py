import json
from sentence_transformers.util import cos_sim
from sqlalchemy import text
from db import SessionLocal
from services.embedding_service import encode_vector

def semantic_search(query: str, top_k: int = 5):
    query_emb = encode_vector(query)
    with SessionLocal() as session:
        result = session.execute(text("SELECT id, content, embedding FROM notes")).fetchall()

    scored = []
    for row in result:
        emb = json.loads(row[2])
        sim = cos_sim([query_emb], [emb])[0][0].item()
        scored.append({"id": row[0], "content": row[1], "score": sim})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

def get_top_k_context(query: str, k: int = 3):
    query_emb = encode_vector(query)
    with SessionLocal() as session:
        result = session.execute(text("SELECT id, content, embedding FROM notes")).fetchall()

    scored = []
    for row in result:
        emb = json.loads(row[2])
        sim = cos_sim([query_emb], [emb])[0][0].item()
        scored.append((sim, row[1]))

    scored.sort(reverse=True)
    top_notes = [note for _, note in scored[:k]]
    return top_notes
