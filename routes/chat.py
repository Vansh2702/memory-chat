from fastapi import APIRouter
from schemas.search import SearchQuery
from services.chat_utils import call_claude

router = APIRouter()

@router.post("/chat")
def chat_with_claude(input: SearchQuery):
    return {"answer": call_claude(input.query)}
