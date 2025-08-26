from fastapi import FastAPI
from db import init_db
from routes import notes, chat

app = FastAPI()
init_db()

app.include_router(notes.router)
app.include_router(chat.router)
