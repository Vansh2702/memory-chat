import uuid
import os
import fitz  # PyMuPDF
from fastapi import UploadFile

async def extract_text_from_file(file: UploadFile) -> str:
    extension = file.filename.split(".")[-1].lower()
    text = ""

    if extension == "txt":
        text = (await file.read()).decode("utf-8")

    elif extension == "pdf":
        tmp_path = f"/tmp/{uuid.uuid4()}.pdf"
        with open(tmp_path, "wb") as tmp:
            tmp.write(await file.read())

        with fitz.open(tmp_path) as doc:
            for page in doc:
                text += page.get_text()

        os.remove(tmp_path)

    else:
        raise ValueError(f"Unsupported file type: {extension}")

    return text

def chunk_text(text: str, chunk_size: int = 600):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
