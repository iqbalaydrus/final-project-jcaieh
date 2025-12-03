import os
from typing import Optional, Annotated
from contextlib import asynccontextmanager

import aiosqlite
import magic
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    status,
    UploadFile,
    Form,
    File,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.storage.blob import Blob

import schema
import agents

load_dotenv()

API_KEY = os.getenv("API_KEY")
GCS_BUCKET = os.getenv("GCS_BUCKET")

db: Optional[aiosqlite.Connection] = None
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db
    db = await aiosqlite.connect("database.db")
    yield


app = FastAPI(lifespan=lifespan)
bearer_scheme = HTTPBearer()


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = credentials.credentials
    if token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return True


@app.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload(
    session_id: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    _: bool = Depends(verify_api_key),
):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided"
        )
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type"
        )
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file extension"
        )
    mime_type = magic.from_buffer(file.file.read(1024), mime=True)
    if mime_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type"
        )
    blob_name = f"uploads/{session_id}.pdf"
    blob: Blob = bucket.blob(blob_name)
    blob.upload_from_file(
        file.file,
        rewind=True,
        content_type=file.content_type,
    )
    return {"id": blob_name}


@app.post("/chat")
async def chat(
    request: schema.ChatRequest,
    _: bool = Depends(verify_api_key),
) -> schema.ChatResponse:
    return schema.ChatResponse(message=schema.ChatMessage(role="ai", content="Hello world"))
