import os
import tempfile
from typing import Annotated
from contextlib import asynccontextmanager

import sqlite3
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
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

import schema
import agents

API_KEY = os.getenv("API_KEY")
GCS_BUCKET = os.getenv("GCS_BUCKET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET)


@asynccontextmanager
async def lifespan(app: FastAPI):
    agents.db = sqlite3.connect("database.db")
    # embeddings = OpenAIEmbeddings(
    #     model="text-embedding-3-small",
    #     api_key=OPENAI_API_KEY,
    # )
    # agents.qdrant = QdrantVectorStore.from_existing_collection(
    #     embedding=embeddings,
    #     collection_name=QDRANT_COLLECTION,
    #     url=QDRANT_URL,
    #     api_key=QDRANT_API_KEY,
    # )
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
    req: schema.ChatRequest,
    _: bool = Depends(verify_api_key),
) -> schema.ChatResponse:
    blob_name = f"uploads/{req.session_id}.pdf"
    blob: Blob = bucket.blob(blob_name)
    if blob.exists():
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_file:
            blob.download_to_file(tmp_file)
            tmp_file.flush()
            tmp_file.seek(0)
            loader = PyPDFLoader(tmp_file.name)
            docs = loader.load()
        content = ""
        for page in docs:
            content += page.page_content + "\n"
        resp = agents.chat(req, content)
    else:
        resp = agents.chat(req, None)
    return schema.ChatResponse(message=schema.ChatMessage(role="ai", content=resp))
