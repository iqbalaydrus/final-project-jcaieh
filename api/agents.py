import os
from typing import Optional

import sqlite3

from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore

import schema

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

db: Optional[sqlite3.Connection] = None
qdrant: Optional[QdrantVectorStore] = None


# cv_file_contents can be None if the user didn't upload a CV
def chat(req: schema.ChatRequest, cv_file_contents: Optional[str]) -> str:
    return "Hello World"
