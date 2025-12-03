from typing import Optional

import sqlite3

import schema

db: Optional[sqlite3.Connection] = None


# cv_file_path can be None if the user didn't upload a CV
def chat(req: schema.ChatRequest, cv_file_path: Optional[str]) -> str:
    return "Hello World"
