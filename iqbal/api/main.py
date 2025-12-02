import os
from typing import Optional
from contextlib import asynccontextmanager

import aiosqlite
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

load_dotenv()

db: Optional[aiosqlite.Connection] = None
API_KEY = os.getenv("API_KEY")

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

@app.get("/something")
async def something(_: bool = Depends(verify_api_key)):
    return {"message": "Hello World"}
