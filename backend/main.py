from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import os
import io
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from collections import defaultdict, Counter
import re

load_dotenv()

app = FastAPI(title="MANOBHAV API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase setup
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# ==========================
# MODELS
# ==========================

class LoginRequest(BaseModel):
    email: str
    password: str


class CommentCreate(BaseModel):
    text: str
    user_name: Optional[str] = "Anonymous"
    user_type: Optional[str] = "individual"
    sentiment: Optional[str] = "neutral"
    confidence: Optional[float] = 0.7
    risk_level: Optional[str] = "MEDIUM"
    policy_ref: Optional[str] = None


class LegalTextRequest(BaseModel):
    text: str


class LegalTrainRequest(BaseModel):
    model_type: str
    use_default_data: bool = True

    # FIX for Pydantic protected namespace warning
    model_config = {
        "protected_namespaces": ()
    }


class AIRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = ""


class SummaryRequest(BaseModel):
    business: Optional[List[str]] = []
    individual: Optional[List[str]] = []
    all_comments: Optional[List[str]] = []
    output_format: Optional[str] = "narrative"

# ==========================
# BASIC ROUTES
# ==========================

@app.get("/")
def root():
    return {"message": "MANOBHAV API is running!", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}

# ==========================
# AUTH
# ==========================

@app.post("/api/auth/login")
def login(req: LoginRequest):
    result = supabase.table("profiles").select("*").eq("email", req.email).eq("password", req.password).execute()
    if not result.data:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"success": True, "user": result.data[0]}


@app.post("/api/auth/register")
def register(req: LoginRequest):
    existing = supabase.table("profiles").select("id").eq("email", req.email).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Email already registered")
    result = supabase.table("profiles").insert({
        "email": req.email,
        "password": req.password,
        "full_name": req.email.split("@")[0],
        "role": "individual"
    }).execute()
    return {"success": True, "user": result.data[0]}

# ==========================
# LEGAL ROUTES
# ==========================

try:
    from legal_sentiment import (
        analyze_single,
        analyze_csv as legal_analyze_csv,
        get_model_status,
        SAMPLE_COMMENTS
    )
    LEGAL_ENGINE_READY = True
except ImportError:
    LEGAL_ENGINE_READY = False


@app.get("/api/legal/models/status")
def legal_model_status():
    if not LEGAL_ENGINE_READY:
        return {"success": False}
    return {"success": True, "models": get_model_status(), "device": "cpu"}


@app.post("/api/legal/analyze")
def legal_analyze(req: LegalTextRequest):
    if not LEGAL_ENGINE_READY:
        raise HTTPException(status_code=503, detail="Legal engine not available")
    return {"success": True, "data": analyze_single(req.text)}


@app.post("/api/legal/analyze/batch")
async def legal_batch(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not LEGAL_ENGINE_READY:
        raise HTTPException(status_code=503, detail="Legal engine not available")

    content = await file.read()
    rows = legal_analyze_csv(content)

    df = pd.DataFrame(rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    fname = f"legal_results_{ts}.csv"

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / fname
    df.to_csv(output_path, index=False)

    return {
        "success": True,
        "filename": fname,
        "download_url": f"/api/legal/download/{fname}",
        "summary": {"total_entries": len(df)}
    }


@app.get("/api/legal/download/{filename}")
def legal_download(filename: str):
    file_path = Path("output") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(file_path), filename=filename)


@app.post("/api/legal/train/default")
def legal_train_default(req: LegalTrainRequest):
    return {
        "success": True,
        "data": {"message": f"Rule-based {req.model_type} is always ready."}
    }


@app.post("/api/legal/train/upload")
async def legal_train_upload(file: UploadFile = File(...), model_type: str = Form(...)):
    return {
        "success": True,
        "data": {
            "message": f"CSV received for {model_type}. Full BERT training requires ML pipeline setup."
        }
    }


@app.get("/api/legal/test/samples")
def legal_test_samples():
    if not LEGAL_ENGINE_READY:
        raise HTTPException(status_code=503, detail="Legal engine not available")
    results = [analyze_single(c) for c in SAMPLE_COMMENTS]
    return {"success": True, "data": results}

# ==========================
# RUN SERVER
# ==========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)