from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
from datetime import datetime
import os
import io
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from collections import defaultdict
import re
from collections import Counter

load_dotenv()

app = FastAPI(title="MANOBHAV API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Vercel frontend + local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "MANOBHAV API",
        "supabase": supabase is not None,
        "legal_engine": LEGAL_ENGINE_READY if 'LEGAL_ENGINE_READY' in dir() else False
    }


supabase_url = os.getenv("SUPABASE_URL", "")
supabase_key = os.getenv("SUPABASE_KEY", "")

# Safe Supabase init — won't crash server if env vars are missing
try:
    supabase: Client = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None
except Exception as e:
    print(f"[WARN] Supabase init failed: {e}")
    supabase = None


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


def analyze_sentiment(text: str):
    t = text.lower()
    positive_words = ["good","great","excellent","support","approve","beneficial","clear",
                      "transparent","improved","appreciate","well","fast","straightforward",
                      "positive","helpful","efficient","useful","simple","easy"]
    negative_words = ["bad","poor","burdensome","harsh","unclear","difficult","complex",
                      "tight","unfair","oppose","concern","problem","issue","negative",
                      "challenge","burden","penalty","costly","confusing","excessive"]

    pos = sum(1 for w in positive_words if w in t)
    neg = sum(1 for w in negative_words if w in t)

    if pos > neg:
        sentiment = "positive"
        confidence = min(0.95, 0.6 + (pos - neg) * 0.1)
    elif neg > pos:
        sentiment = "negative"
        confidence = min(0.95, 0.6 + (neg - pos) * 0.1)
    else:
        sentiment = "neutral"
        confidence = 0.6

    business_keywords = ["company","gst","compliance","enterprise","business",
                         "corporate","llp","pvt","ltd","startup","msme","director"]
    user_type = "business" if any(w in t for w in business_keywords) else "individual"
    risk_map = {"negative": "HIGH", "neutral": "MEDIUM", "positive": "LOW"}

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 2),
        "user_type": user_type,
        "risk_level": risk_map[sentiment]
    }


@app.get("/")
def root():
    return {"message": "MANOBHAV API is running!", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/auth/login")
def login(req: LoginRequest):
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not configured. Add SUPABASE_URL and SUPABASE_KEY to Render environment variables.")
    result = supabase.table("profiles").select("*").eq("email", req.email).eq("password", req.password).execute()
    if not result.data:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    user = result.data[0]
    return {"success": True, "user": user}

@app.post("/api/auth/register")
def register(req: LoginRequest):
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not configured. Add SUPABASE_URL and SUPABASE_KEY to Render environment variables.")
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


@app.get("/api/comments")
def get_comments(limit: int = 100):
    if not supabase:
        return {"comments": [], "total": 0}
    result = supabase.table("comments").select("*").order("created_at", desc=True).limit(limit).execute()
    return {"comments": result.data, "total": len(result.data)}

@app.post("/api/comments")
def create_comment(comment: CommentCreate):
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not configured.")
    result = supabase.table("comments").insert(comment.dict()).execute()
    return {"success": True, "comment": result.data[0]}

@app.delete("/api/comments/{comment_id}")
def delete_comment(comment_id: int):
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not configured.")
    supabase.table("comments").delete().eq("id", comment_id).execute()
    return {"success": True}


@app.post("/api/sentiment/analyze")
def analyze_single(body: dict):
    text = body.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    result = analyze_sentiment(text)
    result["text"] = text
    return result

@app.post("/api/sentiment/bulk")
async def analyze_bulk(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")

    text_col = next((c for c in ["comment","text","feedback"] if c in df.columns), None)
    if not text_col:
        raise HTTPException(status_code=400, detail="CSV must have 'comment', 'text', or 'feedback' column")

    results = []
    for _, row in df.iterrows():
        text = str(row[text_col])
        analysis = analyze_sentiment(text)
        analysis["text"] = text
        results.append(analysis)

    positive = sum(1 for r in results if r["sentiment"] == "positive")
    negative = sum(1 for r in results if r["sentiment"] == "negative")
    neutral  = sum(1 for r in results if r["sentiment"] == "neutral")

    return {
        "total": len(results),
        "results": results,
        "summary": {"positive": positive, "negative": negative, "neutral": neutral}
    }


@app.post("/api/wordcloud/generate")
async def generate_wordcloud(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")

    text_col = next((c for c in ["comment","text","feedback"] if c in df.columns), None)
    if not text_col:
        raise HTTPException(status_code=400, detail="CSV must have 'comment', 'text', or 'feedback' column")

    stopwords = {"the","a","an","is","in","it","of","and","or","to","that","this","was",
                 "for","on","are","with","as","at","be","by","from","have","has","had",
                 "not","but","we","they","you","i","he","she","do","did","will","would",
                 "can","could","may","should","their","our","its","also","been","very"}

    positive_w = {"good","great","excellent","clear","transparent","improved","helpful",
                  "efficient","well","fast","support","appreciate","easy","simple"}
    negative_w = {"burdensome","harsh","unclear","difficult","complex","burden","penalty",
                  "concern","problem","tight","unfair","costly","excessive","confusing"}

    all_text = " ".join(df[text_col].astype(str)).lower()
    words = re.findall(r'\b[a-z]{3,}\b', all_text)
    filtered = [w for w in words if w not in stopwords]
    counts = Counter(filtered).most_common(50)

    result = []
    for word, freq in counts:
        sentiment = "positive" if word in positive_w else ("negative" if word in negative_w else "neutral")
        result.append({"word": word, "frequency": freq, "sentiment": sentiment})

    return {"words": result, "total_words": len(filtered)}


@app.post("/api/policy/analyze")
async def analyze_policy(body: dict):
    text = body.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Policy text is required")

    positive_words = ["transparent","clear","efficient","simplified","beneficial","support","easy","improve","optional"]
    negative_words = ["must","shall","required","mandatory","penalty","strict","burden","complex","difficult","prohibit"]

    words = text.lower().split()
    pos = sum(1 for w in words if w in positive_words)
    neg = sum(1 for w in words if w in negative_words)

    overall_score = max(20, min(95, 50 + (pos - neg) * 3))

    problems = [
        {"issue": "Compliance Burden", "severity": min(100, 40 + neg * 5), "description": "Multiple mandatory requirements identified"},
        {"issue": "Complexity", "severity": min(100, 30 + len(words) // 100), "description": "Document complexity may hinder understanding"},
        {"issue": "Timeline Pressure", "severity": 55, "description": "Tight deadlines for implementation"}
    ]
    recommendations = [
        {"title": "Simplify Language", "priority": "High", "effort": "Medium", "timeline": "3 months"},
        {"title": "Extend Deadlines", "priority": "High", "effort": "Low", "timeline": "1 month"},
        {"title": "Add MSME Exemptions", "priority": "Medium", "effort": "High", "timeline": "6 months"}
    ]

    supabase.table("analysis_sessions").insert({
        "policy_text": text[:500],
        "overall_score": overall_score,
        "business_score": round(overall_score * 0.92, 1),
        "user_score": round(min(100, overall_score * 1.05), 1),
        "problems": problems,
        "recommendations": recommendations
    }).execute()

    return {
        "overall_score": overall_score,
        "business_score": round(overall_score * 0.92, 1),
        "user_score": round(min(100, overall_score * 1.05), 1),
        "problems": problems,
        "recommendations": recommendations
    }


@app.get("/api/trends")
def get_trends():
    result = supabase.table("comments").select("sentiment,created_at").execute()
    comments = result.data

    monthly = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})
    for c in comments:
        month = c["created_at"][:7]
        monthly[month][c["sentiment"]] += 1

    trend_data = [{"period": m, **monthly[m]} for m in sorted(monthly.keys())]

    pos = sum(1 for c in comments if c["sentiment"] == "positive")
    neg = sum(1 for c in comments if c["sentiment"] == "negative")
    neu = sum(1 for c in comments if c["sentiment"] == "neutral")

    return {
        "trend_data": trend_data,
        "overall": {"positive": pos, "negative": neg, "neutral": neu, "total": len(comments)}
    }


@app.get("/api/users")
def get_users():
    result = supabase.table("profiles").select("id,email,full_name,role,created_at").execute()
    return {"users": result.data}

@app.delete("/api/users/{user_id}")
def delete_user(user_id: int):
    supabase.table("profiles").delete().eq("id", user_id).execute()
    return {"success": True}


# ════════════════════════════════════════════════════════════
# LEGAL SENTIMENT ANALYSIS ROUTES
# Friend's model — ported as pure rule-based (no BERT needed)
# Add real BERT later by updating legal_sentiment.py
# ════════════════════════════════════════════════════════════

try:
    from legal_sentiment import (
        analyze_single, analyze_csv as legal_analyze_csv,
        get_model_status, SAMPLE_COMMENTS
    )
    LEGAL_ENGINE_READY = True
except ImportError as e:
    LEGAL_ENGINE_READY = False
    print(f"[WARN] legal_sentiment not loaded: {e}")


class LegalTextRequest(BaseModel):
    text: str

class LegalTrainRequest(BaseModel):
    model_type: str
    use_default_data: bool = True


@app.get("/api/legal/models/status")
def legal_model_status():
    if not LEGAL_ENGINE_READY:
        return {"success": False, "models": {}, "device": "unavailable"}
    status = get_model_status()
    return {"success": True, "models": status, "device": "cpu"}


@app.post("/api/legal/analyze")
def legal_analyze(req: LegalTextRequest):
    if not LEGAL_ENGINE_READY:
        raise HTTPException(status_code=503, detail="Legal engine not available. Place legal_sentiment.py next to main.py")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    result = analyze_single(req.text)
    return {"success": True, "data": result}


@app.post("/api/legal/analyze/batch")
async def legal_batch(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not LEGAL_ENGINE_READY:
        raise HTTPException(status_code=503, detail="Legal engine not available")
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files supported")

    content = await file.read()
    try:
        rows = legal_analyze_csv(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not rows:
        return {"success": False, "error": "No results generated"}

    df = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"legal_results_{ts}.csv"
    out_dir = Path("output"); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / fname
    df.to_csv(out_path, index=False)

    summary = {"total_entries": len(df)}
    if "sentiment" in df.columns:
        summary["sentiment_distribution"] = df["sentiment"].value_counts().to_dict()
    if "advanced_sentiment" in df.columns:
        summary["advanced_sentiment_distribution"] = df["advanced_sentiment"].value_counts().head(8).to_dict()
    if "model_used" in df.columns:
        summary["model_usage"] = df["model_used"].value_counts().to_dict()
    spam_rows = df[df["spam_score"] != "0.00%"] if "spam_score" in df.columns else pd.DataFrame()
    summary["spam_detected"] = len(spam_rows)

    return {
        "success": True,
        "filename": fname,
        "download_url": f"/api/legal/download/{fname}",
        "summary": summary,
    }


@app.get("/api/legal/download/{filename}")
def legal_download(filename: str):
    path = Path("output") / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(path), filename=filename, media_type="text/csv")


@app.post("/api/legal/train/default")
def legal_train_default(req: LegalTrainRequest):
    # Rule-based model is always "trained" — BERT training needs full pipeline
    # This endpoint is kept for future BERT integration
    return {"success": True, "data": {"message": f"Rule-based {req.model_type} is always ready. BERT training requires full ML pipeline setup."}}


@app.post("/api/legal/train/upload")
async def legal_train_upload(file: UploadFile = File(...), model_type: str = Form(...)):
    return {"success": True, "data": {"message": f"CSV received for {model_type}. Full BERT training requires ML pipeline setup."}}


@app.get("/api/legal/test/samples")
def legal_test_samples():
    if not LEGAL_ENGINE_READY:
        raise HTTPException(status_code=503, detail="Legal engine not available")
    results = [analyze_single(c) for c in SAMPLE_COMMENTS]
    return {"success": True, "data": {"samples": results, "count": len(results)}}


# ════════════════════════════════════════════════════════════
# HYBRID SUMMARY ENGINE
# ════════════════════════════════════════════════════════════

try:
    from summary_engine import HybridSummaryEngine
    _engine = HybridSummaryEngine()
    ENGINE_READY = True
except Exception as e:
    ENGINE_READY = False
    print(f"[WARN] Summary engine not loaded: {e}")


class AIRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = ""

class SummaryRequest(BaseModel):
    business:      Optional[List[str]] = []
    individual:    Optional[List[str]] = []
    all_comments:  Optional[List[str]] = []
    output_format: Optional[str] = "narrative"


BIZ_KW = [
    "company","gst","compliance","enterprise","corporate","llp","pvt","ltd",
    "msme","director","audit","board","shareholder","startup","business",
]

def _auto_split(comments: List[str]):
    business, individual = [], []
    for c in comments:
        t = c.lower() if isinstance(c, str) else ""
        (business if any(k in t for k in BIZ_KW) else individual).append(c)
    if not business:
        half = len(individual) // 2
        business, individual = individual[:half], individual[half:]
    if not individual:
        half = len(business) // 2
        individual, business = business[:half], business[half:]
    return business, individual


def _simple_fallback(text: str) -> str:
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 10]
    total = len(lines)
    pos_w = {"excellent","clear","helpful","good","great","improved","easy","transparent",
              "fast","appreciate","simple","beneficial","support","efficient"}
    neg_w = {"burdensome","harsh","unclear","difficult","complex","penalty","tight","unfair",
              "problem","confusing","slow","frustrating","costly","excessive","burden"}
    pos_h, neg_h = set(), set()
    p = n = 0
    for line in lines:
        t = line.lower()
        for w in pos_w:
            if w in t: p += 1; pos_h.add(w)
        for w in neg_w:
            if w in t: n += 1; neg_h.add(w)
    sentiment = "largely positive" if p > n else ("largely critical" if n > p else "mixed")
    out = f"Analysis of {total} submissions reveals a {sentiment} response. "
    if pos_h: out += f"Positive themes: {', '.join(list(pos_h)[:3])}. "
    if neg_h: out += f"Key concerns: {', '.join(list(neg_h)[:3])}. "
    out += "Stakeholders recommend simplified guidelines, extended timelines, and improved digital infrastructure."
    return out


# ─── ENDPOINT 1: Legacy endpoint — used by current frontend ──
@app.post("/api/ai/summarize")
def ai_summarize(req: AIRequest):
    lines = [l.strip() for l in req.prompt.split("\n") if len(l.strip()) > 8]
    business, individual = _auto_split(lines)
    if ENGINE_READY:
        result = _engine.generate({"business": business, "individual": individual}, "narrative")
        return {"result": result["summary"], "source": result["source"], "stats": result.get("stats", {})}
    return {"result": _simple_fallback(req.prompt), "source": "rule-based", "stats": {}}


# ─── ENDPOINT 2: Rich endpoint — accepts pre-split data ──────
@app.post("/api/summary/generate")
def generate_summary(req: SummaryRequest):
    business   = list(req.business   or [])
    individual = list(req.individual or [])
    if not business and not individual:
        if not req.all_comments:
            raise HTTPException(status_code=400, detail="No comments provided.")
        business, individual = _auto_split(req.all_comments)
    if not business and not individual:
        raise HTTPException(status_code=400, detail="No valid comments after processing.")
    fmt = req.output_format or "narrative"
    if ENGINE_READY:
        result = _engine.generate({"business": business, "individual": individual}, fmt)
        return result
    all_text = "\n".join(business + individual)
    return {"summary": _simple_fallback(all_text), "source": "rule-based", "stats": {}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)