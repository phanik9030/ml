# server.py
import uvicorn
from fastapi import FastAPI, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from aa_play_new_1 import (
    Catalog, Tools, run_agent, initial_state, tools as global_tools
)

app = FastAPI(title="Assistant API")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared catalog/tools already in insight_assistant_simple (global_tools)
catalog = Catalog()  # not strictly needed here but handy for endpoints
tools = global_tools

# In-memory session store: session_id -> state
SESSIONS: Dict[str, Dict[str, Any]] = {}

def get_session(session_id: Optional[str]) -> Dict[str, Any]:
    if not session_id:
        session_id = "default"
    if session_id not in SESSIONS:
        SESSIONS[session_id] = initial_state()
    return SESSIONS[session_id]

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sid = req.session_id or "default"
    state = get_session(sid)
    result = run_agent(req.message, state)
    # Persist the updated state
    SESSIONS[sid] = result
    # Last assistant message
    msgs = result.get("messages", [])
    reply = msgs[-1]["content"] if msgs else "..."
    return ChatResponse(reply=reply, session_id=sid)

@app.get("/schemas")
def schemas():
    return {"schemas": catalog.list_schemas()}

@app.get("/tables/{schema}")
def tables(schema: str):
    return {"tables": tools.list_tables(schema)}

@app.get("/insights")
def list_insights():
    if not tools.saved:
        return {"insights": []}
    # present minimal metadata
    out=[]
    for v in tools.saved.values():
        out.append({
            "id": v["id"],
            "name": v["name"],
            "ttl": v["ttl"],
            "active": v["active"],
            "schedule": v.get("schedule",""),
            "columns": v.get("columns",[])
        })
    return {"insights": sorted(out, key=lambda x: x["name"].lower())}

@app.get("/insights/{name}/explain")
def explain_insight(name: str):
    rec = tools.get_saved(name)
    if not rec:
        return {"error": f"Insight '{name}' not found"}
    meta = {k: rec[k] for k in ["name", "ttl", "active", "schedule", "columns"] if k in rec}
    exp = tools.explain_sql_llm(rec.get("sql", ""))
    return {"meta": meta, "explanation": exp}

class UpdateInsightRequest(BaseModel):
    ttl: Optional[int] = None
    active: Optional[bool] = None
    name: Optional[str] = None
    schedule: Optional[str] = None
    sql: Optional[str] = None

@app.post("/insights/{name}/update")
def update_insight(name: str, body: UpdateInsightRequest):
    rec = tools.get_saved(name)
    if not rec:
        return {"error": f"Insight '{name}' not found"}
    updated = tools.update(rec["id"], body.dict(exclude_none=True))
    return {"updated": updated}

@app.get("/healthz")
def health():
    return {"ok": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
