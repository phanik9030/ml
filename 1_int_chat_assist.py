import os, json, re, uuid
from typing import List, Dict, Any, Optional, Tuple, Set
from flask import Flask, request, jsonify
from flask_cors import CORS

from catalog import CATALOG, dump_catalog_for_prompt  # unchanged
from db import get_connection

# -----------------------------
# Config
# -----------------------------
OPENAI_MODEL = "gpt-4o"
ROW_LIMIT     = 50
MAX_TURNS     = 12          # keep 12 full user+assistant turns in memory
PORT          = int(os.getenv("PORT", "5000"))

# -----------------------------
# DB connection (built & seeded in db.py)
# -----------------------------
con = get_connection()

# -----------------------------
# Conversation store (in-memory)
# -----------------------------
# conv: { id, history, selected_schemas: [..], selected_tables: ["s.t", ..] }
CONVS: Dict[str, Dict[str, Any]] = {}

# -----------------------------
# Utilities
# -----------------------------
identifier_re = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
def qi(x: str) -> str:
    if not identifier_re.match(x):
        return '"' + x.replace('"','""') + '"'
    return '"' + x + '"'

def ensure_row_limit(sql: str) -> str:
    """
    Ensure the final SELECT has a LIMIT. Works for:
      - SELECT ...
      - WITH cte AS (...) SELECT ...
    If a LIMIT already exists (case-insensitive) in the outer query, do nothing.
    """
    s = sql.strip().rstrip(";")
    # crude check: if the outermost query already has LIMIT, skip
    if re.search(r'(?is)\bLIMIT\s+\d+\b', s):
        return s
    return s + f"\nLIMIT {ROW_LIMIT};"


def run_sql_safely(sql: str) -> Tuple[List[str], List[List[Any]], Optional[str]]:
    """
    Execute read-only SQL (CTEs and SELECT). Rejects any write/control statements.
    Returns (columns, rows, error).
    """
    raw = (sql or "").strip()
    if not raw:
        return [], [], "Empty SQL."

    # 1) Reject multiple statements (very simple split on ';' ignoring trailing)
    cleaned = raw.rstrip(";").strip()
    # If there is a semicolon in the middle, treat as multiple statements
    if ";" in cleaned:
        return [], [], "Only a single SQL statement is allowed."

    # 2) Allow only queries starting with WITH or SELECT (case-insensitive)
    if not re.match(r'(?is)^\s*(WITH|SELECT)\b', cleaned):
        return [], [], "Only read-only SELECT/CTE queries are allowed."

    # 3) Block write/control keywords anywhere (very defensive)
    forbidden = re.compile(
        r'(?is)\b(INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|REPLACE|COPY|ATTACH|DETACH|BEGIN|COMMIT|ROLLBACK|MERGE|TRUNCATE|VACUUM|EXPORT|IMPORT|LOAD|PRAGMA|SET)\b'
    )
    if forbidden.search(cleaned):
        return [], [], "Write/control statements are not allowed."

    try:
        final_sql = ensure_row_limit(cleaned)
        res = con.execute(final_sql)
        cols = [d[0] for d in res.description]
        rows = res.fetchall()
        return cols, rows, None
    except Exception as e:
        return [], [], f"SQL execution error: {e}"


def filter_catalog(full: Dict[str, Any], schemas: Optional[List[str]], tables: Optional[List[str]]) -> Dict[str, Any]:
    """
    Return a new catalog dict filtered to only the selected schemas and tables.
    - schemas: list of schema names (or None -> all)
    - tables: list of "schema.table" (or None -> all tables within selected schemas)
    FKs are kept only when both sides are present in the filtered set.
    """
    schemas_set: Optional[Set[str]] = set(schemas) if schemas else None
    tables_set: Optional[Set[str]] = set(tables) if tables else None

    out: Dict[str, Any] = {}
    present_tables: Set[str] = set()

    # Copy schemas/tables
    for s, body in full.items():
        if s.startswith("__"):
            continue
        if schemas_set and s not in schemas_set:
            continue
        out[s] = {"__desc": body.get("__desc", s)}
        for t, meta in body.items():
            if t.startswith("__"):
                continue
            fq = f"{s}.{t}"
            if tables_set and fq not in tables_set:
                continue
            # include the table
            out[s][t] = {
                "__desc": meta.get("__desc", fq),
                "columns": meta.get("columns", []),
                "pk": meta.get("pk", [])
            }
            present_tables.add(fq)

    # Prune empty schemas
    for s in list(out.keys()):
        if s.startswith("__"):
            continue
        # schema with just __desc?
        if all(k.startswith("__") for k in out[s].keys()):
            out.pop(s, None)

    # Filter FKs
    fks = []
    for fk in full.get("__fk", []):
        lsch, ltab, _ = fk["left"]
        rsch, rtab, _ = fk["right"]
        if f"{lsch}.{ltab}" in present_tables and f"{rsch}.{rtab}" in present_tables:
            fks.append(fk)
    if fks:
        out["__fk"] = fks

    return out

def build_system_prompt(filtered_catalog: Dict[str, Any]) -> str:
    """
    Strict instruction with required blocks + (filtered) catalog context.
    """
    return (
        "You are a SQL analyst agent. Maintain conversational context across turns.\n"
        "The user is exploring data and may ask follow-ups or switch topics, then return later.\n"
        "Always use ONLY the given catalog below (schemas, tables, columns). Do not invent names.\n"
        "Prefer explicit JOINs using hints. Output must use a Postgres/Snowflake-like dialect compatible with DuckDB where possible.\n"
        "STRICTLY return the exact block structure below — no extra prose outside the blocks:\n"
        "<<<REWRITTEN_PROMPT>>>\n"
        "(one concise, precise English rewrite of the user's latest request, grounding it in the catalog)\n"
        "<<<END_REWRITTEN_PROMPT>>>\n"
        "<<<ENGLISH_DSL_START>>>\n"
        "Title:\nSource:\nJoins:\nFilters:\nAggregations:\nGroup By:\nHaving:\nWindow Functions:\nOrder:\nLimit:\nAssumptions:\nSecurity Notes:\n"
        "<<<ENGLISH_DSL_END>>>\n"
        "<<<SQL_START>>>\n"
        "(ONE single SELECT statement; fully qualify tables with schema, e.g., sales.ord_hdr. "
        "If a time window like 'last 30 days' is requested, use ord_dt >= CURRENT_DATE - INTERVAL '30 day'.)\n"
        f"(Ensure LIMIT {ROW_LIMIT} at most if needed.)\n"
        "<<<SQL_END>>>\n"
        "<<<EXPLANATION_START>>>\n"
        "(plain-English explanation of what the result shows; mention filters and joins clearly)\n"
        "<<<EXPLANATION_END>>>\n\n"
        "CATALOG:\n" + dump_catalog_for_prompt(filtered_catalog)
    )

def parse_block(text: str, start: str, end: str) -> Optional[str]:
    a, b = text.find(start), text.find(end)
    if a == -1 or b == -1 or b <= a:
        return None
    return text[a+len(start):b].strip()

# --- Salvage helpers when the model forgets blocks ---
CODE_FENCE_RE = re.compile(r"```(?:sql)?\s*([\s\S]*?)```", re.IGNORECASE)
def salvage_sql_from_code(text: str) -> Optional[str]:
    # Prefer fenced code blocks
    for m in CODE_FENCE_RE.finditer(text or ""):
        code = m.group(1).strip()
        if re.search(r'(?is)\bSELECT\b', code):
            return code
    # Otherwise, grab first SELECT...
    m = re.search(r'(?is)(SELECT[\s\S]+?);?$', text or "")
    if m:
        return m.group(1).strip()
    return None

def salvage_rewritten(text: str) -> Optional[str]:
    m = re.search(r"(?is)Rewritten prompt[:\n]+([\s\S]+?)(?:\n\n|$|SQL|Explanation)", text or "")
    if m:
        return m.group(1).strip()
    return None

def salvage_explanation(text: str) -> Optional[str]:
    m = re.search(r"(?is)Explanation[:\n]+([\s\S]+?)(?:\n\n|$|SQL)", text or "")
    if m:
        return m.group(1).strip()
    return None

# --- OpenAI call ---
def oai_chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> Optional[str]:
    """
    Single OpenAI chat completions call. Returns content or None.
    """
    api_key = ""
    if not api_key:
        return None
    import urllib.request
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps({
            "model": OPENAI_MODEL,
            "messages": messages,
            "temperature": temperature
        }).encode("utf-8")
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"]
    except Exception:
        return None

# --- Format enforcement & SQL repair loop ---
def generate_with_repairs(conv_history: List[Dict[str,str]], filtered_catalog: Dict[str, Any]) -> Tuple[str,str,str,str]:
    """
    Returns (revised, english_dsl, sql, explanation) using the filtered catalog.
    """
    system = {"role": "system", "content": build_system_prompt(filtered_catalog)}

    # Attempt 1
    messages = [system] + conv_history
    content = oai_chat(messages) or ""

    revised = parse_block(content, "<<<REWRITTEN_PROMPT>>>", "<<<END_REWRITTEN_PROMPT>>>")
    english = parse_block(content, "<<<ENGLISH_DSL_START>>>", "<<<ENGLISH_DSL_END>>>")
    sql     = parse_block(content, "<<<SQL_START>>>", "<<<SQL_END>>>")
    expl    = parse_block(content, "<<<EXPLANATION_START>>>", "<<<EXPLANATION_END>>>")

    missing = not all([revised, english, sql, expl])

    if missing:
        reminder = {
            "role": "user",
            "content": (
                "Your previous reply did not include ALL required blocks. "
                "Reply again NOW with the EXACT blocks only (no extra text)."
            )
        }
        content2 = oai_chat([system] + conv_history + [reminder]) or ""
        revised2 = parse_block(content2, "<<<REWRITTEN_PROMPT>>>", "<<<END_REWRITTEN_PROMPT>>>")
        english2 = parse_block(content2, "<<<ENGLISH_DSL_START>>>", "<<<ENGLISH_DSL_END>>>")
        sql2     = parse_block(content2, "<<<SQL_START>>>", "<<<SQL_END>>>")
        expl2    = parse_block(content2, "<<<EXPLANATION_START>>>", "<<<EXPLANATION_END>>>")

        revised = revised2 or revised
        english = english2 or english
        sql     = sql2 or sql
        expl    = expl2 or expl

        # If still no SQL, try salvaging from any response text
        if not sql:
            sql = salvage_sql_from_code(content2) or salvage_sql_from_code(content) or ""

    # Salvage minimal fields if still missing
    if not revised:
        revised = salvage_rewritten(content) or ""
    if not expl:
        expl = salvage_explanation(content) or ""

    return revised or "", english or "", sql or "", expl or ""

def trim_history(hist: List[Dict[str,str]]) -> List[Dict[str,str]]:
    # keep last MAX_TURNS * 2 messages (user+assistant)
    if len(hist) <= MAX_TURNS * 2:
        return hist
    return hist[-MAX_TURNS * 2:]

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)
CORS(app)

@app.post("/api/chat/start")
def chat_start():
    """
    Start a new conversation with a filtered context.
    Body (optional):
      {
        "schemas": ["sales","mkt"],        # default: all schemas
        "tables":  ["sales.ord_hdr", ...]  # default: all tables inside selected schemas
      }
    """
    body = request.get_json(silent=True) or {}
    sel_schemas = body.get("schemas")
    sel_tables  = body.get("tables")

    # Validate tables against schemas if both given
    if sel_schemas and sel_tables:
        valid = []
        for fq in sel_tables:
            if "." not in fq:
                continue
            s, t = fq.split(".", 1)
            if s in sel_schemas:
                valid.append(fq)
        sel_tables = valid

    # Build filtered catalog used for this conversation
    filtered = filter_catalog(CATALOG, sel_schemas, sel_tables)

    conv_id = str(uuid.uuid4())
    CONVS[conv_id] = {
        "id": conv_id,
        "history": [],
        "selected_schemas": sel_schemas,
        "selected_tables": sel_tables,
        "catalog": filtered
    }
    hello = (
        "Context set. I’ll use only the selected schemas/tables.\n"
        "Ask me questions like:\n"
        "• “List US customers with total amount > 50 in USD since 2025 and the campaign name.”\n"
        "• “Group revenue by customer segment last 30 days.”"
    )
    CONVS[conv_id]["history"].append({"role": "assistant", "content": hello})
    return jsonify({"conversation_id": conv_id, "message": hello})

@app.get("/api/chat/history")
def chat_history():
    conv_id = request.args.get("conversation_id")
    conv = CONVS.get(conv_id)
    if not conv:
        return jsonify({"error": "Conversation not found."}), 404
    return jsonify({
        "conversation_id": conv_id,
        "history": conv["history"],
        "selected_schemas": conv.get("selected_schemas"),
        "selected_tables": conv.get("selected_tables")
    })

@app.post("/api/chat/message")
def chat_message():
    """
    Body: { "conversation_id": "...", "message": "user text" }
    Returns: assistant message (explanation only), plus structured artifacts & table data.
    """
    body = request.get_json(force=True)
    conv_id = body.get("conversation_id")
    user_msg = (body.get("message") or "").strip()
    if not conv_id or conv_id not in CONVS:
        return jsonify({"error": "Conversation not found. Start one at /api/chat/start."}), 400
    if not user_msg:
        return jsonify({"error": "Message is empty."}), 400

    conv = CONVS[conv_id]
    conv["history"].append({"role": "user", "content": user_msg})
    conv["history"] = trim_history(conv["history"])

    # Generate (with strict formatting & salvage) using the per-conversation filtered catalog
    revised, english, sql, expl = generate_with_repairs(conv["history"], conv["catalog"])

    # If no SQL produced, reply with explanation only
    if not sql:
        assistant_text = expl or "I couldn’t obtain a valid SQL block from the model. Please rephrase and try again."
        conv["history"].append({"role": "assistant", "content": assistant_text})
        return jsonify({
            "assistant": assistant_text,
            "revised_prompt": revised,
            "english_dsl": english,
            "sql": "",
            "explanation": expl,
            "table": {"columns": [], "rows": []},
            "error": "No SQL produced."
        })

    # Execute SQL; if error, ask model once to correct using the error message
    cols, rows, err = run_sql_safely(sql)
    if err:
        repair_msg = {
            "role": "user",
            "content": (
                "The SQL you produced failed when executed against DuckDB.\n"
                f"Error: {err}\n"
                f"Your original SQL was:\n```\n{sql}\n```\n"
                "Please return the SAME block structure again with a corrected single SELECT query."
            )
        }
        revised2, english2, sql2, expl2 = generate_with_repairs(conv["history"] + [repair_msg], conv["catalog"])
        if sql2:
            cols2, rows2, err2 = run_sql_safely(sql2)
            cols, rows, err = cols2, rows2, err2
            revised = revised2 or revised
            english = english2 or english
            sql = sql2 or sql
            expl = expl2 or expl

    if err:
        assistant_text = ((expl + "\n") if expl else "") + f"SQL error: {err}"
        conv["history"].append({"role": "assistant", "content": assistant_text})
        return jsonify({
            "assistant": assistant_text,
            "revised_prompt": revised,
            "english_dsl": english,
            "sql": sql,
            "explanation": expl,
            "table": {"columns": [], "rows": []},
            "error": err
        })

    # Success: only explanation + row count in assistant message
    assistant_text = (expl + "\n" if expl else "") + f"Returned {len(rows)} rows."
    conv["history"].append({"role": "assistant", "content": assistant_text})

    return jsonify({
        "assistant": assistant_text,
        "revised_prompt": revised,
        "english_dsl": english,
        "sql": sql,
        "explanation": expl,
        "table": {"columns": cols, "rows": rows},
        "error": None
    })

# Full catalog for UI selection
@app.get("/api/catalog")
def api_catalog():
    skinny = {}
    for s, body in CATALOG.items():
        if s.startswith("__"):
            continue
        skinny[s] = {"__desc": body.get("__desc", s), "tables": {}}
        for t, meta in body.items():
            if t.startswith("__"):
                continue
            skinny[s]["tables"][t] = {
                "__desc": meta.get("__desc", f"{s}.{t}"),
                "columns": [c["name"] for c in meta["columns"]]
            }
    return jsonify(skinny)

if __name__ == "__main__":
    # Requirements:
    #   pip install flask flask-cors duckdb
    #   export OPENAI_API_KEY=sk-...  # required
    app.run(host="0.0.0.0", port=PORT, debug=True)
