import os, re, json, datetime as dt
from typing import List, Dict, Any, Tuple, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS

# =========================
# Catalog with optional descriptions + FK graph INSIDE catalog
# =========================
CATALOG: Dict[str, Any] = {
    "sales": {
        "__desc": "Transactional sales data including orders and customers.",
        "ord_hdr": {
            "__desc": "Order header facts (one row per order).",
            "columns": [
                {"name": "ord_id",  "type": "int",     "desc": "Order identifier"},
                {"name": "cust_id", "type": "int",     "desc": "Customer identifier"},
                {"name": "ord_dt",  "type": "date",    "desc": "Date when the order was placed (UTC)."},
                {"name": "tot_amt", "type": "numeric", "desc": "Total order amount in the original currency."},
                {"name": "curr",    "type": "text",    "desc": "Order currency code (e.g., USD)."},
                {"name": "seg",     "type": "text",    "desc": "Customer segment label at order time."},
                {"name": "is_rfnd", "type": "boolean", "desc": "True if the order was refunded."}
            ],
            "pk": ["ord_id"]
        },
        "cust_dim": {
            "__desc": "Customer dimension (one row per customer).",
            "columns": [
                {"name": "cust_id",   "type": "int",  "desc": "Customer identifier."},
                {"name": "cntry",     "type": "text", "desc": "Customer country (ISO code or name)."},
                {"name": "seg",       "type": "text", "desc": "Customer segment label."},
                {"name": "signup_dt", "type": "date", "desc": "When the customer signed up."},
                {"name": "eml",       "type": "text", "desc": "Customer email address (PII)."}
            ],
            "pk": ["cust_id"]
        }
    },
    "mkt": {
        "__desc": "Marketing and attribution data.",
        "camp": {
            "__desc": "Marketing campaigns master data.",
            "columns": [
                {"name": "camp_id",   "type": "int",  "desc": "Campaign identifier."},
                {"name": "nm",        "type": "text", "desc": "Campaign name."},
                {"name": "start_dt",  "type": "date", "desc": "Campaign start date."},
                {"name": "end_dt",    "type": "date", "desc": "Campaign end date."},
                {"name": "chnl",      "type": "text", "desc": "Marketing channel (email, social, etc.)."}
            ],
            "pk": ["camp_id"]
        },
        "attr": {
            "__desc": "Attribution: links orders to campaigns (many-to-many).",
            "columns": [
                {"name": "ord_id",   "type": "int",       "desc": "Order identifier."},
                {"name": "camp_id",  "type": "int",       "desc": "Campaign identifier."},
                {"name": "touch_ts", "type": "timestamp", "desc": "Attribution touch timestamp (UTC)."}
            ],
            "pk": ["ord_id", "camp_id"]
        }
    },

    # FK graph lives inside the catalog (schema-aware)
    "__fk": [
        # sales.cust_id join
        {"left": ["sales", "ord_hdr", "cust_id"], "right": ["sales", "cust_dim", "cust_id"]},
        # mkt.attr -> sales.ord_hdr
        {"left": ["mkt", "attr", "ord_id"], "right": ["sales", "ord_hdr", "ord_id"]},
        # mkt.attr -> mkt.camp
        {"left": ["mkt", "attr", "camp_id"], "right": ["mkt", "camp", "camp_id"]},
    ]
}

# =========================
# Utilities
# =========================
identifier_re = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
def qi(x: str) -> str:
    if not identifier_re.match(x):
        return '"' + x.replace('"','""') + '"'
    return '"' + x + '"'

def fq(schema: str, table: str) -> str:
    return f"{qi(schema)}.{qi(table)}"

def split_tokens(s: str) -> List[str]:
    """Split identifiers into readable words (underscores / camelCase), lowercase."""
    s = s.replace("-", "_")
    parts = re.split(r'[_\W]+', s)
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        out += re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+', p)
    return [t.lower() for t in out if t]

def prettify_identifier(name: str) -> str:
    toks = split_tokens(name)
    if not toks:
        return name
    return " ".join(toks)

def get_schema_desc(schema: str) -> Optional[str]:
    val = CATALOG.get(schema, {}).get("__desc")
    return val

def get_table_desc(schema: str, table: str) -> Optional[str]:
    val = CATALOG.get(schema, {}).get(table, {}).get("__desc")
    return val

def get_column_desc(schema: str, table: str, column: str) -> Optional[str]:
    cols = CATALOG.get(schema, {}).get(table, {}).get("columns", [])
    for c in cols:
        if c.get("name") == column:
            return c.get("desc")
    return None

def describe(schema: str, table: str = None, column: str = None) -> str:
    """
    Prefer explicit desc. If missing, fall back to a neutral prettified name.
    No abbreviation heuristics.
    """
    if column:
        d = get_column_desc(schema, table, column)
        if d:
            return d
        return f"{prettify_identifier(column).capitalize()}."

    if table and not column:
        d = get_table_desc(schema, table)
        if d:
            return d
        return f"{prettify_identifier(table).capitalize()} table."

    d = get_schema_desc(schema)
    if d:
        return d
    return f"{prettify_identifier(schema).capitalize()} schema."

def list_columns(schema: str, table: str) -> List[str]:
    return [c["name"] for c in CATALOG[schema][table]["columns"]]

def fmt_cols(schema: str, table: str, selected_cols: List[str]) -> List[str]:
    all_cols = list_columns(schema, table)
    if not selected_cols:
        return all_cols
    safe = [c for c in selected_cols if c in all_cols]
    return safe or all_cols

# =========================
# Embedding fallback (lazy load) — uses name + desc
# =========================
EMB_MODEL = None
EMB_ACTIVE = True  # set False to disable embedding fallback

def get_emb_model():
    global EMB_MODEL
    if EMB_MODEL is None:
        from sentence_transformers import SentenceTransformer
        EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return EMB_MODEL

def embed(texts: List[str]):
    model = get_emb_model()
    vecs = model.encode(texts, show_progress_bar=False)
    # l2 normalize
    import numpy as np
    from sklearn.preprocessing import normalize
    return normalize(np.asarray(vecs))

def best_match(term: str, candidates: List[Tuple[str, str]]) -> Optional[str]:
    """
    candidates: [(id, text), ...] where text includes name + desc.
    Returns id with highest cosine similarity (or None).
    """
    if not EMB_ACTIVE or not term or not candidates:
        return candidates[0][0] if candidates else None
    vecs = embed([term] + [c[1] for c in candidates])
    q, pool = vecs[0], vecs[1:]
    import numpy as np
    sims = pool @ q
    idx = int(np.argmax(sims))
    return candidates[idx][0]

# =========================
# Build bounded context & fallbacks
# =========================
def build_allowed_context(schemas: List[str], tables: List[str], columnsByTable: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Returns:
      {"tables": [s.t,...],
       "columns": [s.t.c,...],
       "tbls_text": {s.t: "<name — desc>"},
       "cols_text": {s.t.c: "<name — desc>"}}
    """
    valid_tables: List[str] = []
    for st in tables:
        if not isinstance(st, str) or "." not in st:
            continue
        s, t = st.split(".", 1)
        if s in CATALOG and t in CATALOG[s]:
            valid_tables.append(st)

    tbls_text: Dict[str, str] = {}
    cols_text: Dict[str, str] = {}
    cols: List[str] = []

    for st in valid_tables:
        s, t = st.split(".", 1)
        tdesc = describe(s, t)  # prefer explicit desc, else prettified
        tbls_text[st] = f"{s}.{t} — {tdesc}"

        allowed_cols = fmt_cols(s, t, columnsByTable.get(st, []))
        for c in allowed_cols:
            cid = f"{s}.{t}.{c}"
            cdesc = describe(s, t, c)
            cols.append(cid)
            cols_text[cid] = f"{cid} — {cdesc}"

    return {"tables": valid_tables, "columns": cols, "tbls_text": tbls_text, "cols_text": cols_text}

def get_fk_hints() -> List[Tuple[str, str, str, str, str, str]]:
    """Read FK edges from CATALOG['__fk'] as tuples (lsch, ltab, lcol, rsch, rtab, rcol)."""
    edges = []
    for edge in CATALOG.get("__fk", []):
        left = edge.get("left") or []
        right = edge.get("right") or []
        if len(left) == 3 and len(right) == 3:
            edges.append((left[0], left[1], left[2], right[0], right[1], right[2]))
    return edges

def english_header() -> str:
    today = dt.date.today().isoformat()
    return f"English DSL (generated on {today})\n"

def compile_fallback_sql(tables: List[str], columnsByTable: Dict[str, List[str]]) -> Tuple[str, str]:
    """Simple SELECT fallback when LLM is not available or errors."""
    if not tables:
        return "No selection.", "-- No selection.\n"
    base = tables[0]
    if not isinstance(base, str) or "." not in base:
        return "Invalid selection.", "-- Invalid selection.\n"
    bs, bt = base.split(".", 1)

    # SELECT list (qualified)
    select_cols: List[str] = []
    for st in tables:
        if not isinstance(st, str) or "." not in st:
            continue
        s, t = st.split(".", 1)
        allowed = columnsByTable.get(st) or list_columns(s, t)
        for c in allowed:
            select_cols.append(f"{s}.{t}.{c}")

    if not select_cols:
        select_cols = [f"{bs}.{bt}.*"]

    # English summary
    schemas_set = sorted(set(st.split(".", 1)[0] for st in tables if isinstance(st, str) and "." in st))
    english_lines = [
        english_header(), "Title:", f"Draft query for {', '.join(tables)}",
        "Source:", f"  - Schemas: {', '.join(schemas_set)}", f"  - Tables: {', '.join(tables)}",
        "Selected Columns:",
        *[f"  - {c}" for c in select_cols],
        "Joins:"
    ]

    # FK hints filtered to selected tables
    hints: List[str] = []
    selected_set = set(tables)
    for (lsch, ltab, lcol, rsch, rtab, rcol) in get_fk_hints():
        if f"{lsch}.{ltab}" in selected_set and f"{rsch}.{rtab}" in selected_set:
            hints.append(f"  - {lsch}.{ltab}.{lcol} = {rsch}.{rtab}.{rcol}")

    english_lines += hints if hints else ["  - (none detected)"]
    english_lines += [
        "Filters:", "  - (none in fallback mode)",
        "Aggregations:", "  - (none in fallback mode)",
        "Group By:", "  - (none in fallback mode)",
        "Having:", "  - (none)",
        "Window Functions:", "  - (none)",
        "Order:", "  - (none)",
        "Limit:", "  - (none)",
        "Assumptions:", "  - Fallback avoided interpreting NL deeply.",
        "Security Notes:", "  - Apply RLS/PII controls before execution."
    ]

    # SQL
    def qid(qname: str) -> str:
        # "schema.table.column" -> "schema"."table"."column"
        parts = qname.split(".")
        return ".".join(qi(p) for p in parts)

    sel = ", ".join(qid(x) for x in select_cols)
    sql = f"/* Heuristic fallback */\nSELECT {sel}\nFROM {fq(bs, bt)}\n/* Add JOIN/WHERE/GROUP BY in NL mode */\n"
    return "\n".join(english_lines), sql

def resolve_with_embeddings(nl: str, allowed: Dict[str, Any]) -> Dict[str, Any]:
    """
    If NL mentions common concepts that don't exactly match selected names,
    map them to the closest allowed table/column using embeddings (name + desc).
    """
    intents = []
    s = (nl or "").lower()
    if "revenue" in s or "sales" in s or "gmv" in s: intents.append(("metric", "revenue"))
    if "segment" in s: intents.append(("group", "segment"))
    if "country" in s: intents.append(("filter", "country"))
    if "last 30" in s or "past 30" in s or "last thirty" in s: intents.append(("time", "last 30 days"))
    if "order date" in s or "purchase date" in s: intents.append(("column", "order date"))
    if "campaign" in s: intents.append(("join", "campaign"))

    picks: Dict[str, str] = {}
    for kind, term in intents:
        # try exact match against allowed ids first
        exact_cols = [c for c in allowed["columns"] if term.replace(" ", "_") in c or term in c]
        if exact_cols:
            picks[kind] = exact_cols[0]
            continue
        # embed against columns
        cands = [(cid, allowed["cols_text"][cid]) for cid in allowed["columns"]]
        if cands:
            best = best_match(term, cands)
            if best:
                picks[kind] = best
    return picks

# =========================
# LLM (GPT-4o) — NL → English DSL + SQL
# =========================
def english_sql_from_llm(nl: str, allowed: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    api_key = ""
    if not api_key:
        return None

    # Bounded context
    lines = ["- Allowed tables:"]
    for t in allowed["tables"]:
        lines.append(f"  * {t}: {allowed['tbls_text'][t]}")
    lines.append("- Allowed columns:")
    for c in allowed["columns"]:
        lines.append(f"  * {c}: {allowed['cols_text'][c]}")

    # Join hints from catalog
    lines.append("- Join hints (use only if both sides selected):")
    any_hint = False
    selected = set(allowed["tables"])
    for (lsch, ltab, lcol, rsch, rtab, rcol) in get_fk_hints():
        if f"{lsch}.{ltab}" in selected and f"{rsch}.{rtab}" in selected:
            lines.append(f"  * {lsch}.{ltab}.{lcol} = {rsch}.{rtab}.{rcol}")
            any_hint = True
    if not any_hint:
        lines.append("  * (none)")

    system = (
        "Translate the user's request into:\n"
        "1) An English, structured query plan (NOT JSON) with these headings:\n"
        "Title:\nSource:\nJoins:\nFilters:\nAggregations:\nGroup By:\nHaving:\nWindow Functions:\nOrder:\nLimit:\nAssumptions:\nSecurity Notes:\n"
        "2) A single SQL query implementing that plan (Postgres/Snowflake-like).\n"
        "Rules: Use ONLY the provided tables/columns. Prefer DATE_TRUNC/CURRENT_DATE/INTERVAL. Use join hints when relevant."
    )

    user = (
        "User request:\n" + (nl or "") + "\n\n" +
        "Context (bounded):\n" + "\n".join(lines) + "\n\n" +
        "Output format:\n"
        "<<<ENGLISH_DSL_START>>>\n(your English plan here)\n<<<ENGLISH_DSL_END>>>\n"
        "<<<SQL_START>>>\n(your SQL here)\n<<<SQL_END>>>"
    )

    import urllib.request
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps({
            "model": "gpt-4o",
            "messages": [{"role":"system","content":system},{"role":"user","content":user}],
            "temperature": 0.2
        }).encode("utf-8")
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
    except Exception:
        return None

    def extract(text: str, start: str, end: str) -> Optional[str]:
        a, b = text.find(start), text.find(end)
        if a == -1 or b == -1 or b <= a:
            return None
        return text[a+len(start):b].strip()

    english = extract(content, "<<<ENGLISH_DSL_START>>>", "<<<ENGLISH_DSL_END>>>")
    sql = extract(content, "<<<SQL_START>>>", "<<<SQL_END>>>")
    if not english or not sql:
        return None
    return english, sql

# =========================
# Flask app & routes
# =========================
app = Flask(__name__)
CORS(app)

@app.get("/api/catalog/schemas")
def api_schemas():
    # Only real schemas (exclude __fk)
    schemas = sorted([k for k in CATALOG.keys() if not k.startswith("__")])
    return jsonify(schemas)

@app.get("/api/catalog/tables")
def api_tables():
    schema = request.args.get("schema")
    if not schema or schema not in CATALOG or schema.startswith("__"):
        return jsonify([])
    tables = sorted([t for t in CATALOG[schema].keys() if not t.startswith("__")])
    return jsonify(tables)

@app.get("/api/catalog/columns")
def api_columns():
    schema = request.args.get("schema"); table = request.args.get("table")
    if (not schema or schema not in CATALOG or schema.startswith("__") or
        not table or table not in CATALOG[schema]):
        return jsonify([])
    cols = [c["name"] for c in CATALOG[schema][table]["columns"]]
    return jsonify(cols)

@app.post("/api/describe")
def api_describe():
    """
    Body: { "items": ["schema:sales", "table:sales.ord_hdr", "column:sales.ord_hdr.tot_amt"] }
    Returns: { "<key>": "<description>", ... }
    """
    body = request.get_json(force=True)
    items = body.get("items") or []
    out: Dict[str, str] = {}
    for it in items:
        if not isinstance(it, str) or ":" not in it:
            continue
        typ, key = it.split(":", 1)
        if typ == "schema":
            out[it] = describe(key)
        elif typ == "table" and "." in key:
            s, t = key.split(".", 1); out[it] = describe(s, t)
        elif typ == "column" and key.count(".") == 2:
            s, t, c = key.split(".", 2); out[it] = describe(s, t, c)
    return jsonify(out)

@app.post("/api/generate")
def api_generate():
    """
    Request:
    {
      "schemas": ["sales","mkt"],
      "tables": ["sales.ord_hdr","sales.cust_dim","mkt.attr"],
      "columnsByTable": {"sales.ord_hdr":["tot_amt","ord_dt","seg"], "sales.cust_dim":["cntry","seg"]},
      "prompt": "US revenue by segment in last 30 days attributed to campaign; include totals, top 10"
    }
    """
    body = request.get_json(force=True)
    schemas = body.get("schemas") or []
    tables  = [st for st in (body.get("tables") or []) if isinstance(st, str) and "." in st]
    columnsByTable = body.get("columnsByTable") or {}
    prompt = (body.get("prompt") or "").strip()

    if not schemas or not tables:
        return jsonify({"error": "Select at least one schema and one table."}), 400

    allowed = build_allowed_context(schemas, tables, columnsByTable)

    # 1) Try LLM path (bounded context)
    out = english_sql_from_llm(prompt, allowed)
    if out:
        english, sql = out
        return jsonify({"english_dsl": english, "sql": sql})

    # 2) Fallback: simple English + SELECT, plus embedding intent mapping
    picks = resolve_with_embeddings(prompt, allowed)
    english, sql = compile_fallback_sql(allowed["tables"], columnsByTable)

    if picks:
        english += "\nIntent Mapping (fallback):\n"
        for k, v in picks.items():
            english += f"  - {k}: {v}\n"

    return jsonify({"english_dsl": english, "sql": sql})

if __name__ == "__main__":
    # Install deps:
    # pip install flask flask-cors sentence-transformers scikit-learn numpy
    # (optional) export OPENAI_API_KEY=sk-...  # enables GPT-4o English plan + SQL
    app.run(host="0.0.0.0", port=5000, debug=True)
