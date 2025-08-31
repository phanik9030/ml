import os, json, re, uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set, Callable, Literal, TypedDict
from difflib import SequenceMatcher
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS

from catalog import CATALOG, dump_catalog_for_prompt
from db import get_connection

# =========================
# Config
# =========================
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ROW_LIMIT      = int(os.getenv("ROW_LIMIT", "50"))
MAX_TURNS      = int(os.getenv("MAX_TURNS", "16"))
PORT           = int(os.getenv("PORT", "5000"))

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY is required for this backend.")

# =========================
# DB connection
# =========================
con = get_connection()

# =========================
# Conversation store (in-memory)
# =========================
CONVS: Dict[str, Dict[str, Any]] = {}

# =========================
# SQL Safety & Catalog Filter
# =========================
def ensure_row_limit(sql: str) -> str:
    s = sql.strip().rstrip(";")
    if re.search(r'(?is)\bLIMIT\s+\d+\b', s):
        return s + ";"
    return s + f"\nLIMIT {ROW_LIMIT};"

def run_sql_safely(sql: str) -> Tuple[List[str], List[List[Any]], Optional[str]]:
    raw = (sql or "").strip()
    if not raw:
        return [], [], "Empty SQL."

    cleaned = raw.rstrip(";").strip()
    if ";" in cleaned:
        return [], [], "Only a single SQL statement is allowed."

    if not re.match(r'(?is)^\s*(WITH|SELECT)\b', cleaned):
        return [], [], "Only read-only SELECT/CTE queries are allowed."

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
    schemas_set: Optional[Set[str]] = set(schemas) if schemas else None
    tables_set: Optional[Set[str]] = set(tables) if tables else None

    out: Dict[str, Any] = {}
    present_tables: Set[str] = set()

    for s, body in full.items():
        if s.startswith("__"): continue
        if schemas_set and s not in schemas_set: continue
        out[s] = {"__desc": body.get("__desc", s)}
        for t, meta in body.items():
            if t.startswith("__"): continue
            fq = f"{s}.{t}"
            if tables_set and fq not in tables_set: continue
            out[s][t] = {
                "__desc": meta.get("__desc", fq),
                "columns": meta.get("columns", []),
                "pk": meta.get("pk", []),
            }
            present_tables.add(fq)

    for s in list(out.keys()):
        if s.startswith("__"): continue
        if all(k.startswith("__") for k in out[s].keys()):
            out.pop(s, None)

    fks = []
    for fk in full.get("__fk", []):
        lsch, ltab, _ = fk["left"]
        rsch, rtab, _ = fk["right"]
        if f"{lsch}.{ltab}" in present_tables and f"{rsch}.{rtab}" in present_tables:
            fks.append(fk)
    if fks:
        out["__fk"] = fks

    return out

# =========================
# LLM helpers
# =========================
BLOCKS = {
    "rew_start": "<<<REWRITTEN_PROMPT>>>",
    "rew_end":   "<<<END_REWRITTEN_PROMPT>>>",
    "dsl_start": "<<<ENGLISH_DSL_START>>>",
    "dsl_end":   "<<<ENGLISH_DSL_END>>>",
    "sql_start": "<<<SQL_START>>>",
    "sql_end":   "<<<SQL_END>>>",
    "exp_start": "<<<EXPLANATION_START>>>",
    "exp_end":   "<<<EXPLANATION_END>>>",
}

def oai_chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    import urllib.request
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        data=json.dumps({
            "model": OPENAI_MODEL,
            "messages": messages,
            "temperature": temperature
        }).encode("utf-8")
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None

def llm_json(system: str, user: str) -> Optional[Dict[str, Any]]:
    content = oai_chat([{"role":"system","content":system},{"role":"user","content":user}], temperature=0.0) or ""
    m = re.search(r"\{[\s\S]*\}", content)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def extract_attribute_mapping(user_text: str, available_cols: List[str]) -> Optional[Dict[str, str]]:
    system = (
        "You parse a user's instruction to map columns for an attribute export.\n"
        "Output strict JSON:\n"
        "{\n"
        '  "mapping": {\n'
        '     "cust_key_value": "<column-name or null>",\n'
        '     "att_val_text": "<column-name or null>",\n'
        '     "as_of_date": "<column-name or null>"\n'
        "  },\n"
        '  "confidence": 0..1\n'
        "}\n"
        "Rules:\n"
        "- Choose column names ONLY from this list (case-sensitive): "
        + ", ".join(available_cols) + "\n"
        "- If a field isn't specified by the user, set it to null.\n"
        "- Never invent names."
    )
    obj = llm_json(system, f"User: {user_text}") or {}
    mapping = (obj.get("mapping") or {})
    if not mapping:
        return None
    for k in ("cust_key_value","att_val_text","as_of_date"):
        mapping.setdefault(k, None)
    for k, v in list(mapping.items()):
        if v is None:
            continue
        if v not in available_cols:
            mapping[k] = None
    return mapping if any(mapping.values()) else None

def parse_block(text: str, start: str, end: str) -> Optional[str]:
    a, b = text.find(start), text.find(end)
    if a == -1 or b == -1 or b <= a: return None
    return text[a+len(start):b].strip()

CODE_FENCE_RE = re.compile(r"```(?:sql)?\s*([\s\S]*?)```", re.IGNORECASE)
def salvage_sql_from_code(text: str) -> Optional[str]:
    for m in CODE_FENCE_RE.finditer(text or ""):
        code = m.group(1).strip()
        if re.search(r'(?is)\bSELECT\b', code):
            return code
    m = re.search(r'(?is)(WITH\s+[\s\S]+|SELECT[\s\S]+?)$', text or "")
    if m:
        return m.group(1).strip()
    return None

def build_attribute_sql(base_sql: str, mapping: Dict[str, str]) -> str:
    id_col   = mapping["cust_key_value"]
    val_col  = mapping["att_val_text"]
    date_col = mapping["as_of_date"]
    base = base_sql.rstrip(";")
    return (
        "SELECT\n"
        f"  base.{id_col}   AS cust_key_value,\n"
        f"  CAST(base.{val_col}  AS TEXT) AS att_val_text,\n"
        f"  CAST(base.{date_col} AS TEXT) AS as_of_date\n"
        f"FROM (\n{base}\n) base\n"
        f"LIMIT {ROW_LIMIT};"
    )

# ---------- Prompts ----------
def build_intent_system() -> str:
    return (
        "You classify a user's message for a data assistant. "
        "Return strict JSON: {\"intent\": str, \"thread_title\": str, \"params\": {...}}.\n"
        "Allowed intents (choose exactly one):\n"
        " - \"tour\"            : how the app works, capabilities, help, onboarding\n"
        " - \"nl2sql\"          : user wants data/analysis query in SQL\n"
        " - \"modify_sql\"      : change the last SQL in the active thread\n"
        " - \"explain_sql\"     : user pasted SQL to explain/validate\n"
        " - \"adopt_sql\"       : user pasted SQL and wants to use it as the working query now\n"
        " - \"schema_qa\"       : ask about available schemas/tables/columns\n"
        " - \"context_update\"  : change working context (schemas/tables in scope)\n"
        " - \"attribute_save\"  : save discovery as 3-column attribute output\n"
        " - \"resume\"          : resume/return to where the user left off previously\n"
        " - \"reset\"           : start fresh thread\n"
        " - \"chat\"            : general conversation or chit-chat\n"
        " - \"unknown\"         : anything else\n"
        "Infer from meaning and from any pasted SQL. Prefer 'adopt_sql' when user intent is to continue using their SQL."
    )

def build_sql_system(filtered_catalog: Dict[str, Any]) -> str:
    return (
        "You are a senior SQL analyst. Use ONLY the provided catalog (schemas/tables/columns). "
        "Do NOT invent names. You may use WITH CTEs, window functions, subqueries, and joins. "
        f"Ensure final output is ONE SELECT (or WITH..SELECT) and respects LIMIT {ROW_LIMIT} max.\n"
        "Produce exactly these blocks (no extra text):\n"
        f"{BLOCKS['rew_start']}\n(one precise English rewrite)\n{BLOCKS['rew_end']}\n"
        f"{BLOCKS['dsl_start']}\n"
        "Title:\nSource (schema.table names):\nCTEs (if any):\nJoin Graph:\nFilters:\nAggregations:\nGroup By:\nHaving:\nWindow Functions:\nOrder:\nLimit:\nAssumptions:\nSecurity Notes:\n"
        f"{BLOCKS['dsl_end']}\n"
        f"{BLOCKS['sql_start']}\n(ONE single statement; fully qualify schema.table; use catalog names verbatim)\n{BLOCKS['sql_end']}\n"
        f"{BLOCKS['exp_start']}\n(plain-English explanation for non-technical users)\n{BLOCKS['exp_end']}\n\n"
        "CATALOG:\n" + dump_catalog_for_prompt(filtered_catalog)
    )

def build_planner_system(filtered_catalog: Dict[str, Any]) -> str:
    return (
        "Plan an analytics query from the user's request. Return strict JSON:\n"
        "{ \"tables\": [\"schema.table\", ...], \"columns\": {\"schema.table\": [\"col\", ...]}, "
        "\"metrics\": [\"...\"], \"filters\": [\"...\"], \"group_by\": [\"...\"], "
        "\"windows\": [\"...\"], \"joins\": [{\"left\":\"schema.table.col\",\"right\":\"schema.table.col\",\"type\":\"inner|left|right|full\"}], "
        "\"notes\": [\"...\"] }\n"
        "Only use names from the provided catalog; be liberal in extracting intent.\n\n"
        "CATALOG:\n" + dump_catalog_for_prompt(filtered_catalog)
    )

def build_schema_qa_system(filtered_catalog: Dict[str, Any]) -> str:
    return (
        "Answer questions about available schemas/tables/columns and how to use them. "
        "Base answers ONLY on this catalog, and propose 2-3 example queries.\n\n"
        "CATALOG:\n" + dump_catalog_for_prompt(filtered_catalog)
    )

def build_explain_sql_system(filtered_catalog: Dict[str, Any]) -> str:
    return (
        "Explain an input SQL in simple English, validate it against the catalog, "
        "and suggest improvements. Use only catalog names.\n\n"
        "CATALOG:\n" + dump_catalog_for_prompt(filtered_catalog)
    )

def build_chat_system() -> str:
    return (
        "You are a helpful data assistant. Respond naturally and engagingly to general questions or chit-chat. "
        "Keep responses concise, friendly, and on-topic. If relevant, tie back to data analysis capabilities."
    )

def build_tour_router_system() -> str:
    return (
        "Classify a help/tour-style user question into one subtopic. "
        "Return strict JSON: {\"subtopic\": str} where subtopic ∈ "
        "[\"onboarding\",\"exploration\",\"filters\",\"joins\",\"save_attribute\",\"attribute_conditions\",\"resume\",\"attribute_framework\",\"examples\"] "
        "Pick the closest meaning; infer intent."
    )

def build_tour_writer_system(filtered_catalog: Dict[str, Any], selected_schemas: Optional[List[str]], selected_tables: Optional[List[str]]) -> str:
    lines = []
    for s, body in filtered_catalog.items():
        if s.startswith("__"):
            continue
        tables = [t for t in body.keys() if not t.startswith("__")]
        if not tables:
            continue
        lines.append(f"- schema {s}: " + body.get("__desc", s))
        for t in tables:
            cols = [c["name"] for c in body[t].get("columns", [])]
            lines.append(f"  • {s}.{t} (cols: {', '.join(cols[:8])}{'…' if len(cols)>8 else ''})")
    scope = "\n".join(lines) if lines else "(no tables selected)"

    return (
        "You are a product tour guide for an NL→SQL multi-agent data assistant. "
        "Write a concise, practical, actionable answer tailored to the user's help question (subtopic provided). "
        "Include:\n"
        "  1) What the user can do in this subtopic (short steps)\n"
        "  2) 2–4 example natural language prompts users can type\n"
        "  3) 1–2 short SQL skeletons that align with the selected catalog (no fake tables)\n"
        "  4) How to switch intents & how to resume previous work\n"
        "  5) If attribute-related, explain the required 3-column framework exactly:\n"
        "     - cust_key_value (10 digit only), att_val_text (string), as_of_date (string)\n"
        "     - Show how to choose mapping: “use X for cust_key_value, Y for att_val_text, Z for as_of_date”\n"
        "     - Mention that only these three columns will be returned when saving attributes\n"
        "Keep it under ~250–350 words, use bullet points where it helps, and never invent table/column names.\n\n"
        "CATALOG IN SCOPE:\n" + scope + "\n\n"
        f"Selected schemas: {selected_schemas or 'all'}\n"
        f"Selected tables: {selected_tables or 'all in selected schemas'}\n"
    )

# =========================
# “Did you mean…?” helpers
# =========================
def update_thread_snapshot(thread: Dict[str, Any], *, last_user: str = "", last_expl: str = "", last_sql: str = ""):
    thread.setdefault("summary", "")
    thread.setdefault("breadcrumbs", [])
    if last_user:
        thread["breadcrumbs"].append({"type":"user", "text": last_user[-300:]})
    if last_expl:
        thread["breadcrumbs"].append({"type":"assistant", "text": last_expl[-300:]})
    if last_sql:
        thread.setdefault("artifacts", {}).setdefault("last_sql", last_sql)
    if len(thread["breadcrumbs"]) > 12:
        thread["breadcrumbs"] = thread["breadcrumbs"][-12:]
    thread["summary"] = (last_expl or thread.get("summary") or "")[:400]

def _all_tables_with_desc(cat: Dict[str, Any]) -> List[Tuple[str,str]]:
    out = []
    for s, body in cat.items():
        if s.startswith("__"): continue
        for t, meta in body.items():
            if t.startswith("__"): continue
            fq = f"{s}.{t}"
            desc = meta.get("__desc", fq)
            out.append((fq, desc))
    return out

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

def _token_set(s: str) -> Set[str]:
    return set(_norm(s).split())

def _score_did_you_mean(query: str, name: str, desc: str) -> float:
    qn = _norm(query); nn = _norm(name); dn = _norm(desc)
    seq = SequenceMatcher(None, qn, nn).ratio()
    q_tokens = _token_set(query); n_tokens = _token_set(name) | _token_set(desc)
    jacc = len(q_tokens & n_tokens) / (len(q_tokens | n_tokens) or 1)
    return 0.65*seq + 0.35*jacc

def did_you_mean_tables(query: str, cat: Dict[str, Any], k: int = 5) -> List[str]:
    candidates = _all_tables_with_desc(cat)
    scored = sorted(
        ((fq, _score_did_you_mean(query, fq, desc)) for fq, desc in candidates),
        key=lambda x: x[1],
        reverse=True
    )
    return [fq for fq, _ in scored[:k]]

# =========================
# Minimal agent framework
# =========================
class State(TypedDict, total=False):
    user_query: str
    catalog: Dict[str, Any]
    history: List[Dict[str,str]]
    thread: Dict[str, Any]

    # intent
    intent: Optional[str]
    intent_params: Dict[str, Any]

    # artifacts
    plan_json: Dict[str, Any]
    revised_prompt: Optional[str]
    english_dsl: Optional[str]
    sql: Optional[str]
    explanation: Optional[str]
    lint_issues: List[str]
    rows: Optional[List[Dict[str, Any]]]
    table_cols: List[str]
    did_you_mean: List[str]
    attr_candidates: Dict[str, List[str]]

    # control
    route: Literal["intent","tour","planner","coder","linter","executor","fixer","schema_qa","explain_sql","modify_sql","context_update","attribute_save","reset","chat","resume","adopt_sql","end"]
    tries: int
    trace: List[str]

@dataclass
class Tools:
    execute_sql: Callable[[str], Tuple[List[str], List[List[Any]], Optional[str]]]
    llm_chat: Callable[[List[Dict[str,str]], float], Optional[str]]
    llm_json: Callable[[str,str], Optional[Dict[str,Any]]]
    sql_system: Callable[[Dict[str,Any]], str]
    planner_system: Callable[[Dict[str,Any]], str]
    schema_qa_system: Callable[[Dict[str,Any]], str]
    explain_sql_system: Callable[[Dict[str,Any]], str]
    chat_system: Callable[[], str]

class Agent:
    name: str
    def run(self, state: State, tools: Tools) -> State:
        raise NotImplementedError

class AgentGraph:
    def __init__(self):
        self.nodes: Dict[str, Agent] = {}
        self.entry: str = "intent"
        self.router: Callable[[State], str] = lambda st: st.get("route","end")
    def add_node(self, name: str, agent: Agent):
        self.nodes[name] = agent
    def set_entry(self, name: str): self.entry = name
    def set_router(self, fn: Callable[[State], str]): self.router = fn
    def invoke(self, state: State, tools: Tools) -> State:
        current = self.entry
        while True:
            if current == "end" or state.get("route") == "end":
                break
            agent = self.nodes.get(current)
            if not agent:
                state["trace"].append(f"Error: No agent for {current}")
                break
            state["trace"].append(f"[{agent.name}] start")
            try:
                state = agent.run(state, tools)
            except Exception as e:
                state["trace"].append(f"[{agent.name}] error: {e}")
                state["explanation"] = f"Internal error in {agent.name}: {e}"
                state["route"] = "end"
                break
            state["trace"].append(f"[{agent.name}] end -> {state.get('route')}")
            current = self.router(state)
        return state

# =========================
# Agents
# =========================
class IntentAgent(Agent):
    name = "Intent"
    def run(self, state: State, tools: Tools) -> State:
        system = build_intent_system()
        recent_history = "\n".join([f"{msg['role']}: {msg['content'][-200:]}" for msg in state["history"][-4:]])
        user = f"User message:\n{state.get('user_query','')}\n\n" \
               f"Recent history:\n{recent_history}\n\n" \
               f"Active thread title: {state['thread'].get('title','(none)')}\n" \
               f"Active thread has last_sql? {bool(state['thread'].get('artifacts',{}).get('last_sql'))}"
        obj = tools.llm_json(system, user) or {"intent":"unknown","thread_title":"Conversation","params":{}}
        state["intent"] = obj.get("intent","unknown")
        state["intent_params"] = obj.get("params",{})
        maybe_title = obj.get("thread_title") or state["thread"].get("title") or "Conversation"
        state["thread"]["title"] = maybe_title[:120]

        nxt = {
            "tour": "tour",
            "nl2sql": "planner",
            "modify_sql": "modify_sql",
            "explain_sql": "explain_sql",
            "adopt_sql": "adopt_sql",
            "schema_qa": "schema_qa",
            "context_update": "context_update",
            "attribute_save": "attribute_save",
            "resume": "resume",
            "reset": "reset",
            "chat": "chat",
        }.get(state["intent"], "end")
        state["route"] = nxt
        return state

class ResetAgent(Agent):
    name = "Reset"
    def run(self, state: State, tools: Tools) -> State:
        state["explanation"] = "Starting a fresh analysis thread. Previous threads remain available. What would you like to explore next?"
        state["route"] = "end"
        return state

class TourAgent(Agent):
    name = "Tour"
    def run(self, state: State, tools: Tools) -> State:
        subtopic_obj = tools.llm_json(build_tour_router_system(), state.get("user_query","")) or {}
        subtopic = subtopic_obj.get("subtopic", "onboarding")

        writer_system = build_tour_writer_system(
            filtered_catalog=state["catalog"],
            selected_schemas=state.get("__selected_schemas__"),
            selected_tables=state.get("__selected_tables__"),
        )

        user = (
            "Subtopic: " + subtopic + "\n"
            "Write the guidance now. Make the examples use real tables/columns from the catalog above.\n"
            "If discussing attributes, show the exact mapping phrase format the user can send.\n"
            "Also mention that the user can type 'show where I left' to resume."
        )
        text = tools.llm_chat([{"role":"system","content": writer_system},
                               {"role":"user","content": user}], 0.3) \
               or "This assistant converts natural language to SQL, validates, runs, and returns results."

        state["explanation"] = text.strip()
        update_thread_snapshot(
            state["thread"],
            last_user=state.get("user_query",""),
            last_expl=state["explanation"]
        )
        state["route"] = "end"
        return state

class PlannerAgent(Agent):
    name = "Planner"
    def run(self, state: State, tools: Tools) -> State:
        system = tools.planner_system(state["catalog"])
        recent_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["history"][-3:] if msg["role"] == "user"])
        user = f"User request:\n{state.get('user_query','')}\n\nRecent context:\n{recent_history}"
        plan = tools.llm_json(system, user) or {}
        state["plan_json"] = plan
        state["route"] = "coder"
        return state

class CoderAgent(Agent):
    name = "Coder"
    def run(self, state: State, tools: Tools) -> State:
        system = tools.sql_system(state["catalog"])
        plan_snippet = json.dumps(state.get("plan_json", {}), ensure_ascii=False)
        last_sql = state["thread"].get("artifacts",{}).get("last_sql","")
        recent_history = state["history"][-6:]
        messages = [{"role":"system","content": system}] + recent_history + [
            {"role":"user","content": f"User request:\n{state.get('user_query','')}\n\n"
                                      f"Prior SQL in this thread (if useful):\n```\n{last_sql}\n```\n"
                                      f"Proposed Plan JSON:\n{plan_snippet}\n"}
        ]
        raw = tools.llm_chat(messages, 0.2) or ""
        revised = parse_block(raw, BLOCKS["rew_start"], BLOCKS["rew_end"]) or ""
        english = parse_block(raw, BLOCKS["dsl_start"], BLOCKS["dsl_end"]) or ""
        sql     = parse_block(raw, BLOCKS["sql_start"], BLOCKS["sql_end"]) or salvage_sql_from_code(raw) or ""
        expl    = parse_block(raw, BLOCKS["exp_start"], BLOCKS["exp_end"]) or ""
        state.update({"revised_prompt": revised, "english_dsl": english, "sql": sql, "explanation": expl})
        state["route"] = "linter"
        return state

class ModifySQLAgent(Agent):
    name = "ModifySQL"
    def run(self, state: State, tools: Tools) -> State:
        last_sql = state["thread"].get("artifacts",{}).get("last_sql","")
        if not last_sql:
            state["explanation"] = "There is no previous SQL in this thread to modify. You can ask for a new query instead."
            state["route"] = "end"
            return state

        system = tools.sql_system(state["catalog"])
        recent_history = state["history"][-6:]
        messages = [{"role":"system","content": system}] + recent_history + [
            {"role":"user","content":
                f"Modify the existing SQL per the user's new instruction.\n\n"
                f"Existing SQL:\n```\n{last_sql}\n```\n"
                f"New instruction:\n{state.get('user_query','')}\n"}
        ]
        raw = tools.llm_chat(messages, 0.2) or ""
        revised = parse_block(raw, BLOCKS["rew_start"], BLOCKS["rew_end"]) or ""
        english = parse_block(raw, BLOCKS["dsl_start"], BLOCKS["dsl_end"]) or ""
        sql     = parse_block(raw, BLOCKS["sql_start"], BLOCKS["sql_end"]) or salvage_sql_from_code(raw) or ""
        expl    = parse_block(raw, BLOCKS["exp_start"], BLOCKS["exp_end"]) or ""
        state.update({"revised_prompt": revised, "english_dsl": english, "sql": sql, "explanation": expl})
        state["route"] = "linter"
        return state

class LinterAgent(Agent):
    name = "Linter"
    def run(self, state: State, tools: Tools) -> State:
        sql = (state.get("sql") or "").strip()
        issues: List[str] = []
        dym: List[str] = []

        if not sql:
            issues.append("No SQL produced.")
        elif not sql.lower().startswith(("select","with")):
            issues.append("Only SELECT/CTE allowed.")
        if sql.count(";") > 1:
            issues.append("Multiple statements detected.")

        m = re.search(r"from\s+([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)", sql, re.IGNORECASE)
        if not m:
            issues.append("Missing explicit schema.table in FROM.")
        else:
            sch, tab = m.group(1), m.group(2)
            if sch not in state["catalog"] or tab not in state["catalog"].get(sch, {}):
                issues.append(f"Unknown table {sch}.{tab} in current context.")
                dym = did_you_mean_tables(f"{sch}.{tab}", state["catalog"], k=5)

        state["lint_issues"] = issues
        state["did_you_mean"] = dym
        state["route"] = "executor" if not issues else "fixer"
        return state

class ExecutorAgent(Agent):
    name = "Executor"
    def run(self, state: State, tools: Tools) -> State:
        sql = state.get("sql") or ""
        cols, rows, err = tools.execute_sql(sql)
        if err:
            state["lint_issues"] = [f"Executor error: {err}"]
            state["route"] = "fixer"
            return state

        state["table_cols"] = cols
        state["rows"] = rows

        state["thread"].setdefault("artifacts", {})["last_sql"] = sql
        state["thread"]["artifacts"]["last_dsl"] = state.get("english_dsl", "")
        state["thread"]["artifacts"]["last_expl"] = state.get("explanation", "")

        if state.get("intent") == "attribute_save":
            required = ["cust_key_value", "att_val_text", "as_of_date"]
            if cols != required:
                state["lint_issues"] = [f"Attribute export must return exactly {required}, got {cols}."]
                state["route"] = "fixer"
                return state
            state["attribute_ready"] = True
            state["attribute_sql"] = sql

        update_thread_snapshot(
            state["thread"],
            last_user=state.get("user_query", ""),
            last_expl=state.get("explanation", ""),
            last_sql=sql
        )

        state["route"] = "end"
        return state

class ResumeAgent(Agent):
    name = "Resume"
    def run(self, state: State, tools: Tools) -> State:
        conv_threads = state["__conv_threads__"]
        items = []
        for t in conv_threads:
            items.append({
                "id": t["id"],
                "title": t.get("title",""),
                "has_last_sql": bool(t.get("artifacts",{}).get("last_sql")),
                "summary": t.get("summary",""),
                "tail": [b.get("text","") for b in t.get("breadcrumbs", [])[-2:]]
            })

        system = (
            "Given prior threads and the user's 'resume' request, choose the SINGLE best thread. "
            "Prefer threads that match the request; if user implies 'my data' or 'where I left off analyzing', "
            "prefer those with last_sql.\n"
            "Return strict JSON: {\"thread_id\": \"...\", \"reason\": \"...\", "
            "\"next_intent_hint\": \"nl2sql|modify_sql|schema_qa|tour|attribute_save|unknown\"}"
        )
        user = json.dumps({
            "resume_request": state.get("user_query",""),
            "candidates": items
        }, ensure_ascii=False)

        obj   = tools.llm_json(system, user) or {}
        t_id  = obj.get("thread_id")
        reason= obj.get("reason","")
        hint  = obj.get("next_intent_hint","unknown")

        if not t_id:
            with_sql = [t for t in conv_threads if t.get("artifacts",{}).get("last_sql")]
            target   = (with_sql[-1] if with_sql else conv_threads[-1])
        else:
            target   = next((t for t in conv_threads if t["id"] == t_id), conv_threads[-1])

        state["intent_params"] = {"resume_thread_id": target["id"], "reason": reason, "hint": hint}

        art   = target.get("artifacts", {})
        last_sql = (art.get("last_sql") or "").strip()
        last_expl= art.get("last_expl") or ""
        last_dsl = art.get("last_dsl") or ""

        if last_sql:
            state["sql"] = last_sql
            state["revised_prompt"] = "Resume: re-running your last analysis query."
            state["english_dsl"] = last_dsl
            state["explanation"] = (
                # "Resuming your previous work.\n" +
                    "\n" +
                (last_expl or "Re-running your last query and showing the latest results.")
            )
            update_thread_snapshot(
                target,
                last_user=state.get("user_query",""),
                last_expl="(Resume) " + (last_expl or "Re-ran last query"),
                last_sql=last_sql
            )
            state["route"] = "executor"
            return state

        state["explanation"] = (
            "Resuming your previous work.\n"
            + (reason or "Switched back to your earlier thread.")
            + "\nYou can continue with: “show customer data”, “add only US customers”, "
              "“limit to last 30 days”, or “save as attributes”."
        )
        update_thread_snapshot(
            target,
            last_user=state.get("user_query",""),
            last_expl="(Resume) No prior SQL; ready to continue."
        )
        state["route"] = "end"
        return state

class FixerAgent(Agent):
    name = "Fixer"
    def run(self, state: State, tools: Tools) -> State:
        tries = state.get("tries", 0)
        if any("Unknown table" in i for i in state.get("lint_issues", [])):
            state["route"] = "end"
            return state
        if tries >= 3:
            state["route"] = "end"
            return state

        system = tools.sql_system(state["catalog"])
        issues = "\n".join(f"- {i}" for i in state.get("lint_issues", []))
        recent_history = state["history"][-4:]
        messages = [{"role":"system","content": system}] + recent_history + [
            {"role":"user","content":
                "The previous SQL has issues:\n" + issues + "\n\n"
                "Original SQL:\n```\n" + (state.get('sql') or '') + "\n```\n"
                "Return the same exact blocks with a corrected single statement."}
        ]
        raw = tools.llm_chat(messages, 0.2) or ""
        revised = parse_block(raw, BLOCKS["rew_start"], BLOCKS["rew_end"]) or state.get("revised_prompt","")
        english = parse_block(raw, BLOCKS["dsl_start"], BLOCKS["dsl_end"]) or state.get("english_dsl","")
        sql     = parse_block(raw, BLOCKS["sql_start"], BLOCKS["sql_end"]) or salvage_sql_from_code(raw) or state.get("sql","")
        expl    = parse_block(raw, BLOCKS["exp_start"], BLOCKS["exp_end"]) or state.get("explanation","")

        state.update({"revised_prompt": revised, "english_dsl": english, "sql": sql, "explanation": expl})
        state["tries"] = tries + 1
        state["route"] = "linter"
        return state

class SchemaQAAgent(Agent):
    name = "SchemaQA"
    def run(self, state: State, tools: Tools) -> State:
        system = tools.schema_qa_system(state["catalog"])
        recent_history = state["history"][-4:]
        messages = [{"role":"system","content":system}] + recent_history + [{"role":"user","content":state.get("user_query","")}]
        text = tools.llm_chat(messages, 0.2) or ""
        state["explanation"] = text.strip()
        state["did_you_mean"] = did_you_mean_tables(state.get("user_query",""), state["catalog"], k=5)

        update_thread_snapshot(
            state["thread"],
            last_user=state.get("user_query",""),
            last_expl=state["explanation"]
        )
        state["route"] = "end"
        return state

class ExplainSQLAgent(Agent):
    name = "ExplainSQL"
    def run(self, state: State, tools: Tools) -> State:
        system = tools.explain_sql_system(state["catalog"])
        user = "Explain this SQL in simple English, validate columns/tables, and suggest improvements:\n" + state.get("user_query","")
        recent_history = state["history"][-4:]
        messages = [{"role":"system","content":system}] + recent_history + [{"role":"user","content":user}]
        text = tools.llm_chat(messages, 0.2) or ""
        state["explanation"] = text.strip()
        state["route"] = "end"
        return state

class ContextUpdateAgent(Agent):
    name = "ContextUpdate"
    def run(self, state: State, tools: Tools) -> State:
        system = (
            "Propose a context update. Return strict JSON: {\"schemas\": [\"...\"], \"tables\": [\"schema.table\", ...], \"message\": \"...\"}. "
            f"Choose only from names available below.\n\nCATALOG:\n" + dump_catalog_for_prompt(state["catalog"])
        )
        user = state.get("user_query","")
        obj = tools.llm_json(system, user) or {"schemas": [], "tables": [], "message": "Context updated."}
        state["intent_params"] = obj
        state["explanation"] = obj.get("message","Context updated.")
        state["route"] = "end"
        return state

class AttributeAgent(Agent):
    name = "AttributeSave"
    def _suggest_example(self, cols: List[str]) -> Dict[str, Optional[str]]:
        lc = {c.lower(): c for c in cols}
        def find_any(candidates):
            for k in candidates:
                if k in lc:
                    return lc[k]
            return None
        example = {
            "cust_key_value": find_any(["cust_id","customer_id","id","cci","customer_key","cust_key","customer"]),
            "att_val_text":   find_any(["seg","segment","status","cat","name","sku","eml","email","cntry","country"]),
            "as_of_date":     find_any(["ord_dt","signup_dt","start_dt","end_dt","touch_ts","date","dt"])
        }
        return example

    def run(self, state: State, tools: Tools) -> State:
        last_sql = state["thread"].get("artifacts",{}).get("last_sql","")
        if not last_sql:
            state["explanation"] = (
                "I don’t have a prior discovery result in this thread. "
                "Run a discovery query first, then ask me to save it as attributes."
            )
            state["route"] = "end"
            return state

        probe_sql = f"SELECT * FROM ({last_sql.rstrip(';')}) AS base LIMIT 0"
        cols, _, err = tools.execute_sql(probe_sql)
        if err:
            state["explanation"] = f"Couldn’t inspect the last query’s columns: {err}"
            state["route"] = "end"
            return state

        user_map = extract_attribute_mapping(state.get("user_query",""), cols)
        if user_map and all(user_map.get(k) for k in ("cust_key_value","att_val_text","as_of_date")):
            mapping = user_map
            transformed_sql = build_attribute_sql(last_sql, mapping)
            state["sql"] = transformed_sql
            state["revised_prompt"] = "Save discovery as attribute dataset (cust_key_value, att_val_text, as_of_date)."
            state["english_dsl"] = "Standardized attribute export using user-selected mapping."
            state["explanation"] = "Generated the attribute dataset with your chosen columns."
            state["intent_params"] = {"attr_mapping": mapping}
            state["route"] = "linter"
            return state

        example = self._suggest_example(cols)
        state["attr_candidates"] = {
            "cust_key_value": cols,
            "att_val_text":   cols,
            "as_of_date":     cols,
        }
        msg_lines = [
            "Attribute framework required in 3-column format (cust_key_value (10 digit only), att_val_text (string), as_of_date (string)).",
            "",
            "You can select the mapping by saying, for example:",
        ]
        eg_id   = example.get("cust_key_value") or (cols[0] if cols else "cust_id")
        eg_val  = example.get("att_val_text")   or (cols[1] if len(cols) > 1 else "seg")
        eg_date = example.get("as_of_date")     or (cols[2] if len(cols) > 2 else "signup_dt")
        msg_lines.append(f'“use {eg_id} for cust_key_value, {eg_val} for att_val_text, {eg_date} for as_of_date and remove other columns”.')
        msg_lines.append("")
        msg_lines.append("You can pick any of the columns from your last result:")
        msg_lines.append(", ".join(cols) if cols else "(no columns found)")
        state["explanation"] = "\n".join(msg_lines)
        state["intent_params"] = {"attr_example": example}
        state["route"] = "end"
        return state

class ChatAgent(Agent):
    name = "Chat"
    def run(self, state: State, tools: Tools) -> State:
        system = tools.chat_system()
        recent_history = state["history"][-8:]
        messages = [{"role":"system","content":system}] + recent_history + [{"role":"user","content":state.get("user_query","")}]
        text = tools.llm_chat(messages, 0.7) or "I'm here to help with data queries! What can I do for you?"
        state["explanation"] = text.strip()
        update_thread_snapshot(
            state["thread"],
            last_user=state.get("user_query",""),
            last_expl=state["explanation"]
        )
        state["route"] = "end"
        return state

class AdoptSQLAgent(Agent):
    """
    NEW: Adopt a pasted SQL as the working query.
    - Validates it's read-only
    - Executes it with LIMIT safety
    - Saves as last_sql in the thread
    - Produces a short explanation via LLM
    """
    name = "AdoptSQL"
    def run(self, state: State, tools: Tools) -> State:
        pasted = state.get("user_query","")
        # Try to extract SQL if wrapped in code fence or accompanied by prose
        sql = salvage_sql_from_code(pasted) or pasted.strip()

        # Basic guard; Executor will re-check:
        if not re.match(r'(?is)^\s*(WITH|SELECT)\b', sql or ""):
            state["explanation"] = "I can only adopt read-only queries (WITH/SELECT). Please paste a SELECT or WITH..SELECT."
            state["route"] = "end"
            return state

        # Short explanation using the explain system
        expl_sys = build_explain_sql_system(state["catalog"])
        expl_msg = tools.llm_chat(
            [{"role":"system","content": expl_sys},
             {"role":"user","content": f"Explain in simple terms what this SQL returns:\n```sql\n{sql}\n```"}],
            0.2
        ) or "Adopted your SQL and executed it."

        state["sql"] = sql
        state["revised_prompt"] = "Adopt user-supplied SQL as working query."
        state["english_dsl"] = "User provided SQL; explanation generated."
        state["explanation"] = "Starting from your SQL.\n" + expl_msg.strip()
        state["route"] = "executor"
        return state

# =========================
# Build the graph
# =========================
graph = AgentGraph()
graph.add_node("intent",         IntentAgent())
graph.add_node("reset",          ResetAgent())
graph.add_node("tour",           TourAgent())
graph.add_node("planner",        PlannerAgent())
graph.add_node("coder",          CoderAgent())
graph.add_node("modify_sql",     ModifySQLAgent())
graph.add_node("linter",         LinterAgent())
graph.add_node("executor",       ExecutorAgent())
graph.add_node("fixer",          FixerAgent())
graph.add_node("schema_qa",      SchemaQAAgent())
graph.add_node("explain_sql",    ExplainSQLAgent())
graph.add_node("context_update", ContextUpdateAgent())
graph.add_node("attribute_save", AttributeAgent())
graph.add_node("chat",           ChatAgent())
graph.add_node("resume",         ResumeAgent())
graph.add_node("adopt_sql",      AdoptSQLAgent())

graph.set_entry("intent")
graph.set_router(lambda st: st.get("route","end"))

TOOLS = Tools(
    execute_sql=run_sql_safely,
    llm_chat=oai_chat,
    llm_json=llm_json,
    sql_system=build_sql_system,
    planner_system=build_planner_system,
    schema_qa_system=build_schema_qa_system,
    explain_sql_system=build_explain_sql_system,
    chat_system=build_chat_system,
)

# =========================
# Flask API
# =========================
app = Flask(__name__)
CORS(app)

def trim_history(hist: List[Dict[str,str]]) -> List[Dict[str,str]]:
    return hist if len(hist) <= MAX_TURNS * 2 else hist[-MAX_TURNS * 2:]

def _new_thread(title: str) -> Dict[str, Any]:
    return {"id": str(uuid.uuid4()), "title": title or "Conversation", "artifacts": {}}

def _maybe_debug_trace(req, trace: List[str]) -> Optional[Dict[str, Any]]:
    want = (os.getenv("DEBUG_TRACE","0") == "1") or (req.args.get("debug") == "1")
    return {"trace": trace} if want else None

# ---- Mock attribute examples registry (for tour/ask/explain) ----
def seed_mock_attributes():
    con.execute("""
        CREATE TABLE IF NOT EXISTS attributes_registry (
            id UUID,
            name TEXT,
            sql_text TEXT,
            mapping JSON,
            tags JSON,
            created_at TIMESTAMP
        )
    """)
    # Seed a few examples if table empty
    rows = con.execute("SELECT COUNT(*) FROM attributes_registry").fetchone()
    if rows and rows[0] == 0:
        examples = [
            ("High Value Buyers (L30D, USD>100)",
             "SELECT cust_id AS cust_key_value, CAST('high_value_l30d' AS TEXT) AS att_val_text, CAST(CURRENT_DATE AS TEXT) AS as_of_date FROM sales.ord_hdr WHERE curr='USD' AND tot_amt>100 AND ord_dt >= CURRENT_DATE - INTERVAL '30 day' GROUP BY cust_id",
             {"cust_key_value": "cust_id", "att_val_text": "(literal:'high_value_l30d')", "as_of_date": "(CURRENT_DATE)"},
             ["example","buyers"]
            ),
            ("New Signups (L7D)",
             "SELECT cust_id AS cust_key_value, CAST('new_signup_l7d' AS TEXT) AS att_val_text, CAST(CURRENT_DATE AS TEXT) AS as_of_date FROM sales.cust_dim WHERE signup_dt >= CURRENT_DATE - INTERVAL '7 day'",
             {"cust_key_value": "cust_id", "att_val_text": "(literal:'new_signup_l7d')", "as_of_date": "(CURRENT_DATE)"},
             ["example","signups"]
            ),
            ("Email Domain: Gmail",
             "SELECT cust_id AS cust_key_value, CAST('gmail' AS TEXT) AS att_val_text, CAST(CURRENT_DATE AS TEXT) AS as_of_date FROM sales.cust_dim WHERE eml LIKE '%@gmail.%'",
             {"cust_key_value": "cust_id", "att_val_text": "(literal:'gmail')", "as_of_date": "(CURRENT_DATE)"},
             ["example","email"]
            )
        ]
        for name, sql_text, mapping, tags in examples:
            con.execute(
                "INSERT INTO attributes_registry VALUES (?, ?, ?, ?, ?, ?)",
                [str(uuid.uuid4()), name, sql_text, json.dumps(mapping), json.dumps(tags), datetime.utcnow()]
            )

seed_mock_attributes()

@app.post("/api/chat/start")
def chat_start():
    body = request.get_json(silent=True) or {}
    sel_schemas = body.get("schemas")
    sel_tables  = body.get("tables")

    if sel_schemas and sel_tables:
        valid = []
        for fq in sel_tables:
            if "." not in fq:
                continue
            s, t = fq.split(".", 1)
            if s in sel_schemas:
                valid.append(fq)
        sel_tables = valid

    filtered = filter_catalog(CATALOG, sel_schemas, sel_tables)
    conv_id = str(uuid.uuid4())

    hello = (
        "Context set. I’ll use only the selected schemas/tables.\n"
        "Ask in your own words — I will detect your intent dynamically (tour, build SQL, modify SQL, schema Q&A, attribute save, adopt SQL, resume, etc.)."
    )
    CONVS[conv_id] = {
        "id": conv_id,
        "history": [{"role":"assistant","content": hello}],
        "selected_schemas": sel_schemas,
        "selected_tables": sel_tables,
        "catalog": filtered,
        "threads": [],
        "active_thread_id": None
    }
    th = _new_thread("Conversation")
    CONVS[conv_id]["threads"].append(th)
    CONVS[conv_id]["active_thread_id"] = th["id"]

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
        "selected_tables": conv.get("selected_tables"),
        "threads": [{"id": t["id"], "title": t["title"]} for t in conv.get("threads", [])],
        "active_thread_id": conv.get("active_thread_id")
    })

@app.post("/api/chat/message")
def chat_message():
    body = request.get_json(force=True)
    conv_id = body.get("conversation_id")
    user_msg = (body.get("message") or "").strip()
    if not conv_id or conv_id not in CONVS:
        return jsonify({"error": "Conversation not found. Start one at /api/chat/start."}), 400
    if not user_msg:
        return jsonify({"error": "Message is empty."}), 400
    if not OPENAI_API_KEY:
        return jsonify({"assistant":"Server missing OPENAI_API_KEY.",
                        "revised_prompt":"", "english_dsl":"", "sql":"", "explanation":"",
                        "table":{"columns":[],"rows":[]}, "did_you_mean": [], "attr_candidates": {},
                        "debug": {"trace":[]}, "error":"Missing OPENAI_API_KEY"}), 500

    conv = CONVS[conv_id]
    conv["history"].append({"role":"user","content": user_msg})
    conv["history"] = trim_history(conv["history"])

    active_id = conv.get("active_thread_id")
    thread = next((t for t in conv["threads"] if t["id"] == active_id), None)
    if thread is None:
        thread = _new_thread("Conversation")
        conv["threads"].append(thread)
        conv["active_thread_id"] = thread["id"]

    state: State = {
        "user_query": user_msg,
        "catalog": conv["catalog"],
        "history": conv["history"],
        "thread": thread,
        "__conv_threads__": conv["threads"],
        "__selected_schemas__": conv.get("selected_schemas"),
        "__selected_tables__": conv.get("selected_tables"),
        "intent": None,
        "intent_params": {},
        "plan_json": {},
        "revised_prompt": None,
        "english_dsl": None,
        "sql": None,
        "explanation": None,
        "lint_issues": [],
        "rows": None,
        "table_cols": [],
        "did_you_mean": [],
        "attr_candidates": {},
        "route": "intent",
        "tries": 0,
        "trace": [],
    }
    final = graph.invoke(state, TOOLS)

    # resume → switch active thread id
    if final.get("intent") == "resume":
        params = final.get("intent_params", {})
        target_id = params.get("resume_thread_id")
        reason = params.get("reason", "")
        if target_id and any(t["id"] == target_id for t in conv["threads"]):
            conv["active_thread_id"] = target_id
        if not final.get("sql"):
            msg = final.get("explanation") or "Resumed your previous work."
            conv["history"].append({"role": "assistant", "content": msg})
            return jsonify({
                "assistant": msg,
                "revised_prompt": "",
                "english_dsl": "",
                "sql": "",
                "explanation": msg,
                "table": {"columns": [], "rows": []},
                "did_you_mean": [],
                "attr_candidates": {},
                "debug": _maybe_debug_trace(request, final["trace"]),
                "error": None
            })

    # reset → new thread
    if final.get("intent") == "reset":
        new_th = _new_thread(final["thread"].get("title") or "New Analysis")
        conv["threads"].append(new_th)
        conv["active_thread_id"] = new_th["id"]
        msg = final.get("explanation") or "Started a fresh analysis."
        conv["history"].append({"role":"assistant","content": msg})
        return jsonify({
            "assistant": msg,
            "revised_prompt": "",
            "english_dsl": "",
            "sql": "",
            "explanation": msg,
            "table": {"columns": [], "rows": []},
            "did_you_mean": [],
            "attr_candidates": {},
            "debug": _maybe_debug_trace(request, final["trace"]),
            "error": None
        })

    # apply context update
    if final.get("intent") == "context_update":
        params = final.get("intent_params", {})
        new_schemas = params.get("schemas")
        new_tables  = params.get("tables")
        if new_schemas or new_tables:
            filtered = filter_catalog(CATALOG, new_schemas, new_tables)
            conv["catalog"] = filtered
            conv["selected_schemas"] = new_schemas
            conv["selected_tables"]  = new_tables
        msg = final.get("explanation") or "Context updated."
        conv["history"].append({"role":"assistant","content": msg})
        return jsonify({
            "assistant": msg,
            "revised_prompt": "",
            "english_dsl": "",
            "sql": "",
            "explanation": msg,
            "table": {"columns": [], "rows": []},
            "did_you_mean": [],
            "attr_candidates": {},
            "debug": _maybe_debug_trace(request, final["trace"]),
            "error": None
        })

    # tour / schema_qa / explain_sql / chat / attribute_save (text-only)
    if final.get("intent") in ("tour", "schema_qa", "explain_sql", "chat", "attribute_save") and not final.get("sql"):
        assistant_text = final.get("explanation") or "Done."
        conv["history"].append({"role": "assistant", "content": assistant_text})
        # Suggestions for UI (optional; e.g., after adopt or tour you can add variants)
        suggestions = []
        if final.get("intent") == "tour":
            suggestions = [
                "show customer data",
                "group revenue by segment last 30 days",
                "save as attributes",
                "resume my last data",
            ]
        return jsonify({
            "assistant": assistant_text,
            "revised_prompt": final.get("revised_prompt") or "",
            "english_dsl": final.get("english_dsl") or "",
            "sql": "",
            "explanation": assistant_text,
            "table": {"columns": [], "rows": []},
            "did_you_mean": final.get("did_you_mean", []),
            "attr_candidates": final.get("attr_candidates", {}),
            "attr_example": final.get("intent_params", {}).get("attr_example"),
            "suggestions": suggestions,
            "debug": _maybe_debug_trace(request, final["trace"]),
            "error": None
        })

    # NL→SQL / modify_sql / attribute_save / adopt_sql go through executor
    sql_out = (final.get("sql") or "").strip()
    expl    = final.get("explanation") or ""
    cols    = final.get("table_cols") or []
    rows    = final.get("rows") or []
    issues  = final.get("lint_issues") or []
    didym   = final.get("did_you_mean", [])
    attr_c  = final.get("attr_candidates", {})

    if sql_out and rows:
        assistant_text = (expl + "\n" if expl else "") + f"Returned {len(rows)} rows."
        conv["history"].append({"role":"assistant","content": assistant_text})
        return jsonify({
            "assistant": assistant_text,
            "revised_prompt": final.get("revised_prompt") or "",
            "english_dsl": final.get("english_dsl") or "",
            "sql": sql_out,
            "explanation": expl,
            "table": {"columns": cols, "rows": rows},
            "did_you_mean": [],
            "attr_candidates": attr_c,
            "attr_mapping": final.get("intent_params", {}).get("attr_mapping"),
            "suggestions": ["add only US customers","limit to last 30 days","save as attributes"],
            "debug": _maybe_debug_trace(request, final["trace"]),
            "error": None
        })

    if sql_out and not rows and not issues:
        assistant_text = (expl + "\n" if expl else "") + "No rows returned."
        conv["history"].append({"role":"assistant","content": assistant_text})
        return jsonify({
            "assistant": assistant_text,
            "revised_prompt": final.get("revised_prompt") or "",
            "english_dsl": final.get("english_dsl") or "",
            "sql": sql_out,
            "explanation": expl,
            "table": {"columns": cols, "rows": rows},
            "did_you_mean": [],
            "attr_candidates": attr_c,
            "suggestions": ["add a filter","remove a filter","save as attributes"],
            "debug": _maybe_debug_trace(request, final["trace"]),
            "error": None
        })

    err_msg = "; ".join(issues) if issues else "No valid SQL was produced."
    assistant_text = ((expl + "\n") if expl else "") + f"SQL error: {err_msg}\nLet me know if you'd like me to try something else."
    conv["history"].append({"role":"assistant","content": assistant_text})
    return jsonify({
        "assistant": assistant_text,
        "revised_prompt": final.get("revised_prompt") or "",
        "english_dsl": final.get("english_dsl") or "",
        "sql": sql_out,
        "explanation": expl,
        "table": {"columns": [], "rows": []},
        "did_you_mean": didym,
        "attr_candidates": attr_c,
        "suggestions": ["show customer data","help","resume my last data"],
        "debug": _maybe_debug_trace(request, final["trace"]),
        "error": err_msg
    })

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

@app.post("/api/attributes/save")
def save_attribute_query():
    body = request.get_json(force=True) or {}
    name = (body.get("name") or "").strip()
    sql  = (body.get("sql") or "").strip()
    mapping = body.get("mapping") or {}
    tags = body.get("tags") or []

    if not sql:
        return jsonify({"error": "Missing sql"}), 400
    if not name:
        name = f"Attribute Export {datetime.utcnow().isoformat(timespec='seconds')}Z"

    try:
        probe = f"SELECT * FROM ({sql.rstrip(';')}) AS _v LIMIT 0"
        res = con.execute(probe)
        cols = [d[0] for d in res.description]
    except Exception as e:
        return jsonify({"error": f"SQL validation failed: {e}"}), 400

    required = ["cust_key_value", "att_val_text", "as_of_date"]
    if cols != required:
        return jsonify({"error": f"Attribute output must be exactly {required}, got {cols}"}), 400

    con.execute("""
        CREATE TABLE IF NOT EXISTS attributes_registry (
            id UUID,
            name TEXT,
            sql_text TEXT,
            mapping JSON,
            tags JSON,
            created_at TIMESTAMP
        )
    """)
    rec_id = str(uuid.uuid4())
    con.execute(
        "INSERT INTO attributes_registry VALUES (?, ?, ?, ?, ?, ?)",
        [rec_id, name, sql, json.dumps(mapping), json.dumps(tags), datetime.utcnow()]
    )

    return jsonify({"ok": True, "id": rec_id, "name": name})

@app.get("/api/attributes/examples")
def attributes_examples():
    rows = con.execute("SELECT id, name, sql_text, mapping, tags, created_at FROM attributes_registry").fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r[0], "name": r[1], "sql": r[2],
            "mapping": json.loads(r[3]) if r[3] else {},
            "tags": json.loads(r[4]) if r[4] else [],
            "created_at": str(r[5])
        })
    return jsonify({"examples": out})

@app.get("/api/attributes/explain")
def attributes_explain():
    """
    Explain one of the saved/example attributes by id or name.
    """
    q = (request.args.get("id") or "").strip()
    name = (request.args.get("name") or "").strip()
    row = None
    if q:
        row = con.execute("SELECT name, sql_text FROM attributes_registry WHERE id = ?", [q]).fetchone()
    elif name:
        row = con.execute("SELECT name, sql_text FROM attributes_registry WHERE name = ?", [name]).fetchone()
    if not row:
        return jsonify({"error": "Attribute not found"}), 404
    att_name, sql_text = row[0], row[1]
    system = build_explain_sql_system(CATALOG)
    text = oai_chat(
        [{"role":"system","content": system},
         {"role":"user","content": f"Explain what this attribute does and who qualifies:\n```sql\n{sql_text}\n```"}],
        0.2
    ) or f"{att_name}: attribute explanation not available."
    return jsonify({"name": att_name, "explanation": text})

if __name__ == "__main__":
    # pip install flask flask-cors duckdb
    # export OPENAI_API_KEY=sk-...
    app.run(host="0.0.0.0", port=PORT, debug=True)
