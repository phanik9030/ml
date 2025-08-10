# insight_assistant_simple.py
from __future__ import annotations
import os, re, json, uuid, math, difflib
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# OpenAI helpers (GPT-4o + embeddings)
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # inexpensive & solid

try:
    from openai import OpenAI  # pip install openai>=1.30.0
except Exception:
    OpenAI = None  # type: ignore

def _client() -> Optional[OpenAI]:
    if not OPENAI_API_KEY or not OpenAI:
        return None
    return OpenAI(api_key=OPENAI_API_KEY)

def gpt_json(messages: List[Dict[str,str]], temperature=0.2) -> Optional[Dict[str,Any]]:
    cli = _client()
    if not cli:
        return None
    try:
        r = cli.chat.completions.create(
            model="gpt-4o",
            temperature=temperature,
            response_format={"type":"json_object"},
            messages=messages,
        )
        return json.loads(r.choices[0].message.content)
    except Exception:
        return None

def gpt_text(messages: List[Dict[str,str]], temperature=0.2) -> Optional[str]:
    cli = _client()
    if not cli:
        return None
    try:
        r = cli.chat.completions.create(model="gpt-4o", temperature=temperature, messages=messages)
        return r.choices[0].message.content
    except Exception:
        return None

def embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    cli = _client()
    if not cli:
        return None
    try:
        r = cli.embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in r.data]
    except Exception:
        return None

# -----------------------------
# Catalog & Mock Data (expanded descriptions for better embeddings)
# -----------------------------
class Catalog:
    def __init__(self):
        self.schemas = [
            {"name": "public", "description": "DVD rental dataset — films, rentals, customers, inventory. Operational video store data used in analytics examples."},
            {"name": "sales",  "description": "Retail sales — customers, orders, order items, products. Revenue analysis, segmentation, basket-level details."},
            {"name": "mkt",    "description": "Marketing — campaigns, touches (sends), responses. Attribution, channel performance, conversion outcomes."},
        ]
        self.tables = [
            # public
            {"schema":"public","name":"customer","description":"DVD rental customers list (people who rent movies). Profiles with status and store affiliation.",
             "columns":[
                 {"name":"customer_id","type":"int","desc":"Primary key; unique customer id"},
                 {"name":"first_name","type":"text","desc":"Given name"},
                 {"name":"last_name","type":"text","desc":"Family name"},
                 {"name":"store_id","type":"int","desc":"Home store identifier"},
                 {"name":"activebool","type":"bool","desc":"Active status flag (True/False)"},
             ]},
            {"schema":"public","name":"rental","description":"Each rental event (who rented what and when). Transactional history.",
             "columns":[
                 {"name":"rental_id","type":"int","desc":"Primary key"},
                 {"name":"customer_id","type":"int","desc":"FK to public.customer"},
                 {"name":"inventory_id","type":"int","desc":"FK to public.inventory"},
                 {"name":"rental_date","type":"timestamptz","desc":"When rental started"},
                 {"name":"return_date","type":"timestamptz","desc":"When returned"},
             ]},
            {"schema":"public","name":"inventory","description":"Inventory copies of films in stores. Stock keeping for titles per store.",
             "columns":[
                 {"name":"inventory_id","type":"int","desc":"Primary key"},
                 {"name":"film_id","type":"int","desc":"FK to public.film"},
                 {"name":"store_id","type":"int","desc":"Store location id"},
             ]},
            {"schema":"public","name":"film","description":"Movies with MPAA rating and titles. Catalog for rental content.",
             "columns":[
                 {"name":"film_id","type":"int","desc":"Primary key"},
                 {"name":"title","type":"text","desc":"Movie title"},
                 {"name":"rating","type":"text","desc":"MPAA rating (G, PG, PG-13, R)"},
             ]},
            # sales
            {"schema":"sales","name":"customers","description":"Customer master with country and business segment. Demographics for retail buyers.",
             "columns":[
                 {"name":"customer_id","type":"int","desc":"Primary key"},
                 {"name":"first_name","type":"text","desc":"Given name"},
                 {"name":"last_name","type":"text","desc":"Family name"},
                 {"name":"country","type":"text","desc":"Customer country (e.g., US, CA)"},
                 {"name":"segment","type":"text","desc":"Business segment (Retail, SMB, Enterprise)"},
             ]},
            {"schema":"sales","name":"orders","description":"Order headers: customer, date, total amount, order country. Core revenue facts.",
             "columns":[
                 {"name":"order_id","type":"int","desc":"Primary key"},
                 {"name":"customer_id","type":"int","desc":"FK to sales.customers"},
                 {"name":"order_date","type":"date","desc":"When the order was placed"},
                 {"name":"total_amount","type":"numeric","desc":"Order total USD"},
                 {"name":"country","type":"text","desc":"Country of order"},
             ]},
            {"schema":"sales","name":"order_items","description":"Order line items with product, quantity, and unit price. Basket composition.",
             "columns":[
                 {"name":"item_id","type":"int","desc":"Primary key"},
                 {"name":"order_id","type":"int","desc":"FK to sales.orders"},
                 {"name":"product_id","type":"int","desc":"FK to sales.products"},
                 {"name":"qty","type":"int","desc":"Quantity ordered"},
                 {"name":"price","type":"numeric","desc":"Unit price"},
             ]},
            {"schema":"sales","name":"products","description":"Product catalog with SKU, name and category. Item master for orders.",
             "columns":[
                 {"name":"product_id","type":"int","desc":"Primary key"},
                 {"name":"sku","type":"text","desc":"Stock keeping unit"},
                 {"name":"name","type":"text","desc":"Product name"},
                 {"name":"category","type":"text","desc":"Category (Apparel, Footwear, etc.)"},
             ]},
            # mkt
            {"schema":"mkt","name":"campaigns","description":"Marketing campaigns and channels. Campaign metadata for attribution.",
             "columns":[
                 {"name":"campaign_id","type":"int","desc":"Primary key"},
                 {"name":"name","type":"text","desc":"Campaign name"},
                 {"name":"channel","type":"text","desc":"Channel (Email, SMS, Push)"},
             ]},
            {"schema":"mkt","name":"touches","description":"Campaign sends to customers. Delivery log for outbound messages.",
             "columns":[
                 {"name":"touch_id","type":"int","desc":"Primary key"},
                 {"name":"campaign_id","type":"int","desc":"FK to mkt.campaigns"},
                 {"name":"customer_id","type":"int","desc":"FK to sales.customers"},
                 {"name":"touch_date","type":"date","desc":"When the touch happened"},
             ]},
            {"schema":"mkt","name":"responses","description":"Touch outcomes (clicked, purchased). Conversion signals.",
             "columns":[
                 {"name":"response_id","type":"int","desc":"Primary key"},
                 {"name":"touch_id","type":"int","desc":"FK to mkt.touches"},
                 {"name":"outcome","type":"text","desc":"Response outcome"},
             ]},
        ]

    def list_schemas(self) -> List[Dict[str, Any]]:
        return self.schemas

    def list_tables(self, schema: str) -> List[Dict[str, Any]]:
        return [t for t in self.tables if t["schema"] == schema]

    def get_table(self, schema: str, name: str) -> Dict[str, Any]:
        t = next((t for t in self.tables if t["schema"] == schema and t["name"] == name), None)
        if not t:
            raise KeyError(f"Unknown table {schema}.{name}")
        return t

# Mock rows for previews
MOCK_DATA: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "sales": {
        "customers": [
            {"customer_id":1,"first_name":"Ana","last_name":"Smith","country":"US","segment":"Retail"},
            {"customer_id":2,"first_name":"Ben","last_name":"Lee","country":"US","segment":"SMB"},
            {"customer_id":3,"first_name":"Cara","last_name":"Ng","country":"CA","segment":"Enterprise"},
        ],
        "orders": [
            {"order_id":1001,"customer_id":1,"order_date":"2025-07-01","total_amount":120.5,"country":"US"},
            {"order_id":1002,"customer_id":2,"order_date":"2025-07-02","total_amount":75.0,"country":"US"},
            {"order_id":1003,"customer_id":1,"order_date":"2025-07-04","total_amount":220.0,"country":"US"},
        ],
        "order_items": [
            {"item_id":1,"order_id":1001,"product_id":501,"qty":1,"price":120.5},
            {"item_id":2,"order_id":1002,"product_id":502,"qty":3,"price":25.0},
            {"item_id":3,"order_id":1003,"product_id":501,"qty":2,"price":110.0},
        ],
        "products": [
            {"product_id":501,"sku":"A-RED-XL","name":"Alpha Shirt","category":"Apparel"},
            {"product_id":502,"sku":"B-BLK-42","name":"Beta Boots","category":"Footwear"},
        ],
    },
    "public": {
        "film":[{"film_id":900,"title":"Ocean Whisper","rating":"PG-13"}],
        "customer":[{"customer_id":1,"first_name":"Ana","last_name":"Smith","store_id":1,"activebool":True}],
        "rental":[],
        "inventory":[]
    },
    "mkt": {"campaigns":[{"campaign_id":200,"name":"Summer Sale","channel":"Email"}]}
}

# -----------------------------
# Embedding index for grounding
# -----------------------------
def _cos(a: List[float], b: List[float]) -> float:
    num = sum(x*y for x,y in zip(a,b))
    da = math.sqrt(sum(x*x for x in a)) or 1e-8
    db = math.sqrt(sum(y*y for y in b)) or 1e-8
    return num / (da*db)

class EmbeddingIndex:
    def __init__(self):
        self.items: List[Dict[str,Any]] = []   # {id, type, text, vec, meta}
        self.has_embeddings = False

    def build_from_catalog(self, catalog: Catalog):
        entries = []
        for s in catalog.schemas:
            entries.append({
                "id": f"schema:{s['name']}",
                "type": "schema",
                "text": f"schema {s['name']}. {s.get('description','')}",
                "meta": {"schema": s["name"]}
            })
        for t in catalog.tables:
            fq = f"{t['schema']}.{t['name']}"
            entries.append({
                "id": f"table:{fq}",
                "type": "table",
                "text": f"table {fq}. {t.get('description','')}",
                "meta": {"schema": t["schema"], "table": t["name"], "fq": fq}
            })
            for c in t.get("columns", []):
                fqcol = f"{fq}.{c['name']}"
                desc = c.get("desc","")
                entries.append({
                    "id": f"column:{fqcol}",
                    "type": "column",
                    "text": f"column {fqcol}. {desc}",
                    "meta": {"schema": t["schema"], "table": t["name"], "column": c["name"], "fq": fq, "fqcol": fqcol}
                })
        texts = [e["text"] for e in entries]
        vecs = embed_texts(texts)
        if vecs is None:
            self.items = [{"vec": None, **e} for e in entries]
            self.has_embeddings = False
        else:
            self.items = [{"vec": v, **e} for v, e in zip(vecs, entries)]
            self.has_embeddings = True

    def search(self, query: str, top_k: int = 5, types: Optional[List[str]] = None) -> List[Dict[str,Any]]:
        if not self.items:
            return []
        types = types or ["schema","table","column"]
        # fallback fuzzy
        if not self.has_embeddings:
            choices = [it for it in self.items if it["type"] in types]
            scored = []
            for it in choices:
                s = difflib.SequenceMatcher(None, query.lower(), it["text"].lower()).ratio()
                scored.append((s, it))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [it for score,it in scored[:top_k]]
        qv_list = embed_texts([query]) or []
        if not qv_list:
            return []
        qv = qv_list[0]
        scored = []
        for it in self.items:
            if it["type"] not in types or it["vec"] is None:
                continue
            s = _cos(qv, it["vec"])
            scored.append((s, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for score,it in scored[:top_k]]

# Build global indices
catalog = Catalog()
EMBED_INDEX = EmbeddingIndex()
EMBED_INDEX.build_from_catalog(catalog)

# ==============================
# Embedding-based intent mapping
# ==============================
CANON_INTENTS = [
    ("discover-schemas", "list schemas, what data do you have, show available schemas"),
    ("select-schemas",   "pick or switch schema: use schema sales, set schema public or multiple schemas"),
    ("discover-tables",  "list tables in a schema, browse tables"),
    ("describe-table",   "describe a table structure, columns, types"),
    ("create",           "create or build sql from natural language"),
    ("refine",           "refine or modify previous sql"),
    ("preview",          "preview data or fetch rows or get records or show results"),
    ("validate",         "validate or lint the last sql"),
    ("save",             "save insight with name ttl active schedule"),
    ("update",           "update or edit an existing insight details"),
    ("explain-insight",  "explain an existing insight in natural language"),
    ("explain-sql",      "explain a sql query"),
    ("help",             "how to use, what can I do, help"),
    ("unknown",          "unknown or unsupported request"),
]

class IntentIndex:
    def __init__(self):
        self.items = []   # [{id, text, vec}]
        self.has_embeddings = False
        self._build()

    def _build(self):
        texts = [f"{iid}: {desc}" for iid, desc in CANON_INTENTS]
        vecs = embed_texts(texts)
        if vecs is None:
            self.items = [{"id": iid, "text": txt, "vec": None}
                          for (iid, _), txt in zip(CANON_INTENTS, texts)]
            self.has_embeddings = False
        else:
            self.items = [{"id": iid, "text": txt, "vec": v}
                          for (iid, _), txt, v in zip(CANON_INTENTS, texts, vecs)]
            self.has_embeddings = True

    def normalize(self, label_or_text: str) -> str:
        label_or_text = (label_or_text or "").strip()
        if not label_or_text:
            return "unknown"
        canon_ids = {iid for iid, _ in CANON_INTENTS}
        if label_or_text in canon_ids:
            return label_or_text
        # fallback fuzzy if no embeddings
        if not self.has_embeddings:
            candidates = [it["text"] for it in self.items]
            best = difflib.get_close_matches(label_or_text, candidates, n=1, cutoff=0.0)
            if best:
                for it in self.items:
                    if it["text"] == best[0]:
                        return it["id"]
            return "unknown"
        qv_list = embed_texts([label_or_text]) or []
        if not qv_list:
            return "unknown"
        qv = qv_list[0]
        best_id, best_s = "unknown", -1.0
        for it in self.items:
            if it["vec"] is None:
                continue
            num = sum(a*b for a,b in zip(qv, it["vec"]))
            da = math.sqrt(sum(a*a for a in qv)) or 1e-8
            db = math.sqrt(sum(b*b for b in it["vec"])) or 1e-8
            s = num/(da*db)
            if s > best_s:
                best_id, best_s = it["id"], s
        return best_id

INTENT_INDEX = IntentIndex()

# -----------------------------
# Mock SQL Preview Engine
# -----------------------------
class QueryJoin:
    def __init__(self, right: str, left_key: str, right_key: str):
        self.right = right
        self.left_key = left_key
        self.right_key = right_key

class QueryBuilder:
    def __init__(self, base: Optional[str]=None, joins: Optional[List[QueryJoin]]=None, limit: int=5):
        self.base = base
        self.joins = joins or []
        self.limit = limit

class MockRunner:
    def __init__(self, data: Dict[str, Dict[str, List[Dict[str,Any]]]]): self.data=data
    def fetch(self, fq: str) -> List[Dict[str,Any]]:
        s,t=fq.split(".",1); return [dict(r) for r in self.data.get(s,{}).get(t,[])]
    def exec(self, qb: QueryBuilder) -> Dict[str,Any]:
        if not qb.base: return {"columns":[],"rows":[]}
        cur=[{f"{qb.base}.{k}":v for k,v in r.items()} for r in self.fetch(qb.base)]
        for j in qb.joins:
            rr=[{f"{j.right}.{k}":v for k,v in r.items()} for r in self.fetch(j.right)]
            idx={}
            for r in rr:
                idx.setdefault(r[f"{j.right}.{j.right_key}"],[]).append(r)
            new=[]
            for l in cur:
                lk=l.get(f"{qb.base}.{j.left_key}")
                for r in idx.get(lk,[]):
                    z=dict(l); z.update(r); new.append(z)
            cur=new
        rows=cur[:qb.limit]
        cols=sorted(set().union(*[set(r) for r in rows])) if rows else []
        simple_rows=[{k.split(".")[-1]:v for k,v in r.items()} for r in rows]
        return {"columns":[c.split(".")[-1] for c in cols],"rows":simple_rows}

# -----------------------------
# Tools (LLM SQL + ops)
# -----------------------------
class Tools:
    READ_ONLY = re.compile(r"\b(UPDATE|DELETE|INSERT|MERGE|DROP|TRUNCATE|ALTER|CREATE)\b", re.I)
    def __init__(self, catalog: Catalog):
        self.catalog=catalog
        self.runner=MockRunner(MOCK_DATA)
        self.saved: Dict[str,Dict[str,Any]] = {}

    # Data discovery
    def list_schemas(self)->List[Dict[str,Any]]:
        return self.catalog.list_schemas()
    def list_tables(self, schema:str)->List[Dict[str,Any]]:
        return self.catalog.list_tables(schema)
    def describe_table(self, fq:str)->Dict[str,Any]:
        s,t=fq.split(".",1)
        return self.catalog.get_table(s,t)
    def preview_table(self, fq:str, n:int)->Dict[str,Any]:
        qb=QueryBuilder(base=fq, limit=n)
        return self.runner.exec(qb)

    # Safety & lint
    def is_safe(self, sql:str)->Tuple[bool,str]:
        return (False,"Only read-only SELECT/CTE queries allowed.") if self.READ_ONLY.search(sql) else (True,"ok")
    def lint(self, sql:str)->Dict[str,Any]:
        refs=set()
        for m in re.finditer(r"\b(from|join)\s+([a-z_0-9]+\.[a-z_0-9]+)", sql, re.I):
            refs.add(m.group(2).lower())
        unknown=[fq for fq in refs if not any(t for t in self.catalog.tables if f"{t['schema']}.{t['name']}"==fq)]
        return {"tables":sorted(refs), "unknown_tables":unknown, "ok": len(unknown)==0}

    # LLM SQL build/refine with grounding
    def plan_sql(self, mode:str, ask:str, prior_sql:Optional[str], grounding: Optional[Dict[str,List[str]]] = None)->Dict[str,Any]:
        cli=_client()
        if not cli:
            return {"error":"OPENAI_API_KEY not set; cannot generate SQL"}
        schemas="\n".join([f"- {s['name']}: {s['description']}" for s in self.catalog.schemas])
        tables=[]
        for t in self.catalog.tables:
            cols=", ".join([f"{c['name']}:{c.get('type','')}" for c in t.get("columns",[])])
            tables.append(f"- {t['schema']}.{t['name']}: {t['description']} | {cols}")
        grounding_txt = ""
        if grounding:
            if grounding.get("tables") or grounding.get("columns"):
                grounding_txt = "\n\nGROUNDING:\n"
                if grounding.get("tables"):
                    grounding_txt += "Candidate tables: " + ", ".join(grounding["tables"]) + "\n"
                if grounding.get("columns"):
                    grounding_txt += "Candidate columns: " + ", ".join(grounding["columns"]) + "\n"

        sys=("You write production-grade PostgreSQL. Use CTEs, avoid SELECT *, never mutate data. "
             "Prefer grounded tables/columns when provided.")
        user=(f"MODE:{mode}\nSCHEMAS:\n{schemas}\nTABLES:\n"+"\n".join(tables)+
              f"\nPRIOR_SQL:\n{prior_sql or '(none)'}{grounding_txt}\n"
              "Return STRICT JSON: {mode, sql, explanation, expected_columns}.\n"
              f"USER_ASK: {ask}")
        out=gpt_json([{"role":"system","content":sys},{"role":"user","content":user}]) or {}
        sql=out.get("sql","")
        safe,why=self.is_safe(sql)
        if not safe: return {"error":why}
        lint=self.lint(sql)
        return {"sql":sql,"explanation":out.get("explanation",""),"expected_columns":out.get("expected_columns",[]),"lint":lint}

    def preview_sql(self, sql:str, limit:int=5)->Dict[str,Any]:
        m=re.search(r"from\s+([a-z0-9_]+\.[a-z0-9_]+)", sql, re.I)
        if not m: return {"error":"preview engine couldn't detect base table"}
        base=m.group(1)
        qb=QueryBuilder(base=base, limit=limit)
        for j in re.finditer(r"join\s+([a-z0-9_]+\.[a-z0-9_]+)\s+on\s+([a-z0-9_\.]+)\s*=\s*([a-z0-9_\.]+)", sql, re.I):
            right=j.group(1); left_key=j.group(2).split(".")[-1]; right_key=j.group(3).split(".")[-1]
            qb.joins.append(QueryJoin(right=right,left_key=left_key,right_key=right_key))
        return self.runner.exec(qb)

    def save(self, name:str, sql:str, ttl:int, active:bool, schedule:Optional[str], columns:List[str])->Dict[str,Any]:
        iid=str(uuid.uuid4())
        rec={"id":iid,"name":name,"sql":sql,"ttl":ttl,"active":active,"schedule":(schedule or ""), "columns":columns}
        self.saved[iid]=rec
        return {"id":iid,"url":f"https://example.local/insights/{iid}", "record":rec}

    def get_saved(self, key:str)->Optional[Dict[str,Any]]:
        if key in self.saved: return self.saved[key]
        for v in self.saved.values():
            if v["name"]==key: return v
        return None

    def update(self, insight_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        rec = self.saved.get(insight_id)
        if not rec:
            for v in self.saved.values():
                if v["name"] == insight_id:
                    rec = v; break
        if not rec:
            raise KeyError("Insight not found")
        if "name" in fields and fields["name"]:
            rec["name"] = fields["name"]
        if "ttl" in fields and fields["ttl"] is not None:
            rec["ttl"] = int(fields["ttl"])
        if "active" in fields and fields["active"] is not None:
            rec["active"] = bool(fields["active"])
        if "schedule" in fields and fields["schedule"] is not None:
            rec["schedule"] = fields["schedule"]
        if "sql" in fields and fields["sql"]:
            rec["sql"] = fields["sql"]
        return rec

    def explain_sql_llm(self, sql: str) -> str:
        cli=_client()
        if not cli:
            return "Set OPENAI_API_KEY to enable natural-language SQL explanations."
        prompt = (
            "Explain this SQL in two parts:\n"
            "1) Technical Breakdown: sources, joins, filters, metrics, groupings, ordering, limits.\n"
            "2) Plain English Summary: 1–2 sentences, what business question it answers.\n\n"
            f"SQL:\n{sql}"
        )
        out = gpt_text([
            {"role":"system","content":"You explain SQL clearly and concisely for technical and non-technical readers."},
            {"role":"user","content":prompt}
        ], temperature=0.2)
        return out or "Explanation unavailable."

# Global tools instance
tools = Tools(catalog)

# -----------------------------
# Intent + Slot Parsing (LLM + fallback)
# -----------------------------
RE_UPDATE = re.compile(r"\b(update|modify|change|edit)\s+(?:insight\s+)?([a-z0-9_\-]+)", re.I)
RE_EXPLAIN_INSIGHT = re.compile(r"\b(explain|describe|summarize)\s+(?:insight|report|metric)\s+([a-z0-9_\-]+)", re.I)
RE_EXPLAIN_SQL = re.compile(r"\b(explain|describe|summarize)\s+sql\b[:\s]*(.+)$", re.I | re.S)

def gpt_parse_intent(user_input: str, schemas: List[Dict[str, Any]], insights: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    cli = _client()
    if not cli:
        return None
    schema_names = [s["name"] for s in schemas]
    insight_names = [i["name"] for i in insights] if insights else []
    system = (
        "You are a robust intent and slot parser for a data insight assistant.\n"
        "Normalize varied phrasings. Output STRICT JSON with fields {intent, confidence, slots}.\n"
        "If unsure, set intent='unknown' and confidence<0.5. Resolve names if present."
    )
    user = json.dumps({
        "input": user_input,
        "available_schemas": schema_names,
        "available_insights": insight_names,
        "examples": [
            {"in":"update us_sales_by_customer insight", "intent":"update_insight", "slots":{"insight_name":"us_sales_by_customer"}},
            {"in":"modify insight us_sales ttl 45 and make active", "intent":"update_insight", "slots":{"insight_name":"us_sales","updates":{"ttl":45,"active":True}}},
            {"in":"explain insight us_sales_by_segment", "intent":"explain_insight", "slots":{"insight_name":"us_sales_by_segment"}},
            {"in":"what does this sql do: SELECT 1", "intent":"explain_sql", "slots":{"sql":"SELECT 1"}},
            {"in":"show 3 rows from sales.orders", "intent":"preview_rows","slots":{"fq_table":"sales.orders","n_rows":3}},
            {"in":"describe sales.orders", "intent":"describe_table","slots":{"fq_table":"sales.orders","schema":"sales","table":"orders"}},
            {"in":"list tables in sales", "intent":"list_tables","slots":{"schema":"sales"}},
            {"in":"use schema sales and public", "intent":"select_schema","slots":{"schemas":["sales","public"]}},
            {"in":"what data do you have", "intent":"discover_schemas","slots":{}},
            {"in":"create revenue by segment last 30 days", "intent":"create_sql","slots":{}},
            {"in":"refine also only country='US'", "intent":"refine_sql","slots":{}},
            {"in":"preview", "intent":"preview_last_sql","slots":{}},
            {"in":"validate", "intent":"validate","slots":{}},
            {"in":"save as us_rev ttl 14 active true", "intent":"save","slots":{"save":{"name":"us_rev","ttl":14,"active":True}}},
        ]
    })
    try:
        r = cli.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":system},{"role":"user","content":user}]
        )
        return json.loads(r.choices[0].message.content)
    except Exception:
        return None

# -----------------------------
# Markdown & ASCII helpers
# -----------------------------
def md_list(title: str, lines: List[str]) -> str:
    return title + "\n" + "\n".join(f"- {ln}" for ln in lines)

def ascii_table(cols: List[str], rows: List[Dict[str, Any]], max_rows: int = 10) -> str:
    if not rows: return "(no rows)"
    widths = [max(len(c), max((len(str(r.get(c,''))) for r in rows[:max_rows]), default=0)) for c in cols]
    header=" | ".join(c.ljust(w) for c,w in zip(cols,widths))
    sep   ="-+-".join("-"*w for w in widths)
    body=[" | ".join(str(r.get(c,'')).ljust(w) for c,w in zip(cols,widths)) for r in rows[:max_rows]]
    return "\n".join([header,sep]+body)

# -----------------------------
# LangGraph Agent
# -----------------------------
from langgraph.graph import StateGraph, END

def initial_state() -> Dict[str, Any]:
    return {
        "messages": [],
        "intent": None,
        "selected_schemas": [],
        "last_sql": None,
        "pending_save": None,
        "preview_cache": None,
        "error": None,
        "user_input": "",
        "_slots": {},
    }

HELP_CARD = (
    "I can describe your data, list examples, and build new attributes/insights.\n\n"
    "Things you can do:\n"
    "- Discover data: list schemas/tables, describe columns\n"
    "- Build insights: natural language → SQL (multi-CTE)\n"
    "- Validate & preview: lint for safety, preview sample rows\n"
    "- Save: name + TTL + active (schedule optional)\n"
    "- Explain: summarize what an existing insight’s SQL does\n\n"
    "Try:\n"
    "- what data do you have?\n"
    "- use schema `sales`\n"
    "- list tables in `sales`\n"
    "- describe `sales.orders`\n"
    "- show 5 rows from `sales.orders`\n"
    "- create revenue by segment last 30 days\n"
    "- preview / validate / save as my_attr ttl 14 active true\n"
    "- explain insight my_attr\n"
)

# --- Grounding helpers ---
def resolve_schema_freeform(text: str) -> Optional[str]:
    hits = EMBED_INDEX.search(text, top_k=3, types=["schema"])
    if hits:
        return hits[0]["meta"]["schema"]
    names = [s["name"] for s in catalog.list_schemas()]
    close = difflib.get_close_matches(text, names, n=1, cutoff=0.6)
    return close[0] if close else None

def resolve_table_freeform(text: str, prefer_schema: Optional[str] = None) -> Optional[str]:
    if re.match(r"^[a-z0-9_]+\.[a-z0-9_]+$", text):
        return text
    hits = EMBED_INDEX.search(text, top_k=5, types=["table"])
    if prefer_schema:
        hits = [h for h in hits if h["meta"]["schema"] == prefer_schema] or hits
    if hits:
        return hits[0]["meta"]["fq"]
    fqs = [f"{t['schema']}.{t['name']}" for t in catalog.tables]
    close = difflib.get_close_matches(text, fqs, n=1, cutoff=0.5)
    if close:
        return close[0]
    names = [t["name"] for t in catalog.tables]
    close = difflib.get_close_matches(text, names, n=1, cutoff=0.6)
    if close:
        tname = close[0]
        for t in catalog.tables:
            if t["name"] == tname and (prefer_schema is None or t["schema"] == prefer_schema):
                return f"{t['schema']}.{t['name']}"
    return None

def grounding_for_ask(ask: str, selected_schemas: List[str]) -> Dict[str,List[str]]:
    tables_hits = EMBED_INDEX.search(ask, top_k=6, types=["table"])
    if selected_schemas:
        filtered = [h for h in tables_hits if h["meta"]["schema"] in selected_schemas]
        if filtered:
            tables_hits = filtered[:6]
    cols_hits = EMBED_INDEX.search(ask, top_k=8, types=["column"])
    if selected_schemas:
        filteredc = [h for h in cols_hits if h["meta"]["schema"] in selected_schemas]
        if filteredc:
            cols_hits = filteredc[:8]
    tables = [h["meta"]["fq"] for h in tables_hits]
    columns = [h["meta"]["fqcol"] for h in cols_hits]
    return {"tables": tables, "columns": columns}

# --- Preview target extraction (natural phrasings) ---
def extract_preview_target(text: str, selected_schemas: List[str]) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Returns (fq_table, limit, reason)
    - fq_table: resolved 'schema.table' or None if can't resolve
    - limit: int (rows to fetch) or None to use default
    - reason: short note for debugging/logging
    Supports:
      - "show 10 rows from sales.orders"
      - "fetch all records from campaigns table"
      - "get records from mkt.campaigns"
      - "show all rows from campaigns"
      - "show records from campaigns table"
    """
    t = text.strip()
    low = t.lower()

    # Explicit number like "show 10 rows/records"
    n = None
    m_n = re.search(r"\b(\d+)\s+(rows|records)\b", low)
    if m_n:
        try:
            n = int(m_n.group(1))
        except Exception:
            n = None

    # "all records/rows" → cap later if needed
    all_req = bool(re.search(r"\ball\b\s+(rows|records)\b", low)) or bool(re.search(r"\b(show|get|fetch)\s+all\b", low))

    # Extract table token after 'from ...'
    cand = None
    m_from = re.search(r"\bfrom\s+([a-z0-9_\.]+)(?:\s+table)?\b", low)
    if m_from:
        cand = m_from.group(1).strip()
    else:
        # "from campaigns table" might be caught above; try looser grammar:
        m_tail = re.search(r"\b(show|get|fetch)\b.*\b([a-z0-9_\.]+)\s+table\b", low)
        if m_tail:
            cand = m_tail.group(2).strip()
        else:
            # last resort: if they just say "show records from campaigns" w/out 'table' keyword
            m_rec = re.search(r"\b(show|get|fetch)\s+(?:all\s+)?(?:rows|records|data)?\s*from\s+([a-z0-9_\.]+)\b", low)
            if m_rec:
                cand = m_rec.group(2).strip()

    if not cand:
        return None, None, "no-table-found"

    # Resolve via embeddings → schema.table
    prefer = selected_schemas[0] if selected_schemas else None
    fq = resolve_table_freeform(cand, prefer_schema=prefer)
    if not fq:
        return None, None, "resolve-failed"

    # Decide limit
    if all_req:
        # try to size from mock data; else cap at 100
        try:
            sch, tbl = fq.split(".", 1)
            max_len = len(MOCK_DATA.get(sch, {}).get(tbl, []))
            limit = max_len if max_len > 0 else 100
        except Exception:
            limit = 100
    else:
        limit = n if n is not None else 5  # default 5

    return fq, limit, "ok"

# Router using LLM + embedding normalization + fallback regex
def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    txt = state["user_input"].strip()
    low = txt.lower()
    state["_slots"] = {}

    # off-topic/abuse quick gate
    if re.search(r"\b(idiot|stupid|useless|waste|dumb|trash|shut up)\b", low):
        state["intent"] = "offtopic"
        return state

    # LLM semantic parse (intent + slots)
    parsed = gpt_parse_intent(txt, catalog.list_schemas(), list(tools.saved.values()))
    if parsed:
        raw_intent = parsed.get("intent","unknown")
        conf = float(parsed.get("confidence",0.0))
        slots = parsed.get("slots",{}) or {}
        # Normalize via embeddings; if low-confidence, normalize the utterance instead
        norm_from_label = INTENT_INDEX.normalize(raw_intent)
        if conf < 0.6 or norm_from_label == "unknown":
            norm_intent = INTENT_INDEX.normalize(txt)
        else:
            norm_intent = norm_from_label
        state["_slots"] = slots
        state["intent"] = norm_intent
        return state

    # FINAL fallback regex + simple heuristics
    if RE_UPDATE.search(txt):                state["intent"] = "update"; return state
    if RE_EXPLAIN_INSIGHT.search(txt):       state["intent"] = "explain-insight"; return state
    if RE_EXPLAIN_SQL.search(txt):           state["intent"] = "explain-sql"; return state
    if re.search(r"\b(fetch|get|show)\s+(rows|records)\b", low):
        state["intent"] = "preview"; return state
    if re.search(r"\b(what data|list schemas|show schemas)\b", low):
        state["intent"] = "discover-schemas"; return state
    if re.search(r"\bhow (to )?use|help|what can i do\b", low):
        state["intent"] = "help"; return state
    if re.search(r"\b(fetch|get|show)\s+(?:all\s+)?(?:rows|records|data)\s+from\b", low):
        state["intent"] = "preview";
        return state
    # If still nothing, embedding normalize whole utterance
    state["intent"] = INTENT_INDEX.normalize(txt) or "unknown"
    return state

def help_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state["messages"].append({"role":"assistant","content":HELP_CARD})
    return state

def offtopic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    msg = ("I can’t help with that.\n"
           "Try:\n- what data do you have?\n- list tables in sales\n- create revenue by segment last 30 days\n"
           "- preview / validate / save as my_attr ttl 14 active true")
    state["messages"].append({"role":"assistant","content":msg})
    return state

def discover_schemas_node(state: Dict[str, Any]) -> Dict[str, Any]:
    lines = [f"`{s['name']}` — {s['description']}" for s in catalog.list_schemas()]
    reply = md_list("**Available schemas:**", lines)
    state["messages"].append({"role":"assistant","content":reply})
    return state

def select_schemas_node(state: Dict[str, Any]) -> Dict[str, Any]:
    txt = state["user_input"]
    slots = state.get("_slots", {})
    if slots.get("schemas"):
        picked_raw = slots["schemas"]
    else:
        if "use schema" in txt.lower():
            picked_raw = [s.strip() for s in txt.split("use schema",1)[1].replace(",", " and ").split("and") if s.strip()]
        else:
            picked_raw = []
    resolved = []
    for token in picked_raw:
        names = {s["name"] for s in catalog.list_schemas()}
        if token in names:
            resolved.append(token); continue
        sc = resolve_schema_freeform(token)
        if sc: resolved.append(sc)
    resolved = list(dict.fromkeys(resolved))  # unique preserve order
    if not resolved:
        names = ", ".join(sorted([s["name"] for s in catalog.list_schemas()]))
        state["messages"].append({"role":"assistant","content":f"I don't recognize those schemas. Try one of: {names}"})
        return state
    state["selected_schemas"] = resolved
    state["messages"].append({"role":"assistant","content":f"Using schemas: {', '.join(f'`{x}`' for x in resolved)}."})
    return state

def discover_tables_node(state: Dict[str, Any]) -> Dict[str, Any]:
    low = state["user_input"].lower()
    m = re.search(r"list tables(?: in ([a-z0-9_]+))?", low)
    schema = (m.group(1) if m and m.group(1) else (state["selected_schemas"][0] if state["selected_schemas"] else None))
    if not schema:
        state["messages"].append({"role":"assistant","content":"Tell me which schema: `list tables in sales`"})
        return state
    tabs = tools.list_tables(schema)
    if not tabs:
        state["messages"].append({"role":"assistant","content":f"No tables found in `{schema}`."})
        return state
    lines = [f"`{schema}.{t['name']}` — {t['description']}" for t in tabs]
    reply = md_list(f"**Tables in `{schema}`:**", lines)
    state["messages"].append({"role":"assistant","content":reply})
    return state

def describe_table_node(state: Dict[str, Any]) -> Dict[str, Any]:
    txt = state["user_input"].strip()
    slots = state.get("_slots", {})
    fq = slots.get("fq_table")
    if not fq:
        parts = txt.split()
        cand = None
        for p in parts[1:]:
            if "." in p:
                cand = p; break
        if not cand:
            prefer = state["selected_schemas"][0] if state["selected_schemas"] else None
            cand = resolve_table_freeform(txt, prefer_schema=prefer)
        fq = cand
    if not fq:
        state["messages"].append({"role":"assistant","content":"Please specify as `describe <schema>.<table>` (e.g., `describe sales.orders`)."})
        return state
    try:
        meta = tools.describe_table(fq)
    except Exception:
        state["messages"].append({"role":"assistant","content":f"I don't recognize `{fq}`."})
        return state
    cols = meta.get("columns", [])
    header = f"**{fq}** — {meta.get('description','')}"
    if not cols:
        state["messages"].append({"role":"assistant","content":header})
        return state
    lines = [f"`{c['name']}` (*{c.get('type','')}*) — {c.get('desc','') or ''}".rstrip() for c in cols]
    reply = header + "\n\n" + md_list("**Columns:**", lines)
    state["messages"].append({"role":"assistant","content":reply})
    return state

def create_node(state: Dict[str, Any]) -> Dict[str, Any]:
    ask = state["user_input"]
    grounding = grounding_for_ask(ask, state.get("selected_schemas", []))
    plan = tools.plan_sql("create", ask, state["last_sql"], grounding=grounding)
    if plan.get("error"):
        state["messages"].append({"role":"assistant","content":f"Could not generate SQL: {plan['error']}"})
        return state
    state["last_sql"] = plan["sql"]
    msg = f"Draft SQL created.\n\n{plan.get('explanation','')}\n\n```sql\n{plan['sql']}\n```\n\nLint: {'ok' if plan['lint']['ok'] else 'issues: ' + ', '.join(plan['lint']['unknown_tables'])}\nNext: say `preview` or add refinements."
    state["messages"].append({"role":"assistant","content":msg})
    return state

def refine_node(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state["last_sql"]:
        state["messages"].append({"role":"assistant","content":"No previous SQL to refine. Say what you want to create first."})
        return state
    ask = state["user_input"]
    grounding = grounding_for_ask(ask, state.get("selected_schemas", []))
    plan = tools.plan_sql("refine", ask, state["last_sql"], grounding=grounding)
    if plan.get("error"):
        state["messages"].append({"role":"assistant","content":f"Could not refine SQL: {plan['error']}"})
        return state
    state["last_sql"] = plan["sql"]
    msg = f"SQL refined.\n\n{plan.get('explanation','')}\n\n```sql\n{plan['sql']}\n```\n\nLint: {'ok' if plan['lint']['ok'] else 'issues: ' + ', '.join(plan['lint']['unknown_tables'])}\nNext: `preview` or `validate`."
    state["messages"].append({"role":"assistant","content":msg})
    return state

def validate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state["last_sql"]:
        state["messages"].append({"role":"assistant","content":"Nothing to validate yet."})
        return state
    lint = tools.lint(state["last_sql"])
    msg = "ok" if lint["ok"] else "Unknown tables: " + ", ".join(lint["unknown_tables"])
    reply = f"Validation: {msg}. Tables referenced: {', '.join(lint['tables']) or '(none)'}"
    state["messages"].append({"role":"assistant","content":reply})
    return state

def preview_node(state: Dict[str, Any]) -> Dict[str, Any]:
    txt = state["user_input"]
    low = txt.lower()

    # First, try natural-language extraction for table previews
    fq, limit, reason = extract_preview_target(txt, state.get("selected_schemas", []))
    if fq:
        out = tools.preview_table(fq, limit or 5)
        table = ascii_table(out["columns"], out["rows"])
        state["preview_cache"] = out
        # Include a tiny header so users see what we ran
        hdr = f"Previewing `{fq}` (limit {limit}):\n" if limit else f"Previewing `{fq}`:\n"
        state["messages"].append({"role":"assistant","content":hdr + table})
        return state

    # Backward-compatible explicit pattern: "show N rows from schema.table"
    m = re.search(r"show\s+(\d+)\s*rows\s+from\s+([a-z0-9_]+\.[a-z0-9_]+)", low)
    if m:
        n = int(m.group(1))
        fq = m.group(2)
        out = tools.preview_table(fq, n)
        table = ascii_table(out["columns"], out["rows"])
        state["preview_cache"] = out
        state["messages"].append({"role":"assistant","content":f"Previewing `{fq}` (limit {n}):\n{table}"})
        return state

    # Otherwise, preview last_sql if available
    if state.get("last_sql"):
        out = tools.preview_sql(state["last_sql"], 5)
        if "error" in out:
            state["messages"].append({"role":"assistant","content":f"Preview error: {out['error']}"})
            return state
        table = ascii_table(out["columns"], out["rows"])
        state["preview_cache"] = out
        state["messages"].append({"role":"assistant","content":table})
        return state

    # Nothing to preview
    state["messages"].append({"role":"assistant","content":"I couldn’t find a table or a previous query to preview. Try `show 3 rows from sales.orders` or `preview` after creating SQL."})
    return state

def re_search(pat: str, text: str) -> Optional[str]:
    m = re.search(pat, text, re.I)
    return m.group(1).strip() if m else None

def save_node(state: Dict[str, Any]) -> Dict[str, Any]:
    txt = state["user_input"]
    pending = state.get("pending_save") or {}
    slots = state.get("_slots", {})

    # incorporate slot-based save (from LLM)
    if "save" in slots and isinstance(slots["save"], dict):
        for k,v in slots["save"].items():
            if v is not None:
                pending[k] = v

    name = re_search(r"(?:save as|name)\s+([a-z0-9_\-]+)", txt) or pending.get("name")
    ttl  = re_search(r"ttl\s+(\d+)", txt)
    ttl  = int(ttl) if ttl else pending.get("ttl")
    active = pending.get("active")
    if re.search(r"\bactive\b", txt, re.I):
        active = bool(re.search(r"(true|yes|on|1)", txt, re.I))
    schedule = re_search(r"schedule\s+([^\n]+)$", txt) or pending.get("schedule")

    missing = []
    if not name: missing.append("name")
    if ttl is None: missing.append("ttl")
    if active is None: missing.append("active")
    if missing:
        state["pending_save"] = {"name":name,"ttl":ttl,"active":active,"schedule":schedule}
        state["messages"].append({"role":"assistant","content":f"I can save this insight. Please provide: {', '.join(missing)}. For example: `name us_sales_by_segment; ttl 14; active true`"})
        return state
    if not state["last_sql"]:
        state["messages"].append({"role":"assistant","content":"There is no SQL to save yet."})
        return state

    prev = tools.preview_sql(state["last_sql"], 1)
    cols = prev.get("columns", [])
    out = tools.save(name, state["last_sql"], int(ttl), bool(active), schedule, cols)
    state["pending_save"] = None
    state["messages"].append({"role":"assistant","content":f"Saved `{name}` (TTL {ttl}d, active {active}). Open in UI: {out['url']}"})
    return state

def update_node(state: Dict[str, Any]) -> Dict[str, Any]:
    txt = state["user_input"]
    key = None
    slots = state.get("_slots", {})
    if slots.get("insight_name"):
        key = slots["insight_name"]
    else:
        m = RE_UPDATE.search(txt)
        if m:
            key = m.group(2)
    if not key:
        state["messages"].append({"role":"assistant","content":"Which insight should I update? e.g., `update <name> ttl 30 active true`"})
        return state
    rec = tools.get_saved(key)
    if not rec:
        state["messages"].append({"role":"assistant","content":f"Insight `{key}` not found."})
        return state
    fields = {}
    if "updates" in slots and isinstance(slots["updates"], dict):
        fields.update(slots["updates"])
    if m := re.search(r"\bttl\s+(\d+)", txt, re.I): fields["ttl"] = int(m.group(1))
    if re.search(r"\bactive\b", txt, re.I): fields["active"] = bool(re.search(r"(true|yes|on|1)", txt, re.I))
    if m := re.search(r"\bname\s+([a-z0-9_\-]+)", txt, re.I): fields["name"] = m.group(1)
    if m := re.search(r"\bschedule\s+([^\n]+)$", txt, re.I): fields["schedule"] = m.group(1)

    updated = tools.update(rec["id"], fields)
    state["messages"].append({"role":"assistant","content":f"Updated `{updated['name']}` (TTL {updated['ttl']}d, active {updated['active']})."})
    return state

def explain_insight_node(state: Dict[str, Any]) -> Dict[str, Any]:
    txt = state["user_input"]
    slots = state.get("_slots", {})
    key = slots.get("insight_name")
    if not key:
        m = RE_EXPLAIN_INSIGHT.search(txt)
        if m:
            key = m.group(2)
    if not key:
        state["messages"].append({"role":"assistant","content":"Which insight should I explain? e.g., `explain insight us_sales_by_segment`"})
        return state
    rec = tools.get_saved(key)
    if not rec:
        state["messages"].append({"role":"assistant","content":f"Insight '{key}' not found."})
        return state
    meta = {k: rec[k] for k in ["name","ttl","active","schedule","columns"] if k in rec}
    expl = tools.explain_sql_llm(rec["sql"])
    card = (f"# Insight: {meta['name']}\n"
            f"- Status: {'Active' if meta.get('active') else 'Inactive'} | TTL: {meta.get('ttl','-')} days | Schedule: {meta.get('schedule') or '(none)'}\n"
            f"- Output columns: {', '.join(meta.get('columns') or []) or '(unknown)'}\n\n{expl}")
    state["messages"].append({"role":"assistant","content":card})
    return state

def explain_sql_node(state: Dict[str, Any]) -> Dict[str, Any]:
    txt = state["user_input"]
    slots = state.get("_slots", {})
    sql = slots.get("sql")
    if not sql:
        m = RE_EXPLAIN_SQL.search(txt)
        if m:
            sql = m.group(2).strip()
    if not sql:
        state["messages"].append({"role":"assistant","content":"Please provide a SQL query after `explain sql`."})
        return state
    explanation = tools.explain_sql_llm(sql)
    state["messages"].append({"role":"assistant","content":explanation})
    return state

def unknown_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state["messages"].append({"role":"assistant","content":"I’m not sure I can help with that.\nTry:\n- what data do you have?\n- list tables in sales\n- describe sales.orders\n- create revenue by segment last 30 days\n- preview / validate / save as my_attr ttl 14 active true"})
    return state

# Build LangGraph
from langgraph.graph import StateGraph, END

graph = StateGraph(dict)

graph.add_node("router", router_node)
graph.add_node("help", help_node)
graph.add_node("offtopic", offtopic_node)
graph.add_node("discover-schemas", discover_schemas_node)
graph.add_node("select-schemas", select_schemas_node)
graph.add_node("discover-tables", discover_tables_node)
graph.add_node("describe-table", describe_table_node)
graph.add_node("create", create_node)
graph.add_node("refine", refine_node)
graph.add_node("validate", validate_node)
graph.add_node("preview", preview_node)
graph.add_node("save", save_node)
graph.add_node("update", update_node)
graph.add_node("explain-insight", explain_insight_node)
graph.add_node("explain-sql", explain_sql_node)
graph.add_node("unknown", unknown_node)

graph.set_entry_point("router")

def route(state: Dict[str, Any]):
    return state.get("intent") or "unknown"

graph.add_conditional_edges("router", route, {
  "help":"help",
  "offtopic":"offtopic",
  "discover-schemas":"discover-schemas",
  "select-schemas":"select-schemas",
  "discover-tables":"discover-tables",
  "describe-table":"describe-table",
  "create":"create",
  "refine":"refine",
  "validate":"validate",
  "preview":"preview",
  "save":"save",
  "update":"update",
  "explain-insight":"explain-insight",
  "explain-sql":"explain-sql",
  "unknown":"unknown",
})

for n in ["help","offtopic","discover-schemas","select-schemas","discover-tables","describe-table",
          "create","refine","validate","preview","save","update","explain-insight","explain-sql","unknown"]:
    graph.add_edge(n, END)

app = graph.compile()

# Public runner (e.g., FastAPI uses this)
def initial_state_public() -> Dict[str, Any]:
    return initial_state()

def run_agent(user_input: str, session_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    state = session_state or initial_state()
    state["user_input"] = user_input
    result = app.invoke(state)
    return result

# --- Optional local demo ---
if __name__ == "__main__":
    s = initial_state()
    for u in [
        "how can you help?",
        "what data do you have?",
        "use schema sales and public",
        "list tables in sales",
        "describe sales.orders",
        "show 3 rows from sales.orders",
        "create US revenue by segment in last 30 days joining orders and customers; include totals",
        "preview",
        "refine also include only country='US' and top 5 segments by revenue",
        "validate",
        "save as us_sales_by_segment ttl 14 active true",
        "explain insight us_sales_by_segment",
        "explain sql SELECT c.segment, SUM(o.total_amount) AS revenue FROM sales.orders o JOIN sales.customers c ON o.customer_id=c.customer_id WHERE o.country='US' GROUP BY c.segment ORDER BY revenue DESC LIMIT 5;"
    ]:
        s["user_input"] = u
        s = app.invoke(s)
        print(f"\n[User] {u}")
        print(f"[Assistant] {s['messages'][-1]['content']}")
