from typing import Dict, Any

# Richer catalog for experimentation (more tables + FKs)
CATALOG: Dict[str, Any] = {
    "sales": {
        "__desc": "Transactional sales data including orders, items, products, and customers.",
        "ord_hdr": {
            "__desc": "Order header facts (one row per order).",
            "columns": [
                {"name": "ord_id",  "type": "INTEGER", "desc": "Order identifier."},
                {"name": "cust_id", "type": "INTEGER", "desc": "Customer identifier."},
                {"name": "ord_dt",  "type": "DATE",    "desc": "Date when the order was placed (UTC)."},
                {"name": "tot_amt", "type": "DECIMAL", "desc": "Total order amount in the original currency."},
                {"name": "curr",    "type": "TEXT",    "desc": "Currency code (e.g., USD)."},
                {"name": "seg",     "type": "TEXT",    "desc": "Customer segment at order time."},
                {"name": "is_rfnd", "type": "BOOLEAN", "desc": "True if the order was refunded."}
            ],
            "pk": ["ord_id"]
        },
        "order_items": {
            "__desc": "Order line items (one row per product per order).",
            "columns": [
                {"name": "ord_id",   "type": "INTEGER", "desc": "Order identifier."},
                {"name": "line_id",  "type": "INTEGER", "desc": "Line sequence within the order."},
                {"name": "prod_id",  "type": "INTEGER", "desc": "Product identifier."},
                {"name": "qty",      "type": "INTEGER", "desc": "Quantity ordered."},
                {"name": "unit_amt", "type": "DECIMAL", "desc": "Unit price in order currency."}
            ],
            "pk": ["ord_id", "line_id"]
        },
        "prod_dim": {
            "__desc": "Product dimension (one row per product).",
            "columns": [
                {"name": "prod_id", "type": "INTEGER", "desc": "Product identifier."},
                {"name": "sku",     "type": "TEXT",    "desc": "Stock-keeping unit."},
                {"name": "name",    "type": "TEXT",    "desc": "Product name."},
                {"name": "cat",     "type": "TEXT",    "desc": "Product category."}
            ],
            "pk": ["prod_id"]
        },
        "cust_dim": {
            "__desc": "Customer dimension (one row per customer).",
            "columns": [
                {"name": "cust_id",   "type": "INTEGER", "desc": "Customer identifier."},
                {"name": "cntry",     "type": "TEXT",    "desc": "Customer country (ISO code or name)."},
                {"name": "seg",       "type": "TEXT",    "desc": "Customer segment label."},
                {"name": "signup_dt", "type": "DATE",    "desc": "When the customer signed up."},
                {"name": "eml",       "type": "TEXT",    "desc": "Customer email address (PII)."}
            ],
            "pk": ["cust_id"]
        }
    },
    "mkt": {
        "__desc": "Marketing and attribution data.",
        "camp": {
            "__desc": "Marketing campaigns master data.",
            "columns": [
                {"name": "camp_id",  "type": "INTEGER", "desc": "Campaign identifier."},
                {"name": "nm",       "type": "TEXT",    "desc": "Campaign name."},
                {"name": "start_dt", "type": "DATE",    "desc": "Campaign start date."},
                {"name": "end_dt",   "type": "DATE",    "desc": "Campaign end date."},
                {"name": "chnl",     "type": "TEXT",    "desc": "Marketing channel (email, social, etc.)."}
            ],
            "pk": ["camp_id"]
        },
        "attr": {
            "__desc": "Attribution: links orders to campaigns (many-to-many).",
            "columns": [
                {"name": "ord_id",   "type": "INTEGER",   "desc": "Order identifier."},
                {"name": "camp_id",  "type": "INTEGER",   "desc": "Campaign identifier."},
                {"name": "touch_ts", "type": "TIMESTAMP", "desc": "Attribution touch timestamp (UTC)."}
            ],
            "pk": ["ord_id", "camp_id"]
        }
    },

    # FK graph
    "__fk": [
        {"left": ["sales","ord_hdr","cust_id"],   "right": ["sales","cust_dim","cust_id"]},
        {"left": ["sales","order_items","ord_id"],"right": ["sales","ord_hdr","ord_id"]},
        {"left": ["sales","order_items","prod_id"],"right": ["sales","prod_dim","prod_id"]},
        {"left": ["mkt","attr","ord_id"],         "right": ["sales","ord_hdr","ord_id"]},
        {"left": ["mkt","attr","camp_id"],        "right": ["mkt","camp","camp_id"]},
    ]
}

def dump_catalog_for_prompt(cat: Dict[str, Any]) -> str:
    lines = []
    for s, body in cat.items():
        if s.startswith("__"): continue
        s_desc = body.get("__desc", s)
        lines.append(f"- schema {s}: {s_desc}")
        for t, meta in body.items():
            if t.startswith("__"): continue
            t_desc = meta.get("__desc", f"{s}.{t}")
            cols = ", ".join(f"{c['name']}({c['type']})" for c in meta["columns"])
            lines.append(f"  * table {s}.{t}: {t_desc}")
            lines.append(f"    columns: {cols}")
    if cat.get("__fk"):
        lines.append("Join hints:")
        for fk in cat["__fk"]:
            lsch, ltab, lcol = fk["left"]
            rsch, rtab, rcol = fk["right"]
            lines.append(f"  - {lsch}.{ltab}.{lcol} = {rsch}.{rtab}.{rcol}")
    return "\n".join(lines)
