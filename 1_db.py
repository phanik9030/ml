from typing import Dict, Any, List
import duckdb
from catalog import CATALOG

_con = None  # singleton DuckDB connection

def _create_schemas_and_tables(con):
    # Create schemas
    schemas = [s for s in CATALOG.keys() if not s.startswith("__")]
    for s in schemas:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {s};")

    # Create tables
    for s in schemas:
        for t, meta in CATALOG[s].items():
            if t.startswith("__"):
                continue
            cols_sql = ", ".join(f"{c['name']} {c['type']}" for c in meta["columns"])
            con.execute(f"CREATE TABLE IF NOT EXISTS {s}.{t} ({cols_sql});")

def _seed_data(con):
    # Clear any existing rows (idempotent-ish for dev)
    for s in CATALOG.keys():
        if s.startswith("__"): continue
        for t in CATALOG[s].keys():
            if t.startswith("__"): continue
            con.execute(f"DELETE FROM {s}.{t};")

    # --- Seed sales.cust_dim ---
    con.execute("""
    INSERT INTO sales.cust_dim VALUES
    (100, 'US', 'SMB', '2024-01-10', 'alice@example.com'),
    (101, 'US', 'ENT', '2023-08-05', 'bob@example.com'),
    (102, 'DE', 'SMB', '2022-03-21', 'chen@example.com'),
    (103, 'US', 'MID', '2024-05-05', 'dina@example.com'),
    (104, 'IN', 'SMB', '2025-02-01', 'eli@example.com'),
    (105, 'US', 'ENT', '2025-03-12', 'fay@example.com');
    """)

    # --- Seed sales.prod_dim ---
    con.execute("""
    INSERT INTO sales.prod_dim VALUES
    (200, 'SKU-1', 'Widget A', 'Gadgets'),
    (201, 'SKU-2', 'Widget B', 'Gadgets'),
    (202, 'SKU-3', 'Cable C',  'Accessories'),
    (203, 'SKU-4', 'Cable D',  'Accessories'),
    (204, 'SKU-5', 'Adapter E','Accessories');
    """)

    # --- Seed sales.ord_hdr ---
    con.execute("""
    INSERT INTO sales.ord_hdr VALUES
    (1, 100, '2025-08-01', 120.50, 'USD', 'SMB', false),
    (2, 101, '2025-07-22',  45.00, 'USD', 'ENT', false),
    (3, 102, '2025-06-30',  60.00, 'EUR', 'SMB', false),
    (4, 100, '2025-01-15',  75.25, 'USD', 'SMB', true),
    (5, 103, '2024-12-01',  90.00, 'USD', 'MID', false),
    (6, 104, '2025-04-10', 150.00, 'USD', 'SMB', false),
    (7, 105, '2025-05-05', 220.00, 'USD', 'ENT', false),
    (8, 104, '2025-08-10',  35.00, 'INR', 'SMB', false),
    (9, 101, '2025-08-15',  80.00, 'USD', 'ENT', false),
    (10,105, '2025-08-20', 300.00, 'USD', 'ENT', false);
    """)

    # --- Seed sales.order_items ---
    con.execute("""
    INSERT INTO sales.order_items VALUES
    (1, 1, 200, 2, 50.00),
    (1, 2, 202, 1, 20.50),
    (2, 1, 201, 1, 45.00),
    (3, 1, 203, 2, 30.00),
    (4, 1, 200, 1, 75.25),
    (5, 1, 204, 3, 30.00),
    (6, 1, 201, 2, 75.00),
    (7, 1, 200, 4, 55.00),
    (7, 2, 204, 1,  5.00),
    (8, 1, 202, 1, 35.00),
    (9, 1, 203, 1, 80.00),
    (10,1, 200, 5, 60.00);
    """)

    # --- Seed mkt.camp ---
    con.execute("""
    INSERT INTO mkt.camp VALUES
    (10, 'My First Campaign', '2025-01-01', '2025-12-31', 'email'),
    (11, 'Holiday Push',      '2024-11-01', '2025-01-15', 'social'),
    (12, 'New Product Launch','2025-05-01', '2025-06-30', 'paid_search');
    """)

    # --- Seed mkt.attr ---
    con.execute("""
    INSERT INTO mkt.attr VALUES
    (1, 10, '2025-08-01 09:00:00'),
    (2, 10, '2025-07-22 11:30:00'),
    (3, 11, '2025-06-30 08:15:00'),
    (6, 12, '2025-04-10 10:00:00'),
    (7, 12, '2025-05-05 13:45:00'),
    (9, 10, '2025-08-15 07:40:00');
    """)

def get_connection():
    global _con
    if _con is None:
        _con = duckdb.connect(database=":memory:")
        _create_schemas_and_tables(_con)
        _seed_data(_con)
    return _con
