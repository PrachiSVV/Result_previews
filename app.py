import os
import re
import json
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection


# =========================
# ========== Auth =========
# =========================
def init_env():
    # Load .env for local; on Streamlit Cloud use Secrets
    load_dotenv(override=False)

def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    # Priority: st.secrets -> env var -> default
    try:
        if "secrets" in dir(st) and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

def check_credentials(username: str, password: str) -> bool:
    valid_user = get_secret("AUTH_USER", "")
    valid_pass = get_secret("AUTH_PASS", "")
    return (username == valid_user) and (password == valid_pass)

def login_page() -> bool:
    st.title("🔐 Login")
    with st.form("login_form", clear_on_submit=False):
        u = st.text_input("Username", value="", autocomplete="username")
        p = st.text_input("Password", value="", type="password", autocomplete="current-password")
        ok = st.form_submit_button("Sign In")
    if ok:
        if check_credentials(u, p):
            st.session_state["auth_ok"] = True
            st.success("Login successful.")
            return True
        else:
            st.error("Invalid credentials.")
            st.session_state["auth_ok"] = False
            return False
    return st.session_state.get("auth_ok", False)


# =========================
# ====== DB & Caching =====
# =========================
@st.cache_resource(show_spinner=False)
def get_collection() -> Collection:
    mongo_uri = get_secret("MONGO_URI", "")
    if not mongo_uri:
        st.error("Missing MONGO_URI. Set it in .env or Streamlit Secrets.")
        st.stop()
    client = MongoClient(mongo_uri)
    return client["CAG_CHATBOT"]["MOFSL_Data"]

@st.cache_data(show_spinner=True, ttl=300)
def load_docs(regex_str: str) -> List[Dict[str, Any]]:
    """Load docs whose _id matches regex_str (case-insensitive)."""
    coll = get_collection()
    # Sanitize bad regex; fallback to literal substring
    try:
        regex = {"$regex": regex_str, "$options": "i"} if regex_str else {"$regex": ".*"}
    except Exception:
        regex = {"$regex": re.escape(regex_str), "$options": "i"}
    cursor = coll.find({"_id": regex})
    return list(cursor)


# =========================
# ====== UI Helpers =======
# =========================
NUM_KEYS_SORT_DEFAULT = [
    "expected_sales", "expected_ebitda", "expected_pat",
    "sales_yoy_pct", "ebitda_yoy_pct", "pat_yoy_pct",
    "ebitda_margin_percent", "pat_margin_percent"
]

def to_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return None
        return float(x)
    except Exception:
        return None

def flatten_for_df(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a flat dict with key fields for table view."""
    company = doc.get("company") or {}
    smap = doc.get("symbol_map_raw") or {}
    flat = {
        "_id": doc.get("_id"),
        "broker_name": doc.get("broker_name"),
        "report_period": doc.get("report_period"),
        "basis": doc.get("basis"),
        "company_name": company.get("name") or doc.get("company_name"),
        "nse": company.get("nse") or smap.get("NSE"),
        "bse": company.get("bse") or smap.get("BSE"),
        "isin": company.get("isin") or smap.get("company"),
        "expected_sales": to_float(doc.get("expected_sales")),
        "expected_ebitda": to_float(doc.get("expected_ebitda")),
        "expected_pat": to_float(doc.get("expected_pat")),
        "sales_yoy_pct": to_float(doc.get("sales_yoy_pct")),
        "ebitda_yoy_pct": to_float(doc.get("ebitda_yoy_pct")),
        "pat_yoy_pct": to_float(doc.get("pat_yoy_pct")),
        "ebitda_margin_percent": to_float(doc.get("ebitda_margin_percent")),
        "pat_margin_percent": to_float(doc.get("pat_margin_percent")),
        "source_file": doc.get("source_file"),
        "source_unit": doc.get("source_unit"),
    }
    return flat

def build_dataframe(docs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = [flatten_for_df(d) for d in docs]
    df = pd.DataFrame(rows)
    return df

def multi_select_options(df: pd.DataFrame, col: str) -> List[str]:
    vals = sorted([str(v) for v in df[col].dropna().unique().tolist()])
    return vals

def range_of(df: pd.DataFrame, col: str) -> Tuple[float, float]:
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return (0.0, 0.0)
    return (float(series.min()), float(series.max()))

def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    # text/regex filters
    if filters["id_regex"]:
        try:
            out = out[out["_id"].str.contains(filters["id_regex"], case=False, na=False, regex=True)]
        except Exception:
            out = out[out["_id"].str.contains(re.escape(filters["id_regex"]), case=False, na=False, regex=True)]
    if filters["company_contains"]:
        out = out[out["company_name"].fillna("").str.contains(filters["company_contains"], case=False, na=False)]
    if filters["nse_contains"]:
        out = out[out["nse"].fillna("").str.contains(filters["nse_contains"], case=False, na=False)]

    # categorical filters
    if filters["broker"]:
        out = out[out["broker_name"].isin(filters["broker"])]
    if filters["period"]:
        out = out[out["report_period"].isin(filters["period"])]
    if filters["basis"]:
        out = out[out["basis"].isin(filters["basis"])]

    # numeric ranges
    for key in ["sales_yoy_pct", "ebitda_yoy_pct", "pat_yoy_pct",
                "ebitda_margin_percent", "pat_margin_percent"]:
        lo, hi = filters[key]
        out = out[(pd.to_numeric(out[key], errors="coerce").fillna(float("-inf")) >= lo) &
                  (pd.to_numeric(out[key], errors="coerce").fillna(float("inf")) <= hi)]

    # headline value ranges (optional)
    for key in ["expected_sales", "expected_ebitda", "expected_pat"]:
        lo, hi = filters[key]
        out = out[(pd.to_numeric(out[key], errors="coerce").fillna(float("-inf")) >= lo) &
                  (pd.to_numeric(out[key], errors="coerce").fillna(float("inf")) <= hi)]
    return out

def sort_df(df: pd.DataFrame, sort_key: str, ascending: bool) -> pd.DataFrame:
    if sort_key in df.columns:
        return df.sort_values(by=sort_key, ascending=ascending, kind="mergesort", na_position="last")
    return df


# =========================
# ========== App ==========
# =========================
st.set_page_config(page_title="MOFSL Research Browser", page_icon="📄", layout="wide")
init_env()

# --- Auth gate ---
if not login_page():
    st.stop()

# --- Defaults & Sidebar ---
DEFAULT_REGEX = "Q2FY26E"
st.sidebar.title("🔎 Filters & Sorting")

id_regex = st.sidebar.text_input("Regex on _id", value=DEFAULT_REGEX, help="Case-insensitive regex; leave blank to load all.")
docs = load_docs(id_regex)

if not docs:
    st.warning("No documents found for the given filter.")
    st.stop()

df_all = build_dataframe(docs)

# Categorical filters
broker_opts = multi_select_options(df_all, "broker_name")
period_opts = multi_select_options(df_all, "report_period")
basis_opts = multi_select_options(df_all, "basis")

broker_sel = st.sidebar.multiselect("Broker(s)", options=broker_opts, default=[])
period_sel = st.sidebar.multiselect("Report Period(s)", options=period_opts, default=[])
basis_sel = st.sidebar.multiselect("Basis", options=basis_opts, default=[])

company_contains = st.sidebar.text_input("Company contains", value="")
nse_contains = st.sidebar.text_input("NSE contains", value="")

# Numeric sliders (compute ranges from data)
ranges = {}
for col in ["sales_yoy_pct", "ebitda_yoy_pct", "pat_yoy_pct",
            "ebitda_margin_percent", "pat_margin_percent",
            "expected_sales", "expected_ebitda", "expected_pat"]:
    lo, hi = range_of(df_all, col)
    # Avoid degenerate slider (lo==hi)
    if lo == hi:
        lo, hi = float(lo), float(hi if hi != 0 else 1.0)
    ranges[col] = st.sidebar.slider(
        f"{col}", min_value=float(lo), max_value=float(hi), value=(float(lo), float(hi))
    )

filters = {
    "id_regex": id_regex,
    "company_contains": company_contains,
    "nse_contains": nse_contains,
    "broker": broker_sel,
    "period": period_sel,
    "basis": basis_sel,
    "sales_yoy_pct": ranges["sales_yoy_pct"],
    "ebitda_yoy_pct": ranges["ebitda_yoy_pct"],
    "pat_yoy_pct": ranges["pat_yoy_pct"],
    "ebitda_margin_percent": ranges["ebitda_margin_percent"],
    "pat_margin_percent": ranges["pat_margin_percent"],
    "expected_sales": ranges["expected_sales"],
    "expected_ebitda": ranges["expected_ebitda"],
    "expected_pat": ranges["expected_pat"],
}

# Sorting controls
sort_field = st.sidebar.selectbox("Sort by", options=["_id", "company_name", "nse"] + NUM_KEYS_SORT_DEFAULT, index=0)
ascending = st.sidebar.toggle("Ascending", value=True)

# --- Apply filters & sorting ---
df_filtered = apply_filters(df_all, filters)
df_sorted = sort_df(df_filtered, sort_field, ascending)

# --- Main UI ---
st.title("📄 MOFSL Research Browser")
top_k = st.slider("Rows per page", 10, 1000, 100, step=10)
st.caption(f"Loaded: {len(df_all)} docs • After filters: {len(df_sorted)}")

# Paginate
page = st.number_input("Page", min_value=1, value=1, step=1)
start = (page - 1) * top_k
end = start + top_k
view = df_sorted.iloc[start:end].reset_index(drop=True)

# Reorder visible columns for clarity
visible_cols = [
    "_id", "broker_name", "report_period", "basis", "company_name", "nse", "bse", "isin",
    "expected_sales", "expected_ebitda", "expected_pat",
    "sales_yoy_pct", "ebitda_yoy_pct", "pat_yoy_pct",
    "ebitda_margin_percent", "pat_margin_percent",
    "source_file", "source_unit",
]
existing_cols = [c for c in visible_cols if c in view.columns]
st.dataframe(view[existing_cols], use_container_width=True, hide_index=True)

# Details panel
st.subheader("📘 Document Details")
if not view.empty:
    row_idx = st.number_input("Select row # (1-based in current page)", min_value=1, max_value=len(view), value=1, step=1)
    selected_row = view.iloc[row_idx - 1]
    # find original doc
    sel_id = str(selected_row["_id"])
    doc = next((d for d in docs if str(d.get("_id")) == sel_id), None)
    if doc:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Broker", doc.get("broker_name", "-"))
        c2.metric("Period", doc.get("report_period", "-"))
        c3.metric("Basis", doc.get("basis", "-"))
        c4.metric("Unit", doc.get("source_unit", "-"))

        # Company block
        st.markdown("**Company**")
        company = doc.get("company") or {}
        co1, co2, co3, co4 = st.columns(4)
        co1.write(company.get("name") or doc.get("company_name", "-"))
        co2.write(company.get("nse") or (doc.get("symbol_map_raw") or {}).get("NSE", ""))
        co3.write(company.get("bse") or (doc.get("symbol_map_raw") or {}).get("BSE", ""))
        co4.write(company.get("isin") or (doc.get("symbol_map_raw") or {}).get("company", ""))

        # KPIs
        k = st.columns(5)
        k[0].metric("Sales (mn)", f"{to_float(doc.get('expected_sales')) or 0:,.0f}",
                    f"{to_float(doc.get('sales_yoy_pct')) or 0:.2f}% YoY")
        k[1].metric("EBITDA (mn)", f"{to_float(doc.get('expected_ebitda')) or 0:,.0f}",
                    f"{to_float(doc.get('ebitda_yoy_pct')) or 0:.2f}% YoY")
        k[2].metric("PAT (mn)", f"{to_float(doc.get('expected_pat')) or 0:,.0f}",
                    f"{to_float(doc.get('pat_yoy_pct')) or 0:.2f}% YoY")
        k[3].metric("EBITDA Margin %", f"{to_float(doc.get('ebitda_margin_percent')) or 0:.2f}")
        k[4].metric("PAT Margin %", f"{to_float(doc.get('pat_margin_percent')) or 0:.2f}")

        # Breakdown tables/charts
        def cols_to_df(doc_: Dict[str, Any], mapping: Dict[str, str]) -> pd.DataFrame:
            rows_ = []
            for label, key in mapping.items():
                dct = doc_.get(key) or {}
                if isinstance(dct, dict) and dct:
                    rows_.append({
                        "Metric": label,
                        "1Q": to_float(dct.get("1Q")),
                        "1QE": to_float(dct.get("1QE")),
                    })
            return pd.DataFrame(rows_) if rows_ else pd.DataFrame(columns=["Metric", "1Q", "1QE"])

        st.markdown("**1Q vs 1QE — Sales / EBITDA / PAT**")
        key_map_main = {
            "Sales": "expected_sales_cols",
            "EBITDA": "expected_ebitda_cols",
            "PAT": "expected_pat_cols",
        }
        df_main = cols_to_df(doc, key_map_main)
        if not df_main.empty:
            st.dataframe(df_main, use_container_width=True)
            df_long = df_main.melt(id_vars=["Metric"], var_name="Period", value_name="Value")
            st.bar_chart(df_long.pivot(index="Metric", columns="Period", values="Value"))
        else:
            st.info("No Sales/EBITDA/PAT breakdown found.")

        st.markdown("**1Q vs 1QE — Margins**")
        key_map_margins = {
            "EBITDA Margin %": "ebitda_margin_percent_cols",
            "PAT Margin %": "pat_margin_percent_cols",
        }
        df_m = cols_to_df(doc, key_map_margins)
        if not df_m.empty:
            st.dataframe(df_m, use_container_width=True)
            df_m_long = df_m.melt(id_vars=["Metric"], var_name="Period", value_name="Value")
            st.bar_chart(df_m_long.pivot(index="Metric", columns="Period", values="Value"))
        else:
            st.info("No margin breakdown found.")

        with st.expander("Raw JSON"):
            st.json(doc)

        # Exports for current selection
        sel_json = json.dumps(doc, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button("Download selected JSON", data=sel_json, file_name=f"{sel_id}.json", mime="application/json")

# Exports for filtered table
csv_bytes = df_sorted[existing_cols].to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv_bytes, file_name="filtered_docs.csv", mime="text/csv")

json_export = json.dumps(df_sorted[existing_cols].to_dict(orient="records"), ensure_ascii=False, indent=2).encode("utf-8")
st.download_button("Download filtered JSON", data=json_export, file_name="filtered_docs.json", mime="application/json")

st.caption("Tip: Add/modify MONGO_URI, AUTH_USER, AUTH_PASS in Secrets or .env. Cache TTL is 5 minutes for queries.")
