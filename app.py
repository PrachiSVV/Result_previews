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
# ====== Boot & Auth ======
# =========================
def init_env():
    load_dotenv(override=False)

def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if "secrets" in dir(st) and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

def check_credentials(username: str, password: str) -> bool:
    return (
        username == get_secret("AUTH_USER", "") and
        password == get_secret("AUTH_PASS", "")
    )

def show_login():
    st.title("🔐 Sign in")
    with st.form("login_form", clear_on_submit=False):
        u = st.text_input("Username", value="", autocomplete="username")
        p = st.text_input("Password", value="", type="password", autocomplete="current-password")
        ok = st.form_submit_button("Sign In")
    if ok:
        if check_credentials(u, p):
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Invalid credentials.")


# =========================
# ====== DB & Cache =======
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
    coll = get_collection()
    # Safe regex
    try:
        query = {"_id": {"$regex": regex_str or ".*", "$options": "i"}}
    except Exception:
        query = {"_id": {"$regex": re.escape(regex_str or ".*"), "$options": "i"}}
    return list(coll.find(query))


# =========================
# ====== Core Utils =======
# =========================

NUM_SORTABLE = [
    "expected_sales", "expected_ebitda", "expected_pat",
    "sales_yoy_pct", "ebitda_yoy_pct", "pat_yoy_pct",
    "ebitda_margin_yoy_bps", "pat_margin_yoy_bps"
]


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return None
        return float(x)
    except Exception:
        return None
def flatten_for_df(doc: Dict[str, Any]) -> Dict[str, Any]:
    company = doc.get("company") or ""
    smap = doc.get("symbolmap") or {}
    return {
        "_id": doc.get("_id"),
        "broker_name": doc.get("broker_name"),
        "report_period": doc.get("report_period"),
        "basis": doc.get("basis"),
        "company_name": doc.get("company_name"),
        "nse": smap.get("NSE"),
        "bse": smap.get("BSE"),
        "isin": company,
        "expected_sales": to_float(doc.get("expected_sales")),
        "expected_ebitda": to_float(doc.get("expected_ebitda")),
        "expected_pat": to_float(doc.get("expected_pat")),
        "sales_yoy_pct": to_float(doc.get("sales_yoy_pct")),
        "ebitda_yoy_pct": to_float(doc.get("ebitda_yoy_pct")),
        "pat_yoy_pct": to_float(doc.get("pat_yoy_pct")),
        # use new BPS fields instead of margin %
        "ebitda_margin_yoy_bps": to_float(doc.get("ebitda_margin_yoy_bps")),
        "pat_margin_yoy_bps": to_float(doc.get("pat_margin_yoy_bps")),
        "source_file": doc.get("source_file"),
        "source_unit": doc.get("source_unit"),
    }

def build_dataframe(docs: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame([flatten_for_df(d) for d in docs])
    # drop rows where *all numeric fields* are NaN
    numeric_cols = [
        "expected_sales", "expected_ebitda", "expected_pat",
        "sales_yoy_pct", "ebitda_yoy_pct", "pat_yoy_pct",
        "ebitda_margin_yoy_bps", "pat_margin_yoy_bps"
    ]
    if not df.empty:
        df = df.dropna(subset=numeric_cols, how="all")
    return df


# def flatten_for_df(doc: Dict[str, Any]) -> Dict[str, Any]:
#     company = doc.get("company") or {}
#     smap = doc.get("symbol_map_raw") or {}
#     return {
#         "_id": doc.get("_id"),
#         "broker_name": doc.get("broker_name"),
#         "report_period": doc.get("report_period"),
#         "basis": doc.get("basis"),
#         "company_name": smap.get("name") or doc.get("company_name"),
#         "nse": smap.get("nse") or smap.get("NSE"),
#         "bse": smap.get("bse") or smap.get("BSE"),
#         "isin": smap.get("isin") or company,
#         "expected_sales": to_float(doc.get("expected_sales")),
#         "expected_ebitda": to_float(doc.get("expected_ebitda")),
#         "expected_pat": to_float(doc.get("expected_pat")),
#         "sales_yoy_pct": to_float(doc.get("sales_yoy_pct")),
#         "ebitda_yoy_pct": to_float(doc.get("ebitda_yoy_pct")),
#         "pat_yoy_pct": to_float(doc.get("pat_yoy_pct")),
#         "ebitda_margin_percent": to_float(doc.get("ebitda_margin_percent")),
#         "pat_margin_percent": to_float(doc.get("pat_margin_percent")),
#         "source_file": doc.get("source_file"),
#         "source_unit": doc.get("source_unit"),
#     }

# def build_dataframe(docs: List[Dict[str, Any]]) -> pd.DataFrame:
#     return pd.DataFrame([flatten_for_df(d) for d in docs])

def ms_options(df: pd.DataFrame, col: str) -> List[str]:
    return sorted([str(v) for v in df[col].dropna().unique().tolist()])

def col_range(df: pd.DataFrame, col: str) -> Tuple[float, float]:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return (0.0, 0.0)
    return float(s.min()), float(s.max())
def apply_filters(df: pd.DataFrame, f: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()

    # Text/regex
    if f["company_contains"]:
        out = out[out["company_name"].fillna("").str.contains(f["company_contains"], case=False, na=False)]
    if f["nse_contains"]:
        out = out[out["nse"].fillna("").str.contains(f["nse_contains"], case=False, na=False)]

    # Categorical
    if f["broker"]:
        out = out[out["broker_name"].isin(f["broker"])]
    if f["period"]:
        out = out[out["report_period"].isin(f["period"])]
    if f["basis"]:
        out = out[out["basis"].isin(f["basis"])]

    # Numeric ranges — keep NaNs (pass-through)
    num_cols = [
        "sales_yoy_pct", "ebitda_yoy_pct", "pat_yoy_pct",
        "ebitda_margin_yoy_bps", "pat_margin_yoy_bps",
        "expected_sales", "expected_ebitda", "expected_pat"
    ]
    for key in num_cols:
        lo, hi = f[key]
        s = pd.to_numeric(out[key], errors="coerce")
        mask = s.between(lo, hi) | s.isna()   # <= THIS keeps rows with NaN
        out = out[mask]

    return out


def apply_filters_old(df: pd.DataFrame, f: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()

    # Text/regex
    if f["company_contains"]:
        out = out[out["company_name"].fillna("").str.contains(f["company_contains"], case=False, na=False)]
    if f["nse_contains"]:
        out = out[out["nse"].fillna("").str.contains(f["nse_contains"], case=False, na=False)]

    # Categorical
    if f["broker"]:
        out = out[out["broker_name"].isin(f["broker"])]
    if f["period"]:
        out = out[out["report_period"].isin(f["period"])]
    if f["basis"]:
        out = out[out["basis"].isin(f["basis"])]

    # Numeric sliders
    for key in ["sales_yoy_pct", "ebitda_yoy_pct", "pat_yoy_pct",
                "ebitda_margin_yoy_bps", "pat_margin_yoy_bps",
                "expected_sales", "expected_ebitda", "expected_pat"]:
        lo, hi = f[key]
        series = pd.to_numeric(out[key], errors="coerce")
        out = out[(series >= lo) & (series <= hi)]

    return out

def sort_df(df: pd.DataFrame, key: str, asc: bool) -> pd.DataFrame:
    if key in df.columns:
        return df.sort_values(key, ascending=asc, kind="mergesort", na_position="last")
    return df


# =========================
# ========= UI ============ 
# =========================
st.set_page_config(page_title="MOFSL Research Browser", page_icon="📊", layout="wide")
init_env()

# Global CSS (compact, trader-friendly)
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    [data-testid="stMetricValue"] { font-size: 1.15rem; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; color: #6b7280; }
    .company-text p { margin: 0.1rem 0; font-size: 0.92rem; }
    .section-h { font-weight: 600; font-size: 1.05rem; margin: 0.4rem 0 0.3rem; }
    .small-cap { font-size: 0.85rem; color: #6b7280; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- AUTH GATE ---
if not st.session_state.get("auth_ok", False):
    show_login()
    st.stop()

# ===== Sidebar: Filters & Sorting =====
DEFAULT_REGEX = "Q2FY26E"

with st.sidebar:
    st.title("🔎 Filters & Sorting")
    # Clear filters button
    if st.button("🧹 Clear All Filters"):
        for k in list(st.session_state.keys()):
            if k.startswith("flt_") or k in ("sort_field", "sort_asc"):
                del st.session_state[k]
        st.session_state["flt_regex"] = DEFAULT_REGEX
        st.session_state["sort_field"] = "_id"
        st.session_state["sort_asc"] = True
        st.rerun()


    # Regex on _id (server query)
    flt_regex = st.text_input("Regex on _id", value=st.session_state.get("flt_regex", DEFAULT_REGEX), key="flt_regex")

# Query DB by regex first (fast narrowing)
docs = load_docs(flt_regex)
if not docs:
    st.warning("No documents found for the given _id filter.")
    st.stop()

df_all = build_dataframe(docs)

with st.sidebar:
    broker_sel = st.multiselect("Broker(s)", options=ms_options(df_all, "broker_name"),
                                default=st.session_state.get("flt_broker", []), key="flt_broker")
    period_sel = st.multiselect("Report Period(s)", options=ms_options(df_all, "report_period"),
                                default=st.session_state.get("flt_period", []), key="flt_period")
    basis_sel = st.multiselect("Basis", options=ms_options(df_all, "basis"),
                               default=st.session_state.get("flt_basis", []), key="flt_basis")

    company_contains = st.text_input("Company contains", value=st.session_state.get("flt_company", ""), key="flt_company")
    nse_contains = st.text_input("NSE contains", value=st.session_state.get("flt_nse", ""), key="flt_nse")

    # Numeric ranges (derive ranges from data)
    def slider_for(col: str, label: str):
        lo, hi = col_range(df_all, col)
        if lo == hi:
            hi = hi if hi != 0 else 1.0
        return st.slider(label, min_value=float(lo), max_value=float(hi),
                         value=st.session_state.get(f"flt_{col}", (float(lo), float(hi))),
                         key=f"flt_{col}")

    sales_y = slider_for("sales_yoy_pct", "Sales YoY %")
    ebitda_y = slider_for("ebitda_yoy_pct", "EBITDA YoY %")
    pat_y = slider_for("pat_yoy_pct", "PAT YoY %")
    ebitda_mg = slider_for("ebitda_margin_yoy_bps", "EBITDA Margin YoY (bps)")
    pat_mg    = slider_for("pat_margin_yoy_bps", "PAT Margin YoY (bps)")

    exp_sales = slider_for("expected_sales", "Sales (mn)")
    exp_ebitda = slider_for("expected_ebitda", "EBITDA (mn)")
    exp_pat = slider_for("expected_pat", "PAT (mn)")

    st.divider()
    sort_field = st.selectbox("Sort by", options=["_id", "company_name", "nse"] + NUM_SORTABLE,
                              index=["_id", "company_name", "nse"].index(st.session_state.get("sort_field", "_id")) if st.session_state.get("sort_field", "_id") in ["_id","company_name","nse"] else 0,
                              key="sort_field")
    sort_asc = st.toggle("Ascending", value=st.session_state.get("sort_asc", True), key="sort_asc")

# Apply in-memory filters
filters = {
    "company_contains": company_contains,
    "nse_contains": nse_contains,
    "broker": broker_sel,
    "period": period_sel,
    "basis": basis_sel,
    "sales_yoy_pct": sales_y,
    "ebitda_yoy_pct": ebitda_y,
    "pat_yoy_pct": pat_y,
    "ebitda_margin_yoy_bps": ebitda_mg,
    "pat_margin_yoy_bps": pat_mg,
    "expected_sales": exp_sales,
    "expected_ebitda": exp_ebitda,
    "expected_pat": exp_pat,
}
df_filtered = apply_filters(df_all, filters)
df_sorted = sort_df(df_filtered, sort_field, sort_asc)

# ====== Top toolbar ======
# ====== Top toolbar ======
# ====== Header ======
st.subheader("📊 MOFSL Research Browser")
st.caption(f"Loaded: {len(df_all)} • After filters: {len(df_sorted)}")

# tcol1, tcol2, tcol3, tcol4 = st.columns([2, 1, 1, 2])
# with tcol1:
#     st.subheader("📊 MOFSL Research Browser")
# with tcol2:
#     rows = st.number_input("Rows", min_value=10, max_value=1000, value=st.session_state.get("rows", 100), step=10, key="rows")
# with tcol3:
#     total_pages = max(1, (len(df_sorted) + rows - 1) // rows)
#     page = st.number_input("Page", min_value=1, max_value=total_pages, value=min(st.session_state.get("page", 1), total_pages), step=1, key="page")
# with tcol4:
#     st.caption(f"Loaded: {len(df_all)} • Filtered: {len(df_sorted)} • Pages: {total_pages}")

# ====== Main table ======
view = df_sorted.reset_index(drop=True)
# start = (page - 1) * rows
# end = start + rows
# view = df_sorted.iloc[start:end].reset_index(drop=True)


visible_cols = [
    "_id", "broker_name", "report_period", "basis",
    "company_name", "nse", "bse", "isin",
    "expected_sales", "expected_ebitda", "expected_pat",
    "sales_yoy_pct", "ebitda_yoy_pct", "pat_yoy_pct",
    "ebitda_margin_yoy_bps", "pat_margin_yoy_bps",
    "source_file", "source_unit",
]
existing_cols = [c for c in visible_cols if c in view.columns]
st.dataframe(view[existing_cols], use_container_width=True, hide_index=True)

# ====== Detail panel ======
st.markdown('<div class="section-h">📘 Document Details</div>', unsafe_allow_html=True)

if not view.empty:
    row_idx = st.number_input("Select row # (1-based in current page)", min_value=1, max_value=len(view), value=1, step=1)
    selected_row = view.iloc[row_idx - 1]
    sel_id = str(selected_row["_id"])
    doc = next((d for d in docs if str(d.get("_id")) == sel_id), None)

    if doc:
        # Top badges
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Broker", doc.get("broker_name", "-"))
        c2.metric("Period", doc.get("report_period", "-"))
        c3.metric("Basis", doc.get("basis", "-"))
        c4.metric("Unit", doc.get("source_unit", "-"))

        # Company compact block
        st.markdown('<div class="section-h">Company</div>', unsafe_allow_html=True)
        co = doc.get("company") or {}
        sm = doc.get("symbol_map_raw") or {}
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.markdown(f'<div class="company-text"><p><b>{sm.get("name") or doc.get("company_name","-")}</b></p></div>', unsafe_allow_html=True)
        cc2.markdown(f'<div class="company-text"><p>NSE: {sm.get("nse") or sm.get("NSE","-")}</p></div>', unsafe_allow_html=True)
        cc3.markdown(f'<div class="company-text"><p>BSE: {sm.get("bse") or sm.get("BSE","-")}</p></div>', unsafe_allow_html=True)
        cc4.markdown(f'<div class="company-text"><p>ISIN: {sm.get("isin") or sm.get("company","-")}</p></div>', unsafe_allow_html=True)

        # Compact KPIs
        st.markdown('<div class="section-h">Key Figures</div>', unsafe_allow_html=True)
        k = st.columns(5)
        k[0].metric("Sales (mn)", f"{to_float(doc.get('expected_sales')) or 0:,.0f}",
                    f"{to_float(doc.get('sales_yoy_pct')) or 0:.2f}% YoY")
        k[1].metric("EBITDA (mn)", f"{to_float(doc.get('expected_ebitda')) or 0:,.0f}",
                    f"{to_float(doc.get('ebitda_yoy_pct')) or 0:.2f}% YoY")
        k[2].metric("PAT (mn)", f"{to_float(doc.get('expected_pat')) or 0:,.0f}",
                    f"{to_float(doc.get('pat_yoy_pct')) or 0:.2f}% YoY")
        k[3].metric("EBITDA Margin YoY (bps)", f"{to_float(doc.get('ebitda_margin_yoy_bps')) or 0:,.0f}")
        k[4].metric("PAT Margin YoY (bps)", f"{to_float(doc.get('pat_margin_yoy_bps')) or 0:,.0f}")


        # Helper for breakdown tables
        def cols_to_df(d: Dict[str, Any], mapping: Dict[str, str]) -> pd.DataFrame:
            rows_ = []
            for label, key in mapping.items():
                dct = d.get(key) or {}
                if isinstance(dct, dict) and dct:
                    rows_.append({"Metric": label, "2Q": to_float(dct.get("2Q")), "2QE": to_float(dct.get("2QE"))})
            return pd.DataFrame(rows_) if rows_ else pd.DataFrame(columns=["Metric", "2Q", "2QE"])

        # Breakdowns
        st.markdown('<div class="section-h">2Q vs 2QE — Sales / EBITDA / PAT</div>', unsafe_allow_html=True)
        df_main = cols_to_df(doc, {
            "Sales": "expected_sales_cols",
            "EBITDA": "expected_ebitda_cols",
            "PAT": "expected_pat_cols",
        })
        if not df_main.empty:
            st.dataframe(df_main, use_container_width=True, hide_index=True)
        else:
            st.caption("No Sales/EBITDA/PAT breakdown found.", help="expected_*_cols missing")

        st.markdown('<div class="section-h">2Q vs 2QE — Margins</div>', unsafe_allow_html=True)
        df_m = cols_to_df(doc, {
            "EBITDA Margin %": "ebitda_margin_percent_cols",
            "PAT Margin %": "pat_margin_percent_cols",
        })
        if not df_m.empty:
            st.dataframe(df_m, use_container_width=True, hide_index=True)
        else:
            st.caption("No margin breakdown found.", help="*_margin_percent_cols missing")

        # Raw + export
        with st.expander("Raw JSON"):
            st.json(doc)

        sel_json = json.dumps(doc, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button("Download selected JSON", data=sel_json, file_name=f"{sel_id}.json", mime="application/json")

# ===== Exports for filtered table =====
csv_bytes = df_sorted[existing_cols].to_csv(index=False).encode("utf-8")
json_bytes = json.dumps(df_sorted[existing_cols].to_dict(orient="records"), ensure_ascii=False, indent=2).encode("utf-8")

exp1, exp2 = st.columns(2)
with exp1:
    st.download_button("⬇️ Download filtered CSV", data=csv_bytes, file_name="filtered_docs.csv", mime="text/csv")
with exp2:
    st.download_button("⬇️ Download filtered JSON", data=json_bytes, file_name="filtered_docs.json", mime="application/json")
