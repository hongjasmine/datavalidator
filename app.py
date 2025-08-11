# app.py
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Excel & DB Data Validation", layout="wide")
st.title("Data Validator")

# ---------- helpers ----------
def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.astype(str)
                  .str.strip()
                  .str.replace(r"\s+", "_", regex=True)
                  .str.replace(r"[^\w\-]", "", regex=True))
    return df

def schema_compare(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    cols = sorted(set(a.columns).union(b.columns))
    return pd.DataFrame([{
        "column": c,
        "in_A": c in a.columns,
        "in_B": c in b.columns,
        "dtype_A": str(a[c].dtype) if c in a.columns else None,
        "dtype_B": str(b[c].dtype) if c in b.columns else None,
    } for c in cols])

def column_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({
            "column": c, "dtype": str(s.dtype),
            "rows": len(s),
            "non_null": int(s.notna().sum()),
            "nulls": int(s.isna().sum()),
            "null_pct": float(s.isna().mean()) if len(s) else 0.0,
            "distinct": int(s.nunique(dropna=True)),
        })
    return pd.DataFrame(rows)

def find_duplicates(df: pd.DataFrame, keys):
    if not keys or not set(keys).issubset(df.columns): return pd.DataFrame()
    return df[df.duplicated(subset=keys, keep=False)].sort_values(keys)

def key_join_stats(a: pd.DataFrame, b: pd.DataFrame, keys):
    if not keys or not set(keys).issubset(a.columns) or not set(keys).issubset(b.columns):
        return {"matched": 0, "only_in_A": len(a), "only_in_B": len(b)}
    A = a[keys].drop_duplicates()
    B = b[keys].drop_duplicates()
    m = A.merge(B, on=keys, how="outer", indicator=True)["_merge"]
    return {
        "matched": int((m=="both").sum()),
        "only_in_A": int((m=="left_only").sum()),
        "only_in_B": int((m=="right_only").sum()),
    }

def row_diffs(a: pd.DataFrame, b: pd.DataFrame, keys):
    if not keys or not set(keys).issubset(a.columns) or not set(keys).issubset(b.columns):
        return pd.DataFrame()
    common = [c for c in a.columns if c in b.columns and c not in keys]
    if not common: return pd.DataFrame()
    a_idx = a.set_index(keys)[common]; b_idx = b.set_index(keys)[common]
    shared = a_idx.index.intersection(b_idx.index)
    diffs = []
    for idx in shared:
        ra, rb = a_idx.loc[idx], b_idx.loc[idx]
        changed = [c for c in common if not (pd.isna(ra[c]) and pd.isna(rb[c])) and str(ra[c]) != str(rb[c])]
        if changed:
            rec = {k:v for k,v in zip(keys, idx if isinstance(idx, tuple) else (idx,))}
            rec["changed_columns"] = ", ".join(changed)
            rec["changes"] = json.dumps({c: {"A": ra[c], "B": rb[c]} for c in changed}, default=str)
            diffs.append(rec)
    return pd.DataFrame(diffs)

# ---------- sidebar ----------
st.sidebar.header("Upload")
fA = st.sidebar.file_uploader("File A (.xlsx/.xls)", type=["xlsx","xls"])
fB = st.sidebar.file_uploader("File B (.xlsx/.xls)", type=["xlsx","xls"])

st.sidebar.header("Options")
normalize = st.sidebar.checkbox("Normalize headers", value=True)
keys_raw = st.sidebar.text_input("Key columns", value="")
out_fmt = st.sidebar.radio("Download format", ["CSV", "JSON"], index=0)

# ---------- main ----------
if fA and fB:
    dfA = pd.read_excel(fA)
    dfB = pd.read_excel(fB)
    if normalize:
        dfA = normalize_headers(dfA)
        dfB = normalize_headers(dfB)

    key_cols = [k.strip() for k in keys_raw.split(",") if k.strip()]
    st.markdown("### Schema Comparison")
    sch = schema_compare(dfA, dfB)
    st.dataframe(sch, use_container_width=True, height=240)

    st.markdown("### Column Quality")
    qa = column_quality(dfA).rename(columns=lambda c: f"A_{c}" if c!="column" else c)
    qb = column_quality(dfB).rename(columns=lambda c: f"B_{c}" if c!="column" else c)
    qual = qa.merge(qb, on="column", how="outer")
    st.dataframe(qual, use_container_width=True, height=280)

    st.markdown("### Key Integrity & Duplicates")
    stats = key_join_stats(dfA, dfB, key_cols)
    c1, c2, c3 = st.columns(3)
    c1.metric("Matched keys", stats["matched"])
    c2.metric("Only in A", stats["only_in_A"])
    c3.metric("Only in B", stats["only_in_B"])
    dupA = find_duplicates(dfA, key_cols); dupB = find_duplicates(dfB, key_cols)
    if not dupA.empty: st.caption("Duplicate keys in A"); st.dataframe(dupA, use_container_width=True)
    if not dupB.empty: st.caption("Duplicate keys in B"); st.dataframe(dupB, use_container_width=True)

    st.markdown("### Row Differences (by key)")
    diffs = row_diffs(dfA, dfB, key_cols)
    if diffs.empty:
        st.info("No diffs for matched keys (or keys missing).")
    else:
        st.dataframe(diffs, use_container_width=True, height=320)

    st.markdown("---")
    st.subheader("Download")
    if out_fmt == "CSV":
        st.download_button("⬇️ schema.csv", sch.to_csv(index=False), "schema.csv", "text/csv")
        st.download_button("⬇️ quality.csv", qual.to_csv(index=False), "quality.csv", "text/csv")
        st.download_button("⬇️ diffs.csv", diffs.to_csv(index=False), "diffs.csv", "text/csv")
    else:
        bundle = {
            "schema": json.loads(sch.to_json(orient="records")),
            "quality": json.loads(qual.to_json(orient="records")),
            "diffs": json.loads(diffs.to_json(orient="records")),
        }
        report = json.dumps(bundle, indent=2, default=str).encode("utf-8")
        st.download_button("⬇️ validation_report.json", report, "validation_report.json", "application/json")
else:
    st.info("Upload two Excel files in the sidebar to begin.")
