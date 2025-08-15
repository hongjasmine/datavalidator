# app.py
import json
import pandas as pd
import streamlit as st



st.set_page_config(page_title="Data Validator", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "SQL Server Integration"])

if page == "SQL Server Integration":
    st.title("SQL Server Integration (Demo)")
    st.write("""
This section demonstrates how you could integrate SQL Server with your Python app using the `pyodbc` library.

**Example connection code:**
    """)
    st.code('''
import pyodbc

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=your_server_name;'
    'DATABASE=your_database_name;'
    'UID=your_username;'
    'PWD=your_password'
)
cursor = conn.cursor()
cursor.execute("SELECT @@VERSION;")
row = cursor.fetchone()
print(row)
conn.close()
''', language='python')
    st.info("Replace the connection parameters with your actual SQL Server details.")
    st.stop()

st.title("Data Validator")

# ---------- helpers ----------
def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.astype(str)
                  .str.strip()
                  .str.replace(r"\s+", "_", regex=True)
                  .str.replace(r"[^\w\-]", "", regex=True))
    return df

def load_file(uploaded_file, csv_sep=",", json_lines=False):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file, sep=csv_sep)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    elif name.endswith(".json"):
        # Try user choice first; then fall back to the other mode
        try:
            return pd.read_json(uploaded_file, lines=json_lines)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_json(uploaded_file, lines=not json_lines)
    else:
        st.error(f"Unsupported file type: {name}. Please upload CSV, Excel, or JSON.")
        return None

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
        total_rows = len(s)
        non_null = int(s.notna().sum())
        nulls = int(s.isna().sum())
        null_pct = (s.isna().mean() * 100) if total_rows else 0.0
        distinct = int(s.nunique(dropna=True))
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "rows": str(total_rows),
            "non_null": str(non_null),
            "nulls": str(nulls),
            "null_pct": f"{int(round(null_pct))}%",
            "distinct": str(distinct),
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
        changed = [c for c in common
                   if not (pd.isna(ra[c]) and pd.isna(rb[c])) and str(ra[c]) != str(rb[c])]
        if changed:
            rec = {k:v for k,v in zip(keys, idx if isinstance(idx, tuple) else (idx,))}
            rec["changed_columns"] = ", ".join(changed)
            rec["changes"] = json.dumps({c: {"A": ra[c], "B": rb[c]} for c in changed}, default=str)
            diffs.append(rec)
    return pd.DataFrame(diffs)

# ---------- sidebar ----------
# Initialize variables to avoid possibly unbound warnings
fA = fB = None
normalize = True
csv_sep = ","
csv_sep_value = ","
json_lines = False
keys_raw = ""
out_fmt = "CSV"

if page == "Home":
    st.sidebar.header("Upload")
    fA = st.sidebar.file_uploader("File A", type=["csv","xlsx","xls","json"])
    fB = st.sidebar.file_uploader("File B", type=["csv","xlsx","xls","json"])

    st.sidebar.header("Options")
    normalize = st.sidebar.checkbox("Normalize headers", value=True)

    st.sidebar.markdown("**CSV options**")
    csv_sep = st.sidebar.selectbox("Delimiter", [",",";","\\t","|"], index=0)
    csv_sep_value = {"\\t":"\t"}.get(csv_sep, csv_sep)

    st.sidebar.markdown("**JSON options**")
    json_lines = st.sidebar.checkbox("JSON Lines (one JSON object per line)", value=False)

    keys_raw = st.sidebar.text_input("Key columns (comma-separated)", value="")
    out_fmt = st.sidebar.radio("Download format", ["CSV", "JSON"], index=0)

# ---------- main ----------
if page == "Home":
    if fA and fB:
        dfA = load_file(fA, csv_sep=csv_sep_value, json_lines=json_lines)
        dfB = load_file(fB, csv_sep=csv_sep_value, json_lines=json_lines)
        if dfA is None or dfB is None:
            st.stop()

        if normalize:
            dfA = normalize_headers(dfA)
            dfB = normalize_headers(dfB)

        key_cols = [k.strip() for k in keys_raw.split(",") if k.strip()]



        # --- Calculate diffs and duplicates before tabs ---
        diffs = row_diffs(dfA, dfB, key_cols)
        dupsA = find_duplicates(dfA, key_cols)
        dupsB = find_duplicates(dfB, key_cols)

        # --- Interactive Filtering and Downloadable Filtered Data ---
        st.markdown("### Data Exploration & Filtering")
        tabs = st.tabs(["File A", "File B", "Mismatches", "Duplicates"])

        # File A tab
        with tabs[0]:
            st.write("#### File A: All Data")
            st.dataframe(dfA, use_container_width=True)
            st.write("#### Rows with Nulls")
            nullsA = dfA[dfA.isnull().any(axis=1)]
            st.dataframe(nullsA, use_container_width=True)
            st.download_button("Download rows with nulls (A)", nullsA.to_csv(index=False), "nulls_A.csv", "text/csv")

        # File B tab
        with tabs[1]:
            st.write("#### File B: All Data")
            st.dataframe(dfB, use_container_width=True)
            st.write("#### Rows with Nulls")
            nullsB = dfB[dfB.isnull().any(axis=1)]
            st.dataframe(nullsB, use_container_width=True)
            st.download_button("Download rows with nulls (B)", nullsB.to_csv(index=False), "nulls_B.csv", "text/csv")

        # Mismatches tab
        with tabs[2]:
            st.write("#### Mismatched Rows (by key)")
            if diffs.empty:
                st.info("No diffs for matched keys (or keys missing).")
            else:
                st.dataframe(diffs, use_container_width=True)
                st.download_button("Download mismatched rows", diffs.to_csv(index=False), "mismatches.csv", "text/csv")

            # --- Schema Comparison and Column Quality only in this tab ---
            st.write("#### Schema Comparison")
            sch = schema_compare(dfA, dfB)
            st.dataframe(sch, use_container_width=True, height=240)

            st.write("#### Column Quality")
            qa = column_quality(dfA).rename(columns=lambda c: f"A_{c}" if c!="column" else c)
            qb = column_quality(dfB).rename(columns=lambda c: f"B_{c}" if c!="column" else c)
            qual = qa.merge(qb, on="column", how="outer")

            def highlight_quality(row):
                highlights = [''] * len(row)
                colnames = list(row.index)
                for i, col in enumerate(colnames):
                    if col.startswith('A_'):
                        b_col = 'B_' + col[2:]
                        if b_col in row and pd.notnull(row[col]) and pd.notnull(row[b_col]) and row[col] != row[b_col]:
                            highlights[i] = 'background-color: #ffe082'
                    elif col.startswith('B_'):
                        a_col = 'A_' + col[2:]
                        if a_col in row and pd.notnull(row[col]) and pd.notnull(row[a_col]) and row[col] != row[a_col]:
                            highlights[i] = 'background-color: #ffe082'
                return highlights

            styled_qual = qual.style.apply(highlight_quality, axis=1)
            st.write(styled_qual)

            # --- Summary generation and downloads ---
            summary = {
                "schema": {
                    "columns_in_A": len(dfA.columns),
                    "columns_in_B": len(dfB.columns),
                    "rows_in_A": len(dfA),
                    "rows_in_B": len(dfB),
                },
                "quality": {
                    "total_columns": len(qual),
                    "total_rows_A": len(dfA),
                    "total_rows_B": len(dfB),
                },
                "diffs": {
                    "total_diffs": len(diffs),
                }
            }

            def add_summary_row(df, summary_dict):
                summary_row = {col: summary_dict.get(col, "") for col in df.columns}
                return pd.concat([
                    pd.DataFrame([summary_row]),
                    df
                ], ignore_index=True)

            sch_csv = add_summary_row(sch, {
                "column": "SUMMARY",
                "in_A": summary["schema"]["columns_in_A"],
                "in_B": summary["schema"]["columns_in_B"],
                "dtype_A": f'Rows: {summary["schema"]["rows_in_A"]}',
                "dtype_B": f'Rows: {summary["schema"]["rows_in_B"]}',
            })
            qual_csv = add_summary_row(qual, {
                "column": "SUMMARY",
                "A_dtype": f'Rows: {summary["quality"]["total_rows_A"]}',
                "B_dtype": f'Rows: {summary["quality"]["total_rows_B"]}',
            }) if not qual.empty else qual
            diffs_csv = add_summary_row(diffs, {
                "changed_columns": "SUMMARY",
                "changes": f'Total diffs: {summary["diffs"]["total_diffs"]}',
            }) if not diffs.empty else diffs

            # --- Text summary for mismatches and nulls ---
            def make_text_summary(dfA, dfB, diffs, key_cols):
                lines = []
                lines.append("Summary of Data Validation\n===========================\n")
                # Mismatched rows by key
                if not diffs.empty:
                    lines.append("Mismatched Rows by Key:\n-----------------------")
                    for _, row in diffs.iterrows():
                        key_info = ', '.join(f"{k}: {row[k]}" for k in key_cols)
                        lines.append(f"Key: {key_info}")
                        lines.append(f"  Changed columns: {row.get('changed_columns','')}")
                        lines.append(f"  Changes: {row.get('changes','')}")
                    lines.append("")
                else:
                    lines.append("No mismatched rows found.\n")

                # Null/None values in A
                nulls_A = dfA.isnull().any(axis=1)
                if nulls_A.any():
                    lines.append("Rows with Null/None values in File A:\n-------------------------------------")
                    for idx, isnull in enumerate(nulls_A):
                        if isnull:
                            row = dfA.iloc[idx]
                            null_cols = [col for col in dfA.columns if pd.isnull(row[col])]
                            lines.append(f"Row {idx+1} (key: {', '.join(str(row[k]) for k in key_cols if k in row)}): nulls in {', '.join(null_cols)}")
                    lines.append("")
                else:
                    lines.append("No null/None values in File A.\n")

                # Null/None values in B
                nulls_B = dfB.isnull().any(axis=1)
                if nulls_B.any():
                    lines.append("Rows with Null/None values in File B:\n-------------------------------------")
                    for idx, isnull in enumerate(nulls_B):
                        if isnull:
                            row = dfB.iloc[idx]
                            null_cols = [col for col in dfB.columns if pd.isnull(row[col])]
                            lines.append(f"Row {idx+1} (key: {', '.join(str(row[k]) for k in key_cols if k in row)}): nulls in {', '.join(null_cols)}")
                    lines.append("")
                else:
                    lines.append("No null/None values in File B.\n")

                return '\n'.join(lines)

            text_summary = make_text_summary(dfA, dfB, diffs, key_cols)
            st.download_button("*summary.txt", text_summary, "summary.txt", "text/plain")

            if out_fmt == "CSV":
                st.download_button("*schema.csv", sch_csv.to_csv(index=False), "schema.csv", "text/csv")
                st.download_button("*quality.csv", qual_csv.to_csv(index=False), "quality.csv", "text/csv")
                st.download_button("*diffs.csv", diffs_csv.to_csv(index=False), "diffs.csv", "text/csv")
            else:
                bundle = {
                    "summary": summary,
                    "schema": json.loads(sch.to_json(orient="records")),
                    "quality": json.loads(qual.to_json(orient="records")),
                    "diffs": json.loads(diffs.to_json(orient="records")),
                }
                report = json.dumps(bundle, indent=2, default=str).encode("utf-8")
                st.download_button("*validation_report.json", report, "validation_report.json", "application/json")

        # Duplicates tab
        with tabs[3]:
            st.write("#### Duplicates in File A")
            st.dataframe(dupsA, use_container_width=True)
            st.download_button("Download duplicates (A)", dupsA.to_csv(index=False), "dups_A.csv", "text/csv")
            st.write("#### Duplicates in File B")
            st.dataframe(dupsB, use_container_width=True)
            st.download_button("Download duplicates (B)", dupsB.to_csv(index=False), "dups_B.csv", "text/csv")
        # No schema comparison or column quality in this tab



        # st.markdown("### Key Integrity & Duplicates")
        # stats = key_join_stats(dfA, dfB, key_cols)
        # c1, c2, c3 = st.columns(3)
        # c1.metric("Matched keys", stats["matched"])
        # c2.metric("Only in A", stats["only_in_A"])
        # c3.metric("Only in B", stats["only_in_B"])
        # dupA = find_duplicates(dfA, key_cols); dupB = find_duplicates(dfB, key_cols)
        # if not dupA.empty: st.caption("Duplicate keys in A"); st.dataframe(dupA, use_container_width=True)
        # if not dupB.empty: st.caption("Duplicate keys in B"); st.dataframe(dupB, use_container_width=True)



    # Row differences by key is now only shown in the Mismatches tab

    st.markdown("---")
    st.subheader("Download")
    # Download logic now only in Mismatches tab

