import os, io, re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import joblib

APP_TITLE = "SteelsGPT – Phase-3 (uses Phase-1 & Phase-2 artifacts; adds process text search)"
st.set_page_config(page_title=APP_TITLE, layout="wide")

# -----------------------
# Default file locations (override via env or sidebar if needed)
# -----------------------
MODEL_PATH   = os.environ.get("MODEL_PATH",   "models/phase2_best_rf.joblib").strip()
META_PATH    = os.environ.get("META_PATH",    "models/phase2_meta.joblib").strip()
P1_PARQUET   = os.environ.get("P1_PARQUET",   "phase1_features.parquet").strip()
REF_CSV      = os.environ.get("REF_CSV",      "steel_dataset 1.csv").strip()
FI_CSV       = os.environ.get("FI_CSV",       "models/rf_feature_importances.csv").strip()

# 18-element composition UI schema (you can type fewer; missing ones default to 0.0)
DEFAULT_COMP = {
    "Al": 0.00, "Cu": 0.00, "Mn": 1.00, "N": 0.00,
    "Ni": 0.30, "Ti": 0.00, "S": 0.00, "Fe": 97.00,
    "Zr": 0.00, "P": 0.00, "Si": 0.50, "V": 0.00,
    "Mo": 0.10, "Co": 0.00, "C": 0.50, "Nb": 0.00,
    "B": 0.00, "Cr": 0.50
}

# -----------------------
# K=12 cluster legend (descriptions)
# -----------------------
CLUSTERS_K12 = [
    {"code":0,  "label":"carburized – quenched (oil) – wear-resistant surface steel – temper"},
    {"code":1,  "label":"hot rolled – general construction steel – thickness"},
    {"code":2,  "label":"quenched (oil) – general-purpose steel – tested"},
    {"code":3,  "label":"normalized – easy-to-machine stock – cold drawn"},
    {"code":4,  "label":"normalized (air cooled) – general-purpose steel – round normalized"},
    {"code":5,  "label":"carburized – quenched (oil) – wear-resistant surface steel – quenched tempered"},
    {"code":6,  "label":"quenched (oil) & tempered – high-strength structural grade – oil"},
    {"code":7,  "label":"annealed – easy to machine – round annealed"},
    {"code":8,  "label":"carburized – quenched (oil) – wear-resistant surface steel – temper [Cluster 8]"},
    {"code":9,  "label":"normalized – general-purpose steel – round normalized"},
    {"code":10, "label":"quenched (water) & tempered – high-strength structural grade – water quenched"},
    {"code":11, "label":"Plate – no special treatment – general-purpose steel – plate grade"},
]
CODE2LABEL = {c["code"]: c["label"] for c in CLUSTERS_K12}

# -----------------------
# Helpers
# -----------------------
def load_model_and_meta(model_path: str, meta_path: str):
    if not Path(model_path).exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()
    if not Path(meta_path).exists():
        st.error(f"Meta file not found: {meta_path}")
        st.stop()
    model = joblib.load(model_path)
    meta  = joblib.load(meta_path)
    features: List[str] = meta.get("features", [])
    targets:  List[str] = meta.get("targets", ["ultimate_tensile","yield","ductility"])
    if not features:
        st.error("Meta file missing `features`; cannot build the input vector.")
        st.stop()
    return model, features, targets

@st.cache_data(show_spinner=False)
def load_phase1_centroids(parquet_path: str, expected_p1_cols: List[str]):
    """Compute per-cluster centroids from the Phase-1 parquet. No file is modified."""
    if not parquet_path or not Path(parquet_path).exists():
        return None, None, []
    p1 = pd.read_parquet(parquet_path)

    # Which column indicates cluster id?
    clcol = None
    for cand in ["macro_cluster", "cluster_id", "cluster", "k_label"]:
        if cand in p1.columns:
            clcol = cand; break
    if clcol is None:
        return None, None, []

    # Phase-1 columns the model expects (present in parquet)
    p1_cols = [c for c in expected_p1_cols if c in p1.columns]

    # Ensure all expected Phase-1 columns exist (fill missing with zeros for centroid calc)
    for c in expected_p1_cols:
        if c not in p1.columns:
            p1[c] = 0.0

    cent = (p1.groupby(clcol)[expected_p1_cols]
              .mean()
              .reset_index()
              .rename(columns={clcol: "cluster_id"}))
    cluster_ids = cent["cluster_id"].tolist()
    return cent, cluster_ids, p1_cols

@st.cache_data(show_spinner=False)
def load_reference_csv(path: str):
    if not path or not Path(path).exists():
        return pd.DataFrame()
    # try common encodings
    for enc in ["utf-8", "ISO-8859-1", "latin1"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        return pd.DataFrame()
    # ensure we have an alloy name column
    if "alloy_name" not in df.columns:
        for alt in ["name","grade","label","steel_name","designation"]:
            if alt in df.columns:
                df = df.rename(columns={alt:"alloy_name"}); break
    if "alloy_name" not in df.columns:
        df.insert(0, "alloy_name", [f"alloy_{i}" for i in range(len(df))])
    return df

def build_feature_row(feature_order: List[str],
                      comp: Dict[str, float],
                      cluster_id: int | None,
                      centroids: pd.DataFrame | None,
                      phase1_cols: List[str]):
    """Build the exact input vector Phase-2 trained on. Uses comp + inferred Phase-1 features."""
    row = {f: float(comp.get(f, 0.0)) for f in feature_order}

    # cl_* one-hots if present
    cl_cols = [c for c in feature_order if c.startswith("cl_")]
    if cl_cols:
        for c in cl_cols: row[c] = 0.0
        if cluster_id is not None and f"cl_{int(cluster_id)}" in cl_cols:
            row[f"cl_{int(cluster_id)}"] = 1.0

    # Inject Phase-1 centroids for any route__/tag__/form__/proc_pca_* columns the model expects
    if centroids is not None and phase1_cols:
        if cluster_id is None and "cluster_id" in centroids.columns and len(centroids):
            sel = centroids.iloc[[0]]
        else:
            sel = centroids.loc[centroids["cluster_id"] == int(cluster_id)] if "cluster_id" in centroids.columns else pd.DataFrame()
            if sel.empty and len(centroids):
                sel = centroids.iloc[[0]]
        for c in phase1_cols:
            if c in sel.columns:
                row[c] = float(sel.iloc[0][c])

    return row

# ---- process text search (for suggested alloy ranking) ----
def tokenize_query(q: str) -> List[str]:
    parts = re.findall(r'"([^"]+)"|(\S+)', q)
    terms = [p[0] or p[1] for p in parts]
    return [t.strip() for t in terms if t.strip()]

def build_regex(terms: List[str], mode: str) -> re.Pattern | None:
    if not terms: return None
    esc = [re.escape(t) for t in terms]
    if mode == "OR":
        return re.compile(r"(?i)(" + "|".join(esc) + r")")
    # AND: require all via lookaheads
    return re.compile(r"(?i)" + "".join([f"(?=.*{e})" for e in esc]))

def excerpt(text: str, pattern: re.Pattern, radius: int = 50) -> str:
    m = pattern.search(text)
    if not m:
        return (text[:2*radius] + "…") if len(text) > 2*radius else text
    start = max(0, m.start() - radius)
    end   = min(len(text), m.end() + radius)
    out = text[start:end]
    return ("…" if start > 0 else "") + out + ("…" if end < len(text) else "")

def rank_suggestions(ref_df: pd.DataFrame,
                     comp_input: Dict[str, float],
                     pred_props: Dict[str, float] | None,
                     process_text_col: str | None,
                     query: str,
                     query_mode: str,
                     topk: int = 10):
    """Rank alloys by: text match score (from query) + composition closeness + (optional) property closeness."""
    if ref_df.empty:
        return pd.DataFrame()

    df = ref_df.copy()
    # --- text score ---
    text_score = np.zeros(len(df), dtype=float)
    if process_text_col and process_text_col in df.columns and query.strip():
        terms = tokenize_query(query)
        rx = build_regex(terms, query_mode)
        tx = df[process_text_col].fillna("").astype(str)
        if rx:
            # base: count total matched characters; boost exact numbers (e.g., 870)
            nums = re.findall(r"\d+\.?\d*", query)
            def s(txt):
                hits = [m.group(0) for m in re.finditer(re.compile("|".join([re.escape(t) for t in terms]), re.I), txt)]
                base = sum(len(h) for h in hits)
                boost = 0
                for n in nums:
                    if re.search(rf"(?i)\b{re.escape(n)}\b", txt): boost += 100
                return base + boost
            text_score = tx.apply(s).to_numpy()
            df["match_excerpt"] = tx.apply(lambda t: excerpt(t, rx))
    else:
        df["match_excerpt"] = ""

    # --- composition distance ---
    comp_cols = [c for c in DEFAULT_COMP if c in df.columns]
    comp_dist = np.zeros(len(df), dtype=float)
    if comp_cols:
        v = np.array([float(comp_input.get(c, 0.0)) for c in comp_cols], dtype=float)
        M = df[comp_cols].to_numpy(dtype=float)
        comp_dist = np.linalg.norm(M - v[None, :], axis=1)
    # normalize to [0,1]
    if comp_dist.max() > comp_dist.min():
        comp_score = 1.0 - (comp_dist - comp_dist.min()) / (comp_dist.max() - comp_dist.min())
    else:
        comp_score = 1.0 - comp_dist

    # --- property distance (optional, uses predicted props from your RF) ---
    prop_cols = [c for c in ["ultimate_tensile","yield","ductility"] if c in df.columns]
    prop_score = np.zeros(len(df), dtype=float)
    if pred_props and prop_cols and all(k in pred_props for k in ["ultimate_tensile","yield","ductility"]):
        v = np.array([float(pred_props.get(k, 0.0)) for k in prop_cols], dtype=float)
        M = df[prop_cols].to_numpy(dtype=float)
        d = np.linalg.norm(M - v[None, :], axis=1)
        if d.max() > d.min():
            prop_score = 1.0 - (d - d.min()) / (d.max() - d.min())
        else:
            prop_score = 1.0 - d

    # Final score: emphasize text if provided; otherwise comp+props
    if query.strip():
        score = 0.60 * (text_score / (text_score.max() or 1.0)) + 0.25 * comp_score + 0.15 * prop_score
    else:
        score = 0.60 * comp_score + 0.40 * prop_score

    out = df.copy()
    out["_score"] = score
    out = out.sort_values("_score", ascending=False).head(topk)
    # choose preview columns
    show_cols = []
    # try to find a process/route column to display
    process_display_col = None
    for c in ["process","route","notes","treatment","processing","description"]:
        if c in out.columns:
            process_display_col = c; break
    for c in ["alloy_name", process_display_col, "match_excerpt", "_score",
              "ultimate_tensile","yield","ductility"]:
        if c and c in out.columns and c not in show_cols:
            show_cols.append(c)
    comp_preview = [c for c in ["C","Si","Mn","Cr","Ni","Mo","V","Nb","Ti","Al","Cu","N","B","W","Co","Fe","Zr","P","S"]
                    if c in out.columns][:8]
    show_cols += [c for c in comp_preview if c not in show_cols]
    return out[show_cols]

# -----------------------
# Load artifacts
# -----------------------
model, feature_order, target_names = load_model_and_meta(MODEL_PATH, META_PATH)

# Phase-1 columns the model expects (based on feature names)
PHASE1_PREFIXES = ("cl_","route__","tag_","form_","proc_pca_")
p1_expected = [c for c in feature_order if c.startswith(PHASE1_PREFIXES)]

centroids, cluster_ids, phase1_cols = load_phase1_centroids(P1_PARQUET, p1_expected)

# Reference dataset (for suggested alloys + process text)
ref_df = load_reference_csv(REF_CSV)

# Feature importances (optional)
fi_df = pd.read_csv(FI_CSV) if Path(FI_CSV).exists() else pd.DataFrame()

# -----------------------
# UI
# -----------------------
st.title("SteelsGPT – Phase-3")
with st.sidebar:
    st.subheader("Files (read-only)")
    st.caption(f"Model: {MODEL_PATH}")
    st.caption(f"Meta: {META_PATH}")
    st.caption(f"Phase-1 parquet: {P1_PARQUET}")
    st.caption(f"Reference CSV: {REF_CSV}")

    if not p1_expected:
        st.info("Model appears to be composition-only (no Phase-1 feature names found)."
                " Predictions will not depend on cluster." )
    else:
        st.caption(f"Phase-1 features in model: {len(p1_expected)}")


# Inputs
left, right = st.columns([0.62, 0.38])
with left:
    st.subheader("Inputs")
    # Optional cluster selection with descriptions
    use_cluster = st.checkbox("Use processing cluster", value=True)
    cluster_id = None
    if use_cluster:
        # Show the 12 fixed codes with readable descriptions
        cluster_id = st.selectbox(
            "Cluster (K=12)",
            [c["code"] for c in CLUSTERS_K12],
            index=1,
            format_func=lambda cid: f"{cid} — {CODE2LABEL.get(cid, 'Cluster')}"
        )

    cols = st.columns(6)
    comp = {}
    for i, (elem, default) in enumerate(DEFAULT_COMP.items()):
        with cols[i % 6]:
            comp[elem] = st.number_input(elem, min_value=0.0, max_value=100.0, value=float(default), step=0.01)

    # Process text search that will influence suggested alloys
    st.markdown("#### Process text (influences **suggested** alloy ranking)")
    process_query = st.text_input('e.g. "cold drawn" 870 degrees pseudo-carburized', "")
    query_mode = st.radio("Match mode", ["AND","OR"], index=0, horizontal=True)

    if st.button("Predict", type="primary"):
        # Build input row strictly in the saved feature order
        row = build_feature_row(feature_order, comp, cluster_id, centroids, phase1_cols)
        X = pd.DataFrame([row], columns=feature_order).astype(float)

        # Basic sanity
        if X.isna().any().any():
            bad = X.columns[X.isna().any()].tolist()
            st.error(f"NaNs in model input: {bad}")
            st.stop()

        y_pred = model.predict(X)
        y_vec = np.array(y_pred).reshape(-1)
        preds = {t: float(v) for t, v in zip(target_names, y_vec)}
        st.session_state["preds"] = preds
        st.session_state["comp"] = comp
        st.session_state["process_query"] = process_query
        st.session_state["query_mode"] = query_mode

with right:
    st.subheader("Results")
    preds = st.session_state.get("preds")
    if preds is None:
        st.info("Set inputs and click **Predict**.")
    else:
        # Display using target names from meta (don’t assume fixed names)
        cols = st.columns(len(target_names))
        for i, t in enumerate(target_names):
            with cols[i]:
                val = preds.get(t, None)
                st.metric(t, f"{val:.3f}" if isinstance(val, (int,float)) else f"{val}")

        # Suggested alloys (influenced by process text, composition, and predicted props if present)
        st.markdown("### Suggested alloys")
        if ref_df.empty:
            st.info("Reference CSV not found or empty. Put your dataset at `steel_dataset 1.csv` or set REF_CSV.")
        else:
            process_col_candidates = [c for c in ["process","route","notes","treatment","processing","description"] if c in ref_df.columns]
            process_col = process_col_candidates[0] if process_col_candidates else None

            ranked = rank_suggestions(
                ref_df, st.session_state.get("comp", {}),
                preds, process_col,
                st.session_state.get("process_query",""),
                st.session_state.get("query_mode","AND"),
                topk=10
            )
            if ranked.empty:
                st.info("No suggestions match your filters. Try adjusting the process text or ensure columns exist in the CSV.")
            else:
                st.dataframe(ranked, use_container_width=True)
                st.download_button("⬇️ Download suggestions", ranked.to_csv(index=False).encode("utf-8"),
                                   "suggested_alloys.csv", "text/csv")


# Optional: quick view of feature importances if provided
if Path(FI_CSV).exists():
    try:
        fi_df = pd.read_csv(FI_CSV)
        with st.expander("Feature importances (from Phase-2 RF)"):
            st.dataframe(fi_df.head(50), use_container_width=True)
    except Exception:
        pass

# Debug panel: inspect first 30 features going into the model
with st.expander("🛠 Debug: model input vector (first 30 features)"):
    X_dbg = None
    if "comp" in st.session_state:
        # reconstruct last input row deterministically for display
        row_dbg = build_feature_row(feature_order,
                                    st.session_state["comp"],
                                    cluster_id=1,
                                    centroids=centroids,
                                    phase1_cols=phase1_cols)
        X_dbg = pd.DataFrame([row_dbg], columns=feature_order).astype(float)
    if X_dbg is not None:
        st.dataframe(X_dbg.iloc[:, :30].T)
        st.download_button("⬇️ Download model input row", X_dbg.to_csv(index=False).encode("utf-8"),
                           "model_input_row.csv", "text/csv")
    else:
        st.caption("Run a prediction to populate this.")
