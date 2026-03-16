#!/usr/bin/env python3
"""
Fish Model Performance Review — Web Version
============================================
Shareable version: loads from pre-built CSVs, no video playback,
no local path configuration required.

Pre-requisite: run  python build_data.py  once to generate:
    data/exports/model_detections.csv
    data/exports/tech_counts.csv

Deploy to Streamlit Community Cloud:
    1. Push this repo (including data/exports/) to GitHub
    2. Go to share.streamlit.io → New app → point at this file
    3. Share the URL with colleagues
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="Model Performance Review",
    page_icon="🐟",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Paths (relative to this file → project root → data/exports/)
# ---------------------------------------------------------------------------
_ROOT        = Path(__file__).parent.parent
_MODEL_CSV   = _ROOT / "data" / "exports" / "model_detections.csv"
_TECH_CSV    = _ROOT / "data" / "exports" / "tech_counts.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPECIES_FISH = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]
SPECIES_ALL  = SPECIES_FISH + ["Non-fish", "Unknown"]

SPECIES_COLORS = {
    "Chinook":       "#2196F3",
    "Coho":          "#FF9800",
    "Atlantic":      "#4CAF50",
    "Rainbow Trout": "#E91E63",
    "Brown Trout":   "#795548",
    "Non-fish":      "#9E9E9E",
    "Unknown":       "#607D8B",
}

_NORM: dict[str, str] = {
    "rainbow": "Rainbow Trout", "rainbow trout": "Rainbow Trout",
    "chinook": "Chinook", "atlantic": "Atlantic", "coho": "Coho",
    "brown": "Brown Trout", "brown trout": "Brown Trout",
    "not fish": "Non-fish", "non fish": "Non-fish",
    "non-fish": "Non-fish", "background": "Non-fish", "unknown": "Unknown",
}

_DIR_NORM: dict[str, str] = {
    "left": "Upstream", "up": "Upstream", "upstream": "Upstream", "u": "Upstream",
    "right": "Downstream", "down": "Downstream", "downstream": "Downstream", "d": "Downstream",
}


def norm_species(s) -> str:
    if not isinstance(s, str):
        return "Unknown"
    return _NORM.get(s.strip().lower(), s.strip())


def norm_direction(d) -> str:
    if not isinstance(d, str):
        return "Unknown"
    return _DIR_NORM.get(d.strip().lower(), "Unknown")


def compute_model_direction(df: pd.DataFrame, logic: str = "full_transit") -> pd.Series:
    has_sides   = all(c in df.columns for c in ["first_side", "last_side", "exited_frame"])
    has_dir_col = "model_direction" in df.columns

    if logic == "full_transit" and has_sides:
        result  = pd.Series("Unknown", index=df.index)
        exited  = df["exited_frame"].astype(str).str.strip().str.lower() == "true"
        first_r = df["first_side"].astype(str).str.strip().str.lower() == "right"
        first_l = df["first_side"].astype(str).str.strip().str.lower() == "left"
        last_l  = df["last_side"].astype(str).str.strip().str.lower() == "left"
        last_r  = df["last_side"].astype(str).str.strip().str.lower() == "right"
        result[exited & first_r & last_l] = "Upstream"
        result[exited & first_l & last_r] = "Downstream"
        return result

    if logic == "exit_side" and has_sides:
        result = pd.Series("Unknown", index=df.index)
        exited = df["exited_frame"].astype(str).str.strip().str.lower() == "true"
        last_l = df["last_side"].astype(str).str.strip().str.lower() == "left"
        last_r = df["last_side"].astype(str).str.strip().str.lower() == "right"
        result[exited & last_l] = "Upstream"
        result[exited & last_r] = "Downstream"
        return result

    if has_dir_col:
        return df["model_direction"].apply(
            lambda v: ("Upstream" if str(v).strip().lower() == "left"
                       else ("Downstream" if str(v).strip().lower() == "right" else "Unknown"))
        )
    return pd.Series("Unknown", index=df.index)


# ---------------------------------------------------------------------------
# Data loading (from pre-built exports)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading model detections…")
def load_model() -> pd.DataFrame:
    if not _MODEL_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(_MODEL_CSV, low_memory=False)
    df["video_recording_time"] = pd.to_datetime(df["video_recording_time"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if "model_species" in df.columns:
        df["model_species"] = df["model_species"].apply(norm_species)
    if "clip_folder" in df.columns:
        df["clip_folder"] = df["clip_folder"].astype(str).str.strip()
    return df


@st.cache_data(show_spinner="Loading technician counts…")
def load_tech() -> pd.DataFrame:
    if not _TECH_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(_TECH_CSV, low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["date"]     = pd.to_datetime(df["date"], errors="coerce").dt.date
    if "species" in df.columns:
        df["species"] = df["species"].apply(norm_species)
    if "count" in df.columns:
        df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(1).astype(int)
    if "false_trigger" in df.columns:
        df["false_trigger"] = pd.to_numeric(df["false_trigger"], errors="coerce").fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Matching & flagging (same logic as app.py)
# ---------------------------------------------------------------------------
def _parse_video_rel(vrel: str):
    if not isinstance(vrel, str):
        return None, None
    vrel = vrel.replace("/", "\\")
    parts = vrel.rsplit("\\", 1)
    if len(parts) != 2:
        return None, None
    return parts[0].strip(), Path(parts[1]).stem.strip()


def build_matches(model_df: pd.DataFrame, tech_df: pd.DataFrame) -> pd.DataFrame:
    if model_df.empty or tech_df.empty:
        return pd.DataFrame()
    fc = tech_df[tech_df["fmt"] == "fish_counter"].copy()
    fc = fc[fc["video_rel"].notna()]
    if fc.empty:
        return pd.DataFrame()
    parsed = fc["video_rel"].apply(_parse_video_rel)
    fc["clip_folder"]  = parsed.apply(lambda x: x[0])
    fc["vid_event_id"] = parsed.apply(lambda x: x[1])
    fc = fc.dropna(subset=["clip_folder", "vid_event_id"])
    fc["match_key"] = fc["clip_folder"].astype(str) + "|" + fc["vid_event_id"].astype(str)
    if "vid_event_id" not in model_df.columns or "clip_folder" not in model_df.columns:
        return pd.DataFrame()
    mdf = model_df.copy()
    mdf["match_key"] = mdf["clip_folder"].astype(str) + "|" + mdf["vid_event_id"].astype(str)
    score_cols = [c for c in ["score_chinook","score_coho","score_atlantic",
                               "score_rainbow","score_brown","score_background"] if c in mdf.columns]
    keep = ["match_key","model_species","model_direction","top_score",
            "video_path","video_recording_time","det_id","clip_folder","vid_event_id"] + score_cols
    keep = [c for c in keep if c in mdf.columns]
    return fc.merge(mdf[keep], on="match_key", how="left", suffixes=("_tech","_model"))


def flag_struggling(matched: pd.DataFrame, conf_thresh: float) -> pd.DataFrame:
    if matched.empty:
        return pd.DataFrame()
    df = matched[matched["false_trigger"] == 0].copy()
    df["wrong_species"]  = (df["model_species"].notna()
                             & (df["model_species"] != df["species"])
                             & (~df["species"].isin(["Non-fish","Unknown"])))
    df["low_confidence"] = df["top_score"].notna() & (df["top_score"] < conf_thresh)
    df["missed"]         = df["model_species"].isna() & (~df["species"].isin(["Non-fish","Unknown"]))
    df["is_struggling"]  = df["wrong_species"] | df["low_confidence"] | df["missed"]
    return df[df["is_struggling"]].copy()


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def _eval_df(matched: pd.DataFrame, species_thresholds: dict) -> pd.DataFrame:
    df = matched[matched["false_trigger"] == 0].copy()
    df = df[df["species"].isin(SPECIES_FISH)]
    df["top_score"] = pd.to_numeric(df["top_score"], errors="coerce").fillna(0)

    def _pred(row):
        sp = row["model_species"]
        if not isinstance(sp, str):
            return "No detection"
        thresh = species_thresholds.get(sp, 0.5)
        if row["top_score"] >= thresh:
            return sp
        return "Flagged for Review"

    df["pred"] = df.apply(_pred, axis=1)
    return df


def chart_confusion_matrix(matched: pd.DataFrame, species_thresholds: dict,
                            normalize: str = "true") -> go.Figure:
    df = _eval_df(matched, species_thresholds)
    if df.empty:
        return go.Figure()
    labels   = SPECIES_FISH + ["Flagged for Review", "No detection"]
    cm       = confusion_matrix(df["species"], df["pred"], labels=labels,
                                normalize=normalize if normalize != "none" else None)
    cm_counts = confusion_matrix(df["species"], df["pred"], labels=labels)
    n = len(labels)
    if normalize != "none":
        text  = [[f"{cm[i][j]*100:.1f}%<br>({cm_counts[i][j]})" for j in range(n)] for i in range(n)]
        zmin, zmax, cb = 0, 1, "Rate"
    else:
        text  = [[str(int(cm_counts[i][j])) for j in range(n)] for i in range(n)]
        zmin, zmax, cb = 0, int(cm_counts.max()) if cm_counts.size else 1, "Count"
    norm_label = {"true": "row-normalised (Recall)", "pred": "col-normalised (Precision)", "none": "raw counts"}
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, text=text, texttemplate="%{text}",
                               colorscale="Blues", zmin=zmin, zmax=zmax,
                               colorbar=dict(title=cb), hoverongaps=False))
    fig.update_layout(
        title=f"Confusion Matrix — {norm_label[normalize]}",
        xaxis=dict(title="Predicted", categoryorder="array", categoryarray=labels, side="bottom"),
        yaxis=dict(title="Actual (Tech)", categoryorder="array", categoryarray=labels[::-1]),
        height=520,
    )
    return fig


def compute_pr_curves(matched: pd.DataFrame) -> dict:
    df = matched[matched["false_trigger"] == 0].copy()
    df = df[df["species"].isin(SPECIES_FISH) | df["model_species"].isin(SPECIES_FISH)]
    df["top_score"] = pd.to_numeric(df["top_score"], errors="coerce").fillna(0)
    thresholds = np.round(np.arange(0.05, 1.00, 0.025), 3)
    results = {}
    for sp in SPECIES_FISH:
        total_pos = (df["species"] == sp).sum()
        if total_pos == 0:
            continue
        rows = []
        for t in thresholds:
            pred_sp  = df[(df["model_species"] == sp) & (df["top_score"] >= t)]
            tp = (pred_sp["species"] == sp).sum()
            fp = (pred_sp["species"] != sp).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall    = tp / total_pos
            f1        = (2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0)
            rows.append(dict(threshold=t, precision=precision, recall=recall,
                             f1=f1, tp=int(tp), fp=int(fp), fn=int(total_pos-tp), n_det=int(len(pred_sp))))
        results[sp] = pd.DataFrame(rows)
    return results


def chart_pr_curve(pr_data: dict, species: list) -> go.Figure:
    fig = go.Figure()
    for sp in species:
        if sp not in pr_data:
            continue
        d = pr_data[sp]
        color = SPECIES_COLORS.get(sp, "#607D8B")
        fig.add_trace(go.Scatter(
            x=d["recall"], y=d["precision"], mode="lines+markers", name=sp,
            line=dict(color=color, width=2), marker=dict(size=5),
            customdata=np.stack([d["threshold"], d["f1"], d["n_det"]], axis=1),
            hovertemplate=(f"<b>{sp}</b><br>Threshold: %{{customdata[0]:.2f}}<br>"
                           "Precision: %{y:.3f}<br>Recall: %{x:.3f}<br>"
                           "F1: %{customdata[1]:.3f}<extra></extra>"),
        ))
    fig.update_layout(title="Precision–Recall Curve", xaxis=dict(title="Recall", range=[0,1.02]),
                      yaxis=dict(title="Precision", range=[0,1.02]), height=480,
                      legend=dict(x=0.01, y=0.01))
    fig.add_shape(type="line", x0=0, x1=1, y0=1, y1=0, line=dict(dash="dot", color="grey", width=1))
    return fig


def chart_metric_vs_threshold(pr_data: dict, metric: str, species: list) -> go.Figure:
    fig = go.Figure()
    for sp in species:
        if sp not in pr_data:
            continue
        d = pr_data[sp]
        fig.add_trace(go.Scatter(x=d["threshold"], y=d[metric], mode="lines+markers", name=sp,
                                 line=dict(color=SPECIES_COLORS.get(sp, "#607D8B"), width=2),
                                 marker=dict(size=4)))
    fig.update_layout(title=f"{metric.capitalize()} vs Confidence Threshold",
                      xaxis_title="Threshold", yaxis=dict(title=metric.capitalize(), range=[0,1.05]),
                      height=400)
    return fig


def metrics_summary_table(pr_data: dict, conf_thresh: float) -> pd.DataFrame:
    rows = []
    for sp, d in pr_data.items():
        r = d.iloc[(d["threshold"] - conf_thresh).abs().argsort()[:1]].iloc[0]
        rows.append(dict(Species=sp, Threshold=round(float(r["threshold"]),3),
                         Precision=round(float(r["precision"]),4), Recall=round(float(r["recall"]),4),
                         F1=round(float(r["f1"]),4), TP=int(r["tp"]), FP=int(r["fp"]),
                         FN=int(r["fn"]), Detections=int(r["n_det"])))
    return pd.DataFrame(rows).sort_values("F1", ascending=False)


def chart_class_scores(row: pd.Series) -> go.Figure:
    mapping = {"Chinook":"score_chinook","Coho":"score_coho","Atlantic":"score_atlantic",
               "Rainbow Trout":"score_rainbow","Brown Trout":"score_brown","Background":"score_background"}
    scores = {sp: float(row.get(col,0) or 0) for sp, col in mapping.items()}
    df = pd.DataFrame(scores.items(), columns=["Species","Score"]).sort_values("Score")
    fig = px.bar(df, x="Score", y="Species", orientation="h",
                 color="Species", color_discrete_map=SPECIES_COLORS,
                 range_x=[0,1], title="Class scores for this detection")
    fig.update_layout(height=300, showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init():
    defaults = {"conf_thresh": 0.5, "page": "📊 Overview"}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
model_df = load_model()
tech_df  = load_tech()
matched  = pd.DataFrame()
struggle = pd.DataFrame()

if model_df.empty:
    st.error("model_detections.csv not found. Run `python build_data.py` first.")
    st.stop()
if tech_df.empty:
    st.error("tech_counts.csv not found. Run `python build_data.py` first.")
    st.stop()

matched = build_matches(model_df, tech_df)
if not matched.empty:
    struggle = flag_struggling(matched, float(st.session_state.conf_thresh))

# ---------------------------------------------------------------------------
# Shared daily aggregations
# ---------------------------------------------------------------------------
model_daily = pd.DataFrame()
tech_daily  = pd.DataFrame()

if not model_df.empty and "model_species" in model_df.columns:
    _m = model_df[model_df["model_species"].isin(SPECIES_FISH)].copy()
    model_daily = _m.groupby(["date","model_species"]).size().reset_index(name="count")
    model_daily.rename(columns={"model_species":"species"}, inplace=True)

if not tech_df.empty:
    _t = tech_df[(tech_df["false_trigger"]==0) & tech_df["species"].isin(SPECIES_FISH)].copy()
    tech_daily = _t.groupby(["date","species"])["count"].sum().reset_index()
    tech_daily["count"] = tech_daily["count"].astype(int)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🐟 Model Performance")
    st.caption("Credit River · 2025 Season")

    st.divider()
    st.session_state.conf_thresh = st.slider(
        "Confidence threshold", 0.1, 0.9,
        float(st.session_state.conf_thresh), 0.05,
        help="Detections below this score are flagged as struggling / sent for review",
    )

    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    _pages = ["📊 Overview", "📈 Time Series", "⚖️ Comparison",
              "🎬 Struggling Videos", "📉 Distributions", "📐 Metrics"]
    page = st.radio(
        "Page", _pages,
        index=_pages.index(st.session_state.page) if st.session_state.page in _pages else 0,
    )
    st.session_state.page = page

    st.divider()
    st.caption(f"Model detections: **{len(model_df):,}**")
    st.caption(f"Tech records: **{len(tech_df):,}**")
    if not model_df.empty and "date" in model_df.columns:
        mn, mx = model_df["date"].min(), model_df["date"].max()
        st.caption(f"Season: {mn} → {mx}")

# ===========================================================================
# PAGE: OVERVIEW
# ===========================================================================
if "Overview" in page:
    st.header("📊 Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Model detections", f"{len(model_df):,}")
    tech_fish = int(tech_df.loc[tech_df["species"].isin(SPECIES_FISH),"count"].sum())
    c2.metric("Tech fish counts", f"{tech_fish:,}")
    c3.metric("Unique clips", f"{model_df['clip_folder'].nunique():,}" if "clip_folder" in model_df.columns else "—")
    c4.metric("Matched events", f"{len(matched):,}" if not matched.empty else "—")
    c5.metric("Struggling events", f"{len(struggle):,}" if not struggle.empty else "—")

    st.divider()
    pie1, pie2 = st.columns(2)
    with pie1:
        st.subheader("Model — species distribution")
        sp_c = model_df[model_df["model_species"].isin(SPECIES_FISH)]["model_species"].value_counts().reset_index()
        sp_c.columns = ["Species","Count"]
        fig = px.pie(sp_c, values="Count", names="Species", color="Species",
                     color_discrete_map=SPECIES_COLORS, hole=0.4)
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
    with pie2:
        st.subheader("Tech — species distribution")
        tc = (tech_df[(tech_df["species"].isin(SPECIES_FISH)) & (tech_df["false_trigger"]==0)]
              .groupby("species")["count"].sum().reset_index())
        tc.columns = ["Species","Count"]
        fig = px.pie(tc, values="Count", names="Species", color="Species",
                     color_discrete_map=SPECIES_COLORS, hole=0.4)
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    if not matched.empty:
        st.divider()
        st.subheader("Species accuracy (direct-matched events)")
        ev = matched[(matched["false_trigger"]==0) & matched["model_species"].notna()
                     & matched["species"].isin(SPECIES_FISH)]
        if not ev.empty:
            correct = (ev["model_species"] == ev["species"]).sum()
            total   = len(ev)
            a1, a2 = st.columns([1,3])
            a1.metric("Overall accuracy", f"{100*correct/total:.1f}%", f"{correct}/{total} correct")
            acc_by_sp = (ev.groupby("species")
                         .apply(lambda g: pd.Series({
                             "Tech count": len(g),
                             "Model correct": int((g["model_species"]==g["species"]).sum()),
                             "Accuracy %": f"{100*(g['model_species']==g['species']).mean():.1f}",
                         })).reset_index())
            a2.dataframe(acc_by_sp, hide_index=True, use_container_width=True)

    st.divider()
    d1, d2 = st.columns(2)
    with d1:
        mn, mx = model_df["date"].min(), model_df["date"].max()
        st.caption(f"**Model data:** {mn} → {mx}")
    with d2:
        mn2 = tech_df["datetime"].min()
        mx2 = tech_df["datetime"].max()
        st.caption(f"**Tech data:** {mn2.date() if pd.notna(mn2) else '?'} → {mx2.date() if pd.notna(mx2) else '?'}")


# ===========================================================================
# PAGE: TIME SERIES
# ===========================================================================
elif "Time Series" in page:
    st.header("📈 Species Counts Through Time")

    sp_sel = st.multiselect("Species", SPECIES_FISH, default=SPECIES_FISH)
    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        granularity = st.radio("Granularity", ["Daily","Weekly"], horizontal=True)
    with ctrl2:
        apply_conf = st.checkbox(
            f"Apply confidence threshold ({float(st.session_state.conf_thresh):.2f})",
            value=False, key="ts_apply_conf")

    with st.expander("Direction / movement filter", expanded=False):
        dc1, dc2 = st.columns(2)
        with dc1:
            st.caption("**Show counts for:**")
            show_total      = st.checkbox("Total detections",          value=True,  key="ts_total")
            show_dir_split  = st.checkbox("Direction split (↑ and ↓)", value=False, key="ts_dir_split")
            show_upstream   = st.checkbox("Upstream only (↑)",         value=False, key="ts_up")
            show_downstream = st.checkbox("Downstream only (↓)",       value=False, key="ts_dn")
        with dc2:
            has_sides = all(c in model_df.columns for c in ["first_side","last_side","exited_frame"])
            dir_opts  = (["Full transit — first=Right → last=Left (recommended)",
                          "Exit side — last_side + exited_frame",
                          "Direction field — model direction tag"]
                         if has_sides else ["Direction field — model direction tag"])
            ts_dir_logic = st.radio("Direction logic", dir_opts, index=0, key="ts_dir_logic")

    dir_mode = ("full_transit" if "Full transit" in ts_dir_logic
                else "exit_side" if "Exit side" in ts_dir_logic else "direction_field")
    selected_dirs = list({
        *(["Upstream","Downstream"] if show_dir_split  else []),
        *(["Upstream"]              if show_upstream   else []),
        *(["Downstream"]            if show_downstream else []),
    })

    mdf_base = model_df.copy()
    if apply_conf and "top_score" in mdf_base.columns:
        mdf_base = mdf_base[pd.to_numeric(mdf_base["top_score"], errors="coerce").fillna(0)
                             >= float(st.session_state.conf_thresh)]

    if not mdf_base.empty and "model_species" in mdf_base.columns:
        _m2 = mdf_base[mdf_base["model_species"].isin(SPECIES_FISH)].copy()
        md_base = _m2.groupby(["date","model_species"]).size().reset_index(name="count")
        md_base.rename(columns={"model_species":"species"}, inplace=True)
    else:
        md_base = model_daily.copy()

    def _to_weekly(df):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.to_period("W").dt.start_time.dt.date
        return df.groupby(["date","species"])["count"].sum().reset_index()

    def _to_weekly_dir(df):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.to_period("W").dt.start_time.dt.date
        return df.groupby(["date","species","direction"])["count"].sum().reset_index()

    if not mdf_base.empty:
        mdf_d_all = mdf_base[mdf_base["model_species"].isin(sp_sel)].copy()
        mdf_d_all["direction"] = compute_model_direction(mdf_d_all, dir_mode)
        model_daily_all_dirs = (mdf_d_all.groupby(["date","model_species","direction"])
                                .size().reset_index(name="count")
                                .rename(columns={"model_species":"species"}))
        n_unknown_dir = int((mdf_d_all["direction"] == "Unknown").sum())
    else:
        model_daily_all_dirs = pd.DataFrame()
        n_unknown_dir = 0

    if not tech_df.empty:
        tdf_d_all = tech_df[(tech_df["false_trigger"]==0) & tech_df["species"].isin(sp_sel)].copy()
        tdf_d_all["direction"] = tdf_d_all["direction"].apply(norm_direction)
        tech_daily_all_dirs = (tdf_d_all.groupby(["date","species","direction"])["count"]
                               .sum().reset_index())
    else:
        tech_daily_all_dirs = pd.DataFrame()

    _dir_dash   = {"Upstream":"solid",  "Downstream":"dot"}
    _dir_marker = {"Upstream":"circle", "Downstream":"square"}
    _dir_label  = {"Upstream":"↑",     "Downstream":"↓"}

    fig = go.Figure()
    if show_total:
        md = (_to_weekly(md_base) if granularity=="Weekly" else md_base).copy()
        td = (_to_weekly(tech_daily) if granularity=="Weekly" else tech_daily).copy()
        for sp in sp_sel:
            color = SPECIES_COLORS.get(sp,"#607D8B")
            m = md[md["species"]==sp] if not md.empty else pd.DataFrame()
            t = td[td["species"]==sp] if not td.empty else pd.DataFrame()
            if not m.empty:
                fig.add_trace(go.Scatter(x=m["date"],y=m["count"],name=f"Model — {sp}",
                    line=dict(color=color,width=2),mode="lines+markers",marker=dict(size=5),legendgroup=sp))
            if not t.empty:
                fig.add_trace(go.Scatter(x=t["date"],y=t["count"],name=f"Tech — {sp}",
                    line=dict(color=color,width=2,dash="dash"),mode="lines+markers",
                    marker=dict(size=6,symbol="diamond"),legendgroup=sp))

    if selected_dirs:
        m_dir = (model_daily_all_dirs[model_daily_all_dirs["direction"].isin(selected_dirs)].copy()
                 if not model_daily_all_dirs.empty else pd.DataFrame())
        t_dir = (tech_daily_all_dirs[tech_daily_all_dirs["direction"].isin(selected_dirs)].copy()
                 if not tech_daily_all_dirs.empty else pd.DataFrame())
        if granularity=="Weekly":
            if not m_dir.empty: m_dir = _to_weekly_dir(m_dir)
            if not t_dir.empty: t_dir = _to_weekly_dir(t_dir)
        for sp in sp_sel:
            color = SPECIES_COLORS.get(sp,"#607D8B")
            for direction in selected_dirs:
                dl = _dir_label[direction]
                mf = (m_dir[(m_dir["species"]==sp)&(m_dir["direction"]==direction)]
                      if not m_dir.empty else pd.DataFrame())
                tf = (t_dir[(t_dir["species"]==sp)&(t_dir["direction"]==direction)]
                      if not t_dir.empty else pd.DataFrame())
                if not mf.empty:
                    fig.add_trace(go.Scatter(x=mf["date"],y=mf["count"],name=f"Model — {sp} {dl}",
                        line=dict(color=color,width=1.5,dash=_dir_dash[direction]),mode="lines+markers",
                        marker=dict(size=5,symbol=_dir_marker[direction]),legendgroup=f"{sp}_{direction}"))
                if not tf.empty:
                    fig.add_trace(go.Scatter(x=tf["date"],y=tf["count"],name=f"Tech — {sp} {dl}",
                        line=dict(color=color,width=1.5,dash="dashdot"),mode="lines+markers",
                        marker=dict(size=6,symbol="diamond"),legendgroup=f"{sp}_{direction}"))
        if n_unknown_dir > 0:
            st.caption(f"ℹ {n_unknown_dir:,} model detections had no determinable direction "
                       f"({dir_mode} logic) and are excluded from ↑/↓ traces but included in totals.")

    fig.update_layout(title="Species counts: Model (solid) vs Technician (dashed)",
                      xaxis_title="Date", yaxis_title="Count",
                      hovermode="x unified", height=520,
                      legend=dict(orientation="v", x=1.01, y=1))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Solid = model · Dashed = tech · ↑ Upstream · ↓ Downstream")

    # ── Cumulative counts ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Cumulative counts over time")
    cum_tab1, cum_tab2 = st.tabs(["📈 Total cumulative", "🔀 Net upstream migration (↑ − ↓)"])

    with cum_tab1:
        fig_cum = go.Figure()
        md_cum = md_base[md_base["species"].isin(sp_sel)].copy() if not md_base.empty else pd.DataFrame()
        td_cum = tech_daily[tech_daily["species"].isin(sp_sel)].copy() if not tech_daily.empty else pd.DataFrame()
        if not md_cum.empty:
            md_cum = md_cum.sort_values("date")
            md_cum["cumcount"] = md_cum.groupby("species")["count"].cumsum()
        if not td_cum.empty:
            td_cum = td_cum.sort_values("date")
            td_cum["cumcount"] = td_cum.groupby("species")["count"].cumsum()
        for sp in sp_sel:
            color = SPECIES_COLORS.get(sp,"#607D8B")
            m = md_cum[md_cum["species"]==sp] if not md_cum.empty else pd.DataFrame()
            t = td_cum[td_cum["species"]==sp] if not td_cum.empty else pd.DataFrame()
            if not m.empty:
                fig_cum.add_trace(go.Scatter(x=m["date"],y=m["cumcount"],name=f"Model — {sp}",
                    line=dict(color=color,width=2),mode="lines",legendgroup=sp))
            if not t.empty:
                fig_cum.add_trace(go.Scatter(x=t["date"],y=t["cumcount"],name=f"Tech — {sp}",
                    line=dict(color=color,width=2,dash="dash"),mode="lines",legendgroup=sp))
        fig_cum.update_layout(title="Cumulative detections: Model vs Technician",
                              xaxis_title="Date", yaxis_title="Cumulative count",
                              hovermode="x unified", height=460,
                              legend=dict(orientation="v", x=1.01, y=1))
        st.plotly_chart(fig_cum, use_container_width=True)

    with cum_tab2:
        st.caption("Net upstream migration = cumulative(↑) − cumulative(↓). "
                   "Positive = more fish travelling upstream than downstream over the season.")
        net_c1, net_c2, net_c3 = st.columns(3)
        show_net       = net_c1.checkbox("Net (↑ − ↓)",           value=True, key="net_show_net")
        show_dir_split2 = net_c2.checkbox("Direction split (↑ & ↓)", value=True, key="net_show_split")
        show_all_fish  = net_c3.checkbox("All fish (total)",        value=True, key="net_show_all")

        if model_daily_all_dirs.empty and not show_all_fish:
            st.info("No directional model data found.")
        else:
            fig_net = go.Figure()
            any_trace = False

            def _ds(df, sp_name, direction):
                sub = df[(df["species"]==sp_name) & (df["direction"]==direction)]
                return sub.set_index("date")["count"] if not sub.empty else pd.Series(dtype=float)

            for sp in sp_sel:
                color = SPECIES_COLORS.get(sp,"#607D8B")
                m_up = _ds(model_daily_all_dirs, sp, "Upstream")
                m_dn = _ds(model_daily_all_dirs, sp, "Downstream")
                t_up = _ds(tech_daily_all_dirs,  sp, "Upstream")
                t_dn = _ds(tech_daily_all_dirs,  sp, "Downstream")

                if show_dir_split2:
                    for series, label, dash, sym in [
                        (m_up, f"Model ↑ — {sp}", "solid",    "circle"),
                        (m_dn, f"Model ↓ — {sp}", "dot",      "square"),
                        (t_up, f"Tech ↑ — {sp}",  "dashdot",  "diamond"),
                        (t_dn, f"Tech ↓ — {sp}",  "longdash", "diamond-open"),
                    ]:
                        if series.empty: continue
                        c = series.sort_index().cumsum()
                        fig_net.add_trace(go.Scatter(x=list(c.index), y=list(c.values), name=label,
                            line=dict(color=color,width=1.5,dash=dash), mode="lines",
                            legendgroup=f"{sp}_split",
                            hovertemplate=f"<b>{label}</b><br>%{{x}}<br>Cumulative: %{{y}}<extra></extra>"))
                        any_trace = True

                if show_net:
                    for up, dn, label, dash in [
                        (m_up, m_dn, f"Model net — {sp}", "solid"),
                        (t_up, t_dn, f"Tech net — {sp}",  "dash"),
                    ]:
                        if up.empty and dn.empty: continue
                        all_d = sorted(set(up.index)|set(dn.index))
                        net = up.reindex(all_d,fill_value=0) - dn.reindex(all_d,fill_value=0)
                        cum = net.cumsum()
                        fig_net.add_trace(go.Scatter(x=list(cum.index), y=list(cum.values), name=label,
                            line=dict(color=color,width=2.5,dash=dash), mode="lines+markers",
                            marker=dict(size=5), legendgroup=f"{sp}_net",
                            hovertemplate=f"<b>{label}</b><br>%{{x}}<br>Net: %{{y}}<extra></extra>"))
                        any_trace = True

            if show_all_fish:
                if not md_base.empty:
                    all_m = (md_base[md_base["species"].isin(sp_sel)].groupby("date")["count"]
                             .sum().sort_index().cumsum())
                    if not all_m.empty:
                        fig_net.add_trace(go.Scatter(x=list(all_m.index), y=list(all_m.values),
                            name="Model — All fish", line=dict(color="#546E7A",width=2.5,dash="solid"),
                            mode="lines", legendgroup="all_fish"))
                        any_trace = True
                if not tech_daily.empty:
                    all_t = (tech_daily[tech_daily["species"].isin(sp_sel)].groupby("date")["count"]
                             .sum().sort_index().cumsum())
                    if not all_t.empty:
                        fig_net.add_trace(go.Scatter(x=list(all_t.index), y=list(all_t.values),
                            name="Tech — All fish", line=dict(color="#546E7A",width=2.5,dash="dash"),
                            mode="lines", legendgroup="all_fish"))
                        any_trace = True

            if not any_trace:
                st.info("No directional data found. Check direction columns exist in the model data.")
            else:
                fig_net.add_hline(y=0, line_dash="dot", line_color="grey", annotation_text="Equal up/down")
                fig_net.update_layout(
                    title="Cumulative counts: Net migration (↑−↓), Direction split, and All fish",
                    xaxis_title="Date", yaxis_title="Cumulative count",
                    hovermode="x unified", height=520,
                    legend=dict(orientation="v", x=1.01, y=1))
                st.plotly_chart(fig_net, use_container_width=True)
                st.caption(f"Direction logic: **{dir_mode.replace('_',' ')}**  ·  "
                           "Thick = net · Thin = individual directions · Grey = all fish total")


# ===========================================================================
# PAGE: COMPARISON
# ===========================================================================
elif "Comparison" in page:
    st.header("⚖️ Model vs Technician Comparison")

    sp_sel2 = st.multiselect("Species", SPECIES_FISH, default=SPECIES_FISH, key="comp_sp")
    gran2   = st.radio("Granularity", ["Daily","Weekly"], horizontal=True, key="comp_gran")

    def _agg_weekly(df):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.to_period("W").dt.start_time.dt.date
        return df.groupby(["date","species"])["count"].sum().reset_index()

    def _agg_weekly_dir(df):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.to_period("W").dt.start_time.dt.date
        return df.groupby(["date","species","direction"])["count"].sum().reset_index()

    md2 = (_agg_weekly(model_daily) if gran2=="Weekly" else model_daily.copy())
    td2 = (_agg_weekly(tech_daily)  if gran2=="Weekly" else tech_daily.copy())
    md2["source"] = "Model"; td2["source"] = "Tech"
    combined = pd.concat([md2[md2["species"].isin(sp_sel2)], td2[td2["species"].isin(sp_sel2)]], ignore_index=True)
    if not combined.empty:
        fig_cmp = px.bar(combined, x="date", y="count", color="source", facet_row="species",
                         barmode="group",
                         title=f"{'Daily' if gran2=='Daily' else 'Weekly'} counts: Model vs Tech",
                         color_discrete_map={"Model":"#1976D2","Tech":"#F57C00"},
                         height=max(200*len(sp_sel2)+80,300),
                         labels={"date":"Date","count":"Count","source":"Source"})
        fig_cmp.update_layout(legend=dict(orientation="h",y=1.02))
        st.plotly_chart(fig_cmp, use_container_width=True)

    st.divider()
    st.subheader("Discrepancy table")
    if not model_daily.empty and not tech_daily.empty:
        m3 = model_daily.groupby(["date","species"])["count"].sum()
        t3 = tech_daily.groupby(["date","species"])["count"].sum()
        disc = pd.concat([m3.rename("model"),t3.rename("tech")],axis=1).reset_index().fillna(0)
        disc["model"]      = disc["model"].astype(int)
        disc["tech"]       = disc["tech"].astype(int)
        disc["difference"] = disc["model"] - disc["tech"]
        disc["abs_diff"]   = disc["difference"].abs()
        disc = disc[disc["species"].isin(sp_sel2)].sort_values("abs_diff",ascending=False)
        thresh_disc   = st.slider("Highlight rows with |difference| ≥", 1, 20, 3)
        abs_diff_vals = disc["abs_diff"].values
        disc_display  = disc.drop(columns="abs_diff").reset_index(drop=True)
        def _hl(row):
            return (["background-color:#FFCDD2"]*len(row) if abs_diff_vals[row.name]>=thresh_disc else [""]*len(row))
        st.dataframe(disc_display.style.apply(_hl,axis=1), hide_index=True, use_container_width=True, height=450)

    st.divider()
    st.subheader("Direction comparison: Model vs Technician")

    has_sides_comp = all(c in model_df.columns for c in ["first_side","last_side","exited_frame"])
    if has_sides_comp:
        dir_logic_comp = st.radio("Model direction logic",
            ["Full transit — first=Right → last=Left (recommended)",
             "Exit side — last_side + exited_frame",
             "Direction field — model direction tag"],
            index=0, key="comp_dir_logic", horizontal=True)
    else:
        dir_logic_comp = "Direction field — model direction tag"
    comp_dir_mode = ("full_transit" if "Full transit" in dir_logic_comp
                     else "exit_side" if "Exit side" in dir_logic_comp else "direction_field")

    mdf_c = model_df[model_df["model_species"].isin(sp_sel2)].copy()
    mdf_c["direction"] = compute_model_direction(mdf_c, comp_dir_mode)
    model_dir_daily = (mdf_c[mdf_c["direction"].isin(["Upstream","Downstream"])]
                       .groupby(["date","model_species","direction"]).size().reset_index(name="count")
                       .rename(columns={"model_species":"species"}))

    tdf_c = tech_df[(tech_df["false_trigger"]==0) & tech_df["species"].isin(sp_sel2)].copy()
    tdf_c["direction"] = tdf_c["direction"].apply(norm_direction)
    tech_dir_daily = (tdf_c[tdf_c["direction"].isin(["Upstream","Downstream"])]
                      .groupby(["date","species","direction"])["count"].sum().reset_index())

    if gran2=="Weekly":
        if not model_dir_daily.empty: model_dir_daily = _agg_weekly_dir(model_dir_daily)
        if not tech_dir_daily.empty:  tech_dir_daily  = _agg_weekly_dir(tech_dir_daily)

    if not model_dir_daily.empty or not tech_dir_daily.empty:
        fig_dir = go.Figure()
        _dc = {("Model","Upstream"):"#1565C0",("Model","Downstream"):"#42A5F5",
               ("Tech","Upstream"):"#E65100",("Tech","Downstream"):"#FFCC02"}
        for sp in sp_sel2:
            color = SPECIES_COLORS.get(sp,"#607D8B")
            for src, ddf in [("Model",model_dir_daily),("Tech",tech_dir_daily)]:
                if ddf.empty: continue
                for direction in ["Upstream","Downstream"]:
                    sub = ddf[(ddf["species"]==sp)&(ddf["direction"]==direction)]
                    if sub.empty: continue
                    dl = "↑" if direction=="Upstream" else "↓"
                    fig_dir.add_trace(go.Scatter(x=sub["date"],y=sub["count"],
                        name=f"{src} — {sp} {dl}",
                        line=dict(color=_dc.get((src,direction),color),width=2 if src=="Model" else 1.5,
                                  dash="solid" if direction=="Upstream" else "dot" if src=="Model" else "dashdot"),
                        mode="lines+markers",
                        marker=dict(size=5,symbol="circle" if direction=="Upstream" else "square"),
                        legendgroup=f"{sp}_{src}_{direction}"))
        fig_dir.update_layout(title="Upstream ↑ / Downstream ↓: Model vs Tech",
                              xaxis_title="Date",yaxis_title="Count",
                              hovermode="x unified",height=520,
                              legend=dict(orientation="v",x=1.01,y=1))
        st.plotly_chart(fig_dir, use_container_width=True)

    st.divider()
    st.subheader("Direction totals by species")
    rows_dir = []
    for sp in sp_sel2:
        def _tot(df, direction):
            if df.empty: return 0
            sub = df[(df["species"]==sp)&(df["direction"]==direction)]
            return int(sub["count"].sum()) if not sub.empty else 0
        m_up = _tot(model_dir_daily,"Upstream"); m_dn = _tot(model_dir_daily,"Downstream")
        t_up = _tot(tech_dir_daily, "Upstream"); t_dn = _tot(tech_dir_daily, "Downstream")
        rows_dir.append({"Species":sp,"Model ↑":m_up,"Tech ↑":t_up,"Diff ↑":m_up-t_up,
                         "Model ↓":m_dn,"Tech ↓":t_dn,"Diff ↓":m_dn-t_dn,
                         "Model net (↑−↓)":m_up-m_dn,"Tech net (↑−↓)":t_up-t_dn,
                         "Net diff":(m_up-m_dn)-(t_up-t_dn)})
    dir_tbl = pd.DataFrame(rows_dir)
    dir_tbl = dir_tbl[dir_tbl[["Model ↑","Tech ↑","Model ↓","Tech ↓"]].sum(axis=1)>0]
    if not dir_tbl.empty:
        def _colour_diff(val):
            if val > 0: return "color: #1976D2"
            if val < 0: return "color: #C62828"
            return ""
        st.dataframe(dir_tbl.style.applymap(_colour_diff, subset=["Diff ↑","Diff ↓","Net diff"]),
                     hide_index=True, use_container_width=True)
        st.caption(f"Blue = model higher · Red = model lower · Logic: **{comp_dir_mode.replace('_',' ')}**")


# ===========================================================================
# PAGE: STRUGGLING VIDEOS
# ===========================================================================
elif "Struggling" in page:
    st.header("🎬 Struggling Videos")

    if struggle.empty:
        if matched.empty:
            st.info("No direct-match data. Requires fish-counter-export tech files with a `video_rel` column.")
        else:
            st.success("No struggling events at the current threshold — try lowering it in the sidebar.")
        st.stop()

    id_cols = ["datetime","event_id","species","direction",
               "model_species","model_direction","top_score",
               "wrong_species","low_confidence","missed",
               "video_path","video_rel","match_key"]
    avail      = [c for c in id_cols if c in struggle.columns]
    event_list = (struggle[avail]
                  .drop_duplicates(subset=["match_key"] if "match_key" in avail else avail[:3])
                  .reset_index(drop=True))

    def _reason(row) -> str:
        r = []
        if row.get("wrong_species"):  r.append("Wrong species")
        if row.get("low_confidence"): r.append(f"Low conf ({row.get('top_score',0):.2f})")
        if row.get("missed"):         r.append("Missed detection")
        return " · ".join(r)

    event_list["Reason"] = event_list.apply(_reason, axis=1)

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader(f"{len(event_list)} struggling events")
        disp_cols = [c for c in ["datetime","event_id","species","model_species","top_score","Reason"]
                     if c in event_list.columns]
        disp = event_list[disp_cols].copy()
        if "top_score" in disp.columns:
            disp["top_score"] = disp["top_score"].round(3)
        if "datetime" in disp.columns:
            disp["datetime"] = pd.to_datetime(disp["datetime"]).dt.strftime("%Y-%m-%d %H:%M")
        sel = st.dataframe(disp, hide_index=True, use_container_width=True,
                           selection_mode="single-row", on_select="rerun",
                           key="struggle_table", height=500)
        selected = (st.session_state.get("struggle_table",{})
                    .get("selection",{}).get("rows",[]))

        st.divider()
        export_cols = [c for c in ["datetime","event_id","species","model_species",
                                    "top_score","Reason","video_path","video_rel"]
                       if c in event_list.columns]
        export_df = event_list[export_cols].copy()
        if "datetime" in export_df.columns:
            export_df["datetime"] = pd.to_datetime(export_df["datetime"]).dt.strftime("%Y-%m-%d %H:%M")
        st.download_button(
            label="⬇ Export struggling videos list (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="struggling_videos.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with right:
        if not selected:
            st.info("👈 Select an event from the list to view details.")
        else:
            row = event_list.iloc[selected[0]]
            st.subheader(f"Event {row.get('event_id','?')}  ·  "
                         f"{pd.to_datetime(row.get('datetime','')).strftime('%Y-%m-%d %H:%M') if row.get('datetime') else ''}")
            inf1, inf2, inf3 = st.columns(3)
            inf1.metric("Tech species",      str(row.get("species","—")))
            inf2.metric("Model prediction",  str(row.get("model_species","—")))
            conf_val = row.get("top_score")
            inf3.metric("Confidence",
                        f"{float(conf_val):.3f}" if conf_val and not pd.isna(conf_val) else "—",
                        delta="⚠ low" if conf_val and float(conf_val) < float(st.session_state.conf_thresh) else None,
                        delta_color="inverse")
            st.caption(f"**Reason flagged:** {row.get('Reason','—')}")

            score_keys = ["score_chinook","score_coho","score_atlantic","score_rainbow",
                          "score_brown","score_background"]
            if any(k in row.index for k in score_keys):
                st.plotly_chart(chart_class_scores(row), use_container_width=True)

            if "video_path" in row.index and str(row.get("video_path","")) not in ("","nan"):
                st.divider()
                st.caption(f"**Video path:** `{row.get('video_path')}`")


# ===========================================================================
# PAGE: DISTRIBUTIONS
# ===========================================================================
elif "Distributions" in page:
    st.header("📉 Distributions & Statistics")

    sp_sel3 = st.multiselect("Species", SPECIES_FISH, default=SPECIES_FISH, key="dist_sp")

    st.subheader("Confidence score distribution")
    sub = model_df[model_df["model_species"].isin(sp_sel3)].dropna(subset=["top_score"])
    if not sub.empty:
        fig_h = px.histogram(sub, x="top_score", color="model_species", nbins=40, opacity=0.75,
                             barmode="overlay", color_discrete_map=SPECIES_COLORS,
                             labels={"top_score":"Confidence Score","model_species":"Species"},
                             title="Distribution of model confidence scores by species")
        fig_h.update_layout(height=400)
        st.plotly_chart(fig_h, use_container_width=True)

    st.divider()
    st.subheader("Monthly detections by species")
    sub2 = model_df[model_df["model_species"].isin(sp_sel3)].copy()
    sub2["month"] = pd.to_datetime(sub2["date"]).dt.to_period("M").astype(str)
    monthly = sub2.groupby(["month","model_species"]).size().reset_index(name="count")
    fig_m = px.bar(monthly, x="month", y="count", color="model_species",
                   color_discrete_map=SPECIES_COLORS, barmode="stack",
                   title="Model detections per month by species",
                   labels={"month":"Month","count":"Detections","model_species":"Species"})
    fig_m.update_layout(height=400)
    st.plotly_chart(fig_m, use_container_width=True)

    st.divider()
    st.subheader("Direction distribution")
    if "model_direction" in model_df.columns:
        dir_df = (model_df[model_df["model_species"].isin(sp_sel3)]
                  .groupby(["model_species","model_direction"]).size().reset_index(name="count"))
        fig_d = px.bar(dir_df, x="model_species", y="count", color="model_direction",
                       barmode="group", title="Model direction predictions by species",
                       labels={"model_species":"Species","count":"Count","model_direction":"Direction"})
        fig_d.update_layout(height=380)
        st.plotly_chart(fig_d, use_container_width=True)

    st.divider()
    st.subheader("Confidence score statistics by species")
    if "top_score" in model_df.columns:
        stats = (model_df[model_df["model_species"].isin(sp_sel3)]
                 .groupby("model_species")["top_score"]
                 .agg(["mean","median","std","min","max"]).round(4).reset_index())
        stats.columns = ["Species","Mean","Median","Std Dev","Min","Max"]
        st.dataframe(stats, hide_index=True, use_container_width=True)


# ===========================================================================
# PAGE: METRICS
# ===========================================================================
elif "Metrics" in page:
    st.header("📐 Classification Metrics")

    if matched.empty:
        st.info("No directly-matched events found. Requires fish-counter tech files with a `video_rel` column.")
        st.stop()

    fish_rows = matched[(matched["false_trigger"]==0) & matched["species"].isin(SPECIES_FISH)]
    if fish_rows.empty:
        st.warning("No fish-only matched rows available.")
        st.stop()

    st.caption(f"Using **{len(fish_rows):,}** directly-matched fish events "
               f"({fish_rows['species'].nunique()} species).")

    default_thresh = float(st.session_state.conf_thresh)
    with st.expander("Per-species confidence thresholds", expanded=True):
        thresh_cols = st.columns(len(SPECIES_FISH))
        species_thresholds = {}
        for col, sp in zip(thresh_cols, SPECIES_FISH):
            species_thresholds[sp] = col.slider(sp, 0.0, 0.95, default_thresh, 0.05, key=f"thresh_{sp}")

    ctrl2, ctrl3 = st.columns(2)
    with ctrl2:
        normalize_opt = st.selectbox("Normalize confusion matrix",
                                     ["Row (Recall)","Column (Precision)","Raw counts"], index=0)
    normalize = {"Row (Recall)":"true","Column (Precision)":"pred","Raw counts":"none"}[normalize_opt]
    with ctrl3:
        sp_for_curves = st.multiselect("Species for PR curves", SPECIES_FISH, default=SPECIES_FISH, key="pr_sp")

    st.divider()
    st.subheader("Confusion Matrix")
    st.plotly_chart(chart_confusion_matrix(matched, species_thresholds, normalize), use_container_width=True)
    st.caption("Rows = actual (tech) · Columns = model predictions · "
               "'Flagged for Review' = detected but below threshold · 'No detection' = no model hit.")

    st.divider()
    st.subheader("Precision–Recall Curves")
    with st.spinner("Computing…"):
        pr_data = compute_pr_curves(matched)
    if pr_data:
        st.plotly_chart(chart_pr_curve(pr_data, sp_for_curves), use_container_width=True)
        st.caption("Each point = one confidence threshold (0.05 → 0.975). Hover for details.")

        st.divider()
        st.subheader("Metric vs Confidence Threshold")
        metric_sel = st.radio("Metric", ["precision","recall","f1"], horizontal=True,
                              format_func=lambda x: x.capitalize())
        st.plotly_chart(chart_metric_vs_threshold(pr_data, metric_sel, sp_for_curves), use_container_width=True)

        st.divider()
        st.subheader("Summary table")
        avg_thresh = float(np.mean(list(species_thresholds.values())))
        summary = metrics_summary_table(pr_data, avg_thresh)
        if not summary.empty:
            st.dataframe(
                summary.style.format({"Precision":"{:.4f}","Recall":"{:.4f}","F1":"{:.4f}","Threshold":"{:.3f}"})
                .background_gradient(subset=["F1"], cmap="RdYlGn", vmin=0, vmax=1),
                hide_index=True, use_container_width=True)
