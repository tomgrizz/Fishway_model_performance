#!/usr/bin/env python3
"""
Build pre-processed data exports for the shareable web app.

Run once locally whenever new model batches or tech files are added:

    python build_data.py

Outputs
-------
    data/exports/model_detections.csv   – one row per detection from every *_count.json
    data/exports/tech_counts.csv        – consolidated, normalised technician counts
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
ROOT       = Path(__file__).parent
MODEL_DIR  = ROOT / "data" / "model"
TECH_DIR   = ROOT / "data" / "tech"
EXPORT_DIR = ROOT / "data" / "exports"

# ---------------------------------------------------------------------------
# Species / direction normalisation  (mirrors app.py)
# ---------------------------------------------------------------------------
_NORM = {
    "rainbow":       "Rainbow Trout",
    "rainbow trout": "Rainbow Trout",
    "chinook":       "Chinook",
    "atlantic":      "Atlantic",
    "coho":          "Coho",
    "brown":         "Brown Trout",
    "brown trout":   "Brown Trout",
    "not fish":      "Non-fish",
    "non fish":      "Non-fish",
    "non-fish":      "Non-fish",
    "background":    "Non-fish",
    "unknown":       "Unknown",
}

def norm_species(s) -> str:
    if not isinstance(s, str):
        return "Unknown"
    return _NORM.get(s.strip().lower(), s.strip())


# ---------------------------------------------------------------------------
# Model JSON → DataFrame
# ---------------------------------------------------------------------------
def build_model_detections(model_dir: Path) -> pd.DataFrame:
    rows = []
    json_files = sorted(model_dir.rglob("*_count.json"))
    total = len(json_files)
    print(f"Scanning {total} JSON files…")

    for i, json_path in enumerate(json_files, 1):
        if i % 1000 == 0:
            print(f"  {i}/{total}")
        try:
            data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue

        clip_folder    = json_path.parent.name
        batch_folder   = json_path.parent.parent.name
        vid_event_id   = json_path.stem.replace("_count", "").strip()
        video_rec_time = data.get("video_recording_time")

        det_keys = [k for k in data if k != "video_recording_time"]
        if not det_keys:
            continue

        for det_id in det_keys:
            det = data[det_id]
            cs  = det.get("class_scores", {})
            top_score = max(cs.values()) if cs else None
            rows.append({
                "batch_folder":         batch_folder,
                "clip_folder":          clip_folder,
                "json_file":            json_path.name,
                "vid_event_id":         vid_event_id,
                "det_id":               det_id,
                "video_path":           det.get("video_path"),
                "video_recording_time": video_rec_time,
                "model_species":        norm_species(det.get("top_class", "")),
                "model_direction":      det.get("direction"),
                "first_side":           det.get("first_side"),
                "last_side":            det.get("last_side"),
                "entered_frame":        det.get("entered_frame"),
                "exited_frame":         det.get("exited_frame"),
                "score_chinook":        cs.get("Chinook"),
                "score_coho":           cs.get("Coho"),
                "score_atlantic":       cs.get("Atlantic"),
                "score_rainbow":        cs.get("Rainbow Trout"),
                "score_brown":          cs.get("Brown Trout"),
                "score_background":     cs.get("Background"),
                "top_score":            top_score,
            })

    if not rows:
        print("WARNING: no detections found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["video_recording_time"] = pd.to_datetime(df["video_recording_time"], errors="coerce")
    df["date"] = df["video_recording_time"].dt.date
    df["clip_folder"] = df["clip_folder"].astype(str).str.strip()
    print(f"  -> {len(df):,} detections across {df['clip_folder'].nunique():,} clips")
    return df


# ---------------------------------------------------------------------------
# Tech CSVs → consolidated DataFrame
# ---------------------------------------------------------------------------
def _detect_format(df: pd.DataFrame) -> str:
    if "sortname" in df.columns:
        return "riverwatcher"
    if "species" in df.columns and "event_id" in df.columns:
        return "fish_counter"
    return "unknown"


def _load_one_tech(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, low_memory=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8", errors="replace")

    fmt = _detect_format(df)
    if fmt == "riverwatcher":
        out = pd.DataFrame()
        out["datetime"]      = pd.to_datetime(df["DateTime"], errors="coerce")
        out["species"]       = df["sortname"].apply(norm_species)
        out["direction"]     = df["Direction"].astype(str).str.strip()
        out["event_id"]      = df["Attrib"].astype(str).str.strip()
        out["video_rel"]     = None
        out["count"]         = 1
        out["false_trigger"] = 0
        out["fmt"]           = "riverwatcher"
    elif fmt == "fish_counter":
        import numpy as np
        out = pd.DataFrame()
        out["datetime"]      = pd.to_datetime(df["ts"], errors="coerce")
        out["species"]       = df["species"].apply(norm_species)
        out["direction"]     = df.get("movement", pd.Series(["Unknown"] * len(df))).astype(str).str.strip()
        out["event_id"]      = df["event_id"].astype(str).str.strip()
        out["video_rel"]     = df.get("video_rel", np.nan)
        out["count"]         = pd.to_numeric(df.get("count", 1), errors="coerce").fillna(1).astype(int)
        out["false_trigger"] = pd.to_numeric(df.get("false_trigger", 0), errors="coerce").fillna(0).astype(int)
        out["fmt"]           = "fish_counter"
    else:
        print(f"  WARNING: unrecognised format in {path.name}")
        return pd.DataFrame()

    out["source_file"] = path.name
    out["date"]        = out["datetime"].dt.date
    return out


def build_tech_counts(tech_dir: Path) -> pd.DataFrame:
    dfs = []
    for p in sorted(tech_dir.glob("*.csv")):
        print(f"  Loading {p.name}…")
        try:
            d = _load_one_tech(p)
            if not d.empty:
                dfs.append(d)
        except Exception as e:
            print(f"  ERROR loading {p.name}: {e}")

    if not dfs:
        print("WARNING: no tech files loaded.")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True).sort_values("datetime")
    print(f"  -> {len(df):,} tech records across {df['source_file'].nunique()} files")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Building model detections ===")
    model_df = build_model_detections(MODEL_DIR)
    if not model_df.empty:
        out = EXPORT_DIR / "model_detections.csv"
        model_df.to_csv(out, index=False)
        print(f"Saved -> {out}  ({out.stat().st_size // 1024} KB)")

    print("\n=== Building tech counts ===")
    tech_df = build_tech_counts(TECH_DIR)
    if not tech_df.empty:
        out = EXPORT_DIR / "tech_counts.csv"
        tech_df.to_csv(out, index=False)
        print(f"Saved -> {out}  ({out.stat().st_size // 1024} KB)")

    print("\nDone. Run:  streamlit run performance_app/app_web.py")
