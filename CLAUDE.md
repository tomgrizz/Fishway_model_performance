# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fish detection model performance review dashboard. Compares a computer vision model's fish species detections against manual technician counts from the Credit River RiverWatcher system. The primary deliverable is an interactive Streamlit web app.

## Running the App

```bash
# From the project root
streamlit run performance_app/app.py
```

App runs at `http://localhost:8501`. Data paths are configured in the sidebar at runtime.

## Installing Dependencies

```bash
pip install -r performance_app/requirements.txt
```

Note: `scikit-learn` is used (for confusion matrix) but not listed in `requirements.txt` — add it if missing.

## Running the Standalone Comparison Script

```bash
python compare.py
```

## Architecture

### Data Sources

Two input data types must be provided via the sidebar:

1. **Model CSV** (`all_model_detections.csv`) — aggregated detections from JSON output files. Key columns: `clip_folder`, `vid_event_id`, `model_species`, `model_direction`, `top_score`, `score_*` (per-species confidence), `video_path`, `video_recording_time`.

2. **Technician directory** (`data/tech/`) — CSV exports in two formats:
   - *RiverWatcher format*: columns `DateTime, sortname, Direction, Attrib`
   - *Fish Counter format*: columns `ts, species, movement, event_id, video_rel, count, false_trigger`

Raw model detections live in `data/model/` as JSON files organized by date-range batch folders. These are pre-aggregated into the model CSV; the app does not read JSON directly.

### Core App Logic (`performance_app/app.py`)

The app is a single ~1100-line Streamlit file organized as:

- **Data loading** (`load_model_csv`, `load_all_tech`, `_detect_format`, `_load_one_tech`) — cached with `@st.cache_data`. Tech files auto-detect format and handle encoding (UTF-8, Windows-1252, Latin-1).
- **Matching** (`build_matches`) — joins model detections to technician events using `clip_folder + vid_event_id` as the key. Only Fish Counter format has `video_rel` for matching.
- **Flagging** (`flag_struggling`) — marks detections as problematic: wrong species, low confidence (configurable threshold), or missed detections.
- **Metrics** (`compute_pr_curves`, `chart_confusion_matrix`) — precision-recall curves and species confusion matrix using scikit-learn.
- **6 UI pages**: Overview, Time Series, Comparison, Struggling Videos, Distributions, Metrics.

### Species Normalization

Five fish species (Chinook, Coho, Atlantic, Rainbow Trout, Brown Trout) plus Non-fish and Unknown. Input variations (e.g. `"rainbow"`, `"Rainbow"`) are normalized to canonical names.

### Video Playback

Videos originate from a network drive (`G:\RiverWatcher\Credit\2025`). The sidebar provides a path remap field to substitute a local path prefix for playback.

### Hardcoded Default Paths

Session state defaults in `app.py` point to local Windows paths:
- Model CSV: `C:\Users\lomu-\Documents\Projects\model_performance\all_model_detections.csv`
- Tech dir: `C:\Users\lomu-\Documents\Projects\model_performance\data\tech`
- Video root: `G:\RiverWatcher\Credit\2025`

These are sidebar defaults only — users override them at runtime.
