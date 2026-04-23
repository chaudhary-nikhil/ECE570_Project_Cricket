# ECE 570 Project: T20 Cricket Score Predictor

This repository contains the code and LaTeX for a T20 first-innings score prediction project with:
- model training (`TensorFlow` + tabular features + categorical embeddings),
- FastAPI backend serving predictions/chat endpoints,
- Streamlit frontend for interactive use,
- conference-style paper (`LLM_570.tex`).

## 1) Repository Structure

### Core app/runtime files (needed for final product)
- `main.py` - FastAPI entrypoint (`/predict`, `/live-scores`, `/chat`)
- `backend.py` - model loading + inference helpers
- `app.py` - Streamlit UI
- `model_io.py` - shared constants for feature schema
- `cricket_model.h5` - trained model artifact used at inference
- `label_encoders.pkl` - categorical encoders + metadata used at inference
- `player_priors.csv` - per-player prior strengths used for roster-aware features
- `ipl_match_team_totals.csv` - historical context used by UI/chat tools
- `weather.py`, `espn_scraper.py`, `espn_roster.py` - external data integrations
- `cricket_chat_service.py`, `cricket_chat_tools.py` - chat toolchain
- `requirements.txt` - dependencies
- `LLM_570.tex` - paper source

### Training/reproducibility files (recommended to keep)
- `train_cricket_model.py` - full training + evaluation pipeline
- `build_player_priors.py` - constructs `player_priors.csv`
- `build_ipl_match_team_totals.py` - constructs `ipl_match_team_totals.csv`

### Large raw datasets (optional in repo; can be external-hosted)
- `kaggle_ipl_dataset/IPL.csv`
- `ipl_data.csv`
- `t20_wc_2024_deliveries.csv`

If repo size is a concern, move raw datasets to external storage and document download steps (see Section 6).

## 2) Dependencies

Primary stack:
- Python 3.11
- TensorFlow
- NumPy, Pandas, scikit-learn
- FastAPI + Uvicorn
- Streamlit
- Requests / httpx

Install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) How to Run (Final Product)

From repository root:

```bash
# Terminal 1 - backend API
source .venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 - frontend UI
source .venv/bin/activate
streamlit run app.py
```

Then open the Streamlit URL printed in terminal (typically `http://localhost:8501`).

## 4) How to Train/Reproduce Model

```bash
source .venv/bin/activate
python train_cricket_model.py
```

This regenerates:
- `cricket_model.h5`
- `label_encoders.pkl`

## 5) Authorship and Provenance Statement

### Written by project authors
- Application code in this repository (`app.py`, `main.py`, `backend.py`, training scripts, utilities) unless noted below.

### Adapted from prior code/templates
- `LLM_570.tex` is adapted from the ICLR 2026 LaTeX template format.
- Common FastAPI/Streamlit usage patterns follow official library documentation.

### Copied from external repositories
- No project source file is directly copied from an external code repository.

### Prior-code edits with exact line references
- `train_cricket_model.py`: lines `500-555`
  - Added percentage-style metrics (MAPE, `1-MAPE` accuracy, `R^2`, tolerance accuracy).
  - Added these metrics to saved metadata (`label_encoders.pkl`).
- `LLM_570.tex`: lines `92-127`
  - Added model architecture diagram figure (TikZ).
- `LLM_570.tex`: line `35`
  - Added required hyperlink placeholder below abstract.

If additional inherited code is used, update this section with exact file paths and line ranges.

## 6) Dataset/Model Availability Policy

Current project state:
- Model artifacts are included locally (`cricket_model.h5`, `label_encoders.pkl`).
- Datasets are currently expected as local files in repository paths shown above.

Automatic download status:
- Automatic download of all datasets is **not fully implemented** in the current codebase.

If not committing large datasets to GitHub:
1. Upload datasets to a public location (Kaggle/Drive/S3).
2. Add a lightweight `download_data.sh` (or Python downloader) that places files in:
   - `kaggle_ipl_dataset/IPL.csv`
   - `ipl_data.csv`
   - `t20_wc_2024_deliveries.csv`
3. Update this README with exact links and checksums.

## 7) Suggested `.gitignore` Entries

At minimum:
- `__pycache__/`
- `.venv/`
- `*.log`
- `.DS_Store`

If datasets are hosted externally:
- `kaggle_ipl_dataset/`
- `ipl_data.csv`
- `t20_wc_2024_deliveries.csv`

