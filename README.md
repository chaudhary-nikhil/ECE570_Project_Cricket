# ECE 570 Project: T20 Cricket Score Predictor

Neural first-innings score predictor for T20 cricket, with a FastAPI backend, Streamlit frontend, and ICLR-format paper.

## 1. Code Structure

```
.
├── main.py                         FastAPI entrypoint (/predict, /live-scores, /chat)
├── backend.py                      Model loading + inference
├── app.py                          Streamlit UI
├── model_io.py                     Shared feature schema / constants
├── weather.py                      Venue weather lookup
├── espn_scraper.py                 Live match scraping
├── espn_roster.py                  Roster lookup
├── cricket_chat_service.py         Chat orchestration
├── cricket_chat_tools.py           Chat tool implementations
├── train_cricket_model.py          Training + evaluation pipeline
├── build_player_priors.py          Builds data/player_priors.csv
├── build_ipl_match_team_totals.py  Builds data/ipl_match_team_totals.csv
├── requirements.txt
├── .gitignore
├── artifacts/
│   ├── cricket_model.h5            Trained model (committed)
│   └── label_encoders.pkl          Encoders + metadata (committed)
├── data/
│   ├── player_priors.csv           Derived; committed
│   ├── ipl_match_team_totals.csv   Derived; committed
│   ├── ipl_data.csv                Raw; gitignored
│   ├── t20_wc_2024_deliveries.csv  Raw; gitignored
│   └── kaggle_ipl_dataset/IPL.csv  Raw; gitignored (kagglehub symlink)
└── paper/LLM_570.tex               ICLR paper source
```

Runtime (final product) requires: all `.py` modules at root, `artifacts/`, and the two derived CSVs in `data/`. Raw CSVs are only needed to retrain.

## 2. Dependencies

Python 3.11. Key packages: `tensorflow`, `numpy`, `pandas`, `scikit-learn`, `fastapi`, `uvicorn`, `streamlit`, `requests`, `beautifulsoup4`. Full list pinned in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Running the Project

From the repository root, with `.venv` activated:

```bash
# Backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Frontend (separate terminal)
streamlit run app.py
```

Open the Streamlit URL printed in the terminal (default `http://localhost:8501`).

### Retraining (optional)

```bash
python train_cricket_model.py
```

Regenerates `artifacts/cricket_model.h5` and `artifacts/label_encoders.pkl`. Requires the raw CSVs in `data/` (see Section 5).

## 4. Authorship and Provenance

All `.py` source files in this repository were written by the project author.

**Adapted:**
- `paper/LLM_570.tex` uses the official ICLR 2026 LaTeX template (`iclr2026_conference.sty`, `iclr2026_conference.bst`, `math_commands.tex`); paper prose and the TikZ figure are original.

**Copied from external repositories:** None.

### Edits to prior code (exact line numbers)

- `train_cricket_model.py`, lines **494–548**: added MAPE, `1−MAPE` accuracy, `R²`, and within-±10/±15 tolerance accuracy; persisted these keys in `artifacts/label_encoders.pkl` metadata.
- `paper/LLM_570.tex`, line **35**: added the required hyperlink to code/LaTeX below the abstract.
- `paper/LLM_570.tex`, lines **92–129**: added the multi-input model-architecture figure (TikZ).

## 5. Dataset and Model Availability

**Model artifacts** (`artifacts/cricket_model.h5`, `artifacts/label_encoders.pkl`) are committed to the repository — no download required for inference.

**Derived data** (`data/player_priors.csv`, `data/ipl_match_team_totals.csv`) is committed — needed by the backend and chat tools at runtime.

**Raw training datasets** are not committed (too large) and must be obtained manually. Automatic download is **not** implemented because one source (`t20_wc_2024_deliveries.csv`) has no stable public URL, and Kaggle requires user-specific API credentials.

| File | Source | Destination |
|---|---|---|
| `IPL.csv` | [kaggle.com/datasets/chaitu20/ipl-dataset2008-2025](https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025) | `data/kaggle_ipl_dataset/IPL.csv` |
| `ipl_data.csv` | Legacy IPL ball-by-ball CSV used in prior coursework | `data/ipl_data.csv` |
| `t20_wc_2024_deliveries.csv` | T20 World Cup 2024 ball-by-ball, publicly circulated CSV | `data/t20_wc_2024_deliveries.csv` |

Kaggle CLI (optional, requires `~/.kaggle/kaggle.json`):

```bash
mkdir -p data/kaggle_ipl_dataset
kaggle datasets download -d chaitu20/ipl-dataset2008-2025 \
    -p data/kaggle_ipl_dataset --unzip
```

Inference and the deployed UI do **not** require any raw dataset — only the committed artifacts and derived CSVs.
