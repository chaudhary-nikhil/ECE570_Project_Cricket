import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from build_player_priors import normalize_name
from model_io import (
    CATEGORICAL_COLS,
    MAX_WICKETS,
    META_KEY_MODEL,
    NUMERIC_COLS,
    PLAYER_PRIORS_PATH,
    T20_MAX_OVERS,
)

# ------------ Config ------------
IPL_CSV_PATH = "data/ipl_data.csv"
T20_CSV_PATH = "data/t20_wc_2024_deliveries.csv"
KAGGLE_IPL_CSV_PATH = "data/kaggle_ipl_dataset/IPL.csv"
MODEL_PATH = "artifacts/cricket_model.h5"
ENCODERS_PATH = "artifacts/label_encoders.pkl"

# Kaggle IPL.csv overlaps heavily with the legacy ipl_data.csv. When both are
# present, prefer the richer Kaggle dataset to avoid training on duplicate matches.
PREFER_KAGGLE_IPL_OVER_LEGACY = True

# Filter out stray rows (super-overs, malformed entries) that don't represent
# a normal first-innings T20 trajectory.
MIN_OVERS_FILTER = 0.0
MAX_OVERS_FILTER = float(T20_MAX_OVERS)

TEST_FRACTION = 0.2
VAL_FRACTION = 0.15  # within training portion
RANDOM_SEED = 42

# In this dataset, the final innings score is in the "total" column,
# and the per-ball features already exist with the following names:
#   venue, bat_team, bowl_team, runs, wickets, overs, runs_last_5
TARGET_COL = "total"

# Map from "logical" feature names (what we conceptually use)
# to the actual column names present in ipl_data.csv.
COLUMN_MAP = {
    "venue": "venue",
    "batting_team": "bat_team",
    "bowling_team": "bowl_team",
    "current_score": "runs",
    "wickets_lost": "wickets",
    "overs_completed": "overs",
    "runs_in_last_5_overs": "runs_last_5",
}

REQUIRED_LOGICAL = [
    "venue",
    "batting_team",
    "bowling_team",
    "current_score",
    "wickets_lost",
    "overs_completed",
    "runs_in_last_5_overs",
    TARGET_COL,
]

# ------------ Load and prepare IPL data ------------
ipl_raw = pd.read_csv(IPL_CSV_PATH)

# Parse match date for recency weighting
ipl_raw["match_date"] = pd.to_datetime(ipl_raw["date"])

# Rename raw IPL columns to match the logical feature names we use below
rename_dict = {v: k for k, v in COLUMN_MAP.items()}
ipl_df = ipl_raw.rename(columns=rename_dict)

# Basic sanity check after renaming
missing_cols_ipl = [c for c in REQUIRED_LOGICAL if c not in ipl_df.columns]
if missing_cols_ipl:
    raise ValueError(f"Missing expected columns in IPL CSV after renaming: {missing_cols_ipl}")

# Attach parsed date and drop rows with missing target
ipl_df["match_date"] = ipl_raw["match_date"]
ipl_df = ipl_df.dropna(subset=[TARGET_COL])

# ------------ Load per-player priors for lineup-strength features ------------
# Produced by build_player_priors.py; missing file is a hard error because the
# new model schema requires these features at training and inference time.
priors_df = pd.read_csv(PLAYER_PRIORS_PATH)
_bat_prior_map = dict(zip(priors_df["player_key"], priors_df["bat_strength"].astype(float)))
_bowl_prior_map = dict(zip(priors_df["player_key"], priors_df["bowl_strength"].astype(float)))


def _bat_prior_for(name: str) -> float:
    return _bat_prior_map.get(normalize_name(name), 0.0)


def _bowl_prior_for(name: str) -> float:
    return _bowl_prior_map.get(normalize_name(name), 0.0)


# Legacy and T20WC sources have no per-ball batter/bowler lookup aligned with
# the Kaggle priors, so give them a neutral lineup strength of 0.0 (league avg).
ipl_df["batting_xi_strength"] = 0.0
ipl_df["bowling_attack_strength"] = 0.0

# Match-level group id for leakage-free splitting; ipl_data.csv has a `mid` column
if "mid" in ipl_df.columns:
    ipl_df["group_id"] = "legacy:" + ipl_df["mid"].astype(str)
else:
    ipl_df["group_id"] = "legacy:" + ipl_df.index.astype(str)


# ------------ Load and prepare T20 World Cup data ------------
try:
    t20_raw = pd.read_csv(T20_CSV_PATH)
except FileNotFoundError:
    t20_raw = None

t20_df_list = []
if t20_raw is not None:
    # Ensure required columns exist
    required_t20_cols = {
        "venue",
        "batting_team",
        "bowling_team",
        "match_id",
        "innings",
        "over",
        "runs_of_bat",
        "extras",
        "wicket_type",
    }
    missing_t20_cols = [c for c in required_t20_cols if c not in t20_raw.columns]
    if missing_t20_cols:
        raise ValueError(f"Missing expected columns in T20 CSV: {missing_t20_cols}")

    # Work on a copy to avoid modifying the original
    t20 = t20_raw.copy()

    # Parse match date for recency weighting
    t20["match_date"] = pd.to_datetime(t20["date"])

    # Total runs for each delivery
    t20["runs_of_bat"] = t20["runs_of_bat"].astype(float)
    t20["extras"] = t20["extras"].astype(float)
    t20["runs_this_ball"] = t20["runs_of_bat"] + t20["extras"]

    # Sort by match/innings/over so cumulative calculations are correct
    t20 = t20.sort_values(["match_id", "innings", "over"])

    # Cumulative runs and wickets within each innings
    t20["current_score"] = t20.groupby(["match_id", "innings"])["runs_this_ball"].cumsum()

    t20["wicket_flag"] = (
        t20["wicket_type"].astype(str).fillna("").str.strip() != ""
    ).astype(int)
    t20["wickets_lost"] = t20.groupby(["match_id", "innings"])["wicket_flag"].cumsum()

    # Overs as a float (already in decimal form like 10.3)
    t20["overs_completed"] = t20["over"].astype(float)

    # Runs in the last 5 overs (~30 balls) within each innings
    t20["runs_in_last_5_overs"] = (
        t20.groupby(["match_id", "innings"])["runs_this_ball"]
        .rolling(window=30, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )

    # Final total for each innings = max cumulative score
    t20[TARGET_COL] = t20.groupby(["match_id", "innings"])["current_score"].transform("max")

    # Build a dataframe with the same logical columns as IPL
    t20_df = t20[
        [
            "venue",
            "batting_team",
            "bowling_team",
            "current_score",
            "wickets_lost",
            "overs_completed",
            "runs_in_last_5_overs",
            "match_date",
            "match_id",
            TARGET_COL,
        ]
    ].copy()

    # Drop any rows with missing target just in case
    t20_df = t20_df.dropna(subset=[TARGET_COL])
    t20_df["group_id"] = "t20wc:" + t20_df["match_id"].astype(str)
    t20_df = t20_df.drop(columns=["match_id"])
    t20_df["batting_xi_strength"] = 0.0
    t20_df["bowling_attack_strength"] = 0.0

    t20_df_list.append(t20_df)


# ------------ Load Kaggle IPL ball-by-ball (IPL.csv) ------------
try:
    kaggle_ipl_raw = pd.read_csv(KAGGLE_IPL_CSV_PATH, low_memory=False)
except FileNotFoundError:
    kaggle_ipl_raw = None

kaggle_ipl_df_list: list[pd.DataFrame] = []
if kaggle_ipl_raw is not None:
    required_k = {
        "match_id",
        "date",
        "innings",
        "batting_team",
        "bowling_team",
        "venue",
        "runs_total",
        "ball_no",
    }
    missing_k = [c for c in required_k if c not in kaggle_ipl_raw.columns]
    if missing_k:
        raise ValueError(f"Kaggle IPL.csv missing columns: {missing_k}")

    kg = kaggle_ipl_raw.copy()
    kg["match_date"] = pd.to_datetime(kg["date"], errors="coerce")
    kg["runs_total"] = pd.to_numeric(kg["runs_total"], errors="coerce").fillna(0.0)
    kg["ball_no"] = pd.to_numeric(kg["ball_no"], errors="coerce")
    kg = kg.dropna(subset=["match_date", "ball_no"])
    kg = kg.sort_values(["match_id", "innings", "ball_no"])

    kg["runs_this_ball"] = kg["runs_total"]
    kg["current_score"] = kg.groupby(["match_id", "innings"])["runs_this_ball"].cumsum()

    if "wicket_kind" in kg.columns:
        wk = kg["wicket_kind"].astype(str).str.strip()
        kg["wicket_flag"] = ((wk != "") & (wk.str.lower() != "nan")).astype(int)
    else:
        kg["wicket_flag"] = 0
    kg["wickets_lost"] = kg.groupby(["match_id", "innings"])["wicket_flag"].cumsum()

    kg["overs_completed"] = kg["ball_no"].astype(float)

    kg["runs_in_last_5_overs"] = (
        kg.groupby(["match_id", "innings"])["runs_this_ball"]
        .rolling(window=30, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )

    kg[TARGET_COL] = kg.groupby(["match_id", "innings"])["current_score"].transform("max")

    # Drop super-overs (innings 3+) that are chase-phase anomalies
    if "innings" in kg.columns:
        kg = kg[kg["innings"].isin([1, 2])]

    # ---- Per-innings XI strength from priors (leakage-aware enough) ----
    # Batting XI strength = mean bat_strength of every distinct batter that
    # actually batted in this (match_id, innings) team innings.
    # Bowling attack strength = mean bowl_strength of every distinct bowler
    # who delivered a ball in that same (match_id, innings).
    bat_prior_series = kg["batter"].fillna("").map(_bat_prior_for)
    bowl_prior_series = kg["bowler"].fillna("").map(_bowl_prior_for)
    kg["_bat_prior"] = bat_prior_series
    kg["_bowl_prior"] = bowl_prior_series
    innings_bat = (
        kg.groupby(["match_id", "innings", "batter"])["_bat_prior"].first().reset_index()
        .groupby(["match_id", "innings"])["_bat_prior"].mean().rename("batting_xi_strength")
    )
    innings_bowl = (
        kg.groupby(["match_id", "innings", "bowler"])["_bowl_prior"].first().reset_index()
        .groupby(["match_id", "innings"])["_bowl_prior"].mean().rename("bowling_attack_strength")
    )
    kg = kg.merge(innings_bat, on=["match_id", "innings"], how="left")
    kg = kg.merge(innings_bowl, on=["match_id", "innings"], how="left")
    kg["batting_xi_strength"] = kg["batting_xi_strength"].fillna(0.0)
    kg["bowling_attack_strength"] = kg["bowling_attack_strength"].fillna(0.0)

    kaggle_df = kg[
        [
            "venue",
            "batting_team",
            "bowling_team",
            "current_score",
            "wickets_lost",
            "overs_completed",
            "runs_in_last_5_overs",
            "batting_xi_strength",
            "bowling_attack_strength",
            "match_date",
            "match_id",
            TARGET_COL,
        ]
    ].copy()
    kaggle_df = kaggle_df.dropna(subset=[TARGET_COL])
    kaggle_df["group_id"] = "kaggle:" + kaggle_df["match_id"].astype(str)
    kaggle_df = kaggle_df.drop(columns=["match_id"])
    kaggle_ipl_df_list.append(kaggle_df)
    print(
        f"Loaded {len(kaggle_df)} ball-by-ball training rows from {KAGGLE_IPL_CSV_PATH} "
        f"with lineup-strength features attached."
    )


# ------------ Combine IPL, T20, and Kaggle IPL deliveries ------------
# Deduplicate IPL coverage: if Kaggle IPL.csv was loaded successfully and the
# preference is set, drop the legacy ipl_data.csv to avoid training on the
# same matches twice under two different schemas.
use_legacy_ipl = not (PREFER_KAGGLE_IPL_OVER_LEGACY and len(kaggle_ipl_df_list) > 0)
if use_legacy_ipl:
    dfs = [ipl_df] + t20_df_list + kaggle_ipl_df_list
    print(f"Using legacy IPL ({len(ipl_df)} rows) + T20 + Kaggle IPL.")
else:
    dfs = t20_df_list + kaggle_ipl_df_list
    print(
        f"Skipping legacy ipl_data.csv ({len(ipl_df)} rows) because Kaggle IPL.csv "
        "is available (PREFER_KAGGLE_IPL_OVER_LEGACY=True)."
    )
df = pd.concat(dfs, ignore_index=True)

# ------------ Filter malformed / non-T20 rows ------------
before = len(df)
df = df[(df["overs_completed"] >= MIN_OVERS_FILTER) & (df["overs_completed"] <= MAX_OVERS_FILTER)].copy()
df = df[df["wickets_lost"] <= MAX_WICKETS].copy()
df = df[df[TARGET_COL] >= df["current_score"]].copy()  # sanity: final >= current
print(f"Filtered out {before - len(df)} malformed rows; {len(df)} rows remain.")

# ------------ Engineered numeric features (T20) ------------
oc = df["overs_completed"].astype(float)
df["overs_remaining"] = (T20_MAX_OVERS - oc).clip(lower=0.0)
df["run_rate"] = df["current_score"].astype(float) / np.maximum(oc.to_numpy(dtype=float), 0.1)
df["wickets_remaining"] = (MAX_WICKETS - df["wickets_lost"].astype(float)).clip(lower=0.0)

# ------------ Recency-based sample weighting ------------
# Newer matches (by match_date) get higher weight; older matches still contribute.
date_numeric = df["match_date"].astype("int64")
min_date = date_numeric.min()
max_date = date_numeric.max()

if max_date == min_date:
    recency = np.ones_like(date_numeric, dtype="float32")
else:
    recency = (date_numeric - min_date) / (max_date - min_date)
    recency = recency.astype("float32")

# Map recency in [0, 1] to weights in [0.2, 1.0]
sample_weight = 0.2 + 0.8 * recency
df["sample_weight"] = sample_weight.astype("float32")

# ------------ Encode categorical features ------------
encoders: dict = {}

for col in CATEGORICAL_COLS:
    df[col] = df[col].astype(str).fillna("Unknown")
    le = LabelEncoder()
    # Guarantee "Unknown" is in the vocabulary so inference can add unseen venues/teams
    # without indices exceeding Embedding(input_dim) saved from this fit.
    vocab = list(dict.fromkeys(df[col].tolist()))
    if "Unknown" not in vocab:
        vocab.append("Unknown")
    le.fit(vocab)
    df[col] = le.transform(df[col])
    encoders[col] = le

# Target: runs still to come in the innings (stable scale for the network)
y_remaining = (df[TARGET_COL].astype(float) - df["current_score"].astype(float)).clip(lower=0.0)
y_remaining = y_remaining.values.astype("float32")
y_total = df[TARGET_COL].values.astype("float32")

X_venue = df["venue"].values.astype(np.int32)
X_bat = df["batting_team"].values.astype(np.int32)
X_bowl = df["bowling_team"].values.astype(np.int32)
X_num = df[NUMERIC_COLS].values.astype("float32")
sample_weights = df["sample_weight"].values.astype("float32")

# ------------ Group-based train/test split (no match leakage) ------------
# Splitting by row would put different balls of the same innings in both train
# and test, producing overoptimistic metrics. Split by match instead.
groups = df["group_id"].astype(str).values
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=RANDOM_SEED)
idx_train, idx_test = next(gss.split(np.zeros(len(df)), groups=groups))
n_train_matches = df.iloc[idx_train]["group_id"].nunique()
n_test_matches = df.iloc[idx_test]["group_id"].nunique()
print(
    f"Train: {len(idx_train)} rows across {n_train_matches} matches | "
    f"Test: {len(idx_test)} rows across {n_test_matches} matches."
)

Xv_train = X_venue[idx_train].reshape(-1, 1)
Xv_test = X_venue[idx_test].reshape(-1, 1)
Xbat_train = X_bat[idx_train].reshape(-1, 1)
Xbat_test = X_bat[idx_test].reshape(-1, 1)
Xbowl_train = X_bowl[idx_train].reshape(-1, 1)
Xbowl_test = X_bowl[idx_test].reshape(-1, 1)
Xnum_train, Xnum_test = X_num[idx_train], X_num[idx_test]
y_train, y_test = y_remaining[idx_train], y_remaining[idx_test]
y_total_test = y_total[idx_test]
sw_train, sw_test = sample_weights[idx_train], sample_weights[idx_test]

# ------------ Model: embeddings + numeric branch ------------
n_venue = int(len(encoders["venue"].classes_))
n_bat = int(len(encoders["batting_team"].classes_))
n_bowl = int(len(encoders["bowling_team"].classes_))

emb_venue = min(40, max(12, n_venue // 3))
emb_bat = min(28, max(10, n_bat // 2))
emb_bowl = min(28, max(10, n_bowl // 2))

inp_v = tf.keras.Input(shape=(1,), dtype="int32", name="venue_id")
inp_bat = tf.keras.Input(shape=(1,), dtype="int32", name="batting_team_id")
inp_bowl = tf.keras.Input(shape=(1,), dtype="int32", name="bowling_team_id")
inp_num = tf.keras.Input(shape=(len(NUMERIC_COLS),), dtype="float32", name="numeric")

ev = tf.keras.layers.Embedding(n_venue, emb_venue, name="emb_venue")(inp_v)
ev = tf.keras.layers.Flatten()(ev)
ebat = tf.keras.layers.Embedding(n_bat, emb_bat, name="emb_bat")(inp_bat)
ebat = tf.keras.layers.Flatten()(ebat)
ebowl = tf.keras.layers.Embedding(n_bowl, emb_bowl, name="emb_bowl")(inp_bowl)
ebowl = tf.keras.layers.Flatten()(ebowl)

num_norm = tf.keras.layers.Normalization(axis=-1, name="num_norm")(inp_num)

merged = tf.keras.layers.Concatenate(name="concat")([ev, ebat, ebowl, num_norm])
x = tf.keras.layers.Dense(128, activation="relu")(merged)
x = tf.keras.layers.Dropout(0.15)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.1)(x)
out = tf.keras.layers.Dense(1, activation="linear", name="remaining_runs")(x)

model = tf.keras.Model(
    inputs=[inp_v, inp_bat, inp_bowl, inp_num],
    outputs=out,
    name="cricket_score_predictor",
)

# Adapt normalization on training numerics only (after train split, before val_split inside fit)
model.get_layer("num_norm").adapt(Xnum_train)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
    loss=tf.keras.losses.Huber(delta=25.0),
    metrics=["mae"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1,
    ),
]

# ------------ Train (recency-weighted; target = remaining runs) ------------
model.fit(
    [Xv_train, Xbat_train, Xbowl_train, Xnum_train],
    y_train,
    sample_weight=sw_train,
    epochs=80,
    batch_size=256,
    validation_split=VAL_FRACTION,
    callbacks=callbacks,
    verbose=1,
)

# ------------ Evaluate on held-out matches ------------
pred_rem = model.predict(
    [Xv_test, Xbat_test, Xbowl_test, Xnum_test],
    verbose=0,
).reshape(-1)
current_test = Xnum_test[:, 0].astype(np.float32)
pred_final = pred_rem + current_test

abs_err_rem = np.abs(pred_rem - y_test)
abs_err_final = np.abs(pred_final - y_total_test)

mae_rem_w = float(np.average(abs_err_rem, weights=sw_test))
mae_final_w = float(np.average(abs_err_final, weights=sw_test))
mae_rem_u = float(np.mean(abs_err_rem))
mae_final_u = float(np.mean(abs_err_final))
rmse_final = float(np.sqrt(np.mean((pred_final - y_total_test) ** 2)))

# Naive baseline: linear projection -> current_score + run_rate * overs_remaining
#   Columns in X_num: [current_score, wickets_lost, overs_completed,
#                      runs_in_last_5_overs, overs_remaining, run_rate, wickets_remaining]
overs_rem_test = Xnum_test[:, 4].astype(np.float32)
run_rate_test = Xnum_test[:, 5].astype(np.float32)
baseline_final = current_test + run_rate_test * overs_rem_test
mae_baseline = float(np.mean(np.abs(baseline_final - y_total_test)))

# Percentage-style accuracy metrics computed on the same match-level held-out
# set, so they remain leakage-free. MAPE excludes any y<=0 just in case.
valid = y_total_test > 0
if int(valid.sum()) > 0:
    mape = float(
        np.mean(np.abs(pred_final[valid] - y_total_test[valid]) / y_total_test[valid])
    )
else:
    mape = float("nan")
accuracy_mape_pct = float((1.0 - mape) * 100.0)

ss_res = float(np.sum((pred_final - y_total_test) ** 2))
ss_tot = float(np.sum((y_total_test - np.mean(y_total_test)) ** 2))
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
r2_pct = float(r2 * 100.0)

within_10_pct = float(np.mean(np.abs(pred_final - y_total_test) <= 10.0) * 100.0)
within_15_pct = float(np.mean(np.abs(pred_final - y_total_test) <= 15.0) * 100.0)

print("\n------ Test metrics (match-level held out) ------")
print(f"Rows: {len(idx_test)} | Matches: {n_test_matches}")
print(f"MAE remaining runs  (weighted): {mae_rem_w:.2f}")
print(f"MAE remaining runs  (uniform):  {mae_rem_u:.2f}")
print(f"MAE final score     (weighted): {mae_final_w:.2f}")
print(f"MAE final score     (uniform):  {mae_final_u:.2f}")
print(f"RMSE final score    (uniform):  {rmse_final:.2f}")
print(f"Baseline MAE (current + RR*overs_remaining): {mae_baseline:.2f}")
print(f"Model beats baseline by: {mae_baseline - mae_final_u:+.2f} runs MAE")
print(f"MAPE (final score):             {mape * 100:.2f}%")
print(f"Accuracy (1 - MAPE):            {accuracy_mape_pct:.2f}%")
print(f"R^2 (final score):              {r2_pct:.2f}%")
print(f"Within +/- 10 runs:             {within_10_pct:.2f}%")
print(f"Within +/- 15 runs:             {within_15_pct:.2f}%")

# ------------ Save model and encoders ------------
model.save(MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")

bundle = {
    "encoders": encoders,
    "meta": {
        "model": META_KEY_MODEL,
        "numeric_cols": list(NUMERIC_COLS),
        "target": "remaining_runs",
        "t20_max_overs": float(T20_MAX_OVERS),
        "max_wickets": float(MAX_WICKETS),
        "test_mae_final_uniform": mae_final_u,
        "test_rmse_final_uniform": rmse_final,
        "baseline_mae_final_uniform": mae_baseline,
        "test_mape": mape,
        "test_accuracy_mape_pct": accuracy_mape_pct,
        "test_r2_pct": r2_pct,
        "test_within_10_pct": within_10_pct,
        "test_within_15_pct": within_15_pct,
        "n_train_matches": int(n_train_matches),
        "n_test_matches": int(n_test_matches),
        "used_legacy_ipl": bool(use_legacy_ipl),
    },
}
with open(ENCODERS_PATH, "wb") as f:
    pickle.dump(bundle, f)
print(f"Saved label encoders + meta to {ENCODERS_PATH}")