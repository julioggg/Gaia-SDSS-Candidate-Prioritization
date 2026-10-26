from pathlib import Path
import argparse
import json
import yaml
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

parser = argparse.ArgumentParser()
parser.add_argument("--catalog", choices=["gaia", "sdss"], required=True)
parser.add_argument("--config", required=True)
args = parser.parse_args()

catalog = args.catalog
with open(args.config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

top_k = int(cfg["optimization"]["top_k"])
seed = int(cfg["optimization"]["random_seed"])
min_eff = float(cfg["optimization"]["min_effective_features"])

in_file = Path(cfg["outputs"]["processed_csv"])
meta_file = Path(f"data/processed/{catalog}_feature_columns.json")
out_dir = Path(f"outputs/baseline_reference/{catalog}/metrics")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(in_file)
with open(meta_file, "r", encoding="utf-8") as f:
    feature_cols = json.load(f)

X = df[feature_cols].to_numpy(dtype=float)
X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
X = np.clip(X, 1e-6, 1.0)
n_features = X.shape[1]

def normalize_weights(w):
    w = np.clip(np.array(w), 0, None)
    s = w.sum()
    if s == 0:
        return np.ones_like(w) / len(w)
    return w / s

def weighted_geometric_score(X, w):
    return np.exp(np.sum(w * np.log(X), axis=1))

def entropy_of_weights(w):
    eps = 1e-12
    h = -np.sum(w * np.log(w + eps))
    h_max = np.log(len(w))
    return h / h_max if h_max > 0 else 0.0

def effective_number_of_features(w):
    return 1.0 / np.sum(w**2)

def objective(w_raw):
    w = normalize_weights(w_raw)
    score = weighted_geometric_score(X, w)
    k = min(top_k, len(score))
    idx = np.argpartition(score, -k)[-k:]
    top_X = X[idx]
    top_scores = score[idx]
    mean_score = np.mean(top_scores)
    diversity_top = np.mean(np.std(top_X, axis=0))
    entropy = entropy_of_weights(w)
    eff_feats = effective_number_of_features(w)
    penalty_concentration = max(0.0, min_eff - eff_feats) / min_eff

    value = (
        0.55 * mean_score +
        0.20 * diversity_top +
        0.20 * entropy -
        0.25 * penalty_concentration
    )
    return -value

bounds = [(0, 1)] * n_features
print(f"Optimizando catalogo: {catalog}")
print(f"Filas disponibles: {len(df)}")
print(f"Top K: {top_k}")

result = differential_evolution(
    objective,
    bounds=bounds,
    seed=seed,
    maxiter=120,
    popsize=15,
    tol=1e-6,
    polish=True,
    workers=1
)

best_w = normalize_weights(result.x)
df["score"] = weighted_geometric_score(X, best_w)
df_sorted = df.sort_values("score", ascending=False).copy()
top = df_sorted.head(top_k)

ranked_file = out_dir / f"ranked_{catalog}.csv"
top_file = out_dir / f"top_{catalog}.csv"
weights_file = out_dir / f"weights_{catalog}.csv"

df_sorted.to_csv(ranked_file, index=False)
top.to_csv(top_file, index=False)

weights = pd.DataFrame({"feature": feature_cols, "weight": best_w}).sort_values("weight", ascending=False)
weights.to_csv(weights_file, index=False)

print("Optimizacion lista.")
print(f"Mejor valor objetivo: {-result.fun}")
print(f"Numero efectivo de variables: {effective_number_of_features(best_w)}")
print(f"Entropia de pesos: {entropy_of_weights(best_w)}")
print(weights.to_string(index=False))
