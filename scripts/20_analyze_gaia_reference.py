from pathlib import Path
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--catalog", choices=["gaia", "sdss"], required=True)
parser.add_argument("--config", required=True)
args = parser.parse_args()

catalog = args.catalog
with open(args.config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

feature_file = Path(cfg["outputs"]["processed_csv"])
metrics_dir = Path(f"outputs/baseline_reference/{catalog}/metrics")
fig_dir = Path(f"outputs/baseline_reference/{catalog}/figures")
metrics_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

ranked = pd.read_csv(metrics_dir / f"ranked_{catalog}.csv")
top = pd.read_csv(metrics_dir / f"top_{catalog}.csv")
weights = pd.read_csv(metrics_dir / f"weights_{catalog}.csv")
features = pd.read_csv(feature_file)

mean_total = ranked["score"].mean()
mean_top = top["score"].mean()
improvement = mean_top / mean_total if mean_total != 0 else float("nan")

w = weights["weight"].to_numpy(dtype=float)
effective_features = 1.0 / np.sum(w**2)
entropy = -np.sum(w * np.log(w + 1e-12)) / np.log(len(w))

summary = []
summary.append(f"Catalogo: {catalog.upper()}")
summary.append(f"Total objetos procesados: {len(features)}")
summary.append(f"Top seleccionados: {len(top)}")
summary.append(f"Media score total: {mean_total:.6f}")
summary.append(f"Media score top: {mean_top:.6f}")
summary.append(f"Mejora relativa top/total: {improvement:.3f}x")
summary.append(f"Numero efectivo de variables: {effective_features:.3f}")
summary.append(f"Entropia normalizada de pesos: {entropy:.3f}")
summary.append("")
summary.append("Pesos optimizados:")
summary.append(weights.to_string(index=False))

summary_text = "\n".join(summary)
(metrics_dir / f"summary_{catalog}.txt").write_text(summary_text, encoding="utf-8")
print(summary_text)

plt.figure(figsize=(8, 4))
plt.hist(ranked["score"], bins=40)
plt.xlabel("Score optimizado")
plt.ylabel("Frecuencia")
plt.title(f"Distribucion del score - {catalog.upper()}")
plt.savefig(fig_dir / f"score_distribution_{catalog}.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 4))
plt.bar(weights["feature"], weights["weight"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Peso optimizado")
plt.title(f"Pesos de la funcion objetivo - {catalog.upper()}")
plt.savefig(fig_dir / f"weights_{catalog}.png", dpi=200, bbox_inches="tight")
plt.close()
