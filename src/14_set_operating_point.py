"""
14_set_operating_point.py
--------------------------
Dat nguong cho UNG DUNG MINH HOA theo mot muc FAR muc tieu, thay cho nguong
toi uu accuracy (theo gop y GVHD: dung nguong theo FAR muc tieu chon hoan
toan tren validation lam operating point).

Doc nguong da chon tren validation trong file
results/tables/threshold_analysis_target_far.csv (do 10_threshold_analysis.py
tao ra), roi ghi vao model_artifacts/decision_config.json. Backend tu doc file
nay khi khoi dong.

Cach dung:
    cd src
    python 14_set_operating_point.py                       # mo hinh ket hop, FAR 10%
    python 14_set_operating_point.py --model combined --target_far 0.05

Sau do KHOI DONG LAI backend.
"""
import argparse, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="combined", choices=["rf", "svm", "siamese", "combined"])
ap.add_argument("--target_far", type=float, default=0.10)
a = ap.parse_args()

df = pd.read_csv(ROOT / "results" / "tables" / "threshold_analysis_target_far.csv")
row = df[(df["model"] == a.model) & ((df["target_FAR"] - a.target_far).abs() < 1e-9)]
if row.empty:
    raise SystemExit(f"Khong co dong {a.model} voi FAR {a.target_far}. Cac muc co san: {sorted(df.target_FAR.unique())}")
r = row.iloc[0]
import math
if not math.isfinite(float(r["threshold"])):
    raise SystemExit(f"Tren validation khong co nguong huu han nao giu FAR <= {a.target_far:.0%} cho mo hinh {a.model}. "
                     "Hay chon muc FAR cao hon, vi du --target_far 0.15")
cfg = {
    "model": a.model,
    "threshold": float(r["threshold"]),
    "selection": f"target_far_{a.target_far:.2f}_on_validation",
    "target_FAR": a.target_far,
    "FAR_val": float(r["FAR_val_dat_duoc"]),
    "accuracy_test": float(r["accuracy_test"]),
    "FAR_test": float(r["FAR_test"]),
    "FRR_test": float(r["FRR_test"]),
}
out = ROOT / "model_artifacts" / "decision_config.json"
out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(json.dumps(cfg, indent=2))
print(f"Da ghi: {out}\nKHOI DONG LAI backend de ap dung.")