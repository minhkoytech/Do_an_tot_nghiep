"""
Bước 8 — Pairwise Classical ML Baseline, chạy qua NHIỀU writer split

(Reference, Query) -> |features_ref - features_query| -> RF/SVM -> Match/Non-match

Khác bản trước: thay vì chạy 1 lần trên 1 cách chia writer, giờ LẶP QUA
n_splits cách chia khác nhau (đã tạo ở 07_writer_split.py), rồi báo cáo
kết quả dạng "trung bình ± độ lệch chuẩn" — đáng tin cậy hơn nhiều so với
1 lần chạy, vì CEDAR chỉ có 55 writer (dễ bị phương sai cao).

Có khả năng RESUME: nếu chạy dở rồi dừng, lần sau chạy lại sẽ bỏ qua các
split đã có kết quả, chỉ chạy tiếp phần còn thiếu.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

from utils_metrics import evaluate_predictions, find_threshold_by_eer, find_threshold_at_target_far

PROC_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables/pairwise_baseline")
ALPHA = 0.05
TARGET_FAR = 0.10


def build_pairs_for_training(df_writers, feature_cols, rng):
    rows = []
    writer_ids = df_writers["writer_id"].unique()

    for wid in writer_ids:
        sub = df_writers[df_writers["writer_id"] == wid]
        genuine = sub[sub["label"] == "genuine"]
        forged = sub[sub["label"] == "skilled_forgery"]

        for i, j in combinations(genuine.index, 2):
            diff = np.abs(df_writers.loc[i, feature_cols].values - df_writers.loc[j, feature_cols].values)
            rows.append({"diff": diff, "y": 1, "pair_type": "positive", "writer_id": wid})

        for i in genuine.index:
            for j in forged.index:
                diff = np.abs(df_writers.loc[i, feature_cols].values - df_writers.loc[j, feature_cols].values)
                rows.append({"diff": diff, "y": 0, "pair_type": "negative_skilled", "writer_id": wid})

    n_random_needed = sum(1 for r in rows if r["pair_type"] == "negative_skilled")
    genuine_all = df_writers[df_writers["label"] == "genuine"]
    for _ in range(n_random_needed):
        w1, w2 = rng.choice(writer_ids, size=2, replace=False)
        s1 = genuine_all[genuine_all["writer_id"] == w1].sample(1, random_state=int(rng.integers(1e9))).index[0]
        s2 = genuine_all[genuine_all["writer_id"] == w2].sample(1, random_state=int(rng.integers(1e9))).index[0]
        diff = np.abs(df_writers.loc[s1, feature_cols].values - df_writers.loc[s2, feature_cols].values)
        rows.append({"diff": diff, "y": 0, "pair_type": "negative_random", "writer_id": f"{w1}-{w2}"})

    return rows


def build_pairs_for_test(df_writers, feature_cols, seed):
    rows = []
    for wid in df_writers["writer_id"].unique():
        sub = df_writers[df_writers["writer_id"] == wid]
        genuine = sub[sub["label"] == "genuine"]
        forged = sub[sub["label"] == "skilled_forgery"]
        if len(genuine) < 2:
            continue

        ref_idx = genuine.sample(1, random_state=seed).index[0]
        query_genuine = genuine.drop(ref_idx)

        for qi in query_genuine.index:
            diff = np.abs(df_writers.loc[ref_idx, feature_cols].values - df_writers.loc[qi, feature_cols].values)
            rows.append({"diff": diff, "y": 1, "pair_type": "positive", "writer_id": wid})
        for qi in forged.index:
            diff = np.abs(df_writers.loc[ref_idx, feature_cols].values - df_writers.loc[qi, feature_cols].values)
            rows.append({"diff": diff, "y": 0, "pair_type": "negative_skilled", "writer_id": wid})

    return rows


def rows_to_arrays(rows):
    X = np.stack([np.asarray(r["diff"], dtype=np.float64) for r in rows])
    y = np.array([r["y"] for r in rows])
    pair_type = np.array([r["pair_type"] for r in rows])
    writer_id = np.array([r["writer_id"] for r in rows])
    valid_mask = np.all(np.isfinite(X), axis=1)
    return X[valid_mask], y[valid_mask], pair_type[valid_mask], writer_id[valid_mask]


def select_diff_features_train_only(X_train, y_train, feature_cols, alpha=ALPHA):
    selected_idx = []
    match_diff = X_train[y_train == 1]
    nonmatch_diff = X_train[y_train == 0]
    for idx in range(len(feature_cols)):
        _, p = mannwhitneyu(match_diff[:, idx], nonmatch_diff[:, idx], alternative="two-sided")
        if p < alpha:
            selected_idx.append(idx)
    return selected_idx if selected_idx else list(range(len(feature_cols)))


def run_one_split(split, features, feature_cols):
    seed = split["seed"]
    rng = np.random.default_rng(seed)

    train_df = features[features["writer_id"].isin(split["train_writers"])].reset_index(drop=True)
    val_df = features[features["writer_id"].isin(split["val_writers"])].reset_index(drop=True)
    test_df = features[features["writer_id"].isin(split["test_writers"])].reset_index(drop=True)

    train_rows = build_pairs_for_training(train_df, feature_cols, rng)
    val_rows = build_pairs_for_training(val_df, feature_cols, rng)
    X_train, y_train, _, _ = rows_to_arrays(train_rows)
    X_val, y_val, _, _ = rows_to_arrays(val_rows)

    selected_idx = select_diff_features_train_only(X_train, y_train, feature_cols)
    X_train_sel, X_val_sel = X_train[:, selected_idx], X_val[:, selected_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_val_scaled = scaler.transform(X_val_sel)

    candidates = {
        "RF_200": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed),
        "RF_400": RandomForestClassifier(n_estimators=400, max_depth=10, class_weight="balanced", random_state=seed),
        "SVM_rbf": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
    }
    best_name, best_model, best_auc = None, None, -1
    for name, clf in candidates.items():
        clf.fit(X_train_scaled, y_train)
        auc = roc_auc_score(y_val, clf.predict_proba(X_val_scaled)[:, 1])
        if auc > best_auc:
            best_name, best_model, best_auc = name, clf, auc

    val_scores = best_model.predict_proba(X_val_scaled)[:, 1]
    thr_eer, eer_val = find_threshold_by_eer(y_val, val_scores)
    thr_far, achieved_far = find_threshold_at_target_far(y_val, val_scores, TARGET_FAR)

    test_rows = build_pairs_for_test(test_df, feature_cols, seed)
    X_test, y_test, test_pair_type, test_writer = rows_to_arrays(test_rows)
    X_test_sel = X_test[:, selected_idx]
    X_test_scaled = scaler.transform(X_test_sel)
    test_scores = best_model.predict_proba(X_test_scaled)[:, 1]

    split_results = {}
    per_writer_all = []
    for thr_name, thr in [("EER_balanced", thr_eer), (f"FAR_target_{TARGET_FAR:.0%}", thr_far)]:
        y_pred = (test_scores >= thr).astype(int)
        result, per_writer_df = evaluate_predictions(
            y_test, y_pred, test_scores, pair_type=test_pair_type, writer_id=test_writer
        )
        result["threshold_name"] = thr_name
        result["best_model"] = best_name
        result["n_selected_features"] = len(selected_idx)
        split_results[thr_name] = result
        per_writer_df["threshold_name"] = thr_name
        per_writer_all.append(per_writer_df)

    return split_results, pd.concat(per_writer_all, ignore_index=True)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(PROC_DIR / "features.csv")
    feature_cols = [c for c in features.columns if c not in ("filename", "writer_id", "label")]

    with open(PROC_DIR / "writer_splits.json", encoding="utf-8") as f:
        splits_data = json.load(f)
    splits = splits_data["splits"]

    all_results = []
    all_per_writer = []

    for split in splits:
        sid = split["split_id"]
        result_path = RESULTS_DIR / f"split_{sid}_results.csv"

        if result_path.exists():
            print(f"[Split {sid}] Đã có kết quả từ trước, bỏ qua (resume).")
            df_existing = pd.read_csv(result_path)
            all_results.append(df_existing)
            pw_path = RESULTS_DIR / f"split_{sid}_per_writer.csv"
            if pw_path.exists():
                all_per_writer.append(pd.read_csv(pw_path))
            continue

        print(f"\n[Split {sid}] Đang chạy (train={len(split['train_writers'])}, "
              f"val={len(split['val_writers'])}, test={len(split['test_writers'])})...")
        split_results, per_writer_df = run_one_split(split, features, feature_cols)

        df_result = pd.DataFrame(split_results.values())
        df_result["split_id"] = sid
        df_result.to_csv(result_path, index=False)
        per_writer_df["split_id"] = sid
        per_writer_df.to_csv(RESULTS_DIR / f"split_{sid}_per_writer.csv", index=False)

        all_results.append(df_result)
        all_per_writer.append(per_writer_df)

        for thr_name, r in split_results.items():
            print(f"  [{thr_name}] Accuracy={r['accuracy']:.4f}  FAR={r['FAR']:.4f}  "
                  f"FRR={r['FRR']:.4f}  ROC_AUC={r['ROC_AUC']:.4f}")

    # ---- Tổng hợp qua tất cả split: trung bình ± độ lệch chuẩn ----
    combined = pd.concat(all_results, ignore_index=True)
    summary = combined.groupby("threshold_name")[["accuracy", "FAR", "FRR", "ROC_AUC"]].agg(["mean", "std"])
    summary.to_csv(RESULTS_DIR / "summary_mean_std.csv")

    print("\n" + "=" * 70)
    print(f"TỔNG HỢP QUA {len(splits)} SPLIT — Pairwise Classical ML Baseline")
    print("=" * 70)
    print(summary.to_string())

    combined.to_csv(RESULTS_DIR / "all_splits_results.csv", index=False)
    pd.concat(all_per_writer, ignore_index=True).to_csv(RESULTS_DIR / "all_splits_per_writer.csv", index=False)
    print(f"\nĐã lưu toàn bộ kết quả vào {RESULTS_DIR}/")


if __name__ == "__main__":
    main()