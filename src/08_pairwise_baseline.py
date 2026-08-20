import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from utils_metrics import evaluate_predictions, find_threshold_by_eer

PROC_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")
RANDOM_SEED = 42
ALPHA = 0.05

rng = np.random.default_rng(RANDOM_SEED)


def load_data():
    features = pd.read_csv(PROC_DIR / "features.csv")
    with open(PROC_DIR / "writer_split.json", encoding="utf-8") as f:
        split = json.load(f)
    return features, split


def build_pairs_for_training(df_writers, feature_cols):
    """Tạo pairs CHO TRAIN/VALIDATION — tận dụng nhiều tổ hợp để có đủ dữ liệu học.
    positive: genuine-genuine cùng writer
    negative_skilled: genuine-skilled_forgery cùng writer (negative khó, ưu tiên)
    negative_random: genuine-genuine khác writer (negative dễ, bổ sung)
    """
    rows = []
    writer_ids = df_writers["writer_id"].unique()

    for wid in writer_ids:
        sub = df_writers[df_writers["writer_id"] == wid]
        genuine = sub[sub["label"] == "genuine"]
        forged = sub[sub["label"] == "skilled_forgery"]

        # positive: mọi cặp genuine-genuine cùng writer
        for i, j in combinations(genuine.index, 2):
            diff = np.abs(df_writers.loc[i, feature_cols].values - df_writers.loc[j, feature_cols].values)
            rows.append({"diff": diff, "y": 1, "pair_type": "positive", "writer_id": wid})

        # negative_skilled: genuine-forgery cùng writer (ưu tiên, trọng số cao hơn)
        for i in genuine.index:
            for j in forged.index:
                diff = np.abs(df_writers.loc[i, feature_cols].values - df_writers.loc[j, feature_cols].values)
                rows.append({"diff": diff, "y": 0, "pair_type": "negative_skilled", "writer_id": wid})

    # negative_random: genuine-genuine khác writer, lấy mẫu (không cần toàn bộ, tránh mất cân bằng quá mức)
    n_random_needed = sum(1 for r in rows if r["pair_type"] == "negative_skilled")
    genuine_all = df_writers[df_writers["label"] == "genuine"]
    for _ in range(n_random_needed):
        w1, w2 = rng.choice(writer_ids, size=2, replace=False)
        s1 = genuine_all[genuine_all["writer_id"] == w1].sample(1, random_state=None).index[0]
        s2 = genuine_all[genuine_all["writer_id"] == w2].sample(1, random_state=None).index[0]
        diff = np.abs(df_writers.loc[s1, feature_cols].values - df_writers.loc[s2, feature_cols].values)
        rows.append({"diff": diff, "y": 1 if False else 0, "pair_type": "negative_random", "writer_id": f"{w1}-{w2}"})

    return rows


def build_pairs_for_test(df_writers, feature_cols):
    """Tạo pairs CHO TEST — mô phỏng đúng kịch bản thực tế:
    mỗi writer có đúng 1 chữ ký reference (enrollment), các chữ ký còn lại
    (genuine + skilled_forgery) là query, chỉ so với reference của ĐÚNG writer đó."""
    rows = []
    writer_ids = df_writers["writer_id"].unique()

    for wid in writer_ids:
        sub = df_writers[df_writers["writer_id"] == wid]
        genuine = sub[sub["label"] == "genuine"]
        forged = sub[sub["label"] == "skilled_forgery"]

        if len(genuine) < 2:
            continue  # cần ít nhất 1 reference + 1 query genuine

        ref_idx = genuine.sample(1, random_state=RANDOM_SEED).index[0]
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

    # Loại bỏ các cặp có NaN/Inf trong đặc trưng (thường do GLCM correlation
    # bị chia cho 0 trên vùng ảnh gần như đồng nhất một màu)
    valid_mask = np.all(np.isfinite(X), axis=1)
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"  [Cảnh báo] Loại {n_dropped} cặp có giá trị NaN/Inf trong đặc trưng")
    return X[valid_mask], y[valid_mask], pair_type[valid_mask], writer_id[valid_mask]


def select_diff_features_train_only(X_train, y_train, feature_cols, alpha=ALPHA):
    """Feature selection CHỈ trên train pairs (không đụng val/test)."""
    selected_idx = []
    match_diff = X_train[y_train == 1]
    nonmatch_diff = X_train[y_train == 0]
    for idx, col in enumerate(feature_cols):
        _, p = mannwhitneyu(match_diff[:, idx], nonmatch_diff[:, idx], alternative="two-sided")
        if p < alpha:
            selected_idx.append(idx)
    return selected_idx if selected_idx else list(range(len(feature_cols)))


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    features, split = load_data()
    feature_cols = [c for c in features.columns if c not in ("filename", "writer_id", "label")]

    train_df = features[features["writer_id"].isin(split["train_writers"])].reset_index(drop=True)
    val_df = features[features["writer_id"].isin(split["val_writers"])].reset_index(drop=True)
    test_df = features[features["writer_id"].isin(split["test_writers"])].reset_index(drop=True)

    print(f"Ảnh: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # ---- 1. Tạo pairs cho train/val (nhiều tổ hợp để đủ dữ liệu học) ----
    train_rows = build_pairs_for_training(train_df, feature_cols)
    val_rows = build_pairs_for_training(val_df, feature_cols)
    X_train, y_train, _, _ = rows_to_arrays(train_rows)
    X_val, y_val, val_pair_type, val_writer = rows_to_arrays(val_rows)
    print(f"Pairs: train={len(y_train)}, val={len(y_val)}")

    # ---- 2. Feature selection CHỈ trên train pairs ----
    selected_idx = select_diff_features_train_only(X_train, y_train, feature_cols)
    selected_names = [feature_cols[i] for i in selected_idx]
    print(f"Chọn {len(selected_idx)}/{len(feature_cols)} đặc trưng diff: {selected_names}")

    X_train_sel = X_train[:, selected_idx]
    X_val_sel = X_val[:, selected_idx]

    # ---- 3. Chuẩn hóa fit trên train ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_val_scaled = scaler.transform(X_val_sel)

    # ---- 4. Thử vài model/hyperparameter, CHỌN dựa trên validation (không đụng test) ----
    candidates = {
        "RF_200": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED),
        "RF_400": RandomForestClassifier(n_estimators=400, max_depth=10, class_weight="balanced", random_state=RANDOM_SEED),
        "SVM_rbf": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_SEED),
    }

    best_name, best_model, best_auc = None, None, -1
    for name, clf in candidates.items():
        clf.fit(X_train_scaled, y_train)
        val_scores = clf.predict_proba(X_val_scaled)[:, 1]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_val, val_scores)
        print(f"  [{name}] validation ROC-AUC = {auc:.4f}")
        if auc > best_auc:
            best_name, best_model, best_auc = name, clf, auc

    print(f"\n>>> Chọn model: {best_name} (val ROC-AUC = {best_auc:.4f})")

    # ---- 5. Chọn threshold bằng EER trên VALIDATION (không đụng test) ----
    val_scores = best_model.predict_proba(X_val_scaled)[:, 1]
    threshold, eer_val = find_threshold_by_eer(y_val, val_scores)
    print(f">>> Threshold chọn qua EER trên validation: {threshold:.4f} (EER={eer_val:.4f})")

    # ==== TỪ ĐÂY TRỞ ĐI: TEST SET, CHỈ ĐỤNG 1 LẦN DUY NHẤT ====
    test_rows = build_pairs_for_test(test_df, feature_cols)
    X_test, y_test, test_pair_type, test_writer = rows_to_arrays(test_rows)
    print(f"\nPairs test (reference-query, KHÓA, chỉ đánh giá 1 lần): {len(y_test)}")

    X_test_sel = X_test[:, selected_idx]
    X_test_scaled = scaler.transform(X_test_sel)

    test_scores = best_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_test = (test_scores >= threshold).astype(int)

    result, per_writer_df = evaluate_predictions(
        y_test, y_pred_test, test_scores, pair_type=test_pair_type, writer_id=test_writer
    )

    print("\n=== KẾT QUẢ CUỐI CÙNG TRÊN TEST SET (Pairwise Classical ML Baseline) ===")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    pd.DataFrame([result]).to_csv(RESULTS_DIR / "pairwise_baseline_test_results.csv", index=False)
    per_writer_df.to_csv(RESULTS_DIR / "pairwise_baseline_per_writer.csv", index=False)

    with open(RESULTS_DIR / "pairwise_baseline_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_model": best_name,
            "selected_features": selected_names,
            "threshold": float(threshold),
            "val_EER": float(eer_val),
        }, f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu kết quả vào {RESULTS_DIR}/")
    print("\nBreakdown theo writer (test set):")
    print(per_writer_df.to_string(index=False))


if __name__ == "__main__":
    main()