import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

PROC_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")
N_FOLDS = 10
ALPHA = 0.05  # ngưỡng ý nghĩa thống kê cho feature selection


def select_features_train_only(X_train, y_train, feature_cols, alpha=ALPHA):
    """Feature selection CHỈ dùng dữ liệu training fold — đây là phần cốt lõi
    khắc phục góp ý của thầy."""
    genuine_mask = y_train == "genuine"
    forged_mask = y_train == "skilled_forgery"
    selected = []
    for col in feature_cols:
        _, p = mannwhitneyu(
            X_train.loc[genuine_mask, col],
            X_train.loc[forged_mask, col],
            alternative="two-sided",
        )
        if p < alpha:
            selected.append(col)
    # fallback: nếu không đặc trưng nào đạt ngưỡng, giữ lại toàn bộ để tránh lỗi
    return selected if selected else list(feature_cols)


def run_cv(df, feature_cols, model_name="svm"):
    groups = df["writer_id"].values
    y = df["label"].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, y, groups)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # --- Feature selection CHỈ trên training fold ---
        selected_features = select_features_train_only(
            train_df[feature_cols], train_df["label"].values, feature_cols
        )

        X_train_raw = train_df[selected_features].values
        X_test_raw = test_df[selected_features].values
        y_train = train_df["label"].values
        y_test = test_df["label"].values

        # --- Chuẩn hóa fit CHỈ trên training fold ---
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        # --- Model ---
        if model_name == "svm":
            clf = SVC(kernel="rbf", probability=True, class_weight="balanced")
        elif model_name == "rf":
            clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
        else:
            raise ValueError(model_name)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, list(clf.classes_).index("skilled_forgery")]

        y_test_bin = (y_test == "skilled_forgery").astype(int)
        y_pred_bin = (y_pred == "skilled_forgery").astype(int)

        fold_results.append({
            "fold": fold_idx,
            "n_selected_features": len(selected_features),
            "selected_features": ",".join(selected_features),
            "accuracy": accuracy_score(y_test_bin, y_pred_bin),
            "precision": precision_score(y_test_bin, y_pred_bin, zero_division=0),
            "recall": recall_score(y_test_bin, y_pred_bin, zero_division=0),
            "f1": f1_score(y_test_bin, y_pred_bin, zero_division=0),
            "roc_auc": roc_auc_score(y_test_bin, y_prob) if len(set(y_test_bin)) > 1 else float("nan"),
        })

    return pd.DataFrame(fold_results)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROC_DIR / "features.csv")
    feature_cols = [c for c in df.columns if c not in ("filename", "writer_id", "label")]

    for model_name in ["svm", "rf"]:
        print(f"\n=== Model: {model_name.upper()} — {N_FOLDS}-fold GroupKFold CV ===")
        results = run_cv(df, feature_cols, model_name=model_name)
        results.to_csv(RESULTS_DIR / f"cv_results_{model_name}.csv", index=False)
        print(results[["fold", "n_selected_features", "accuracy", "precision", "recall", "f1", "roc_auc"]])
        print("\n--- Trung bình qua các fold ---")
        print(results[["accuracy", "precision", "recall", "f1", "roc_auc"]].mean())


if __name__ == "__main__":
    main()