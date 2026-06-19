"""
train.py — Train Random Forest classifier (an toàn, không leakage)
==================================================================
Input  : data/processed/features_final.csv, data/processed/intervals.csv
Output : models/random_forest.pkl

Đồng bộ với config.py mới:
- Import FEATURE_COLS_STATIC thay vì tự định nghĩa.
- Dùng Paths.INTERVALS_FILE thay vì hardcode đường dẫn.
- Validation seed files từ config.
"""

import sys
import json
import logging
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RANDOM_STATE,
    RF_TEST_SIZE, RF_CV_FOLDS,
    Paths, LABEL_SLOP, LABEL_GENUINE, validate,
    IF_CONTAMINATION, IF_N_ESTIMATORS, IF_RANDOM_STATE,
    FEATURE_COLS_STATIC,           # <-- dùng chung danh sách feature tĩnh
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Feature tĩnh (import từ config, không hardcode) ─────────────────────────
STATIC_FEATURE_COLS = FEATURE_COLS_STATIC

# Các cột có thể mang giá trị -1 → sẽ chuyển thành NaN
MISSING_FLAG_COLS = ["time_interval_std", "video_upload_frequency", "subscriber_velocity"]


# ══════════════════════════════════════════════════════════════════════════════
# Custom transformers (giữ nguyên)
# ══════════════════════════════════════════════════════════════════════════════

class IntervalAnomalyTransformer(BaseEstimator, TransformerMixin):
    """
    Nhận DataFrame chứa cột 'intervals' (list các float).
    Fit Isolation Forest trên intervals của train fold.
    Transform: trả về DataFrame có thêm cột 'if_anomaly_score'.
    """
    def __init__(self, contamination=IF_CONTAMINATION, n_estimators=IF_N_ESTIMATORS,
                 random_state=IF_RANDOM_STATE):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.if_model = None

    def fit(self, X: pd.DataFrame, y=None):
        all_intervals = []
        for intervals_list in X["intervals"]:
            if isinstance(intervals_list, str):
                intervals_list = json.loads(intervals_list)
            all_intervals.extend(intervals_list)
        all_intervals = np.array(all_intervals).reshape(-1, 1)
        self.if_model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.if_model.fit(all_intervals)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.if_model is None:
            raise RuntimeError("Transformer chưa được fit.")
        scores = []
        for intervals_list in X["intervals"]:
            if isinstance(intervals_list, str):
                intervals_list = json.loads(intervals_list)
            if len(intervals_list) == 0:
                scores.append(0.5)
            else:
                arr = np.array(intervals_list).reshape(-1, 1)
                raw = self.if_model.decision_function(arr)
                anomaly = 1.0 - (raw + 0.5)  # quy đổi thành score càng cao càng bất thường
                scores.append(float(np.mean(anomaly)))
        X = X.copy()
        X["if_anomaly_score"] = scores
        return X


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Chọn các cột từ DataFrame và trả về numpy array."""
    def __init__(self, columns: list):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.columns].values


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load & chuẩn bị dữ liệu
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, list[str]]:
    """
    Load features_final.csv và intervals.csv, merge, xử lý sơ bộ.
    Trả về DataFrame đầy đủ (gồm label, intervals) và danh sách feature cuối cùng.
    """
    # Đọc features
    df_feat = pd.read_csv(Paths.FEATURES_FINAL)
    log.info(f"Loaded {len(df_feat)} samples | Slop: {(df_feat['label']==1).sum()} | Genuine: {(df_feat['label']==0).sum()}")

    # Đọc intervals (dùng Paths.INTERVALS_FILE thay vì hardcode)
    intervals_path = Paths.INTERVALS_FILE
    if not intervals_path.exists():
        log.error(f"Không tìm thấy {intervals_path}. Hãy chạy features.py mới nhất.")
        sys.exit(1)
    df_int = pd.read_csv(intervals_path)
    df = df_feat.merge(df_int, on="channel_id", how="left")

    # Chuyển intervals từ JSON string thành list
    def parse_intervals(val):
        if isinstance(val, str):
            return json.loads(val)
        if isinstance(val, list):
            return val
        return []
    df["intervals"] = df["intervals_json"].apply(parse_intervals)
    df.drop(columns=["intervals_json"], inplace=True)

    # Bỏ cột if_anomaly_score cũ nếu có
    if "if_anomaly_score" in df.columns:
        log.warning("Cột if_anomaly_score cũ trong CSV - sẽ bị bỏ qua.")
        df.drop(columns=["if_anomaly_score"], inplace=True)

    # Xác định các cột feature tĩnh tồn tại
    available_static = [c for c in STATIC_FEATURE_COLS if c in df.columns]
    missing_static = set(STATIC_FEATURE_COLS) - set(available_static)
    if missing_static:
        log.error(f"Thiếu các cột feature: {missing_static}. Chạy lại features.py.")
        sys.exit(1)

    # Xử lý sub_to_view_velocity_ratio nếu hầu hết genuine là NaN
    nan_gen = df[df["label"] == LABEL_GENUINE]["sub_to_view_velocity_ratio"].isna().sum()
    total_gen = (df["label"] == LABEL_GENUINE).sum()
    nan_slop = df[df["label"] == LABEL_SLOP]["sub_to_view_velocity_ratio"].isna().sum()
    total_slop = (df["label"] == LABEL_SLOP).sum()

    if total_gen > 0 and nan_gen / total_gen > 0.9:
        log.warning(f"sub_to_view_velocity_ratio có {nan_gen}/{total_gen} NaN ở genuine → loại bỏ feature này.")
        available_static.remove("sub_to_view_velocity_ratio")
    else:
        log.info(f"sub_to_view_velocity_ratio NaN: genuine {nan_gen}/{total_gen}, slop {nan_slop}/{total_slop}")

    # Danh sách feature cuối cùng = static + 'if_anomaly_score' (sẽ sinh ra trong pipeline)
    final_feature_cols = available_static + ["if_anomaly_score"]

    # Chuyển giá trị -1 thành NaN để pipeline impute
    for col in MISSING_FLAG_COLS:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)

    log.info(f"Feature matrix sẽ có {len(final_feature_cols)} features: {final_feature_cols}")
    return df, final_feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# 2. Xây dựng pipeline
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline(feature_cols: list[str]) -> Pipeline:
    """
    Tạo pipeline:
    1. IntervalAnomalyTransformer → thêm cột if_anomaly_score.
    2. DataFrameSelector → chọn các cột feature_cols (gồm cả if_anomaly_score).
    3. SimpleImputer → thay NaN bằng median.
    4. StandardScaler.
    5. RandomForestClassifier.
    """
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight="balanced",
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("add_anomaly", IntervalAnomalyTransformer()),
        ("select_features", DataFrameSelector(feature_cols)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", rf),
    ])
    return pipeline


# ══════════════════════════════════════════════════════════════════════════════
# 3. Train
# ══════════════════════════════════════════════════════════════════════════════

def train(df: pd.DataFrame, feature_cols: list[str]) -> tuple[Pipeline, dict]:
    """
    Train/test split, cross-validation, đánh giá.
    Trả về pipeline đã fit và dict các metrics.
    """
    # Chỉ lấy các cột có sẵn (static features + intervals)
    input_cols = [c for c in feature_cols if c in df.columns] + ["intervals"]
    X = df[input_cols].copy()
    y = df["label"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=RF_TEST_SIZE,
        random_state=RF_RANDOM_STATE,
        stratify=y,
    )
    log.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    log.info(f"Train label distribution — Slop: {(y_train==1).sum()} | Genuine: {(y_train==0).sum()}")

    pipeline = build_pipeline(feature_cols)

    # Cross-validation
    log.info(f"Chạy {RF_CV_FOLDS}-fold cross-validation (có sinh anomaly score trong mỗi fold)...")
    cv = StratifiedKFold(n_splits=RF_CV_FOLDS, shuffle=True, random_state=RF_RANDOM_STATE)
    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        return_train_score=False,
        error_score='raise',
    )

    cv_metrics = {
        "cv_accuracy":  cv_results["test_accuracy"].mean(),
        "cv_precision": cv_results["test_precision"].mean(),
        "cv_recall":    cv_results["test_recall"].mean(),
        "cv_f1":        cv_results["test_f1"].mean(),
        "cv_roc_auc":   cv_results["test_roc_auc"].mean(),
        "cv_f1_std":    cv_results["test_f1"].std(),
        "cv_auc_std":   cv_results["test_roc_auc"].std(),
    }

    log.info(f"CV F1: {cv_metrics['cv_f1']:.4f} (±{cv_metrics['cv_f1_std']:.4f})")
    log.info(f"CV ROC-AUC: {cv_metrics['cv_roc_auc']:.4f} (±{cv_metrics['cv_auc_std']:.4f})")

    # Train lại trên toàn bộ train set
    pipeline.fit(X_train, y_train)

    # Đánh giá test set
    y_pred = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    test_metrics = {
        "test_accuracy":  accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall":    recall_score(y_test, y_pred),
        "test_f1":        f1_score(y_test, y_pred),
        "test_roc_auc":   roc_auc_score(y_test, y_pred_prob),
    }

    log.info(f"Test F1: {test_metrics['test_f1']:.4f} | Test ROC-AUC: {test_metrics['test_roc_auc']:.4f}")

    metrics = {**cv_metrics, **test_metrics,
               "y_test": y_test, "y_pred": y_pred,
               "X_train": X_train, "y_train": y_train}
    return pipeline, metrics


# ══════════════════════════════════════════════════════════════════════════════
# 4. Report (giữ nguyên)
# ══════════════════════════════════════════════════════════════════════════════

def print_report(pipeline: Pipeline, metrics: dict, feature_cols: list[str]) -> None:
    print()
    print("=" * 70)
    print(f"KẾT QUẢ TRAINING — ICF DETECTOR ({len(feature_cols)} features)")
    print("=" * 70)

    print(f"\n── {RF_CV_FOLDS}-Fold Cross-Validation (trên train set) ──")
    print(f"  Accuracy  : {metrics['cv_accuracy']:.4f}")
    print(f"  Precision : {metrics['cv_precision']:.4f}")
    print(f"  Recall    : {metrics['cv_recall']:.4f}")
    print(f"  F1-score  : {metrics['cv_f1']:.4f}  (±{metrics['cv_f1_std']:.4f})")
    print(f"  ROC-AUC   : {metrics['cv_roc_auc']:.4f}  (±{metrics['cv_auc_std']:.4f})")

    print(f"\n── Test Set ({int(RF_TEST_SIZE*100)}%) ──")
    print(f"  Accuracy  : {metrics['test_accuracy']:.4f}")
    print(f"  Precision : {metrics['test_precision']:.4f}")
    print(f"  Recall    : {metrics['test_recall']:.4f}")
    print(f"  F1-score  : {metrics['test_f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['test_roc_auc']:.4f}")

    print(f"\n── Confusion Matrix ──")
    cm = confusion_matrix(metrics["y_test"], metrics["y_pred"])
    tn, fp, fn, tp = cm.ravel()
    print(f"  {'':20} Predicted Genuine  Predicted Slop")
    print(f"  {'Actual Genuine':20} {tn:^17} {fp:^14}")
    print(f"  {'Actual Slop':20} {fn:^17} {tp:^14}")
    print(f"\n  ✓ True Negatives  (TN): {tn:2d}  — Genuine được nhận diện đúng")
    print(f"  ✓ True Positives  (TP): {tp:2d}  — Slop bị bắt đúng")
    print(f"  ✗ False Positives (FP): {fp:2d}  — Genuine bị nhầm thành Slop (false alarm)")
    print(f"  ✗ False Negatives (FN): {fn:2d}  — Slop bị sót (miss)")

    print(f"\n── Classification Report ──")
    print(classification_report(
        metrics["y_test"], metrics["y_pred"],
        target_names=["Genuine (0)", "Slop (1)"]
    ))

    # Feature importance
    rf_model = pipeline.named_steps["clf"]
    final_feature_names = pipeline.named_steps["select_features"].columns
    importances = rf_model.feature_importances_

    print(f"\n── Feature Importance (Random Forest) ──")
    print(f"  {'Feature':<30} {'Importance':>10}  {'Bar'}")
    print(f"  {'─'*30} {'─'*10}  {'─'*40}")

    fi_pairs = sorted(zip(final_feature_names, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in fi_pairs:
        bar = "█" * int(imp * 50)
        print(f"  {feat:<30} {imp:>10.4f}  {bar}")

    print(f"\n  🔥 Top 3 features:")
    for i, (feat, imp) in enumerate(fi_pairs[:3], 1):
        print(f"     {i}. {feat} ({imp:.4f})")

    # Cảnh báo
    if metrics["test_f1"] > 0.98:
        print("\n⚠️  F1 > 0.98 — Kiểm tra overfitting!")
    if metrics["cv_f1_std"] > 0.1:
        print(f"\n⚠️  CV F1 std={metrics['cv_f1_std']:.3f} cao — Model không ổn định.")
    f1_gap = abs(metrics["cv_f1"] - metrics["test_f1"])
    if f1_gap > 0.1:
        print(f"\n⚠️  Gap CV-Test F1={f1_gap:.3f} lớn — Có thể overfitting hoặc test set không đại diện.")

    print(f"\n💡 Model hiện tại ({len(final_feature_names)} features) gồm các nhóm:")
    print("   📈 Nhóm chuỗi thời gian & vận tốc đăng bài:")
    print("       time_interval_std, upload_burst_ratio, video_upload_frequency, view_per_video, if_anomaly_score")
    print("   📝 Nhóm dấu vết cấu trúc & định dạng văn bản AI:")
    print("       dash_density, title_length_std, capitalization_ratio, opening_repeat_ratio, temporal_clickbait_ratio")
    print("   🔄 Nhóm độ đa dạng & tương đồng nội dung:")
    print("       type_token_ratio, avg_title_similarity")
    print("   💰 Nhóm chỉ số tài chính & gian lận tương tác:")
    print("       sub_to_view_ratio, subscriber_velocity, sub_to_view_velocity_ratio (nếu có)")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Save
# ══════════════════════════════════════════════════════════════════════════════

def save_model(pipeline: Pipeline, feature_cols: list[str]) -> None:
    Paths.MODELS.mkdir(parents=True, exist_ok=True)
    model_data = {
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
    }
    with open(Paths.MODEL_RF, "wb") as f:
        pickle.dump(model_data, f)
    log.info(f"Model saved → {Paths.MODEL_RF}")
    log.info(f"  Features ({len(feature_cols)}): {', '.join(feature_cols)}")
    log.info(f"  Pipeline: IntervalAnomaly → Select → Imputer → Scaler → RandomForest")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train ICF detector model (an toàn leakage)")
    parser.add_argument("--no-save", action="store_true", help="Không lưu model")
    parser.add_argument("--drop-burst", action="store_true",
                        help="Loại bỏ feature upload_burst_ratio (ít quan trọng)")
    args = parser.parse_args()

    validate()  # kiểm tra API key, seed files, tạo thư mục...

    if not Paths.FEATURES_FINAL.exists():
        log.error(f"Không tìm thấy {Paths.FEATURES_FINAL}. Chạy features.py trước.")
        sys.exit(1)

    log.info("=" * 70)
    log.info("BẮT ĐẦU TRAINING (pipeline an toàn)")
    log.info("=" * 70)

    df, final_feature_cols = load_data()

    if args.drop_burst and "upload_burst_ratio" in final_feature_cols:
        log.info("Loại bỏ upload_burst_ratio theo yêu cầu --drop-burst.")
        final_feature_cols = [c for c in final_feature_cols if c != "upload_burst_ratio"]

    pipeline, metrics = train(df, final_feature_cols)
    print_report(pipeline, metrics, final_feature_cols)

    if not args.no_save:
        save_model(pipeline, final_feature_cols)
        print(f"\n✅ Model đã được lưu tại: {Paths.MODEL_RF}")
    else:
        log.info("\n--no-save: Model không được lưu.")


if __name__ == "__main__":
    main()