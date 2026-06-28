"""
compare_models.py

So sánh nhiều mô hình Machine Learning
trên dataset features_final.csv
"""



import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# ===========================
# Load dataset
# ===========================

def load_data():

    print("=" * 50)
    print("Loading dataset...")

    df = pd.read_csv("data/processed/features_final.csv")

    print(f"Dataset shape : {df.shape}")
    print(f"Total samples : {len(df)}")

    return df

def build_pipeline(model):

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])

    return pipeline

def evaluate_model(name, model, X_train, X_test, y_train, y_test):


    pipeline = build_pipeline(model)
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    result = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "CV F1 Mean": cv_scores.mean(),
        "CV F1 Std": cv_scores.std(),
    }

    print(f"Accuracy : {result['Accuracy']:.4f}")
    print(f"Precision: {result['Precision']:.4f}")
    print(f"Recall   : {result['Recall']:.4f}")
    print(f"F1-score : {result['F1']:.4f}")
    print(f"CV F1 Mean : {result['CV F1 Mean']:.4f}")
    print(f"CV F1 Std  : {result['CV F1 Std']:.4f}")

    return result

def plot_results(results_df):

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        results_df["Model"],
        results_df["F1"],
    )

    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            fontsize=9,
        )

    plt.title("Model Comparison (F1 Score)")
    plt.xlabel("Model")
    plt.ylabel("F1 Score")

    plt.ylim(0, 1)

    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    plt.savefig(
        "data/processed/comparison_chart.png",
        dpi=300,
    )

    plt.close()

    print("Saved comparison_chart.png")
# ===========================
# Main
# ===========================

def main():

    df = load_data()

    feature_cols = [
        "time_interval_std",
        "upload_burst_ratio",
        "video_upload_frequency",
        "view_per_video",
        "dash_density",
        "title_length_std",
        "capitalization_ratio",
        "opening_repeat_ratio",
        "temporal_clickbait_ratio",
        "type_token_ratio",
        "avg_title_similarity",
        "sub_to_view_ratio",
        "subscriber_velocity",
        "sub_to_view_velocity_ratio",
    ]

    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print()
    print("=" * 50)
    print("Dataset summary")
    print("=" * 50)

    print(f"Train samples : {len(X_train)}")
    print(f"Test samples  : {len(X_test)}")

    print()
    print("Train label distribution")
    print(y_train.value_counts())

    print()
    print("Test label distribution")
    print(y_test.value_counts())
    print()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=2000,
            random_state=42,
        ),
    }

    results = []

    for name, model in models.items():
        results.append(
            evaluate_model(
                name,
                model,
                X_train,
                X_test,
                y_train,
                y_test,
            )
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.round(4)
    results_df = results_df.sort_values(
        by="F1",
        ascending=False,
    )
    print()
    print("=" * 70)
    print("Comparison Results")
    print("=" * 70)

    print(results_df)
    print()
    print("=" * 70)

    best = results_df.iloc[0]

    print(
        f"Best Model: {best['Model']} "
        f"(F1 = {best['F1']:.4f})"
    )

    results_df.to_csv(
        "data/processed/comparison_results.csv",
        index=False,
    )
    plot_results(results_df)


if __name__ == "__main__":
    main()