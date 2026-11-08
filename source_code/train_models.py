from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "breast_cancer_wisconsin.csv"
MODEL_DIR = ROOT / "model"
METRICS_PATH = ROOT / "metrics.json"
TARGET = "diagnosis"
DROP_COLUMNS = ["id"]


def build_models():
    scaled = lambda estimator: Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )
    return {
        "Logistic Regression": scaled(
            LogisticRegression(max_iter=5000, random_state=42)
        ),
        "Decision Tree": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DecisionTreeClassifier(max_depth=6, random_state=42)),
            ]
        ),
        "KNN": scaled(KNeighborsClassifier(n_neighbors=7)),
        "Naive Bayes": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GaussianNB()),
            ]
        ),
        "Support Vector Machine": scaled(
            SVC(kernel="rbf", probability=True, random_state=42)
        ),
    }


def calculate_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, predictions), 6),
        "auc": round(roc_auc_score(y_test, probabilities), 6),
        "precision": round(
            precision_score(y_test, predictions, pos_label="M", zero_division=0), 6
        ),
        "recall": round(
            recall_score(y_test, predictions, pos_label="M", zero_division=0), 6
        ),
        "f1": round(
            f1_score(y_test, predictions, pos_label="M", zero_division=0), 6
        ),
        "mcc": round(matthews_corrcoef(y_test, predictions), 6),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=["B", "M"]
        ).tolist(),
    }


def main():
    data = pd.read_csv(DATA_PATH)
    X = data.drop(columns=[TARGET] + DROP_COLUMNS)
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    MODEL_DIR.mkdir(exist_ok=True)
    metrics = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, MODEL_DIR / filename)
        metrics[name] = calculate_metrics(model, X_test, y_test)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
