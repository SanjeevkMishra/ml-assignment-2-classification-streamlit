from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
TARGET = "diagnosis"
DROP_COLUMNS = ["id"]
LABELS = ["B", "M"]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_resource
def load_model(model_name):
    return joblib.load(ROOT / "model" / MODEL_FILES[model_name])


def evaluate(model, data):
    if TARGET not in data.columns:
        raise ValueError(
            f"The uploaded CSV must contain the target column '{TARGET}'."
        )
    feature_columns = [column for column in data.columns if column not in [TARGET, *DROP_COLUMNS]]
    if len(feature_columns) != 30:
        raise ValueError("The uploaded CSV must contain exactly the 30 model features.")
    X = data[feature_columns]
    y = data[TARGET].astype(str)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score(y, probabilities),
        "Precision": precision_score(y, predictions, pos_label="M", zero_division=0),
        "Recall": recall_score(y, predictions, pos_label="M", zero_division=0),
        "F1": f1_score(y, predictions, pos_label="M", zero_division=0),
        "MCC": matthews_corrcoef(y, predictions),
    }, y, predictions


st.set_page_config(page_title="Breast Cancer Classifier", page_icon=":microscope:")
st.title("Breast Cancer Diagnostic Classifier")
st.write(
    "Upload labelled test data, choose a saved classifier, and inspect its "
    "evaluation results. The uploaded data is evaluated only; no retraining occurs."
)

default_path = ROOT / "test_data.csv"
uploaded = st.file_uploader("Upload test_data.csv", type=["csv"])
data = pd.read_csv(uploaded if uploaded else default_path)
st.caption(f"Loaded {len(data)} records with {len(data.columns)} columns.")

model_name = st.selectbox("Select a model", list(MODEL_FILES))

try:
    model = load_model(model_name)
    metrics, y_true, predictions = evaluate(model, data)
except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(str(error))
    st.stop()

st.subheader(f"{model_name} metrics")
st.dataframe(
    pd.DataFrame([metrics]).style.format("{:.4f}"),
    use_container_width=True,
    hide_index=True,
)

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    matrix = confusion_matrix(y_true, predictions, labels=LABELS)
    figure, axis = plt.subplots()
    axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=range(2),
        yticks=range(2),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Predicted label",
        ylabel="True label",
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    st.pyplot(figure)
    plt.close(figure)

with right:
    st.subheader("Classification report")
    st.code(classification_report(y_true, predictions, zero_division=0))

with st.expander("Preview uploaded data"):
    st.dataframe(data.head(10), use_container_width=True)
