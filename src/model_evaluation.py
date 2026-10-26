from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
    }

    report = classification_report(
        y_test,
        predictions,
        target_names=["No Disease", "Disease"],
        zero_division=0,
    )

    cm = confusion_matrix(y_test, predictions)

    return predictions, metrics, report, cm


def plot_kernel_accuracy(results_df, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Kernel"], results_df["Accuracy"])
    plt.title("SVM Kernel Accuracy Comparison")
    plt.xlabel("Kernel")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()


def plot_performance_line(results_df, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(results_df["Kernel"], results_df["Accuracy"], marker="o", label="Accuracy")
    plt.plot(results_df["Kernel"], results_df["F1 Score"], marker="o", label="F1-score")
    plt.title("SVM Performance Comparison")
    plt.xlabel("Kernel")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()


def plot_confusion_matrix(cm, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix - Final SVM Model")
    plt.colorbar()
    plt.xticks([0, 1], ["Predicted: No Disease", "Predicted: Disease"], rotation=15)
    plt.yticks([0, 1], ["Actual: No Disease", "Actual: Disease"])
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > threshold else "black")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()


def save_results(results_df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(path, index=False)
