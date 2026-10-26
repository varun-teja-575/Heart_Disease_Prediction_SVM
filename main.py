from pathlib import Path
import sys

from src.data_preprocessing import (
    load_dataset,
    clean_dataset,
    prepare_features,
    split_data,
)
from src.eda import plot_target_distribution
from src.svm_models import (
    train_kernel_models,
    tune_kernel_models,
    select_best_model,
)
from src.model_evaluation import (
    evaluate_model,
    plot_kernel_accuracy,
    plot_performance_line,
    plot_confusion_matrix,
    save_results,
)
from src.prediction import (
    collect_patient_input,
    predict_patient,
    display_prediction,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "data" / "heart_disease_dataset_named.csv"
GRAPH_DIR = PROJECT_DIR / "outputs" / "graphs"
RESULTS_DIR = PROJECT_DIR / "outputs" / "results"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main(run_prediction=True):
    print("=" * 60)
    print("HEART DISEASE PREDICTION USING SVM")
    print("=" * 60)

    # 1. Load dataset
    df = load_dataset(DATASET_PATH)
    print(f"\nDataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Clean and understand data
    print(f"Missing values before cleaning: {int(df.isnull().sum().sum())}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")

    df = clean_dataset(df)

    print(f"Dataset after cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Missing values after cleaning: {int(df.isnull().sum().sum())}")

    # 3. EDA
    plot_target_distribution(
        df,
        "Target",
        GRAPH_DIR / "01_heart_disease_distribution.png",
    )

    # 4. Prepare features
    (
        X,
        y,
        preprocessor,
        numeric_features,
        categorical_features,
        binary_numeric_features,
    ) = prepare_features(df)

    print("\nNumerical features:", numeric_features)
    print("Categorical features:", categorical_features)
    print("Binary numeric features:", binary_numeric_features)

    # 5. Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"\nTraining records: {len(X_train)}")
    print(f"Testing records : {len(X_test)}")

    # 6. Train all four SVM kernels
    print("\nTraining Linear, Polynomial, RBF and Sigmoid SVM models...")
    kernel_models, results_df = train_kernel_models(
        preprocessor,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    print("\nKernel comparison:")
    print(results_df.to_string(index=False))

    save_results(results_df, RESULTS_DIR / "kernel_results.csv")

    # 7. Kernel visualizations
    plot_kernel_accuracy(
        results_df,
        GRAPH_DIR / "02_svm_kernel_accuracy.png",
    )

    plot_performance_line(
        results_df,
        GRAPH_DIR / "03_svm_performance_comparison.png",
    )

    # 8. Hyperparameter tuning
    print("\nRunning GridSearchCV for all four kernels...")
    tuned_estimators, tuned_results_df = tune_kernel_models(
        preprocessor,
        X_train,
        y_train,
    )

    print("\nTuned kernel results:")
    print(tuned_results_df.to_string(index=False))

    tuned_results_df.to_csv(
        RESULTS_DIR / "tuned_kernel_results.csv",
        index=False,
    )

    # 9. Select best model
    (
        best_kernel,
        best_params,
        best_cv_score,
        final_model,
    ) = select_best_model(
        tuned_estimators,
        tuned_results_df,
    )

    print("\n" + "=" * 60)
    print("BEST SVM MODEL")
    print("=" * 60)
    print(f"Best Kernel: {best_kernel}")
    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validation Accuracy: {best_cv_score:.4f}")

    # 10. Final evaluation
    predictions, metrics, report, cm = evaluate_model(
        final_model,
        X_test,
        y_test,
    )

    print("\nFinal Test Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(cm)

    plot_confusion_matrix(
        cm,
        GRAPH_DIR / "04_confusion_matrix.png",
    )

    # Save final metrics
    with open(RESULTS_DIR / "final_metrics.txt", "w", encoding="utf-8") as file:
        file.write(f"Best Kernel: {best_kernel}\n")
        file.write(f"Best Parameters: {best_params}\n")
        file.write(f"Best CV Accuracy: {best_cv_score:.4f}\n\n")
        for name, value in metrics.items():
            file.write(f"{name}: {value:.4f}\n")
        file.write("\nClassification Report:\n")
        file.write(report)
        file.write("\nConfusion Matrix:\n")
        file.write(str(cm))

    # 11. Console prediction
    if run_prediction:
        patient_input = collect_patient_input(
            X,
            numeric_features,
            categorical_features,
            binary_numeric_features,
        )

        prediction, _ = predict_patient(
            final_model,
            patient_input,
        )

        display_prediction(
            prediction,
            best_kernel,
            patient_input,
            final_model,
        )

    print("\nProject execution completed successfully.")


if __name__ == "__main__":
    skip_input = "--skip-input" in sys.argv
    main(run_prediction=not skip_input)
