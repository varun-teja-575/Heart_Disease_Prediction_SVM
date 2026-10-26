import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

RANDOM_STATE = 42

KERNELS = ["linear", "poly", "rbf", "sigmoid"]

PARAM_GRIDS = {
    "linear": {
        "svm__C": [0.1, 1, 10]
    },
    "poly": {
        "svm__C": [0.1, 1, 10],
        "svm__degree": [2, 3],
        "svm__gamma": ["scale", "auto"]
    },
    "rbf": {
        "svm__C": [0.1, 1, 10],
        "svm__gamma": ["scale", "auto", 0.01, 0.1]
    },
    "sigmoid": {
        "svm__C": [0.1, 1, 10],
        "svm__gamma": ["scale", "auto"]
    },
}


def build_pipeline(preprocessor, kernel):
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("svm", SVC(kernel=kernel, random_state=RANDOM_STATE)),
        ]
    )


def train_kernel_models(preprocessor, X_train, y_train, X_test, y_test):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    models = {}
    rows = []

    for kernel in KERNELS:
        model = build_pipeline(preprocessor, kernel)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        rows.append({
            "Kernel": kernel,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1 Score": f1_score(y_test, predictions, zero_division=0),
        })
        models[kernel] = model

    return models, pd.DataFrame(rows)


def tune_kernel_models(preprocessor, X_train, y_train):
    cv_strategy = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    tuned_estimators = {}
    tuned_rows = []

    for kernel in KERNELS:
        pipeline = build_pipeline(preprocessor, kernel)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=PARAM_GRIDS[kernel],
            cv=cv_strategy,
            scoring="accuracy",
            n_jobs=-1,
        )

        grid_search.fit(X_train, y_train)

        tuned_estimators[kernel] = grid_search.best_estimator_
        tuned_rows.append({
            "Kernel": kernel,
            "Best CV Accuracy": grid_search.best_score_,
            "Best Params": grid_search.best_params_,
        })

    return tuned_estimators, pd.DataFrame(tuned_rows)


def select_best_model(tuned_estimators, tuned_results):
    best_row = tuned_results.loc[tuned_results["Best CV Accuracy"].idxmax()]
    best_kernel = best_row["Kernel"]

    return (
        best_kernel,
        best_row["Best Params"],
        best_row["Best CV Accuracy"],
        tuned_estimators[best_kernel],
    )
