import pandas as pd


def get_numeric_input(prompt_text, cast_type=float):
    while True:
        raw = input(prompt_text)
        try:
            return cast_type(raw)
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def get_categorical_input(prompt_text, allowed_values):
    allowed_str = "/".join(str(value) for value in allowed_values)

    while True:
        raw = input(f"{prompt_text} ({allowed_str}): ").strip()

        for value in allowed_values:
            if raw.lower() == str(value).lower():
                return value

        print(f"Invalid input. Please enter one of: {allowed_str}")


def collect_patient_input(X, numeric_features, categorical_features, binary_numeric_features):
    print("\nEnter the new patient's details below.")
    print("Use the values represented in the training dataset.\n")

    patient_input = {}

    for col in X.columns:
        if col in binary_numeric_features:
            allowed_values = sorted(X[col].dropna().unique().tolist())
            patient_input[col] = float(
                get_categorical_input(f"Enter {col}", allowed_values)
            )

        elif col in numeric_features:
            patient_input[col] = get_numeric_input(f"Enter {col}: ")

        else:
            allowed_values = sorted(X[col].dropna().unique().tolist(), key=str)
            value = get_categorical_input(f"Enter {col}", allowed_values)
            patient_input[col] = str(value)

    return patient_input


def predict_patient(model, patient_input):
    patient_df = pd.DataFrame([patient_input])
    prediction = model.predict(patient_df)[0]

    return prediction, patient_df


def display_prediction(prediction, best_kernel, patient_input, model):
    print("\n" + "=" * 55)
    print("HEART DISEASE PREDICTION")
    print("=" * 55)
    print("Model used   : Support Vector Machine (SVM)")
    print(f"Kernel used  : {best_kernel}")

    print("\nInput Summary:")
    for key, value in patient_input.items():
        print(f"  {key}: {value}")

    print()

    if prediction == 1:
        print("Prediction: Heart Disease Detected (model estimate)")
    else:
        print("Prediction: No Heart Disease Detected (model estimate)")

    if hasattr(model, "decision_function"):
        margin_score = model.decision_function(
            pd.DataFrame([patient_input])
        )[0]
        print(f"Decision margin score: {margin_score:.4f}")

    print("\nDISCLAIMER: This is an educational machine-learning project output.")
    print("It is NOT a medical diagnosis.")
    print("Consult a qualified doctor for real health concerns.")
    print("=" * 55)
