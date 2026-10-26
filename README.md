# Heart Disease Prediction Using SVM

A console-based machine learning project that predicts the presence or absence of heart disease using Support Vector Machine (SVM).

## Dataset

The project uses `data/heart_disease_dataset_named.csv` with 3069 rows and 17 columns.

The dataset uses clear column names such as:

- Age
- Sex
- Chest Pain
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression
- ST Slope
- Number of Major Vessels
- Thalassemia
- Smoking
- Diabetes
- Body Mass Index
- Target

`Target` is the prediction label: `0 = No Disease`, `1 = Disease`.

## Main Steps

1. Dataset loading
2. Data cleaning
3. Basic EDA
4. Feature preprocessing and scaling
5. Train-test split
6. SVM training with Linear, Polynomial, RBF and Sigmoid kernels
7. GridSearchCV hyperparameter tuning
8. Kernel comparison
9. Final model evaluation
10. Console-based patient prediction

## Run the project

Run the complete project without patient input:

```powershell
python main.py --skip-input
```

Run the complete project with patient input:

```powershell
python main.py
```

## Output folders

- `outputs/graphs/` contains generated graphs.
- `outputs/results/` contains kernel comparison and final evaluation results.

## Note

This project is for academic and educational use. Its prediction is not a medical diagnosis.
