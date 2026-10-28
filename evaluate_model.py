import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

FEATURES = [
    "force",
    "temperature",
    "deformation",
    "stiffness",
    "tactile_score",
    "thermal_score",
    "fusion"
]

DATASETS = [
    "dataset1_clean.csv",
    "dataset2_clean.csv"
]

artifact = joblib.load("tactile_thermal_model.pkl")
model = artifact["model"]
model_features = artifact["features"]

print("=" * 65)
print("EXISTING RANDOM FOREST EVALUATION")
print("=" * 65)

print("\nModel features:", model_features)
print("Model classes:", list(model.classes_))

for filename in DATASETS:
    print("\n" + "=" * 65)
    print("DATASET:", filename)
    print("=" * 65)

    df = pd.read_csv(filename)

    X = df[FEATURES]
    y = df["ground_truth"]

    # Hold out 20% of the rows for evaluation.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Align test columns with the saved model's expected order.
    X_test = X_test[model_features]

    predictions = model.predict(X_test)

    print("\nTest samples:", len(y_test))
    print(
        "Accuracy:",
        f"{accuracy_score(y_test, predictions) * 100:.2f}%"
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=model.classes_,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(
        pd.DataFrame(
            confusion_matrix(
                y_test,
                predictions,
                labels=model.classes_
            ),
            index=[f"Actual {c}" for c in model.classes_],
            columns=[f"Predicted {c}" for c in model.classes_]
        ).to_string()
    )

print("\nEvaluation complete.")