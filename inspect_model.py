import joblib
import pandas as pd
from pathlib import Path

print("=" * 60)
print("TACTILE-THERMAL MODEL INSPECTION")
print("=" * 60)

# Load saved model
model_path = Path("tactile_thermal_model.pkl")

if not model_path.exists():
    print("ERROR: Model file not found.")
    raise SystemExit(1)

artifact = joblib.load(model_path)

print("\n1. MODEL ARTIFACT")

if isinstance(artifact, dict):
    print("Artifact keys:", list(artifact.keys()))

    model = artifact.get("model")
    features = artifact.get("features", [])

    print("Model type:", type(model).__name__)
    print("Features:", features)

    if model is not None:
        print("Classes:", getattr(model, "classes_", "Unavailable"))
        print("Number of trees:", len(getattr(model, "estimators_", [])))
else:
    print("Artifact type:", type(artifact).__name__)

# Inspect both datasets
for filename in ["dataset1_clean.csv", "dataset2_clean.csv"]:

    print("\n" + "=" * 60)
    print("DATASET:", filename)
    print("=" * 60)

    path = Path(filename)

    if not path.exists():
        print("File not found.")
        continue

    df = pd.read_csv(path)

    print("\nShape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nNumeric feature statistics:")
    print(df.describe().round(3).to_string())

    print("\nPossible classification columns:")
    for column in df.columns:
        if any(word in column.lower()
               for word in ["class", "label", "target", "diagnosis"]):
            print("\n", column)
            print(df[column].value_counts(dropna=False).to_string())

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)