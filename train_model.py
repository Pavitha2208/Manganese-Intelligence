
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

# Dataset location
data_path = (
    Path.home()
    / "Downloads"
    / "SIH26009_Six_Synthetic_Training_Datasets (2)"
    / "01_Exploration_Geochemistry_Prospectivity.xlsx"
)

# Load synthetic exploration data
df = pd.read_excel(data_path, sheet_name="synthetic_data")

target = "prospectivity_label_simulated"

features = [
    "elevation_m_simulated",
    "manganese_ppm_simulated",
    "iron_pct_simulated",
    "silica_pct_simulated",
    "alumina_pct_simulated",
    "phosphorus_pct_simulated",
    "distance_to_mapped_structure_km_simulated",
    "weathering_index_simulated",
    "sample_method_simulated",
]

X = df[features].copy()
y = df[target]

# Separate numerical and categorical features
numeric_features = [
    col for col in features
    if col != "sample_method_simulated"
]
categorical_features = ["sample_method_simulated"]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ]
)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Train and evaluate
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\n=== MANGANESE PROSPECTIVITY MODEL ===")
print(f"Total samples: {len(df)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Target distribution:\n{y.value_counts().sort_index()}")

print(f"\nTest accuracy: {accuracy_score(y_test, predictions):.3f}")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

# Save the trained model
output_path = Path(__file__).parent / "manganese_model.joblib"
joblib.dump(model, output_path)

print(f"\nModel saved to: {output_path}")
print("Note: This model uses synthetic data for demonstration.")
