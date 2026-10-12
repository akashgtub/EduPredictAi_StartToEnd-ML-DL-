
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("data/data.csv", sep=";")
df.columns = df.columns.str.strip()

X = df.drop(columns=["Target"])
y = df["Target"]

categorical_cols = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

numeric_cols = [
    col for col in X.columns if col not in categorical_cols
]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_cols),
        ("numeric", StandardScaler(), numeric_cols),
    ]
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# Reuse your existing preprocessor from Stage 1

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
}

results = {}
pipelines = {}

# Compare models using 5-fold cross-validation on training data
for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessing", clone(preprocessor)),
        ("classifier", classifier)
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1
    )

    results[name] = scores.mean()
    pipelines[name] = pipeline

    print(f"{name}: Macro F1 = {scores.mean():.4f}")

# Select the best model using cross-validation
best_name = max(results, key=results.get)
best_model = pipelines[best_name]

print("\nBest model:", best_name)
print("Cross-validation Macro F1:", round(results[best_name], 4))

# Train the selected model on all training data
best_model.fit(X_train, y_train)

# Evaluate once on the held-out test data
y_pred = best_model.predict(X_test)

print("\nTest Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Test Macro F1:", round(f1_score(y_test, y_pred, average="macro"), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the selected pipeline
joblib.dump(best_model, "student_success_model.joblib")
print("\nBest model saved!")
