from sklearn.model_selection import GridSearchCV
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

# STAGE 3: HYPERPARAMETER TUNING

# Choose settings based on the Stage 2 winner
if best_name == "Logistic Regression":
    param_grid = {
        "classifier__C": [0.1, 1, 10]
    }

elif best_name == "Decision Tree":
    param_grid = {
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 10],
        "classifier__min_samples_leaf": [1, 3]
    }

else:  # Random Forest
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 20],
        "classifier__min_samples_leaf": [1, 2]
    }

# Search for the best hyperparameters
grid_search = GridSearchCV(
    estimator=best_model,
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Get the tuned model
tuned_model = grid_search.best_estimator_

print("\n--- STAGE 3 RESULTS ---")
print("Selected algorithm:", best_name)
print("Best parameters:", grid_search.best_params_)
print("Best CV Macro F1:", round(grid_search.best_score_, 4))

# Evaluate on the same untouched test set
tuned_predictions = tuned_model.predict(X_test)

print(
    "Tuned Test Accuracy:",
    round(accuracy_score(y_test, tuned_predictions), 4)
)
print(
    "Tuned Test Macro F1:",
    round(f1_score(y_test, tuned_predictions, average="macro"), 4)
)

print("\nTuned Classification Report:")
print(classification_report(
    y_test, tuned_predictions, zero_division=0
))

# Save the improved pipeline
joblib.dump(tuned_model, "student_success_model.joblib")
print("Tuned model saved successfully!")

# STAGE 4: FEATURE IMPORTANCE

from sklearn.inspection import permutation_importance

print("\n--- STAGE 4: FEATURE IMPORTANCE ---")

importance_result = permutation_importance(
    tuned_model,
    X_test,
    y_test,
    scoring="f1_macro",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance_result.importances_mean
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(importance_df.head(15).to_string(index=False))

importance_df.to_csv("feature_importance.csv", index=False)
print("\nFeature importance saved to feature_importance.csv")

# STUDENT DROPOUT RISK ANALYSIS

print("\n--- STUDENT RISK ANALYSIS ---")

class_names = list(tuned_model.classes_)
dropout_index = class_names.index("Dropout")

dropout_probabilities = tuned_model.predict_proba(X_test)[
    :, dropout_index
]

risk_report = pd.DataFrame({
    "Actual Outcome": y_test.to_numpy(),
    "Predicted Outcome": tuned_predictions,
    "Dropout Probability": dropout_probabilities
}, index=X_test.index)

# Illustrative thresholds, not validated intervention cutoffs
risk_report["Risk Level"] = pd.cut(
    risk_report["Dropout Probability"],
    bins=[-0.001, 0.30, 0.60, 1.00],
    labels=["Low", "Medium", "High"]
)

risk_report = risk_report.sort_values(
    "Dropout Probability",
    ascending=False
)

print("\n10 Students with Highest Predicted Dropout Probability:")
print(risk_report.head(10).to_string(index=False))

risk_report.to_csv("student_risk_report.csv", index=False)

print("\nRisk report saved to student_risk_report.csv")
