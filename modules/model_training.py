
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def train_model(df, target, task_type, test_size=0.2):

    if target not in df.columns:
        raise ValueError("Selected target column does not exist.")

    if task_type not in ["Classification", "Regression"]:
        raise ValueError("Invalid machine learning task.")

    data = df.dropna(subset=[target]).copy()

    if len(data) < 5:
        raise ValueError("Not enough rows to train a model.")

    X = data.drop(columns=[target])
    y = data[target]

    if X.empty:
        raise ValueError("No feature columns available.")

    if task_type == "Regression" and not pd.api.types.is_numeric_dtype(y):
        raise ValueError("Regression requires a numeric target column.")

    if task_type == "Classification" and y.nunique() < 2:
        raise ValueError("Classification requires at least two classes.")

    if task_type == "Classification":
        if y.value_counts().min() < 2:
            raise ValueError(
                "Each class needs at least two records for a stratified split."
            )

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    stratify = y if task_type == "Classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify
    )

    if task_type == "Classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        }

    results = []
    trained_models = {}

    for name, algorithm in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", algorithm)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        trained_models[name] = pipeline

        if task_type == "Classification":

            metrics = {
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(
                    y_test, predictions,
                    average="weighted",
                    zero_division=0
                ),
                "Recall": recall_score(
                    y_test, predictions,
                    average="weighted",
                    zero_division=0
                ),
                "F1 Score": f1_score(
                    y_test, predictions,
                    average="weighted",
                    zero_division=0
                )
            }

            results.append({
                "Model": name,
                **metrics
            })

        else:

            metrics = {
                "MAE": mean_absolute_error(y_test, predictions),
                "RMSE": np.sqrt(
                    mean_squared_error(y_test, predictions)
                ),
                "R2 Score": r2_score(y_test, predictions)
            }

            results.append({
                "Model": name,
                **metrics
            })

    comparison = pd.DataFrame(results)

    if task_type == "Classification":
        comparison = comparison.sort_values(
            by="Accuracy",
            ascending=False
        )
    else:
        comparison = comparison.sort_values(
            by="RMSE",
            ascending=True
        )

    best_model_name = comparison.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    predictions = best_model.predict(X_test)

    if task_type == "Classification":

        final_metrics = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions,
                average="weighted",
                zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions,
                average="weighted",
                zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions,
                average="weighted",
                zero_division=0
            )
        }

    else:

        final_metrics = {
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(
                mean_squared_error(y_test, predictions)
            ),
            "R2 Score": r2_score(y_test, predictions)
        }

    return {
    "model": best_model,
    "metrics": final_metrics,
    "predictions": predictions,
    "actual": y_test,
    "X_test": X_test,
    "y_test": y_test,
    "training_rows": len(X_train),
    "testing_rows": len(X_test),
    "feature_columns": X.columns.tolist(),
    "comparison": comparison,
    "best_model_name": best_model_name
}