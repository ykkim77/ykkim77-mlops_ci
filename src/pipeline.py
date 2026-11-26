from typing import Annotated, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from zenml import pipeline, step

from src.functions import preprocess_data, train_model


@step
def preprocess_step(
    test_size: float = 0.2,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:

    X_train, X_test, y_train, y_test = preprocess_data(
        test_size=test_size,
        random_state=42,
    )

    print("Train rows:", len(X_train))
    print("Test rows :", len(X_test))

    return X_train, X_test, y_train, y_test


@step
def train_step(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 5,
) -> Tuple[
    Annotated[RandomForestClassifier, "model"],
    Annotated[float, "accuracy"],
]:

    model, accuracy = train_model(
        X_train,
        X_test,
        y_train,
        y_test,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    print("n_estimators:", n_estimators)
    print("max_depth    :", max_depth)
    print(f"Accuracy     : {accuracy:.4f}")

    return model, accuracy


@step
def evaluate_step(
    accuracy: float,
    min_accuracy: float = 0.8,
) -> bool:

    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Min Accuracy : {min_accuracy:.4f}")

    if accuracy < min_accuracy:
        raise RuntimeError(
            "Model quality gate failed: "
            f"{accuracy:.4f} < {min_accuracy:.4f}"
        )

    print("Model quality gate: PASS")

    return True


@pipeline
def training_pipeline(
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 5,
    min_accuracy: float = 0.8,
):

    X_train, X_test, y_train, y_test = preprocess_step(
        test_size=test_size,
    )

    model, accuracy = train_step(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    evaluate_step(
        accuracy=accuracy,
        min_accuracy=min_accuracy,
    )
