from src.functions import preprocess_data, train_model


def test_preprocess_data():

    X_train, X_test, y_train, y_test = preprocess_data()

    assert len(X_train) == 999
    assert len(X_test) > 0
    assert len(X_train) + len(X_test) == 150
    assert X_train.shape[1] == 4


def test_train_model():

    X_train, X_test, y_train, y_test = preprocess_data()

    model, accuracy = train_model(
        X_train,
        X_test,
        y_train,
        y_test,
        n_estimators=50,
        max_depth=3,
    )

    assert model is not None
    assert 0.0 <= accuracy <= 1.0
    assert accuracy >= 0.8
