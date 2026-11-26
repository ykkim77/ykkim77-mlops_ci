from src.pipeline import training_pipeline


if __name__ == "__main__":

    training_pipeline(
        test_size=0.2,
        n_estimators=100,
        max_depth=5,
        min_accuracy=0.8,
    )
