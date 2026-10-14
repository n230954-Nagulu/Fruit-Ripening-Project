"""Check that the saved model and scaler load and accept one inference input."""

import time

import numpy as np

from app import IMG_SIZE, load_resources, predict_image


def main() -> None:
    started = time.perf_counter()
    model, scaler = load_resources()
    print(f"Model and scaler loaded in {time.perf_counter() - started:.2f}s.")
    print(f"Model inputs: {[tensor.name for tensor in model.inputs]}")
    print(f"Scaler feature count: {scaler.n_features_in_}")

    sample = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
    prediction = predict_image(model, scaler, sample)
    print(f"Smoke-test raw model output: {prediction:.6f}")


if __name__ == "__main__":
    main()
