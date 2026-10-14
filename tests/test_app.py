import io
import unittest

import numpy as np
from PIL import Image, UnidentifiedImageError
from streamlit.testing.v1 import AppTest

import app


class RecordingScaler:
    def __init__(self):
        self.features = None

    def transform(self, features):
        self.features = features.copy()
        return features


class RecordingModel:
    def __init__(self, value):
        self.value = value
        self.inputs = None
        self.calls = 0

    def predict(self, inputs, verbose=0):
        self.inputs = inputs
        self.verbose = verbose
        self.calls += 1
        return np.array([[self.value]], dtype=np.float32)


class FailingModel:
    def predict(self, inputs, verbose=0):
        raise ValueError("internal model failure")


def png_bytes():
    image = Image.new("RGB", (37, 61), color=(186, 127, 42))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class ImagePipelineTests(unittest.TestCase):
    def test_preprocessing_matches_existing_image_pipeline(self):
        encoded = png_bytes()
        preview, batch = app.prepare_uploaded_image(encoded)

        with Image.open(io.BytesIO(encoded)) as original:
            expected = np.array(
                original.convert("RGB").resize(app.IMG_SIZE)
            ).astype(np.float32) / 255.0
        expected = np.expand_dims(expected, axis=0)

        self.assertEqual(preview.shape, (61, 37, 3))
        self.assertEqual(batch.shape, (1, 224, 224, 3))
        np.testing.assert_array_equal(batch, expected)

    def test_invalid_image_bytes_are_rejected(self):
        with self.assertRaises((UnidentifiedImageError, OSError, ValueError)):
            app.prepare_uploaded_image(b"not an image")

    def test_image_cache_uses_uploaded_bytes_as_its_key(self):
        first_bytes = png_bytes()
        other_image = Image.new("RGB", (37, 61), color=(25, 91, 172))
        encoded_other = io.BytesIO()
        other_image.save(encoded_other, format="PNG")

        _, first_batch = app.prepare_uploaded_image(first_bytes)
        _, second_batch = app.prepare_uploaded_image(encoded_other.getvalue())

        self.assertFalse(np.array_equal(first_batch, second_batch))

    def test_inference_preserves_inputs_scaling_and_quiet_prediction(self):
        model = RecordingModel(0.73)
        scaler = RecordingScaler()
        batch = np.ones((1, 224, 224, 3), dtype=np.float32)

        first = app.predict_image(model, scaler, batch)
        second = app.predict_image(model, scaler, batch)

        self.assertEqual(first, second)
        self.assertEqual(model.calls, 2)
        self.assertEqual(model.verbose, 0)
        self.assertIs(model.inputs["mobile_input"], batch)
        self.assertIs(model.inputs["cnn_input"], batch)
        self.assertEqual(model.inputs["feat_input"].shape, (1, 4))
        np.testing.assert_array_equal(
            scaler.features,
            np.zeros((1, 4), dtype=np.float32),
        )

    def test_prediction_errors_are_propagated_and_lock_is_released(self):
        with self.assertRaisesRegex(ValueError, "internal model failure"):
            app.predict_image(
                FailingModel(),
                RecordingScaler(),
                np.zeros((1, 224, 224, 3), dtype=np.float32),
            )
        self.assertTrue(app.PREDICTION_LOCK.acquire(blocking=False))
        app.PREDICTION_LOCK.release()

    def test_existing_class_mapping_and_score_threshold(self):
        self.assertEqual(
            app.interpret_prediction(0.8),
            ("Artificially Ripened Mango", 0.8, "artificial"),
        )
        self.assertEqual(
            app.interpret_prediction(0.2),
            ("Naturally Ripened Mango", 0.8, "natural"),
        )
        self.assertEqual(
            app.interpret_prediction(0.5),
            ("Naturally Ripened Mango", 0.5, "natural"),
        )


class StreamlitWorkflowTests(unittest.TestCase):
    def test_home_theme_detection_upload_predict_repeat_and_return(self):
        app_test = AppTest.from_file(
            str(app.PROJECT_DIR / "app.py"),
            default_timeout=120,
        ).run()
        self.assertFalse(app_test.exception)
        self.assertEqual(app_test.session_state["page"], "home")

        app_test.button(key="nav_detection").click().run()
        self.assertEqual(app_test.session_state["page"], "detection")
        app_test.button(key="nav_home").click().run()
        self.assertEqual(app_test.session_state["page"], "home")

        app_test.toggle(key="dark_theme").set_value(True).run()
        self.assertTrue(app_test.session_state["dark_theme"])

        app_test.button(key="start_detection").click().run()
        self.assertEqual(app_test.session_state["page"], "detection")
        self.assertTrue(app_test.session_state["dark_theme"])

        app_test.file_uploader(key="mango_image").set_value(
            ("broken.png", b"invalid image bytes", "image/png")
        ).run()
        self.assertTrue(any("valid image" in item.value for item in app_test.error))

        image_data = png_bytes()
        app_test.file_uploader(key="mango_image").set_value(
            ("mango.png", image_data, "image/png")
        ).run()
        self.assertFalse(app_test.error)

        app_test.button(key="predict").click().run()
        first_prediction = app_test.session_state["last_prediction"]["raw_prediction"]
        self.assertTrue(
            any(
                "Ripened Mango" in item.value
                for item in app_test.markdown
            )
        )

        app_test.button(key="predict").click().run()
        second_prediction = app_test.session_state["last_prediction"]["raw_prediction"]
        self.assertEqual(first_prediction, second_prediction)

        second_image = Image.new("RGB", (37, 61), color=(25, 91, 172))
        second_image_bytes = io.BytesIO()
        second_image.save(second_image_bytes, format="PNG")
        app_test.file_uploader(key="mango_image").set_value(
            ("other-mango.png", second_image_bytes.getvalue(), "image/png")
        ).run()
        self.assertIsNone(app_test.session_state["last_prediction"])

        app_test.toggle(key="dark_theme").set_value(False).run()
        self.assertFalse(app_test.session_state["dark_theme"])
        app_test.button(key="nav_home").click().run()
        self.assertEqual(app_test.session_state["page"], "home")
        self.assertFalse(app_test.session_state["dark_theme"])


if __name__ == "__main__":
    unittest.main()
