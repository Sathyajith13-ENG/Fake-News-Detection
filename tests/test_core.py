"""Focused tests. Run: python -m unittest discover -s tests -v"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from fake_news_detection.chatbot import answer, local_answer
from fake_news_detection.config import Settings
from fake_news_detection.dataset import DatasetError, load_dataset, validate_frame
from fake_news_detection.model import (ModelNotFoundError, load_model, predict, predict_many, save_model,
                                       top_features, train_and_evaluate)
from fake_news_detection.preprocessing import normalize_text
from scripts.make_demo_data import make_rows


def demo_frame(n: int = 30) -> pd.DataFrame:
    return pd.DataFrame(make_rows(n, seed=1), columns=["text", "label"])


def settings(tmp: Path, **kw) -> Settings:
    base = dict(data_path=tmp / "d.csv", model_path=tmp / "m.joblib", ai_api_url="", ai_api_key="",
                ai_model="", ai_timeout=1.0)
    base.update(kw)
    return Settings(**base)


class PreprocessingTests(unittest.TestCase):
    def test_normalizes(self):
        self.assertEqual(normalize_text("  Hello <b>WORLD</b> https://x.io/a  &amp; more "), "hello world & more")

    def test_none_and_empty(self):
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text("   "), "")


class DatasetTests(unittest.TestCase):
    def test_labels_case_insensitive(self):
        df = pd.DataFrame({"text": list("abcd"), "label": ["fake", "Real", "FAKE", "real"]})
        self.assertEqual(sorted(validate_frame(df)["label"].unique()), ["FAKE", "REAL"])

    def test_rejects_unknown_label(self):
        df = pd.DataFrame({"text": list("abcd"), "label": ["fake", "real", "maybe", "real"]})
        with self.assertRaisesRegex(DatasetError, "Invalid label"):
            validate_frame(df)

    def test_missing_column(self):
        with self.assertRaisesRegex(DatasetError, "Missing column"):
            validate_frame(pd.DataFrame({"body": ["x"], "label": ["REAL"]}))

    def test_needs_two_per_class(self):
        df = pd.DataFrame({"text": list("abc"), "label": ["FAKE", "REAL", "REAL"]})
        with self.assertRaisesRegex(DatasetError, "at least 2 FAKE"):
            validate_frame(df)

    def test_blank_text_dropped(self):
        df = pd.DataFrame({"text": ["a", " ", "b", "c", "d"], "label": ["FAKE", "FAKE", "FAKE", "REAL", "REAL"]})
        self.assertEqual(len(validate_frame(df)), 4)

    def test_missing_file(self):
        with self.assertRaisesRegex(DatasetError, "not found"):
            load_dataset("does/not/exist.csv")


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe, cls.ev = train_and_evaluate(demo_frame())
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = save_model(cls.pipe, cls.ev, Path(cls.tmp.name) / "m.joblib")
        cls.bundle = load_model(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_evaluation_record(self):
        ev = self.ev
        for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            self.assertTrue(0.0 <= ev[k] <= 1.0)
        self.assertEqual(ev["n_train"] + ev["n_test"], 60)
        self.assertEqual(sum(map(sum, ev["confusion_matrix"])), ev["n_test"])

    def test_split_is_reproducible(self):
        _, again = train_and_evaluate(demo_frame())
        self.assertEqual(again["confusion_matrix"], self.ev["confusion_matrix"])

    def test_predict_shape(self):
        out = predict(self.bundle, "SHOCKING!!! Insiders reveal the truth they don't want you to see. Share now!")
        self.assertIn(out["label"], ("FAKE", "REAL"))
        self.assertAlmostEqual(sum(out["scores"].values()), 1.0, places=6)
        self.assertEqual(out["label"], max(out["scores"], key=out["scores"].get))
        self.assertTrue(out["terms"] and {"term", "weight", "leans"} <= set(out["terms"][0]))
        self.assertIn("not a fact-check", out["caution"])

    def test_predict_rejects_empty(self):
        for bad in ("", "   ", None, 5):
            with self.assertRaises(ValueError):
                predict(self.bundle, bad)

    def test_predict_many_handles_blanks(self):
        res = predict_many(self.bundle, ["A council announced a budget on Monday.", "", None])
        self.assertIsNotNone(res[0]["label"])
        self.assertIsNone(res[1]["label"])
        self.assertIsNone(res[2]["label"])

    def test_top_features_cover_both_classes(self):
        tf = top_features(self.bundle, 5)
        self.assertEqual(set(tf), {"FAKE", "REAL"})
        self.assertTrue(all(len(v) == 5 for v in tf.values()))

    def test_missing_and_corrupt_model(self):
        with self.assertRaises(ModelNotFoundError):
            load_model("nope.joblib")
        bad = Path(self.tmp.name) / "bad.joblib"
        bad.write_bytes(b"not a model")
        with self.assertRaises(ModelNotFoundError):
            load_model(bad)

    def test_tiny_dataset_gives_readable_error(self):
        tiny = pd.DataFrame({"text": list("abcd"), "label": ["FAKE", "FAKE", "REAL", "REAL"]})
        with self.assertRaises(DatasetError):
            train_and_evaluate(tiny)


class ChatbotTests(unittest.TestCase):
    def test_local_fallback_identifies_itself(self):
        with tempfile.TemporaryDirectory() as t:
            r = answer("How does TF-IDF work?", settings=settings(Path(t)))
        self.assertEqual(r["source"], "local-fallback")
        self.assertIn("TF-IDF", r["answer"])
        self.assertIn("fallback", r["note"].lower())

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            answer("   ")
        with self.assertRaises(ValueError):
            answer("hi", prediction_context=123)

    def test_provider_failure_falls_back(self):
        with tempfile.TemporaryDirectory() as t:
            s = settings(Path(t), ai_api_url="http://127.0.0.1:9/v1", ai_api_key="k", ai_model="m")
            r = answer("What is recall?", settings=s)
        self.assertEqual(r["source"], "local-fallback")
        self.assertIn("failed", r["note"])

    def test_non_http_url_never_fetched(self):
        with tempfile.TemporaryDirectory() as t:
            s = settings(Path(t), ai_api_url="file:///etc/passwd", ai_api_key="k", ai_model="m")
            self.assertEqual(answer("hello", settings=s)["source"], "local-fallback")

    def test_unknown_question_still_helpful(self):
        self.assertIn("TF-IDF", local_answer("zzz qqq"))


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from api import create_app
        except ImportError as exc:  # Flask missing
            raise unittest.SkipTest(str(exc))
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cfg = settings(Path(cls.tmp.name))
        cls.client = create_app(cls.cfg).test_client()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_01_no_model_gives_503_with_instructions(self):
        self.assertEqual(self.client.get("/health").status_code, 503)
        r = self.client.post("/predict", json={"text": "hello world"})
        self.assertEqual(r.status_code, 503)
        self.assertIn("train_model", r.get_json()["error"])

    def test_02_with_model(self):
        pipe, ev = train_and_evaluate(demo_frame())
        save_model(pipe, ev, self.cfg.model_path)
        h = self.client.get("/health")
        self.assertEqual((h.status_code, h.get_json()["model_loaded"]), (200, True))
        r = self.client.post("/predict", json={"text": "The regional health board announced a new clinic on Friday."})
        self.assertEqual(r.status_code, 200)
        self.assertIn(r.get_json()["label"], ("FAKE", "REAL"))

    def test_03_input_validation(self):
        for body in ({}, {"text": ""}, {"text": 5}, {"text": "   "}):
            self.assertEqual(self.client.post("/predict", json=body).status_code, 400, body)
        self.assertEqual(self.client.post("/predict", data="nope", content_type="text/plain").status_code, 400)
        self.assertEqual(self.client.post("/predict", json=["a"]).status_code, 400)

    def test_04_chat(self):
        ok = self.client.post("/chat", json={"question": "What is a confusion matrix?", "prediction_context": "FAKE"})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.get_json()["source"], "local-fallback")
        self.assertEqual(self.client.post("/chat", json={}).status_code, 400)
        self.assertEqual(self.client.post("/chat", json={"question": "hi", "prediction_context": 3}).status_code, 400)

    def test_05_json_errors(self):
        self.assertEqual(self.client.get("/nope").get_json()["error"], "Not found.")
        self.assertEqual(self.client.get("/predict").status_code, 405)


if __name__ == "__main__":
    unittest.main()
