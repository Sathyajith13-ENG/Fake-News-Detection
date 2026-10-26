"""TF-IDF + Logistic Regression pipeline: training, evaluation, inference."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .dataset import VALID_LABELS, DatasetError
from .preprocessing import normalize_text

RANDOM_STATE = 42
CAUTION = (
    "This is a statistical guess from patterns in the training data, not a fact-check. "
    "Verify claims against original records and several independent sources."
)


class ModelNotFoundError(FileNotFoundError):
    """Raised when no usable trained model is available."""


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2) -> tuple[Pipeline, dict]:
    """Fit on a reproducible stratified split and measure held-out performance."""
    if not 0.05 <= test_size <= 0.5:
        raise DatasetError("test_size must be between 0.05 and 0.5.")
    try:
        x_tr, x_te, y_tr, y_te = train_test_split(
            df["text"].tolist(), df["label"].tolist(),
            test_size=test_size, random_state=RANDOM_STATE, stratify=df["label"].tolist(),
        )
    except ValueError as exc:
        raise DatasetError(
            f"Dataset is too small for a stratified {test_size:.0%} test split ({len(df)} rows): {exc}"
        ) from exc

    pipe = build_pipeline().fit(x_tr, y_tr)
    pred = pipe.predict(x_te)
    labels = list(VALID_LABELS)
    p, r, f, s = precision_recall_fscore_support(y_te, pred, labels=labels, zero_division=0)
    evaluation = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": float(np.mean(f)),
        "per_class": {
            lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, lab in enumerate(labels)
        },
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_te, pred, labels=labels).tolist(),
        "n_train": len(x_tr),
        "n_test": len(x_te),
        "test_size": test_size,
        "class_counts": {k: int(v) for k, v in df["label"].value_counts().items()},
        "random_state": RANDOM_STATE,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return pipe, evaluation


@dataclass
class ModelBundle:
    pipeline: Pipeline
    evaluation: dict


def save_model(pipeline: Pipeline, evaluation: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "evaluation": evaluation}, path)
    return path


def load_model(path: str | Path) -> ModelBundle:
    path = Path(path)
    hint = "Train one first: python -m scripts.train_model --data data/news.csv"
    if not path.is_file():
        raise ModelNotFoundError(f"No trained model at {path}. {hint}")
    try:
        blob = joblib.load(path)
        return ModelBundle(blob["pipeline"], blob.get("evaluation", {}))
    except Exception as exc:  # corrupt / incompatible artifact
        raise ModelNotFoundError(f"Model at {path} could not be loaded ({type(exc).__name__}). {hint}") from exc


def _check_text(text: object) -> str:
    if not isinstance(text, str) or not normalize_text(text):
        raise ValueError("`text` must be a non-empty string.")
    return text


def predict(bundle: ModelBundle, text: str, top_k: int = 12) -> dict:
    """Classify one article and list the terms that pushed the score."""
    text = _check_text(text)
    tfidf, clf = bundle.pipeline.named_steps["tfidf"], bundle.pipeline.named_steps["clf"]
    x = tfidf.transform([text])
    probs = clf.predict_proba(x)[0]
    classes = list(clf.classes_)
    scores = {c: float(p) for c, p in zip(classes, probs)}
    label = classes[int(np.argmax(probs))]

    # Binary LR: positive coefficient favours classes_[1], negative favours classes_[0].
    contrib = x.multiply(clf.coef_[0]).tocsr()
    names = tfidf.get_feature_names_out()
    terms = [
        {"term": str(names[i]), "weight": float(w), "leans": classes[1] if w > 0 else classes[0]}
        for i, w in zip(contrib.indices, contrib.data)
    ]
    terms.sort(key=lambda t: abs(t["weight"]), reverse=True)
    return {
        "label": label,
        "confidence": scores[label],
        "scores": scores,
        "terms": terms[:top_k],
        "word_count": len(text.split()),
        "caution": CAUTION,
    }


def predict_many(bundle: ModelBundle, texts: list[str]) -> list[dict]:
    """Fast batch inference (label + scores, no term explanations)."""
    clean = [t if isinstance(t, str) and normalize_text(t) else "" for t in texts]
    valid = [i for i, t in enumerate(clean) if t]
    out: list[dict] = [{"label": None, "confidence": None, "scores": {}} for _ in clean]
    if valid:
        pipe = bundle.pipeline
        probs = pipe.predict_proba([clean[i] for i in valid])
        classes = list(pipe.classes_)
        for row, i in zip(probs, valid):
            k = int(np.argmax(row))
            out[i] = {"label": classes[k], "confidence": float(row[k]),
                      "scores": {c: float(p) for c, p in zip(classes, row)}}
    return out


def top_features(bundle: ModelBundle, n: int = 12) -> dict[str, list[tuple[str, float]]]:
    """Terms the model associates most strongly with each class."""
    tfidf, clf = bundle.pipeline.named_steps["tfidf"], bundle.pipeline.named_steps["clf"]
    names, coef = tfidf.get_feature_names_out(), clf.coef_[0]
    c0, c1 = clf.classes_
    order = np.argsort(coef)
    return {
        c0: [(str(names[i]), float(-coef[i])) for i in order[:n]],
        c1: [(str(names[i]), float(coef[i])) for i in order[::-1][:n]],
    }
