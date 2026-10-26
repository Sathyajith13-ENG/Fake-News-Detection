"""Optional AI chat with a clearly-labelled local fallback."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .config import Settings, get_settings

MAX_QUESTION = 2000
MAX_CONTEXT = 4000

SYSTEM_PROMPT = (
    "You are a concise assistant for a fake-news text classifier built with TF-IDF and "
    "Logistic Regression. Explain concepts and help interpret results. Never claim the "
    "classifier verifies facts; remind users to check original sources."
)

# (keywords, answer) - the best keyword overlap wins.
_KB: list[tuple[tuple[str, ...], str]] = [
    (("tf-idf", "tfidf", "tf idf", "vector"),
     "TF-IDF turns text into numbers. Each word (or two-word phrase) gets a weight that is high when "
     "it appears often in this article but rarely across all articles, so common words like 'the' count "
     "for little and distinctive words count for more."),
    (("logistic", "classifier", "algorithm", "how does the model", "how it work"),
     "The model is a Logistic Regression on TF-IDF features. It learns one weight per term; the weighted "
     "sum becomes a probability-like score for FAKE and REAL, and the larger score is the prediction. "
     "The 'influential terms' list shows which words pushed the score and in which direction."),
    (("accuracy", "precision", "recall", "f1", "metric"),
     "Accuracy is the share of held-out articles classified correctly. Precision asks 'when it says FAKE, "
     "how often is it right?', recall asks 'of all truly FAKE articles, how many did it catch?', and F1 "
     "balances the two. Macro averages treat both classes equally. They only describe your dataset and split."),
    (("confusion", "matrix"),
     "The confusion matrix counts held-out articles by true label (rows) versus predicted label (columns). "
     "The diagonal is correct predictions; off-diagonal cells are the two kinds of mistakes."),
    (("split", "holdout", "test set", "leak", "overfit"),
     "A stratified split keeps the FAKE/REAL ratio the same in training and test data. Tuning repeatedly "
     "against the same test set leaks information and inflates scores; keep a final untouched set."),
    (("calibrat", "probab", "confidence", "score", "percent"),
     "The scores are classifier outputs, not true probabilities of being fake. They may be over- or "
     "under-confident, especially on topics unlike the training data."),
    (("trust", "reliable", "accurate", "bias", "wrong", "limit", "fact-check", "fact check", "verify", "true"),
     "Treat predictions as a weak signal. The model learns writing style and vocabulary from your labels, "
     "so it can be wrong, biased, or out of date, and it cannot check facts. Confirm claims with original "
     "records and several credible, independent sources."),
    (("dataset", "data", "improve", "better", "train"),
     "To improve results: use a larger, balanced, well-documented dataset with consistent labels; keep "
     "sources and time periods varied; check for duplicates; and judge the model on a held-out split."),
    (("run", "start", "streamlit", "api", "install"),
     "Train with `python -m scripts.train_model --data data/news.csv`, then run `streamlit run app.py` for "
     "the interface and `python api.py` for the JSON API at http://127.0.0.1:5000."),
    (("privacy", "secret", "key", "send"),
     "With a remote AI provider configured, your question and any prediction context are sent to it, so "
     "avoid sensitive content. Never commit .env or API keys."),
]

_FALLBACK = (
    "I can explain TF-IDF, Logistic Regression, the metrics, the confusion matrix, train/test splits, "
    "limitations, and how to run the project. Try asking one of those."
)


def local_answer(question: str, prediction_context: str | None = None) -> str:
    q = question.lower()
    best, best_hits = _FALLBACK, 0
    for keys, answer in _KB:
        hits = sum(k in q for k in keys)
        if hits > best_hits:
            best, best_hits = answer, hits
    if prediction_context and any(w in q for w in ("this", "result", "prediction", "why", "predict")):
        return f"About the result you shared ({prediction_context.strip()[:300]}): {best}"
    return best


def _remote_answer(question: str, context: str | None, s: Settings) -> str:
    if urlparse(s.ai_api_url).scheme not in ("http", "https"):
        raise ValueError("AI_API_URL must start with http:// or https://")
    user = question if not context else f"Prediction context: {context}\n\nQuestion: {question}"
    body = json.dumps({
        "model": s.ai_model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(
        s.ai_api_url, data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {s.ai_api_key}"},
    )
    with urllib.request.urlopen(req, timeout=s.ai_timeout) as resp:  # noqa: S310 (scheme checked above)
        data = json.loads(resp.read().decode("utf-8"))
    text = data["choices"][0]["message"]["content"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty response")
    return text.strip()


def answer(question: str, prediction_context: str | None = None, settings: Settings | None = None) -> dict:
    """Return {'answer', 'source', 'note'}; source is 'provider' or 'local-fallback'."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("`question` must be a non-empty string.")
    if prediction_context is not None and not isinstance(prediction_context, str):
        raise ValueError("`prediction_context` must be a string.")
    question = question.strip()[:MAX_QUESTION]
    context = (prediction_context or "").strip()[:MAX_CONTEXT] or None
    s = settings or get_settings()

    if not s.ai_configured:
        return {"answer": local_answer(question, context), "source": "local-fallback",
                "note": "Local fallback assistant: no AI provider is configured."}
    try:
        return {"answer": _remote_answer(question, context, s), "source": "provider", "note": None}
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
        return {"answer": local_answer(question, context), "source": "local-fallback",
                "note": f"Local fallback assistant: the AI provider failed ({type(exc).__name__})."}
