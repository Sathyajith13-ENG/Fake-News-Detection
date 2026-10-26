"""Flask JSON API: /health, /predict, /chat (local development server)."""
from __future__ import annotations

from flask import Flask, jsonify, request

from fake_news_detection.chatbot import answer as chat_answer
from fake_news_detection.config import Settings, get_settings
from fake_news_detection.model import ModelBundle, ModelNotFoundError, load_model, predict

MAX_TEXT_CHARS = 200_000


def create_app(settings: Settings | None = None) -> Flask:
    app = Flask(__name__)
    cfg = settings or get_settings()
    cache: dict = {"mtime": None, "bundle": None}

    def bundle() -> ModelBundle:
        """Load the model, reloading automatically if the file was retrained."""
        try:
            mtime = cfg.model_path.stat().st_mtime
        except OSError:
            cache.update(mtime=None, bundle=None)
            return load_model(cfg.model_path)  # raises ModelNotFoundError with instructions
        if cache["bundle"] is None or cache["mtime"] != mtime:
            cache.update(bundle=load_model(cfg.model_path), mtime=mtime)
        return cache["bundle"]

    def error(message: str, status: int):
        return jsonify({"error": message}), status

    def json_body():
        data = request.get_json(silent=True)
        return data if isinstance(data, dict) else None

    @app.get("/health")
    def health():
        try:
            ev = bundle().evaluation
        except ModelNotFoundError as exc:
            return jsonify({"status": "unavailable", "model_loaded": False, "error": str(exc)}), 503
        return jsonify({"status": "ok", "model_loaded": True, "accuracy": ev.get("accuracy"),
                        "trained_at": ev.get("trained_at")})

    @app.post("/predict")
    def predict_route():
        data = json_body()
        if data is None:
            return error("Send a JSON object like {\"text\": \"...\"}.", 400)
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            return error("`text` must be a non-empty string.", 400)
        if len(text) > MAX_TEXT_CHARS:
            return error(f"`text` is too long (max {MAX_TEXT_CHARS} characters).", 413)
        try:
            return jsonify(predict(bundle(), text))
        except ModelNotFoundError as exc:
            return error(str(exc), 503)
        except ValueError as exc:
            return error(str(exc), 400)

    @app.post("/chat")
    def chat_route():
        data = json_body()
        if data is None:
            return error("Send a JSON object like {\"question\": \"...\"}.", 400)
        try:
            return jsonify(chat_answer(data.get("question"), data.get("prediction_context"), cfg))
        except ValueError as exc:
            return error(str(exc), 400)

    @app.errorhandler(404)
    def not_found(_):
        return error("Not found.", 404)

    @app.errorhandler(405)
    def bad_method(_):
        return error("Method not allowed.", 405)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
