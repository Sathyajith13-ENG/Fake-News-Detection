"""Environment-backed configuration (reads .env without extra dependencies)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    data_path: Path
    model_path: Path
    ai_api_url: str
    ai_api_key: str
    ai_model: str
    ai_timeout: float

    @property
    def ai_configured(self) -> bool:
        return bool(self.ai_api_url and self.ai_api_key and self.ai_model)


def _resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def get_settings() -> Settings:
    _load_dotenv(ROOT / ".env")
    try:
        timeout = float(os.environ.get("AI_TIMEOUT_SECONDS") or 20)
    except ValueError:
        timeout = 20.0
    return Settings(
        data_path=_resolve(os.environ.get("DATA_PATH") or "data/news.csv"),
        model_path=_resolve(os.environ.get("MODEL_PATH") or "models/fake_news_pipeline.joblib"),
        ai_api_url=(os.environ.get("AI_API_URL") or "").strip(),
        ai_api_key=(os.environ.get("AI_API_KEY") or "").strip(),
        ai_model=(os.environ.get("AI_MODEL") or "").strip(),
        ai_timeout=timeout,
    )
