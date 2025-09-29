from pydantic import BaseModel
from pathlib import Path
import yaml, os

DEFAULT_CFG = Path.home()/".repo_agent/config.yaml"

class Settings(BaseModel):
    repo_path: str
    db_path: str
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    max_chunk_chars: int = 1600
    overlap_chars: int = 200
    exclude_globs: list[str] = ["**/node_modules/**","**/.git/**","**/dist/**","**/build/**",".venv/**"]

def load_settings(path: Path | None = None) -> Settings:
    path = path or DEFAULT_CFG
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return Settings(**data)

def save_settings(s: Settings, path: Path | None = None):
    path = path or DEFAULT_CFG
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(s.model_dump(), f)
