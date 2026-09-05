from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.config.settings import AppSettings


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid YAML settings at {path}")
    return loaded


def load_settings(config_path: str | Path = "config/settings.yaml") -> AppSettings:
    load_dotenv()
    cfg = _read_yaml(Path(config_path))

    env_mapping = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_model": os.getenv("OPENAI_MODEL"),
        "openai_embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL"),
        "langchain_api_key": os.getenv("LANGCHAIN_API_KEY"),
        "langchain_tracing_v2": os.getenv("LANGCHAIN_TRACING_V2"),
        "chroma_persist_dir": os.getenv("CHROMA_PERSIST_DIR"),
        "hybrid_index_dir": os.getenv("HYBRID_INDEX_DIR"),
        "faiss_index_dir": os.getenv("FAISS_INDEX_DIR"),
        "raw_data_dir": os.getenv("DATA_RAW_DIR"),
        "processed_data_dir": os.getenv("DATA_PROCESSED_DIR"),
        "top_k": os.getenv("TOP_K"),
        "chunk_size": os.getenv("CHUNK_SIZE"),
        "chunk_overlap": os.getenv("CHUNK_OVERLAP"),
    }

    merged: dict[str, Any] = {**cfg}
    for key, value in env_mapping.items():
        if value is None or value == "":
            continue

        if key in {"top_k", "chunk_size", "chunk_overlap"} and key in merged:
            continue

        merged[key] = value

    return AppSettings.model_validate(merged)


def load_prompts(path: str | Path = "config/prompts.yaml") -> dict[str, str]:
    prompts = _read_yaml(Path(path))
    return {
        "system_prompt": str(prompts.get("system_prompt", "")),
        "qa_prompt": str(prompts.get("qa_prompt", "")),
    }
