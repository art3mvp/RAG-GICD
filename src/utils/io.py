from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    target.write_text(content + ("\n" if content else ""), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    safe_path = str(path).replace("\\", "/")
    return json.loads(Path(safe_path).read_text(encoding="utf-8"))
