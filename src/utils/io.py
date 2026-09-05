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


def dump_jsonl_atomic(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    dump_jsonl(temporary, rows)
    temporary.replace(target)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Invalid JSONL record at {target}:{line_number}")
        rows.append(value)
    return rows


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
