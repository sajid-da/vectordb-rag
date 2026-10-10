"""Crash-safe local vector persistence using a JSONL write-ahead log."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StoredVector:
    """A vector and its caller-provided ID."""

    id: str
    values: list[float]


class PersistentVectorStore:
    """Append each mutation durably, then compact it into an atomic snapshot."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._snapshot = directory / "snapshot.json"
        self._wal = directory / "wal.jsonl"
        self._vectors: dict[str, StoredVector] = {}
        self._load()

    def add(self, identifier: str, values: list[float]) -> None:
        """Durably add or replace one vector before exposing it to readers."""
        vector = np.asarray(values, dtype=np.float64)
        if vector.ndim != 1 or len(vector) == 0 or float(np.linalg.norm(vector)) == 0:
            raise ValueError("values must be a non-zero one-dimensional vector")
        self._append({"operation": "add", "id": identifier, "values": values})
        self._vectors[identifier] = StoredVector(identifier, values)

    def delete(self, identifier: str) -> None:
        """Durably remove one vector if it exists."""
        self._append({"operation": "delete", "id": identifier})
        self._vectors.pop(identifier, None)

    def get_all(self) -> list[StoredVector]:
        """Return a stable snapshot of the current logical data."""
        return list(self._vectors.values())

    def snapshot(self) -> None:
        """Atomically persist all current data and clear already-applied WAL entries."""
        self._directory.mkdir(parents=True, exist_ok=True)
        temporary = self._snapshot.with_suffix(".tmp")
        temporary.write_text(
            json.dumps([item.__dict__ for item in self.get_all()]), encoding="utf-8"
        )
        os.replace(temporary, self._snapshot)
        self._wal.write_text("", encoding="utf-8")

    def _append(self, operation: dict[str, object]) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        with self._wal.open("a", encoding="utf-8") as file:
            file.write(json.dumps(operation) + "\n")
            file.flush()
            os.fsync(file.fileno())

    def _load(self) -> None:
        if self._snapshot.exists():
            raw_vectors = json.loads(self._snapshot.read_text(encoding="utf-8"))
            self._vectors = {
                item["id"]: StoredVector(item["id"], item["values"])
                for item in raw_vectors
            }
        if not self._wal.exists():
            return

        raw_lines = self._wal.read_text(encoding="utf-8").splitlines()
        lines = [line for line in raw_lines if line.strip()]
        total_lines = len(lines)

        for index, line in enumerate(lines, 1):
            is_last = index == total_lines
            try:
                op = json.loads(line)
                if not isinstance(op, dict) or "operation" not in op or "id" not in op:
                    raise ValueError("invalid WAL record format")

                identifier = str(op["id"])
                op_type = op["operation"]

                if op_type == "add":
                    self._vectors[identifier] = StoredVector(identifier, op["values"])
                elif op_type == "delete":
                    self._vectors.pop(identifier, None)
                else:
                    raise ValueError(f"unknown WAL operation type: {op_type}")

            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as error:
                if is_last:
                    break
                raise RuntimeError(
                    f"Corrupted WAL record at line {index}/{total_lines}: {error}"
                ) from error
