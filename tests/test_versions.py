"""Tests for snapshot version creation and rollback."""

from pathlib import Path

from vectordb_rag.storage import PersistentVectorStore
from vectordb_rag.versions import IndexVersionManager


def test_rollback_restores_a_named_snapshot(tmp_path: Path) -> None:
    store = PersistentVectorStore(tmp_path)
    store.add("first", [1.0, 0.0])
    store.snapshot()
    versions = IndexVersionManager(tmp_path)
    versions.create("v1")
    store.add("second", [0.0, 1.0])
    store.snapshot()

    versions.rollback("v1")

    assert [item.id for item in PersistentVectorStore(tmp_path).get_all()] == ["first"]
