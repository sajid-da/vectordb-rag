"""Crash-recovery tests for snapshots and the write-ahead log."""

from pathlib import Path

from vectordb_rag.storage import PersistentVectorStore


def test_restart_recovers_entries_not_yet_compacted_into_snapshot(
    tmp_path: Path,
) -> None:
    store = PersistentVectorStore(tmp_path)
    store.add("first", [1.0, 0.0])
    store.snapshot()
    store.add("second", [0.0, 1.0])

    restarted = PersistentVectorStore(tmp_path)

    assert restarted.get_all() == [
        store.get_all()[0],
        store.get_all()[1],
    ]


def test_delete_is_replayed_after_restart(tmp_path: Path) -> None:
    store = PersistentVectorStore(tmp_path)
    store.add("remove-me", [1.0, 1.0])
    store.snapshot()
    store.delete("remove-me")

    assert PersistentVectorStore(tmp_path).get_all() == []
