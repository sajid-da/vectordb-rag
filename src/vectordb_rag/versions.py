"""Version and roll back persisted vector snapshots."""

import shutil
from pathlib import Path


class IndexVersionManager:
    """Keep named copies of a store snapshot and restore one on request."""

    def __init__(self, store_directory: Path) -> None:
        self._store_directory = store_directory
        self._versions_directory = store_directory / "versions"

    def create(self, version: str) -> Path:
        """Save the current snapshot as a named immutable version."""
        if not version or "/" in version or "\\" in version:
            raise ValueError("version must be a simple non-empty name")
        source = self._store_directory / "snapshot.json"
        if not source.exists():
            raise FileNotFoundError("create a snapshot before versioning it")
        self._versions_directory.mkdir(parents=True, exist_ok=True)
        destination = self._versions_directory / f"{version}.json"
        shutil.copyfile(source, destination)
        return destination

    def rollback(self, version: str) -> None:
        """Restore a named version as the current snapshot."""
        source = self._versions_directory / f"{version}.json"
        if not source.exists():
            raise FileNotFoundError(f"unknown version: {version}")
        temporary = self._store_directory / "snapshot.rollback.tmp"
        shutil.copyfile(source, temporary)
        temporary.replace(self._store_directory / "snapshot.json")
