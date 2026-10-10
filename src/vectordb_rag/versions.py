"""Version and roll back persisted vector snapshots."""

import re
import shutil
from pathlib import Path


class IndexVersionManager:
    """Keep named copies of a store snapshot and restore one on request."""

    def __init__(self, store_directory: Path) -> None:
        self._store_directory = store_directory
        self._versions_directory = store_directory / "versions"

    def _validate_version_name(self, version: str) -> Path:
        """Enforce strict alphanumeric/hyphen/underscore names and path isolation."""
        if not version or not re.match(r"^[a-zA-Z0-9_-]+$", version):
            msg = (
                "version name must contain only letters, numbers, "
                "hyphens, and underscores"
            )
            raise ValueError(msg)

        self._versions_directory.mkdir(parents=True, exist_ok=True)
        destination = (self._versions_directory / f"{version}.json").resolve()
        versions_root = self._versions_directory.resolve()

        try:
            destination.relative_to(versions_root)
        except ValueError as error:
            raise ValueError("version path traversal detected") from error

        return destination

    def create(self, version: str) -> Path:
        """Save the current snapshot as a named immutable version."""
        destination = self._validate_version_name(version)
        source = self._store_directory / "snapshot.json"
        if not source.exists():
            raise FileNotFoundError("create a snapshot before versioning it")
        shutil.copyfile(source, destination)
        return destination

    def rollback(self, version: str) -> None:
        """Restore a named version as the current snapshot."""
        source = self._validate_version_name(version)
        if not source.exists():
            raise FileNotFoundError(f"unknown version: {version}")
        temporary = self._store_directory / "snapshot.rollback.tmp"
        shutil.copyfile(source, temporary)
        temporary.replace(self._store_directory / "snapshot.json")
