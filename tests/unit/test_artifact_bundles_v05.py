from __future__ import annotations

import io
import tarfile
import tempfile
from pathlib import Path

import pytest

from icframe.orchestration.bundles import (
    create_artifact_bundle,
    import_artifact_bundle,
    verify_artifact_bundle,
)


def test_bundle_verification_and_atomic_import(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "result.json").write_text('{"ok":true}')
    bundle = create_artifact_bundle(source, tmp_path / "artifact.tar.gz", logical_id="shard-1")

    manifest = verify_artifact_bundle(bundle)
    imported = import_artifact_bundle(
        bundle,
        tmp_path / "imported",
        expected_logical_id="shard-1",
    )

    assert manifest.logical_id == "shard-1"
    assert (imported / "result.json").read_text() == '{"ok":true}'
    with pytest.raises(FileExistsError):
        import_artifact_bundle(bundle, imported, expected_logical_id="shard-1")


def test_bundle_rejects_path_traversal(tmp_path) -> None:
    bundle = tmp_path / "unsafe.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        payload = b"escape"
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="unsafe"):
        verify_artifact_bundle(bundle)
    assert not (tmp_path.parent / "escape.txt").exists()


def test_bundle_rejects_checksum_corruption(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "result.json").write_text('{"ok":true}')
    bundle = create_artifact_bundle(source, tmp_path / "artifact.tar.gz", logical_id="shard-1")
    with tempfile.TemporaryDirectory(dir=tmp_path) as directory:
        stage = Path(directory)
        with tarfile.open(bundle, "r:gz") as archive:
            archive.extractall(stage, filter="fully_trusted")
        (stage / "result.json").write_text('{"ok":false}')
        with tarfile.open(bundle, "w:gz") as archive:
            for path in stage.iterdir():
                archive.add(path, arcname=path.name)

    with pytest.raises(ValueError, match="checksum"):
        verify_artifact_bundle(bundle)
