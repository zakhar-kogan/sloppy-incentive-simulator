from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from icframe.orchestration.models import ArtifactBundleManifest, BundleFile

BUNDLE_MANIFEST = "bundle-manifest.json"


def create_artifact_bundle(
    source: str | Path,
    output: str | Path,
    *,
    logical_id: str,
) -> Path:
    source_path = Path(source)
    output_path = Path(output)
    if not source_path.is_dir():
        raise ValueError("artifact bundle source must be a directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for path in sorted(item for item in source_path.rglob("*") if item.is_file()):
        relative = path.relative_to(source_path).as_posix()
        if relative == BUNDLE_MANIFEST:
            continue
        files.append(
            BundleFile(path=relative, size=path.stat().st_size, sha256=_sha256(path))
        )
    manifest = ArtifactBundleManifest(logical_id=logical_id, files=files)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as temp:
        stage = Path(temp)
        for file in files:
            destination = stage / file.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path / file.path, destination)
        (stage / BUNDLE_MANIFEST).write_text(manifest.model_dump_json(indent=2))
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        with tarfile.open(temporary, "w:gz") as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.add(path, arcname=path.relative_to(stage).as_posix(), recursive=False)
        temporary.replace(output_path)
    return output_path


def verify_artifact_bundle(bundle: str | Path) -> ArtifactBundleManifest:
    bundle_path = Path(bundle)
    with tempfile.TemporaryDirectory(dir=bundle_path.parent) as temp:
        stage = Path(temp)
        _safe_extract(bundle_path, stage)
        manifest_path = stage / BUNDLE_MANIFEST
        if not manifest_path.is_file():
            raise ValueError("artifact bundle is missing its manifest")
        manifest = ArtifactBundleManifest.model_validate_json(manifest_path.read_text())
        expected = {item.path: item for item in manifest.files}
        actual = {
            path.relative_to(stage).as_posix(): path
            for path in stage.rglob("*")
            if path.is_file() and path.name != BUNDLE_MANIFEST
        }
        if set(actual) != set(expected):
            raise ValueError("artifact bundle file list does not match its manifest")
        for name, path in actual.items():
            declared = expected[name]
            if path.stat().st_size != declared.size or _sha256(path) != declared.sha256:
                raise ValueError(f"artifact bundle checksum mismatch: {name}")
        return manifest


def import_artifact_bundle(
    bundle: str | Path,
    destination: str | Path,
    *,
    expected_logical_id: str | None = None,
) -> Path:
    bundle_path = Path(bundle)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination_path.parent) as temp:
        stage = Path(temp) / "artifact"
        stage.mkdir()
        _safe_extract(bundle_path, stage)
        manifest_path = stage / BUNDLE_MANIFEST
        if not manifest_path.is_file():
            raise ValueError("artifact bundle is missing its manifest")
        manifest = ArtifactBundleManifest.model_validate_json(manifest_path.read_text())
        if expected_logical_id is not None and manifest.logical_id != expected_logical_id:
            raise ValueError("artifact bundle logical id does not match the requested job")
        verify_artifact_bundle(bundle_path)
        manifest_path.unlink()
        if destination_path.exists():
            raise FileExistsError(f"artifact destination already exists: {destination_path}")
        stage.replace(destination_path)
    return destination_path


def bundle_sha256(bundle: str | Path) -> str:
    return _sha256(Path(bundle))


def write_completion_marker(bundle: str | Path, marker: str | Path) -> Path:
    bundle_path = Path(bundle)
    payload = {
        "schema_version": "1",
        "bundle": bundle_path.name,
        "sha256": bundle_sha256(bundle_path),
        "size": bundle_path.stat().st_size,
    }
    marker_path = Path(marker)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker_path.with_suffix(marker_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True))
    temporary.replace(marker_path)
    return marker_path


def _safe_extract(bundle: Path, destination: Path) -> None:
    with tarfile.open(bundle, "r:gz") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                raise ValueError(f"unsafe artifact bundle member: {member.name}")
            if not member.isfile() and not member.isdir():
                raise ValueError(f"unsupported artifact bundle member: {member.name}")
        try:
            archive.extractall(destination, filter="fully_trusted")
        except TypeError:  # pragma: no cover - Python 3.11 without extraction filters
            archive.extractall(destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
