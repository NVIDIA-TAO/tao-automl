from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

import pytest

from tao_automl.ptm_preflight import (
    ArtifactCacheError,
    AtomicArtifactCache,
    NGCCredential,
    NGCHTTPSClient,
)
from tao_automl.ptm_registry import PTMRegistry, load_ptm_registry

from . import ptm_stage


SECRET = "unit-test-ngc-secret"
CANONICAL_STAGE_ROOT = PurePosixPath(
    "/lustre/fsw/unit-tests/mask-grounding-dino-v1"
)


class _Response:
    def __init__(self, status_code: int, content: bytes):
        self.status_code = status_code
        self.content = content
        self.headers = {"Content-Length": str(len(content))}

    def iter_content(self, chunk_size: int):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset : offset + chunk_size]

    def close(self):
        return None


class _Session:
    def __init__(self, payloads: dict[str, bytes]):
        self.payloads = payloads
        self.calls: list[tuple[str, str, dict]] = []

    def head(self, url: str, **kwargs):
        self.calls.append(("HEAD", url, kwargs))
        payload = self.payloads.get(url)
        return _Response(200 if payload is not None else 404, payload or b"")

    def get(self, url: str, **kwargs):
        self.calls.append(("GET", url, kwargs))
        payload = self.payloads.get(url)
        return _Response(200 if payload is not None else 404, payload or b"")

    @property
    def download_calls(self):
        return [
            item
            for item in self.calls
            if item[0] == "GET"
            and item[2]["headers"].get("Range") is None
        ]


def _registry_and_payloads(
    *,
    corrupt_registered_checksum: bool = False,
) -> tuple[PTMRegistry, dict[str, bytes]]:
    document = load_ptm_registry().to_dict()
    records = document["models"]["mask_grounding_dino"]["checkpoints"]
    payload_by_id = {
        record["id"]: f"verified:{record['id']}".encode("utf-8")
        for record in records
    }
    for index, record in enumerate(records):
        payload = payload_by_id[record["id"]]
        source = record["source"]
        source.update(
            {
                "resource": f"mask_grounding_dino_fixture_{index}",
                "version": "fixture_v1.0",
                "member": f"checkpoints/fixture_{index}.pth",
            }
        )
        source["immutable_identity"] = (
            f"ngc://{source['registry']}/{source['resource']}:"
            f"{source['version']}#{source['member']}"
        )
        record["expected_size_bytes"] = len(payload)
        if index == 0:
            record.pop("sha256", None)
        else:
            record["sha256"] = hashlib.sha256(payload).hexdigest()
    if corrupt_registered_checksum:
        records[1]["sha256"] = "0" * 64
    registry = PTMRegistry(document)
    client = NGCHTTPSClient(
        NGCCredential(SECRET),
        api_base_url="https://ngc.example.test",
    )
    payloads = {
        client.resolve_member(record["source"]).url: payload_by_id[record["id"]]
        for record in records
    }
    return registry, payloads


def _client(session: _Session) -> NGCHTTPSClient:
    return NGCHTTPSClient(
        NGCCredential(SECRET),
        session=session,
        api_base_url="https://ngc.example.test",
    )


def _restore_write_bits(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
    ):
        os.chmod(path, 0o755)
    os.chmod(root, 0o755)


def test_all_four_ptms_stage_atomically_read_only_and_idempotently(
    tmp_path: Path,
):
    registry, payloads = _registry_and_payloads()
    session = _Session(payloads)
    cache = AtomicArtifactCache(tmp_path / "cache")
    stage_root = tmp_path / "lustre" / "mask-grounding-dino"
    publisher = ptm_stage.LustreStagePublisher(
        stage_root,
        enforce_lustre_prefix=False,
    )
    manifest_path = tmp_path / "evidence" / "ptm_stage_manifest.json"
    try:
        first = ptm_stage.stage_official_ptms(
            registry=registry,
            cache=cache,
            ngc_client=_client(session),
            publisher=publisher,
            manifest_path=manifest_path,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert set(manifest) == {
            "schema_version",
            "model",
            "registry_sha256",
            "stage_complete",
            "remote_read_only",
            "cpu_model_runs",
            "smoke_model_runs",
            "mini_step_runs",
            "checkpoints",
            "manifest_sha256",
        }
        assert [item["id"] for item in manifest["checkpoints"]] == list(
            ptm_stage.EXPECTED_CHECKPOINT_IDS
        )
        assert all(
            set(item)
            == {
                "id",
                "path",
                "size_bytes",
                "sha256",
                "immutable_source_identity",
                "remote_read_only",
            }
            and item["remote_read_only"] is True
            and Path(item["path"]).is_file()
            and not Path(item["path"]).stat().st_mode & 0o222
            for item in manifest["checkpoints"]
        )
        assert manifest["manifest_sha256"] == first["manifest_sha256"]
        assert first["remote_manifest_path"] == str(
            publisher.manifest_path
        )
        assert first["physical_manifest_path"] == str(
            publisher.manifest_path
        )
        assert first["execution"] == {
            "data_only": True,
            "model_invoked": False,
            "cpu_model_runs": 0,
            "gpu_model_runs": 0,
            "smoke_model_runs": 0,
            "mini_step_runs": 0,
            "slurm_jobs_submitted": 0,
            "scheduler_client_constructed": False,
        }
        assert not manifest_path.stat().st_mode & 0o222
        assert not publisher.manifest_path.stat().st_mode & 0o222
        assert not stage_root.stat().st_mode & 0o222
        assert len(session.download_calls) == 4
        assert SECRET not in json.dumps(first, sort_keys=True)

        second = ptm_stage.stage_official_ptms(
            registry=registry,
            cache=cache,
            ngc_client=_client(session),
            publisher=publisher,
            manifest_path=manifest_path,
        )
        assert second["manifest_sha256"] == first["manifest_sha256"]
        assert second["local_manifest_reused"] is True
        assert all(second["cache_hits"].values())
        assert all(second["published_reuse"].values())
        assert len(session.download_calls) == 4
    finally:
        _restore_write_bits(stage_root)


def test_sshfs_mapped_stage_is_exact_and_uses_canonical_manifest_paths(
    tmp_path: Path,
):
    registry, payloads = _registry_and_payloads()
    session = _Session(payloads)
    mount = tmp_path / "sshfs-lustre"
    mount.mkdir()
    publisher = ptm_stage.LustreStagePublisher.from_publication_roots(
        CANONICAL_STAGE_ROOT,
        physical_lustre_mount=mount,
        mount_verifier=lambda _: True,
    )
    manifest_path = tmp_path / "evidence" / "ptm_stage_manifest.json"
    expected_physical_root = mount.joinpath(
        *CANONICAL_STAGE_ROOT.relative_to("/lustre").parts
    ).resolve()
    try:
        summary = ptm_stage.stage_official_ptms(
            registry=registry,
            cache=AtomicArtifactCache(tmp_path / "cache"),
            ngc_client=_client(session),
            publisher=publisher,
            manifest_path=manifest_path,
        )
        assert publisher.root == expected_physical_root
        assert publisher.canonical_root == CANONICAL_STAGE_ROOT
        assert manifest_path.read_bytes() == publisher.manifest_path.read_bytes()
        local_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        remote_manifest = json.loads(
            publisher.manifest_path.read_text(encoding="utf-8")
        )
        assert local_manifest == remote_manifest

        expected_files = {publisher.manifest_path.resolve()}
        for checkpoint in local_manifest["checkpoints"]:
            canonical_path = PurePosixPath(checkpoint["path"])
            assert canonical_path.is_relative_to(CANONICAL_STAGE_ROOT)
            physical_path = publisher.physical_path(canonical_path)
            assert publisher.canonical_path(physical_path) == canonical_path
            assert physical_path.is_file()
            assert physical_path.stat().st_size == checkpoint["size_bytes"]
            assert (
                hashlib.sha256(physical_path.read_bytes()).hexdigest()
                == checkpoint["sha256"]
            )
            assert not physical_path.stat().st_mode & 0o222
            expected_files.add(physical_path.resolve())

        observed_files = {
            path.resolve()
            for path in publisher.root.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        assert observed_files == expected_files
        assert summary["remote_manifest_path"] == str(
            CANONICAL_STAGE_ROOT / ptm_stage.REMOTE_MANIFEST_NAME
        )
        assert summary["physical_manifest_path"] == str(
            publisher.manifest_path
        )
        assert not publisher.manifest_path.stat().st_mode & 0o222
        assert not publisher.root.stat().st_mode & 0o222
    finally:
        _restore_write_bits(publisher.root)


def test_publication_path_mapping_is_bijective_and_direct_mode_is_unchanged(
    tmp_path: Path,
):
    mount = tmp_path / "sshfs-lustre"
    mount.mkdir()
    publisher = ptm_stage.LustreStagePublisher.from_publication_roots(
        CANONICAL_STAGE_ROOT,
        physical_lustre_mount=mount,
        mount_verifier=lambda _: True,
    )
    canonical_checkpoint = (
        CANONICAL_STAGE_ROOT / "checkpoint-id" / "model.pth"
    )
    physical_checkpoint = publisher.physical_path(canonical_checkpoint)
    assert physical_checkpoint == (
        publisher.root / "checkpoint-id" / "model.pth"
    )
    assert (
        publisher.canonical_path(physical_checkpoint)
        == canonical_checkpoint
    )
    assert publisher.canonical_manifest_path == (
        CANONICAL_STAGE_ROOT / ptm_stage.REMOTE_MANIFEST_NAME
    )

    with pytest.raises(ptm_stage.PTMStageError, match="physical artifact"):
        publisher.canonical_path(tmp_path / "outside.bin")
    with pytest.raises(ptm_stage.PTMStageError, match="canonical artifact"):
        publisher.physical_path("/lustre/fsw/another-stage/model.pth")

    direct = ptm_stage.LustreStagePublisher.from_publication_roots(
        CANONICAL_STAGE_ROOT
    )
    assert direct.root == Path(str(CANONICAL_STAGE_ROOT))
    assert direct.canonical_root == CANONICAL_STAGE_ROOT
    assert direct.physical_lustre_mount is None
    assert direct.canonical_manifest_path == PurePosixPath(
        str(direct.manifest_path)
    )


@pytest.mark.parametrize(
    "canonical_root",
    (
        "/tmp/not-lustre/stage",
        "/lustre",
        "lustre/relative/stage",
        "/lustre/fsw/../escaped-stage",
    ),
)
def test_sshfs_mapping_rejects_unsafe_canonical_roots(
    tmp_path: Path,
    canonical_root: str,
):
    mount = tmp_path / "sshfs-lustre"
    mount.mkdir()
    with pytest.raises(ptm_stage.PTMStageError, match="canonical PTM root"):
        ptm_stage.LustreStagePublisher.from_publication_roots(
            canonical_root,
            physical_lustre_mount=mount,
            mount_verifier=lambda _: True,
        )


def test_sshfs_mapping_rejects_inactive_symlinked_or_mismatched_roots(
    tmp_path: Path,
):
    mount = tmp_path / "sshfs-lustre"
    mount.mkdir()
    with pytest.raises(ptm_stage.PTMStageError, match="active safe mount"):
        ptm_stage.LustreStagePublisher.from_publication_roots(
            CANONICAL_STAGE_ROOT,
            physical_lustre_mount=mount,
            mount_verifier=lambda _: False,
        )

    with pytest.raises(ptm_stage.PTMStageError, match="active safe mount"):
        ptm_stage.LustreStagePublisher.from_publication_roots(
            CANONICAL_STAGE_ROOT,
            physical_lustre_mount="/",
            mount_verifier=lambda _: True,
        )

    mount_target = tmp_path / "mount-target"
    mount_target.mkdir()
    mount_link = tmp_path / "mount-link"
    mount_link.symlink_to(mount_target, target_is_directory=True)
    with pytest.raises(ptm_stage.PTMStageError, match="active safe mount"):
        ptm_stage.LustreStagePublisher.from_publication_roots(
            CANONICAL_STAGE_ROOT,
            physical_lustre_mount=mount_link,
            mount_verifier=lambda _: True,
        )

    with pytest.raises(
        ptm_stage.PTMStageError,
        match="publication roots do not correspond",
    ):
        ptm_stage.LustreStagePublisher(
            mount / "non-corresponding-stage",
            canonical_root=CANONICAL_STAGE_ROOT,
            physical_lustre_mount=mount,
            mount_verifier=lambda _: True,
        )


def test_checksum_failure_never_emits_completed_stage(tmp_path: Path):
    registry, payloads = _registry_and_payloads(
        corrupt_registered_checksum=True
    )
    session = _Session(payloads)
    stage_root = tmp_path / "lustre" / "mask-grounding-dino"
    manifest = tmp_path / "evidence" / "ptm_stage_manifest.json"
    try:
        with pytest.raises(ArtifactCacheError, match="SHA-256"):
            ptm_stage.stage_official_ptms(
                registry=registry,
                cache=AtomicArtifactCache(tmp_path / "cache"),
                ngc_client=_client(session),
                publisher=ptm_stage.LustreStagePublisher(
                    stage_root,
                    enforce_lustre_prefix=False,
                ),
                manifest_path=manifest,
            )
        assert not manifest.exists()
        assert not (stage_root / ptm_stage.REMOTE_MANIFEST_NAME).exists()
    finally:
        _restore_write_bits(stage_root)


def test_unexpected_existing_stage_content_fails_before_network(
    tmp_path: Path,
):
    registry, payloads = _registry_and_payloads()
    session = _Session(payloads)
    stage_root = tmp_path / "lustre" / "mask-grounding-dino"
    stage_root.mkdir(parents=True)
    (stage_root / "unregistered.bin").write_bytes(b"unexpected")
    with pytest.raises(ptm_stage.PTMStageError, match="unexpected file"):
        ptm_stage.stage_official_ptms(
            registry=registry,
            cache=AtomicArtifactCache(tmp_path / "cache"),
            ngc_client=_client(session),
            publisher=ptm_stage.LustreStagePublisher(
                stage_root,
                enforce_lustre_prefix=False,
            ),
            manifest_path=tmp_path / "manifest.json",
        )
    assert session.calls == []


def test_secret_file_parser_reads_only_ngc_key_without_exposure(
    tmp_path: Path,
):
    path = tmp_path / "config.env"
    path.write_text(
        f"SLURM_USER=ignored\nexport NGC_KEY='{SECRET}'\n",
        encoding="utf-8",
    )
    credential = ptm_stage._read_ngc_credential(path)
    assert credential.authorization_header() == f"Bearer {SECRET}"
    assert SECRET not in repr(credential)


def test_stager_has_no_model_or_scheduler_execution_path():
    source = Path(ptm_stage.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "tao_sdk" not in imported
    assert all(
        module is None or not module.startswith("nvidia_tao_pytorch")
        for module in imported_from
    )
    assert "create_job" not in source
    assert "sbatch" not in source
    assert "srun" not in source


def test_cli_preserves_direct_lustre_default_and_accepts_only_mount_root(
    tmp_path: Path,
):
    direct = ptm_stage._parser().parse_args(())
    assert direct.lustre_root == ptm_stage.DEFAULT_LUSTRE_ROOT
    assert direct.physical_lustre_mount is None

    mount = tmp_path / "sshfs-lustre"
    mapped = ptm_stage._parser().parse_args(
        ("--physical-lustre-mount", str(mount))
    )
    assert mapped.lustre_root == ptm_stage.DEFAULT_LUSTRE_ROOT
    assert mapped.physical_lustre_mount == mount
