"""Data-only OneFormer PTM staging contract tests."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from tao_automl.ptm_preflight import (
    ArtifactCacheError,
    AtomicArtifactCache,
    NGCCredential,
    NGCHTTPSClient,
)
from tao_automl.ptm_registry import (
    PTMRegistry,
    canonical_sha256,
    load_ptm_registry,
)

from . import campaign_contract, ptm_stage


SECRET = "unit-test-ngc-secret"
CANONICAL_ROOT = "/lustre/remote/project/oneformer-v1"


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


def _fixture_snapshot(registry: PTMRegistry) -> dict:
    model = registry.to_dict()["models"]["oneformer"]
    records = sorted(model["checkpoints"], key=lambda item: item["id"])
    return {
        "registry_version": registry.registry_version,
        "registry_sha256": registry.document_sha256,
        "default_ptm": model["default_ptm"],
        "records": [
            {
                "id": record["id"],
                "registry_record_sha256": canonical_sha256(record),
            }
            for record in records
        ],
        "record_count": len(records),
        "supported_ids": [],
        "unverified_ids": [record["id"] for record in records],
    }


def _registry_and_payloads(
    monkeypatch: pytest.MonkeyPatch,
    *,
    corrupt_registered_checksum: bool = False,
) -> tuple[PTMRegistry, dict[str, bytes]]:
    document = load_ptm_registry().to_dict()
    records = document["models"]["oneformer"]["checkpoints"]
    payload_by_id = {
        record["id"]: f"verified:{record['id']}".encode("utf-8")
        for record in records
    }
    for index, record in enumerate(records):
        payload = payload_by_id[record["id"]]
        source = record["source"]
        source.update(
            {
                "resource": f"oneformer_fixture_{index}",
                "version": f"fixture_v{index + 1}.0",
                "member": (
                    f"checkpoints/fixture_{index}=oneformer.pth"
                ),
            }
        )
        source["immutable_identity"] = (
            f"ngc://{source['registry']}/{source['resource']}:"
            f"{source['version']}#{source['member']}"
        )
        record["expected_size_bytes"] = len(payload)
        record["sha256"] = hashlib.sha256(payload).hexdigest()
    if corrupt_registered_checksum:
        records[1]["sha256"] = "0" * 64
    registry = PTMRegistry(document)
    snapshot = _fixture_snapshot(registry)
    monkeypatch.setattr(
        campaign_contract,
        "oneformer_registry_snapshot",
        lambda: copy.deepcopy(snapshot),
    )
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
    os.chmod(root, 0o755)
    for path in root.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            os.chmod(path, 0o755)


def _stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    PTMRegistry,
    _Session,
    ptm_stage.PublicationRoot,
    Path,
    dict,
]:
    registry, payloads = _registry_and_payloads(monkeypatch)
    session = _Session(payloads)
    publisher = ptm_stage.PublicationRoot(
        tmp_path / "sshfs" / "remote-lustre" / "oneformer",
        CANONICAL_ROOT,
    )
    manifest_path = tmp_path / "evidence" / "ptm_stage_manifest.json"
    summary = ptm_stage.stage_official_ptms(
        registry=registry,
        cache=AtomicArtifactCache(tmp_path / "cache"),
        ngc_client=_client(session),
        publisher=publisher,
        manifest_path=manifest_path,
    )
    return registry, session, publisher, manifest_path, summary


def test_all_four_registry_ptms_publish_to_physical_mount_with_canonical_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    registry, session, publisher, manifest_path, first = _stage(
        tmp_path,
        monkeypatch,
    )
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
        expected_ids = [
            record["id"]
            for record in sorted(
                registry.to_dict()["models"]["oneformer"]["checkpoints"],
                key=lambda item: item["id"],
            )
        ]
        assert first["checkpoint_ids"] == expected_ids
        assert first["physical_publication_root"] == str(
            publisher.physical_root
        )
        assert first["canonical_publication_root"] == CANONICAL_ROOT
        assert manifest["publication"] == {
            "canonical_root": CANONICAL_ROOT,
            "manifest_path": (
                f"{CANONICAL_ROOT}/{ptm_stage.REMOTE_MANIFEST_NAME}"
            ),
            "physical_root_recorded": False,
            "path_contract": (
                "canonical_paths_for_cluster_runtime;"
                "physical_root_is_stage_and_verification_only"
            ),
        }
        assert str(publisher.physical_root) not in manifest_text
        assert [row["id"] for row in manifest["checkpoints"]] == expected_ids
        for row in manifest["checkpoints"]:
            assert row["path"].startswith(f"{CANONICAL_ROOT}/")
            physical = publisher.physical_for_canonical(row["path"])
            assert physical.is_file()
            assert physical.stat().st_size == row["size_bytes"]
            assert not physical.stat().st_mode & 0o222
            assert row["mode"] == "444"
            assert row["checkpoint_spec_file"]["source"] == "repository"
            assert row["checkpoint_spec_file"]["path"].startswith(
                "data/ptm_specs/oneformer/"
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
        assert not publisher.manifest_path.physical.stat().st_mode & 0o222
        assert not publisher.physical_root.stat().st_mode & 0o222
        assert len(session.download_calls) == 4
        assert SECRET not in json.dumps(first, sort_keys=True)

        checked = ptm_stage.verify_staged_ptms(
            registry=registry,
            publisher=publisher,
            manifest_path=manifest_path,
        )
        assert checked["all_artifacts_verified"] is True
        assert checked["execution"]["network_accessed"] is False

        second = ptm_stage.stage_official_ptms(
            registry=registry,
            cache=AtomicArtifactCache(tmp_path / "cache"),
            ngc_client=_client(session),
            publisher=publisher,
            manifest_path=manifest_path,
        )
        assert second["manifest_sha256"] == first["manifest_sha256"]
        assert second["local_manifest_reused"] is True
        assert all(second["cache_hits"].values())
        assert all(second["publication_reuse"].values())
        assert len(session.download_calls) == 4
    finally:
        _restore_write_bits(publisher.physical_root)


def test_registered_checksum_failure_never_emits_completed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    registry, payloads = _registry_and_payloads(
        monkeypatch,
        corrupt_registered_checksum=True,
    )
    session = _Session(payloads)
    publisher = ptm_stage.PublicationRoot(
        tmp_path / "sshfs" / "oneformer",
        CANONICAL_ROOT,
    )
    manifest = tmp_path / "evidence" / "ptm_stage_manifest.json"
    try:
        with pytest.raises(ArtifactCacheError, match="SHA-256"):
            ptm_stage.stage_official_ptms(
                registry=registry,
                cache=AtomicArtifactCache(tmp_path / "cache"),
                ngc_client=_client(session),
                publisher=publisher,
                manifest_path=manifest,
            )
        assert not manifest.exists()
        assert not publisher.manifest_path.physical.exists()
    finally:
        _restore_write_bits(publisher.physical_root)


def test_registry_drift_is_rejected_before_network_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    registry, payloads = _registry_and_payloads(monkeypatch)
    changed = registry.to_dict()
    changed["models"]["oneformer"]["checkpoints"][0]["backbone"] = "changed"
    drifted = PTMRegistry(changed)
    session = _Session(payloads)
    with pytest.raises(ptm_stage.PTMStageError, match="exact frozen"):
        ptm_stage.stage_official_ptms(
            registry=drifted,
            cache=AtomicArtifactCache(tmp_path / "cache"),
            ngc_client=_client(session),
            publisher=ptm_stage.PublicationRoot(
                tmp_path / "sshfs" / "oneformer",
                CANONICAL_ROOT,
            ),
            manifest_path=tmp_path / "manifest.json",
        )
    assert session.calls == []


def test_unexpected_physical_content_fails_before_network_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    registry, payloads = _registry_and_payloads(monkeypatch)
    session = _Session(payloads)
    physical = tmp_path / "sshfs" / "oneformer"
    physical.mkdir(parents=True)
    (physical / "unregistered.bin").write_bytes(b"unexpected")
    with pytest.raises(ptm_stage.PTMStageError, match="unexpected file"):
        ptm_stage.stage_official_ptms(
            registry=registry,
            cache=AtomicArtifactCache(tmp_path / "cache"),
            ngc_client=_client(session),
            publisher=ptm_stage.PublicationRoot(
                physical,
                CANONICAL_ROOT,
            ),
            manifest_path=tmp_path / "manifest.json",
        )
    assert session.calls == []


@pytest.mark.parametrize("mutation", ["writable", "content"])
def test_final_stage_rejects_writable_or_changed_physical_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
):
    registry, _, publisher, manifest_path, _ = _stage(
        tmp_path,
        monkeypatch,
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        target = publisher.physical_for_canonical(
            manifest["checkpoints"][0]["path"]
        )
        if mutation == "writable":
            os.chmod(target, 0o644)
        else:
            os.chmod(target.parent, 0o755)
            os.chmod(target, 0o644)
            target.write_bytes(b"changed")
            os.chmod(target, 0o444)
        with pytest.raises(
            ptm_stage.PTMStageError,
            match="writable|identity differs",
        ):
            ptm_stage.verify_staged_ptms(
                registry=registry,
                publisher=publisher,
                manifest_path=manifest_path,
            )
    finally:
        _restore_write_bits(publisher.physical_root)


def test_semantically_changed_manifest_is_rejected_after_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    registry, _, publisher, manifest_path, _ = _stage(
        tmp_path,
        monkeypatch,
    )
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        value["checkpoints"][0]["path"] = (
            f"{CANONICAL_ROOT}/wrong/checkpoint.pth"
        )
        value.pop("manifest_sha256")
        value["manifest_sha256"] = canonical_sha256(value)
        with pytest.raises(ptm_stage.PTMStageError, match="identity changed"):
            ptm_stage.validate_stage_manifest(
                value,
                registry=registry,
                canonical_root=publisher.canonical_root,
            )
    finally:
        _restore_write_bits(publisher.physical_root)


def test_physical_and_canonical_roots_and_member_paths_fail_closed(
    tmp_path: Path,
):
    with pytest.raises(ptm_stage.PTMStageError, match="canonical"):
        ptm_stage.PublicationRoot(tmp_path / "physical", "/tmp/not-lustre")
    with pytest.raises(ptm_stage.PTMStageError, match="physical"):
        ptm_stage.PublicationRoot("relative", CANONICAL_ROOT)
    publisher = ptm_stage.PublicationRoot(
        tmp_path / "physical",
        CANONICAL_ROOT,
    )
    for member in ("../escape.pth", "/absolute.pth", "bad name.pth"):
        with pytest.raises(ptm_stage.PTMStageError, match="safe relative"):
            publisher.checkpoint_path("oneformer.valid", member)
    with pytest.raises(ptm_stage.PTMStageError, match="outside"):
        publisher.physical_for_canonical("/lustre/other/checkpoint.pth")


def test_check_stage_cli_constructs_no_ngc_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    registry, _, publisher, manifest_path, _ = _stage(
        tmp_path,
        monkeypatch,
    )
    try:
        registry_path = tmp_path / "registry.json"
        registry_path.write_text(
            json.dumps(registry.to_dict()),
            encoding="utf-8",
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("check-stage must not construct NGC")

        monkeypatch.setattr(ptm_stage, "NGCHTTPSClient", forbidden)
        assert (
            ptm_stage.main(
                [
                    "--check-stage",
                    "--registry",
                    str(registry_path),
                    "--physical-publication-root",
                    str(publisher.physical_root),
                    "--canonical-publication-root",
                    CANONICAL_ROOT,
                    "--manifest",
                    str(manifest_path),
                ]
            )
            == 0
        )
        output = json.loads(capsys.readouterr().out)
        assert output["execution"]["network_accessed"] is False
    finally:
        _restore_write_bits(publisher.physical_root)


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


def test_stager_has_no_model_scheduler_or_cpu_smoke_execution_path():
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
        module is None
        or (
            not module.startswith("nvidia_tao_pytorch")
            and not module.startswith("tao_sdk")
        )
        for module in imported_from
    )
    assert "create_job" not in source
    assert "sbatch" not in source
    assert "srun" not in source
