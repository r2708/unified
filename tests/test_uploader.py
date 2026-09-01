"""Parallel upload/verify against the mock hub: idempotent skips, failure
propagation, and checksum verification."""

import pytest

from ucc.hashing import sha256_file
from ucc.hf_remote import MockHub
from ucc.uploader import (
    UploadFailure,
    VerificationFailure,
    upload_outputs,
    verify_outputs,
)


def _make_outputs(tmp_path, n=5):
    outputs = []
    for i in range(n):
        local = tmp_path / "local" / f"part-{i}.parquet"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(f"payload-{i}".encode() * (i + 1))
        outputs.append({
            "local": str(local),
            "dest": f"data/full/src/part-{i}.parquet",
            "sha256": sha256_file(local),
            "size": local.stat().st_size,
            "records": 1,
            "subset": "full",
        })
    return outputs


def test_upload_verify_and_idempotent_skip(tmp_path):
    hub = MockHub(tmp_path / "hub")
    hub.ensure_repo()
    outputs = _make_outputs(tmp_path)

    assert upload_outputs(hub, outputs, "shard-1", workers=3) == len(outputs)
    verify_outputs(hub, outputs, workers=3)
    # Second run: everything already present-and-matching -> nothing moves.
    assert upload_outputs(hub, outputs, "shard-1", workers=3) == 0


def test_verify_fails_on_corrupted_remote(tmp_path):
    hub = MockHub(tmp_path / "hub")
    hub.ensure_repo()
    outputs = _make_outputs(tmp_path, n=3)
    upload_outputs(hub, outputs, "shard-1", workers=2)
    (tmp_path / "hub" / outputs[1]["dest"]).write_bytes(b"tampered")
    with pytest.raises(VerificationFailure) as exc:
        verify_outputs(hub, outputs, workers=2)
    assert outputs[1]["dest"] in str(exc.value)
    # The tampered file no longer matches -> it is re-uploaded, others skip.
    assert upload_outputs(hub, outputs, "shard-1", workers=2) == 1
    verify_outputs(hub, outputs, workers=2)


def test_missing_local_file_raises(tmp_path):
    hub = MockHub(tmp_path / "hub")
    hub.ensure_repo()
    outputs = _make_outputs(tmp_path, n=3)
    outputs[2]["local"] = str(tmp_path / "local" / "gone.parquet")
    with pytest.raises(UploadFailure):
        upload_outputs(hub, outputs, "shard-1", workers=3)
    # The other files still landed and remain verifiable.
    verify_outputs(hub, outputs[:2], workers=2)
