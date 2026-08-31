import pytest

from ucc.config import load_config


def _write_cfg(tmp_path, shard_yaml: str):
    path = tmp_path / "cfg.yaml"
    path.write_text(f"shard:\n{shard_yaml}\n", encoding="utf-8")
    return path


def test_gb_keys_convert_to_bytes(tmp_path):
    cfg = load_config(_write_cfg(tmp_path, "  target_gb: 1.5\n  min_gb: 0.5\n  max_gb: 2"))
    assert cfg.shard.target_bytes == 1_500_000_000
    assert cfg.shard.min_bytes == 500_000_000
    assert cfg.shard.max_bytes == 2_000_000_000
    assert "target_gb" not in cfg["shard"]  # canonicalized away


def test_gb_and_byte_forms_hash_identically(tmp_path):
    gb_form = load_config(_write_cfg(tmp_path, "  target_gb: 1.5"))
    (tmp_path / "cfg.yaml").unlink()
    byte_form = load_config(_write_cfg(tmp_path, "  target_bytes: 1500000000"))
    assert gb_form.config_hash == byte_form.config_hash


def test_batch_size_is_hash_exempt(tmp_path):
    small = tmp_path / "small.yaml"
    small.write_text("processing:\n  batch_size: 2048\n", encoding="utf-8")
    large = tmp_path / "large.yaml"
    large.write_text("processing:\n  batch_size: 500000\n", encoding="utf-8")
    a, b = load_config(small), load_config(large)
    assert a.config_hash == b.config_hash          # chunking never gates resume
    assert b.processing.batch_size == 500000        # ...but the value applies


def test_invalid_sizing_rejected(tmp_path):
    with pytest.raises(SystemExit, match="exceeds max"):
        load_config(_write_cfg(tmp_path, "  target_gb: 5"))  # default max is 2 GB
    with pytest.raises(SystemExit, match="larger than target"):
        load_config(_write_cfg(tmp_path, "  target_gb: 1.5\n  min_gb: 1.8\n  max_gb: 2"))
