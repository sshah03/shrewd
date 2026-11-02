"""A checkpoint write that fails (a full disk) leaves the previous epoch and no partial file:
a fine-tune's state is gigabytes. Needs torch, but no model."""

import pytest

torch = pytest.importorskip("torch")

from shrewd.encoder import Checkpoint  # noqa: E402


def test_a_failed_write_leaves_the_previous_epoch_and_no_partial(tmp_path, monkeypatch):
    checkpoint = Checkpoint(tmp_path, "fit")
    checkpoint.save({"epoch": 1})

    def disk_full(state, path):
        path.write_bytes(b"half a checkpoint")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(torch, "save", disk_full)
    with pytest.raises(OSError):
        checkpoint.save({"epoch": 2})
    monkeypatch.undo()

    assert not (tmp_path / "fit.partial").exists()
    assert checkpoint.load() == {"epoch": 1}


def test_clear_removes_a_partial_left_by_a_killed_write(tmp_path):
    checkpoint = Checkpoint(tmp_path, "fit")
    checkpoint.save({"epoch": 1})
    (tmp_path / "fit.partial").write_bytes(b"from a process that was killed mid-write")
    checkpoint.clear()
    assert list(tmp_path.iterdir()) == []
