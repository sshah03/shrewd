"""`load("https://.../panel.tar.gz")` downloads a project tarball once and loads it."""
import io
import json
import tarfile

import pytest

from shrewd import students


def _tarball(tmp_path, wrap=None):
    src = tmp_path / "src"
    (src / "student").mkdir(parents=True)
    (src / "manifest.json").write_text(json.dumps({"kind": "decisions"}))
    (src / "student" / "meta.json").write_text(json.dumps({"type": "nonsense"}))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(src, arcname=wrap or ".")
    out = tmp_path / "panel.tar.gz"
    out.write_bytes(buf.getvalue())
    return out


@pytest.mark.parametrize("wrap", [None, "panel-x"])
def test_url_is_fetched_once_and_unpacked(tmp_path, monkeypatch, wrap):
    archive = _tarball(tmp_path, wrap)
    calls = []

    def fake_retrieve(url, dest):
        calls.append(url)
        dest.write_bytes(archive.read_bytes())

    monkeypatch.setattr("urllib.request.urlretrieve", fake_retrieve)
    monkeypatch.setenv("SHREWD_CACHE", str(tmp_path / "cache"))
    url = "https://example.test/panel-x.tar.gz"
    with pytest.raises(ValueError, match="unknown student type"):   # got as far as the student
        students.load(url)
    with pytest.raises(ValueError, match="unknown student type"):
        students.load(url)
    assert calls == [url]                                             # second load: cache hit
    assert (students.fetch_project(url) / "manifest.json").exists()


def test_local_paths_never_download(tmp_path, monkeypatch):
    def boom(url, dest):
        raise AssertionError("no download for a local path")

    monkeypatch.setattr("urllib.request.urlretrieve", boom)
    with pytest.raises(FileNotFoundError):
        students.load(tmp_path / "missing")
