"""meta.json must keep a generated entrypoint so cloud reload runs build.sh."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cloud_reload_entrypoint_is_not_committed_run_sh():
    meta = json.loads((ROOT / "meta.json").read_text())
    assert meta["entrypoint"] == "start"
    assert meta["first_run"] == "first_run.sh"
    assert meta["build"]["build"] == "./build.sh"
    assert meta["build"]["path"] == "module.tar.gz"
    assert not (ROOT / "start").exists() or "start" in (ROOT / ".gitignore").read_text()
    gitignore = (ROOT / ".gitignore").read_text()
    assert "/start" in gitignore
    assert (ROOT / "first_run.sh").is_file()
    assert (ROOT / "run.sh").is_file()
