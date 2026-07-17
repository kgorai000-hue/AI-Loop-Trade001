from __future__ import annotations

from pathlib import Path

import yaml

from main import load_config
from src.config_paths import normalize_config_paths, resolve_under


def test_resolve_under_joins_relative(tmp_path: Path):
    assert resolve_under("state", tmp_path) == (tmp_path / "state").resolve()
    abs_state = (tmp_path / "elsewhere").resolve()
    assert resolve_under(abs_state, tmp_path) == abs_state


def test_normalize_config_paths_rewrites_relative(tmp_path: Path):
    cfg = {
        "paths": {"state_dir": "state"},
        "loop": {"log_dir": "logs"},
        "mt5": {"path": "terminals\\fxpro\\terminal64.exe"},
    }
    out = normalize_config_paths(cfg, tmp_path)
    assert out["paths"]["state_dir"] == str((tmp_path / "state").resolve())
    assert out["loop"]["log_dir"] == str((tmp_path / "logs").resolve())
    assert out["mt5"]["path"] == str(
        (tmp_path / "terminals" / "fxpro" / "terminal64.exe").resolve()
    )


def test_normalize_keeps_absolute_and_empty_mt5(tmp_path: Path):
    abs_state = (tmp_path / "abs_state").resolve()
    abs_logs = (tmp_path / "abs_logs").resolve()
    abs_mt5 = (tmp_path / "t64.exe").resolve()
    cfg = {
        "paths": {"state_dir": str(abs_state)},
        "loop": {"log_dir": str(abs_logs)},
        "mt5": {"path": str(abs_mt5)},
    }
    normalize_config_paths(cfg, tmp_path / "ignored")
    assert cfg["paths"]["state_dir"] == str(abs_state)
    assert cfg["loop"]["log_dir"] == str(abs_logs)
    assert cfg["mt5"]["path"] == str(abs_mt5)

    cfg2 = {"mt5": {"path": ""}}
    normalize_config_paths(cfg2, tmp_path)
    assert cfg2["mt5"]["path"] == ""
    assert Path(cfg2["paths"]["state_dir"]).is_absolute()
    assert Path(cfg2["loop"]["log_dir"]).is_absolute()


def test_load_config_uses_config_parent_not_cwd(tmp_path: Path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    cfg_path = proj / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "paths": {"state_dir": "state"},
                "loop": {"log_dir": "logs"},
                "mt5": {"path": ""},
            }
        ),
        encoding="utf-8",
    )
    other = tmp_path / "other_cwd"
    other.mkdir()
    monkeypatch.chdir(other)

    loaded = load_config(cfg_path)
    assert loaded["paths"]["state_dir"] == str((proj / "state").resolve())
    assert loaded["loop"]["log_dir"] == str((proj / "logs").resolve())
    # Must not have created state under the foreign cwd.
    assert not (other / "state").exists()
