from __future__ import annotations

import os
from pathlib import Path

import yaml

from src.secrets import apply_mt5_credentials, load_dotenv


def test_load_dotenv_does_not_override(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("MT5_LOGIN=111\nMT5_PASSWORD=from-file\n", encoding="utf-8")
    monkeypatch.setenv("MT5_LOGIN", "999")
    monkeypatch.delenv("MT5_PASSWORD", raising=False)
    assert load_dotenv(env_file) is True
    assert os.environ["MT5_LOGIN"] == "999"
    assert os.environ["MT5_PASSWORD"] == "from-file"


def test_config_password_ignored(tmp_path: Path, caplog):
    cfg = {"mt5": {"login": 4242, "password": "leaked", "server": "FxPro-Demo"}}
    with caplog.at_level("WARNING"):
        apply_mt5_credentials(cfg, config_dir=tmp_path, environ={})
    assert cfg["mt5"]["password"] == ""
    assert cfg["mt5"]["login"] == 0
    assert any("password" in r.message.lower() for r in caplog.records)
    assert any("login" in r.message.lower() for r in caplog.records)


def test_secrets_yaml_then_env_override(tmp_path: Path):
    (tmp_path / "secrets.yaml").write_text(
        yaml.safe_dump({"mt5": {"login": 100, "password": "file-secret"}}),
        encoding="utf-8",
    )
    cfg = {"mt5": {"server": "FxPro-Demo"}}
    apply_mt5_credentials(cfg, config_dir=tmp_path, environ={})
    assert cfg["mt5"]["login"] == 100
    assert cfg["mt5"]["password"] == "file-secret"

    apply_mt5_credentials(
        cfg,
        config_dir=tmp_path,
        environ={"MT5_LOGIN": "200", "MT5_PASSWORD": "env-secret", "MT5_SERVER": "FxPro-Live"},
    )
    assert cfg["mt5"]["login"] == 200
    assert cfg["mt5"]["password"] == "env-secret"
    assert cfg["mt5"]["server"] == "FxPro-Live"


def test_attach_mode_when_no_secrets(tmp_path: Path):
    cfg = {"mt5": {"server": "FxPro-Demo"}}
    apply_mt5_credentials(cfg, config_dir=tmp_path, environ={})
    assert cfg["mt5"]["login"] == 0
    assert cfg["mt5"].get("password", "") == ""
