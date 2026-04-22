"""Tests for ServerConfig."""

import os
import pytest
from nve.serve.config import ServerConfig


def test_defaults():
    cfg = ServerConfig()
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8000
    assert cfg.workers == 1
    assert cfg.max_batch_size == 8
    assert cfg.default_max_tokens == 512
    assert 0.0 < cfg.default_temperature <= 2.0
    assert 0.0 < cfg.default_top_p <= 1.0


def test_from_env(monkeypatch):
    monkeypatch.setenv("NVE_HOST", "127.0.0.1")
    monkeypatch.setenv("NVE_PORT", "9000")
    monkeypatch.setenv("NVE_WORKERS", "4")
    monkeypatch.setenv("NVE_LOG_LEVEL", "debug")
    cfg = ServerConfig.from_env()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 9000
    assert cfg.workers == 4
    assert cfg.log_level == "debug"


def test_from_env_defaults(monkeypatch):
    # Ensure missing env vars fall back to defaults.
    for key in ["NVE_HOST", "NVE_PORT", "NVE_WORKERS"]:
        monkeypatch.delenv(key, raising=False)
    cfg = ServerConfig.from_env()
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8000
