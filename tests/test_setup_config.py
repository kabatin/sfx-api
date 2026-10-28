"""The config writer: TOML emitter round trip, merge rules, atomic 0600 writes."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from sfx_api.setup.config_write import (
    ConfigRequest,
    Peer,
    add_missing,
    dumps,
    peer_token_env,
    plan_config,
    write_config,
)
from sfx_api.setup.process import SetupError


def test_dumps_round_trips_the_subset() -> None:
    data = {
        "host": "127.0.0.1",
        "port": 8769,
        "settle_seconds": 30.5,
        "flag": False,
        "names": ["a", 'b"c', "d\\e"],
        "odd": 'line\nbreak\t"quoted"\x7f',
        "backends": {
            "sa3-medium": {"mlx_dir": "/x/optimized/mlx", "negative_prompt_cfg": 3.0},
            "moss": {"steps": 50},
        },
        "coordinate_with": [
            {"name": "vpipe-api", "health_url": "http://127.0.0.1:8765/v1/health"},
            {"name": "ace", "health_url": "http://127.0.0.1:8768/v1/health", "token": "t"},
        ],
        "has space": "quoted key",
        "empty": {},
    }
    assert tomllib.loads(dumps(data)) == data


@pytest.mark.parametrize("value", [float("inf"), float("nan"), [1, 2], object()])
def test_dumps_refuses_what_it_cannot_write(value: object) -> None:
    with pytest.raises(ValueError, match="cannot write"):
        dumps({"x": value})


def test_add_missing_never_replaces() -> None:
    existing = {"port": 9000, "backends": {"sa3": {"hf_home": "/mine"}}}
    additions = {"port": 1, "backends": {"sa3": {"hf_home": "/new", "mlx_dir": "/m"}, "moss": {}}}
    assert add_missing(existing, additions) == {
        "port": 9000,
        "backends": {"sa3": {"hf_home": "/mine", "mlx_dir": "/m"}, "moss": {}},
    }
    assert existing == {"port": 9000, "backends": {"sa3": {"hf_home": "/mine"}}}


def test_lan_keeps_an_existing_token() -> None:
    existing = {"token": "k" * 40}
    data = plan_config(existing, ConfigRequest(backends={}, lan=True), env={})
    assert data["token"] == "k" * 40 and data["host"] == "0.0.0.0"


def test_lan_env_token_wins_and_is_validated() -> None:
    env = {"SFX_API_TOKEN": "e" * 32}
    data = plan_config({"token": "k" * 40}, ConfigRequest(backends={}, lan=True), env)
    assert data["token"] == "e" * 32
    with pytest.raises(SetupError) as info:
        plan_config({}, ConfigRequest(backends={}, lan=True), {"SFX_API_TOKEN": "e" * 31})
    assert "e" * 31 not in str(info.value)
    with pytest.raises(SetupError):
        plan_config({}, ConfigRequest(backends={}, lan=True), {"SFX_API_TOKEN": "e " * 20})


def test_generated_tokens_differ() -> None:
    first = plan_config({}, ConfigRequest(backends={}, lan=True), {})["token"]
    second = plan_config({}, ConfigRequest(backends={}, lan=True), {})["token"]
    assert first != second and len(first) >= 32


def test_peer_token_env_name() -> None:
    assert peer_token_env("vpipe-api") == "SFX_API_COORD_TOKEN_VPIPE_API"
    assert peer_token_env("a.b_c") == "SFX_API_COORD_TOKEN_A_B_C"


def test_existing_peer_is_left_alone() -> None:
    existing = {"coordinate_with": [{"name": "vpipe-api", "health_url": "http://old/v1/health"}]}
    request = ConfigRequest(backends={}, peers=[Peer("vpipe-api", "http://new/v1/health")])
    data = plan_config(existing, request, {"SFX_API_COORD_TOKEN_VPIPE_API": "t"})
    assert data["coordinate_with"] == existing["coordinate_with"]


def test_write_is_0600_and_validated(tmp_path: Path) -> None:
    path = tmp_path / "new" / "config.toml"
    write_config(path, {"port": 8770})
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert tomllib.loads(path.read_text()) == {"port": 8770}
    # a config the server would refuse never replaces the old one
    with pytest.raises(SetupError, match="would not load"):
        write_config(path, {"port": 70000})
    assert tomllib.loads(path.read_text()) == {"port": 8770}
    assert sorted(p.name for p in path.parent.iterdir()) == ["config.toml"]


def test_write_does_not_echo_a_bad_token(tmp_path: Path) -> None:
    with pytest.raises(SetupError) as info:
        write_config(tmp_path / "c.toml", {"token": "secret-but-short"})
    assert "secret-but-short" not in str(info.value)
