"""Adding what ``setup`` installed to the config file ``settings.py`` reads.

Every existing key and value is kept; only missing keys are added (``--lan`` and ``--port``
set theirs because the user asked for them). The file is written atomically with mode 0600
and validated by the server's own loader before it replaces the old one. The token is
never printed: errors name the key, never the value.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import secrets
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sfx_api.settings import _TOKEN, ENV_PREFIX, PeerServer, SettingsError, load_settings
from sfx_api.setup.process import SetupError

LAN_HOST = "0.0.0.0"  # noqa: S104 - --lan asks for exactly this
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


# -- a TOML emitter for the subset a config uses --------------------------------------------
def _key(name: str) -> str:
    return name if _BARE_KEY.fullmatch(name) else _string(name)


def _string(value: str) -> str:
    out = []
    for char in value:
        if char in ('"', "\\"):
            out.append("\\" + char)
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(char)
    return '"' + "".join(out) + '"'


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cannot write a non-finite float")
        return repr(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return "[" + ", ".join(_string(v) for v in value) + "]"
    raise ValueError(f"cannot write a {type(value).__name__} value")


def _is_table_array(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, dict) for v in value)


def _emit_table(lines: list[str], path: Sequence[str], table: Mapping[str, Any]) -> None:
    for name, value in table.items():
        if not isinstance(value, dict) and not _is_table_array(value):
            lines.append(f"{_key(name)} = {_value(value)}")
    for name, value in table.items():
        child = [*path, _key(name)]
        if isinstance(value, dict):
            scalars = any(
                not isinstance(v, dict) and not _is_table_array(v) for v in value.values()
            )
            if scalars or not value:
                lines.extend(["", f"[{'.'.join(child)}]"])
            _emit_table(lines, child, value)
        elif _is_table_array(value):
            for item in value:
                lines.extend(["", f"[[{'.'.join(child)}]]"])
                _emit_table(lines, child, item)


def dumps(data: Mapping[str, Any]) -> str:
    lines: list[str] = []
    _emit_table(lines, [], data)
    return "\n".join(lines).lstrip("\n") + "\n"


# -- merging ----------------------------------------------------------------------------------
def add_missing(existing: Mapping[str, Any], additions: Mapping[str, Any]) -> dict[str, Any]:
    """``existing`` with the keys of ``additions`` it lacks; tables merge, values stay."""
    merged = dict(existing)
    for key, value in additions.items():
        current = merged.get(key)
        if key not in merged:
            merged[key] = value
        elif isinstance(current, dict) and isinstance(value, dict):
            merged[key] = add_missing(current, value)
    return merged


@dataclass(frozen=True)
class Peer:
    name: str
    health_url: str


def parse_peer(text: str) -> Peer:
    name, sep, url = text.partition("=")
    if not sep or not name or not url:
        raise ValueError(f"--coordinate-with wants NAME=HEALTH_URL, got {text!r}")
    return Peer(name, url)


def peer_token_env(name: str) -> str:
    return f"{ENV_PREFIX}COORD_TOKEN_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper()


@dataclass(frozen=True)
class ConfigRequest:
    backends: Mapping[str, Mapping[str, Any]]
    lan: bool = False
    port: int | None = None
    peers: Sequence[Peer] = ()


def plan_config(
    existing: Mapping[str, Any], request: ConfigRequest, env: Mapping[str, str]
) -> dict[str, Any]:
    data = add_missing(existing, {"backends": request.backends} if request.backends else {})
    if request.port is not None:
        data["port"] = request.port
    if request.lan:
        data["host"] = LAN_HOST
        data["token"] = _lan_token(data.get("token"), env)
    if request.peers:
        data["coordinate_with"] = _with_peers(data.get("coordinate_with", []), request.peers, env)
    return data


def env_token(env: Mapping[str, str]) -> str | None:
    """``$SFX_API_TOKEN`` when set; ValueError (naming the variable only) when it is invalid."""
    from_env = env.get(ENV_PREFIX + "TOKEN", "").strip()
    if from_env and not _TOKEN.fullmatch(from_env):
        raise ValueError(
            f"{ENV_PREFIX}TOKEN must be at least 32 printable ASCII characters without spaces"
        )
    return from_env or None


def check_peer(peer: Peer, env: Mapping[str, str]) -> None:
    """The server's own rules for a ``[[coordinate_with]]`` entry; ValueError without values."""
    data: dict[str, Any] = {"name": peer.name, "health_url": peer.health_url}
    token = env.get(peer_token_env(peer.name), "").strip()
    if token:
        data["token"] = token
    try:
        PeerServer.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ValueError(f"--coordinate-with {peer.name[:64]!r}: {problems}") from exc


def _lan_token(existing: Any, env: Mapping[str, str]) -> str:
    try:
        from_env = env_token(env)
    except ValueError as exc:
        raise SetupError(str(exc)) from exc
    if from_env:
        return from_env
    if isinstance(existing, str) and existing:
        return existing
    return secrets.token_urlsafe(32)


def _with_peers(
    existing: Any, peers: Sequence[Peer], env: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Add each peer, or give the entry of that name the new URL (and the token from the
    environment, when there is one)."""
    entries = [dict(e) for e in existing] if isinstance(existing, list) else []
    for peer in peers:
        token = env.get(peer_token_env(peer.name), "").strip()
        update: dict[str, Any] = {"health_url": peer.health_url}
        if token:
            update["token"] = token
        index = next((i for i, e in enumerate(entries) if e.get("name") == peer.name), None)
        if index is None:
            entries.append({"name": peer.name, **update})
        else:
            entries[index] = {**entries[index], **update}
    return entries


# -- reading and writing --------------------------------------------------------------------
def read_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SetupError(f"cannot read config file {path}: {exc}") from exc


def write_config(path: Path, data: Mapping[str, Any]) -> None:
    """Validate with the server's loader, then replace ``path`` atomically (0600)."""
    parent = path.parent
    if not parent.exists():
        parent.mkdir(mode=0o700, parents=True)
    try:
        text = dumps(data)
    except ValueError as exc:
        raise SetupError(f"cannot write {path}: {exc}") from exc
    # mkstemp creates the file 0600 next to the target, so the rename is atomic
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        try:
            load_settings(env={}, config_path=tmp)
        except SettingsError as exc:
            raise SetupError(f"the new config would not load ({exc}); {path} is unchanged") from exc
        tmp.replace(path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
