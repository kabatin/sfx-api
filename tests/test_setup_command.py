"""``sfx-api setup`` end to end against fakes: check, confirm, install, config, service."""

from __future__ import annotations

import json
import plistlib
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from sfx_api.cli import build_parser, main
from sfx_api.setup import pins
from sfx_api.setup.command import cmd_setup, run_setup
from tests.setup_fakes import APPLE, FakeHttp, FakeInstallers, deps, tree

SA3_SMALL = sum(pins.SA3_FILES["sm-sfx"].values())
SA3_MEDIUM = sum(pins.SA3_FILES["medium"].values())
SHARED = sum(pins.SA3_SHARED_FILES.values()) + pins.SA3_ENV_BYTES
MOSS = pins.MOSS_MODEL_BYTES + pins.MOSS_ENV_BYTES


def _args(tmp_path: Path, *extra: str):
    return build_parser().parse_args(["setup", "--prefix", str(tmp_path / "opt"), *extra])


def _setup(tmp_path: Path, *extra: str, **kw):
    d, lines = deps(tmp_path, **kw)
    code, report = run_setup(None, _args(tmp_path, *extra), d)
    return code, report.to_json(), d, lines


def _by_id(report: dict) -> dict[str, dict]:
    return {c["id"]: c for c in report["components"]}


def test_check_reports_missing_with_sizes_and_changes_nothing(tmp_path: Path) -> None:
    run = FakeInstallers()
    code, report, _, _ = _setup(tmp_path, "--check", run=run)
    assert code == 0 and report["ok"] and report["action"] == "check"
    c = _by_id(report)
    assert [x["status"] for x in report["components"]] == ["missing"] * 3
    # T5Gemma and the MLX environment are counted once, on the first SA3 component
    assert c["sa3-medium"]["downloadBytes"] == SA3_MEDIUM + SHARED
    assert c["sa3-small"]["downloadBytes"] == SA3_SMALL
    assert c["moss"]["downloadBytes"] == MOSS
    assert c["moss"]["fix"]["command"] == "sfx-api setup --models moss --yes"
    assert run.calls == []
    assert tree(tmp_path) == []
    assert report["service"] == {
        "installed": False,
        "label": "com.github.kabatin.sfx-api",
        "healthy": False,
    }


def test_check_of_one_component_counts_shared_files_for_it(tmp_path: Path) -> None:
    _, report, _, _ = _setup(tmp_path, "--check", "--models", "sa3-small")
    assert [c["id"] for c in report["components"]] == ["sa3-small"]
    assert report["components"][0]["downloadBytes"] == SA3_SMALL + SHARED


def test_unknown_component_is_a_bad_argument(tmp_path: Path) -> None:
    code, report, _, _ = _setup(tmp_path, "--check", "--models", "sa3,moss")
    assert code == 2 and not report["ok"]
    assert "sa3-medium, sa3-small, moss" in report["error"]


def test_unsupported_machine(tmp_path: Path) -> None:
    intel = replace(APPLE, arm64=False)
    code, report, _, _ = _setup(tmp_path, "--check", machine=intel)
    assert code == 0
    assert {c["status"] for c in report["components"]} == {"unsupported"}
    # asked for by name on such a Mac: a failure, and nothing is run
    run = FakeInstallers()
    code, report, _, _ = _setup(tmp_path, "--models", "moss", "--yes", machine=intel, run=run)
    assert code == 1 and run.calls == []


def test_moss_needs_32_gb(tmp_path: Path) -> None:
    small = replace(APPLE, ram_gb=16)
    _, report, _, _ = _setup(tmp_path, "--check", machine=small)
    c = _by_id(report)
    assert c["moss"]["status"] == "unsupported" and "32 GB" in c["moss"]["message"]
    assert c["sa3-medium"]["status"] == "missing"


def test_without_yes_and_no_terminal_nothing_happens(tmp_path: Path) -> None:
    run = FakeInstallers()
    code, report, _, _ = _setup(tmp_path, "--models", "sa3-small", "--lan", run=run)
    assert code == 3 and not report["ok"]
    fix = report["components"][0]["fix"]
    assert fix["needsHuman"]
    assert fix["command"].endswith(
        f"setup --models sa3-small --prefix {tmp_path / 'opt'} --lan --yes"
    )
    assert run.calls == [] and tree(tmp_path) == []


def test_terminal_prompt_shows_size_and_licence(tmp_path: Path) -> None:
    run = FakeInstallers()
    code, _, _, lines = _setup(tmp_path, "--models", "sa3-small", run=run, tty=True, answer="n")
    assert code == 3 and run.calls == []
    text = "\n".join(lines)
    assert "Stability AI Community License" in text and "GB" in text
    code, report, _, _ = _setup(
        tmp_path, "--models", "sa3-small", run=FakeInstallers(), tty=True, answer="y"
    )
    assert code == 0 and report["components"][0]["status"] == "installed"


def test_install_all_then_rerun_is_already(tmp_path: Path) -> None:
    run = FakeInstallers()
    code, report, d, _ = _setup(tmp_path, "--yes", run=run)
    assert code == 0 and report["ok"], report
    assert [c["status"] for c in report["components"]] == ["installed"] * 3
    opt = tmp_path / "opt"
    # one clone per repository, pinned, and one install.sh for both SA3 models
    clones = run.commands("git")
    assert [c[1] for c in clones] == ["clone", "-C", "clone", "-C"]
    assert pins.SA3_COMMIT in clones[1] and pins.MOSS_COMMIT in clones[3]
    (installer,) = run.commands("./install.sh")
    assert installer[installer.index("--download") + 1] == "medium,sm-sfx"
    assert run.envs[run.calls.index(installer)]["HF_HOME"] == str(opt / "hf")
    sa3_hf, moss_hf = run.commands("uvx")
    assert pins.MOSS_MODEL_REVISION in moss_hf and pins.HF_CLI in moss_hf
    pip = next(c for c in run.calls if c[:3] == ("uv", "pip", "install"))
    assert "torch==2.9.1" in pip
    assert run.envs[run.calls.index(pip)]["UV_EXCLUDE_NEWER"] == pins.UV_EXCLUDE_NEWER

    # SA3: the weights come at a pinned revision first, then install.sh links them offline
    assert run.calls.index(sa3_hf) < run.calls.index(installer)
    assert sa3_hf[sa3_hf.index("--revision") + 1] == pins.SA3_WEIGHTS_REVISION
    assert pins.SA3_WEIGHTS_REPO in sa3_hf and "--local-dir" not in sa3_hf
    wanted = {f"MLX/{name}" for b in pins.SA3_FILES.values() for name in b}
    assert set(sa3_hf) >= wanted | {f"MLX/{name}" for name in pins.SA3_SHARED_FILES}
    assert run.envs[run.calls.index(sa3_hf)]["HF_HOME"] == str(opt / "hf")
    installer_env = run.envs[run.calls.index(installer)]
    assert installer_env["HF_HUB_OFFLINE"] == "1"
    assert installer_env["UV_EXCLUDE_NEWER"] == pins.UV_EXCLUDE_NEWER
    ref = opt / "hf" / "hub" / "models--stabilityai--stable-audio-3-optimized" / "refs" / "main"
    assert ref.read_text() == pins.SA3_WEIGHTS_REVISION

    config_path = Path(d.env["SFX_API_CONFIG"])
    config = tomllib.loads(config_path.read_text())
    mlx = str(opt / "stable-audio-3" / "optimized" / "mlx")
    assert config["backends"]["sa3"] == {"mlx_dir": mlx, "hf_home": str(opt / "hf")}
    assert config["backends"]["sa3-medium"]["mlx_dir"] == mlx
    assert config["backends"]["moss"] == {
        "code_dir": str(opt / "moss" / "src"),
        "model_dir": str(opt / "moss" / "MOSS-SoundEffect-v2.0"),
    }
    assert config_path.stat().st_mode & 0o777 == 0o600
    assert config_path.parent.stat().st_mode & 0o777 == 0o700

    again = FakeInstallers()
    code, report, _, _ = _setup(tmp_path, "--yes", run=again)
    assert code == 0 and [c["status"] for c in report["components"]] == ["already"] * 3
    assert again.calls == []
    assert tomllib.loads(config_path.read_text()) == config


def test_existing_config_values_are_kept(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "config.toml"
    config_path.parent.mkdir()
    hf = tmp_path / "elsewhere" / "hf"
    config_path.write_text(
        f'port = 9000\nmax_waiting = 4\n\n[backends.sa3]\nhf_home = "{hf}"\n'
        "negative_prompt_cfg = 3.0\n"
    )
    run = FakeInstallers()
    code, report, _, _ = _setup(tmp_path, "--models", "sa3-small", "--yes", run=run)
    assert code == 0, report
    config = tomllib.loads(config_path.read_text())
    assert config["port"] == 9000 and config["max_waiting"] == 4
    assert config["backends"]["sa3"]["hf_home"] == str(hf)
    assert config["backends"]["sa3"]["negative_prompt_cfg"] == 3.0
    assert "mlx_dir" in config["backends"]["sa3"]
    # the configured weights directory is the one install.sh downloads into
    (installer,) = run.commands("./install.sh")
    assert run.envs[run.calls.index(installer)]["HF_HOME"] == str(hf)
    assert report["url"] == "http://127.0.0.1:9000"


def test_failed_step_fails_only_its_component(tmp_path: Path) -> None:
    run = FakeInstallers(fail={pins.MOSS_MODEL_REPO})
    code, report, d, _ = _setup(tmp_path, "--yes", run=run)
    assert code == 1 and not report["ok"]
    c = _by_id(report)
    assert c["moss"]["status"] == "failed" and "3 tries" in c["moss"]["message"]
    assert c["sa3-small"]["status"] == "installed"
    config = tomllib.loads(Path(d.env["SFX_API_CONFIG"]).read_text())
    assert set(config["backends"]) == {"sa3", "sa3-medium"}


def test_moved_commit_is_refused(tmp_path: Path) -> None:
    d, _ = deps(tmp_path)
    d = replace(d, head_commit=lambda _: "0" * 40)
    code, report = run_setup(None, _args(tmp_path, "--models", "moss", "--yes"), d)
    assert code == 1
    assert "expected " + pins.MOSS_COMMIT in report.components[0].message


def test_not_enough_disk(tmp_path: Path) -> None:
    tight = replace(APPLE, free_bytes=10 * 10**9)
    run = FakeInstallers()
    code, report, _, _ = _setup(tmp_path, "--models", "moss", "--yes", machine=tight, run=run)
    assert code == 1 and run.calls == []
    assert "not enough disk" in report["components"][0]["message"]


def test_missing_git_needs_a_human(tmp_path: Path) -> None:
    no_git = replace(APPLE, tools={"git": False, "uv": True})
    code, report, _, _ = _setup(tmp_path, "--models", "moss", "--yes", machine=no_git)
    assert code == 3
    c = report["components"][0]
    assert c["status"] == "needs_human" and c["fix"]["command"] == "xcode-select --install"


def test_lan_generates_a_token_and_never_prints_it(tmp_path: Path, capsys) -> None:
    d, lines = deps(tmp_path)
    code = cmd_setup(None, _args(tmp_path, "--models", "sa3-small", "--yes", "--lan", "--json"), d)
    assert code == 0
    config = tomllib.loads(Path(d.env["SFX_API_CONFIG"]).read_text())
    token = config["token"]
    assert config["host"] == "0.0.0.0" and len(token) >= 32
    out = capsys.readouterr()
    report = json.loads(out.out.strip().splitlines()[-1])
    assert len(out.out.strip().splitlines()) == 1
    assert report["lan"] and report["tokenSet"]
    assert token not in out.out + out.err + "\n".join(lines)


def test_lan_takes_the_token_from_the_environment(tmp_path: Path) -> None:
    token = "t" * 40
    _, report, d, _ = _setup(
        tmp_path, "--models", "sa3-small", "--yes", "--lan", env={"SFX_API_TOKEN": token}
    )
    assert tomllib.loads(Path(d.env["SFX_API_CONFIG"]).read_text())["token"] == token
    code, report, _, _ = _setup(
        tmp_path, "--models", "sa3-small", "--yes", "--lan", env={"SFX_API_TOKEN": "short"}
    )
    assert code == 1 and "SFX_API_TOKEN" in report["error"] and "short" not in report["error"]


def test_port_and_coordinate_with(tmp_path: Path) -> None:
    peer_token = "p" * 20
    args = (
        "--models", "sa3-small", "--yes", "--port", "8800",
        "--coordinate-with", "vpipe-api=http://127.0.0.1:8765/v1/health",
        "--coordinate-with", "ace-step-api=http://127.0.0.1:8768/v1/health",
    )  # fmt: skip
    env = {"SFX_API_COORD_TOKEN_VPIPE_API": peer_token}
    code, report, d, _ = _setup(tmp_path, *args, env=env)
    assert code == 0, report
    config = tomllib.loads(Path(d.env["SFX_API_CONFIG"]).read_text())
    assert config["port"] == 8800
    assert config["coordinate_with"] == [
        {"name": "vpipe-api", "health_url": "http://127.0.0.1:8765/v1/health", "token": peer_token},
        {"name": "ace-step-api", "health_url": "http://127.0.0.1:8768/v1/health"},
    ]
    _setup(tmp_path, *args, env=env)  # again: no duplicates
    config = tomllib.loads(Path(d.env["SFX_API_CONFIG"]).read_text())
    assert [p["name"] for p in config["coordinate_with"]] == ["vpipe-api", "ace-step-api"]


def test_bad_coordinate_with(tmp_path: Path) -> None:
    code, _, _, _ = _setup(tmp_path, "--coordinate-with", "no-url", "--check")
    assert code == 2
    code, report, _, _ = _setup(
        tmp_path, "--models", "sa3-small", "--yes", "--coordinate-with", "x=ftp://h/"
    )
    assert code == 1 and "coordinate_with" in report["error"]


def test_service_writes_the_plist_and_waits_for_health(tmp_path: Path) -> None:
    run = FakeInstallers()
    http = FakeHttp(statuses=[None, None, 200])
    code, report, d, _ = _setup(
        tmp_path, "--models", "sa3-small", "--yes", "--service", "--label-suffix", "test",
        run=run, http=http,
    )  # fmt: skip
    assert code == 0, report
    label = "com.github.kabatin.sfx-api.test"
    assert report["service"] == {"installed": True, "label": label, "healthy": True}
    plist = plistlib.loads((d.home / "Library" / "LaunchAgents" / f"{label}.plist").read_bytes())
    assert plist["ProgramArguments"] == ["/Users/x/.local/bin/sfx-api", "serve"]
    assert plist["EnvironmentVariables"]["SFX_API_CONFIG"] == d.env["SFX_API_CONFIG"]
    assert plist["StandardErrorPath"] == str(d.home / "Library" / "Logs" / "sfx-api.test.log")
    launchctl = run.commands("launchctl")
    assert [c[1] for c in launchctl] == ["bootout", "bootstrap"]
    assert launchctl[0][2].endswith("/" + label)
    assert http.seen[-1][0] == "http://127.0.0.1:8769/v1/health"


def test_service_that_never_answers(tmp_path: Path) -> None:
    code, report, _, _ = _setup(
        tmp_path, "--models", "sa3-small", "--yes", "--service", http=FakeHttp(statuses=[None])
    )
    assert code == 1 and report["service"]["healthy"] is False
    assert "did not answer" in report["error"]


def test_service_needs_the_cli_on_path(tmp_path: Path) -> None:
    code, report, _, _ = _setup(tmp_path, "--models", "sa3-small", "--yes", "--service", which=None)
    assert code == 1 and "uv tool install" in report["error"]


def test_bad_label_suffix(tmp_path: Path) -> None:
    code, _, _, _ = _setup(tmp_path, "--check", "--label-suffix", "a/b")
    assert code == 2


def test_cli_runs_setup_even_when_the_config_does_not_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    config = tmp_path / "config.toml"
    config.write_text('host = "0.0.0.0"\n')  # no token: the server refuses to load this
    monkeypatch.setenv("SFX_API_CONFIG", str(config))
    monkeypatch.setenv("HOME", str(tmp_path))
    code = main(["setup", "--check", "--json", "--prefix", str(tmp_path / "opt")])
    assert code == 0
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["action"] == "check" and report["lan"] is True
    assert config.read_text() == 'host = "0.0.0.0"\n'
