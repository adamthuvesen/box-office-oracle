"""Isolation, evidence, and failure tokens shared by the drives."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SKILL_DIR = Path(__file__).resolve().parent
SECRET_NAME = re.compile("KEY|TOKEN|SECRET|PASSWORD")
TRIPWIRES = ("aws", "snow", "terraform", "dbt", "docker", "curl")
TRIPWIRE_EXIT = 97
# An unroutable proxy makes a Python HTTP client fail fast instead of reaching the network.
DEAD_PROXY = "http://127.0.0.1:9"
CHILD_TIMEOUT_SECONDS = 90.0
# Dummy, non-secret values that let box_office.config import. No drive reaches a real account.
CONFIG_ENV = {
    "AWS_REGION": "eu-north-1",
    "SAGEMAKER_ROLE_ARN": "arn:aws:iam::123456789:role/verify",
    "AWS_S3_BUCKET": "verify-bucket",
    "SAGEMAKER_TRAINING_IMAGE_URI": "123456789012.dkr.ecr.eu-north-1.amazonaws.com/verify:tag",
    "SNOWFLAKE_USER": "verify",
    "SNOWFLAKE_ACCOUNT": "verify",
    "SNOWFLAKE_DATABASE": "VERIFY_DB",
    "SNOWFLAKE_WAREHOUSE": "VERIFY_WH",
}

# Failure tokens: stable identifiers a control or a reader matches on, never message wording.
TIMEOUT = "timeout"
TRIPWIRE_HIT = "tripwire-hit"
CHECKOUT_CHANGED = "checkout-changed"
SKIPPED = "skipped"
EARLY_EXIT = "early-exit"
NO_ASSERTIONS = "no-assertions"
CRASH = "crash"
EXIT_CODE = "unexpected-exit-code"


class DriveFailure(Exception):
    def __init__(self, token: str, detail: str) -> None:
        super().__init__(f"{token}: {detail}")
        self.token = token
        self.detail = detail


def git(*args: str, cwd: Path = ROOT) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    done = subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "absent"


def checkout_state() -> dict[str, str]:
    """HEAD, the index, git config and hooks, and every changed, untracked, or ignored path.

    Ignored paths compare by size and mtime: .venv and node_modules hold too many files to hash.
    """
    git_dir = (ROOT / git("rev-parse", "--git-common-dir")).resolve()
    state = {
        "HEAD": git("rev-parse", "HEAD"),
        "index": git("ls-files", "--stage"),
        ".git/config": _digest(git_dir / "config"),
    }
    for hook in sorted((git_dir / "hooks").glob("*")):
        state[f".git/hooks/{hook.name}"] = _digest(hook)
    listing = git("status", "--porcelain", "-z", "--untracked-files=all", "--ignored")
    for entry in filter(None, listing.split("\0")):
        code, path = entry[:2], entry[3:]
        if path.startswith(".verify/") or "__pycache__" in path.split("/"):
            continue
        target = ROOT / path
        if code == "!!":
            if target.exists():
                info = target.stat()
                state[path] = f"ignored:{info.st_size}:{info.st_mtime_ns}"
        else:
            state[path] = f"{code}:{_digest(target)}"
    return state


def changed_paths(before: Mapping[str, str]) -> list[str]:
    after = checkout_state()
    return sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))


class Session:
    """One temp copy of the checkout, a temp HOME, tripwires, and the evidence for a run."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.scratch = Path(tempfile.mkdtemp(prefix="verify-box-office-oracle-")).resolve()
        self.copy = self.scratch / "checkout"
        self.home = self.scratch / "home"
        self.tripwire_bin = self.scratch / "tripwires"
        self.tripwire_log = self.scratch / "tripwire-hits.log"
        self.memo: dict[str, object] = {}
        self.assertions = 0
        self._lock = threading.Lock()
        self._steps = 0
        self.home.mkdir()
        self._install_tripwires()
        self._copy_checkout()

    def _install_tripwires(self) -> None:
        self.tripwire_bin.mkdir()
        for name in TRIPWIRES:
            shim = self.tripwire_bin / name
            shim.write_text(
                f'#!/bin/sh\necho "{name} $*" >> "{self.tripwire_log}"\nexit {TRIPWIRE_EXIT}\n'
            )
            shim.chmod(0o755)

    def _copy_checkout(self) -> None:
        for rel in git("ls-files", "-co", "--exclude-standard").splitlines():
            source = ROOT / rel
            if not source.exists():
                continue
            target = self.copy / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target, follow_symlinks=False)

    def cleanup(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)

    def env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        tool_dirs = [str(Path(sys.executable).parent), "/usr/bin", "/bin"]
        for tool in ("uv", "node"):
            found = shutil.which(tool)
            if found:
                tool_dirs.append(str(Path(found).parent))
        env = {
            "PATH": os.pathsep.join([str(self.tripwire_bin), *tool_dirs]),
            "HOME": str(self.home),
            "PYTHONPATH": str(self.copy),
            "PYTHONDONTWRITEBYTECODE": "1",
            "TZ": "UTC",
            "NO_COLOR": "1",
            "HTTP_PROXY": DEAD_PROXY,
            "HTTPS_PROXY": DEAD_PROXY,
            "ALL_PROXY": DEAD_PROXY,
            "NO_PROXY": "127.0.0.1,localhost",
            "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_CONFIG_FILE": os.devnull,
            "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
            "XDG_CONFIG_HOME": str(self.home / "config"),
            "XDG_CACHE_HOME": str(self.home / "cache"),
            "XDG_DATA_HOME": str(self.home / "data"),
            **CONFIG_ENV,
        }
        env.update(extra or {})
        return env

    def tripwire_hits(self) -> list[str]:
        return self.tripwire_log.read_text().splitlines() if self.tripwire_log.exists() else []

    def check(self, condition: bool, token: str, detail: str) -> None:
        with self._lock:
            self.assertions += 1
        if not condition:
            raise DriveFailure(token, detail)

    def run(
        self,
        drive: str,
        step: str,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        extra_env: Mapping[str, str] | None = None,
        expect_exit: int | None = 0,
        timeout: float = CHILD_TIMEOUT_SECONDS,
    ) -> subprocess.CompletedProcess[str]:
        """Run one child in the isolated environment and record it. ``expect_exit=None`` skips the code check."""
        env = self.env(extra_env)
        started = time.monotonic()
        try:
            done = subprocess.run(
                list(argv),
                cwd=cwd or self.copy,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                start_new_session=True,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self._record(drive, step, argv, env, None, "", f"timed out after {timeout}s", started)
            raise DriveFailure(TIMEOUT, f"{drive}.{step} exceeded {timeout}s") from exc
        self._record(drive, step, argv, env, done.returncode, done.stdout, done.stderr, started)
        hits = self.tripwire_hits()
        if hits:
            raise DriveFailure(TRIPWIRE_HIT, f"called {hits} (seen after {drive}.{step})")
        if expect_exit is not None and done.returncode != expect_exit:
            tail = (done.stderr or done.stdout)[-600:]
            raise DriveFailure(
                EXIT_CODE,
                f"{drive}.{step} exited {done.returncode}, expected {expect_exit}: {tail}",
            )
        return done

    def _record(
        self,
        drive: str,
        step: str,
        argv: Sequence[str],
        env: Mapping[str, str],
        code: int | None,
        out: str,
        err: str,
        started: float,
    ) -> None:
        secrets = [v for k, v in env.items() if SECRET_NAME.search(k) and len(v) >= 4]

        def redact(text: str) -> str:
            for secret in secrets:
                text = text.replace(secret, "<redacted>")
            return text

        with self._lock:
            self._steps += 1
            number = self._steps
        entry = {
            "argv": [redact(a) for a in argv],
            "exit": code,
            "seconds": round(time.monotonic() - started, 2),
            "stdout": redact(out[-4000:]),
            "stderr": redact(err[-4000:]),
        }
        (self.run_dir / f"{drive}.{number:02d}-{step}.json").write_text(
            json.dumps(entry, indent=2) + "\n"
        )


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
