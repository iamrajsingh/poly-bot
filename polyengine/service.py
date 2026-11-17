"""Run polyengine 24x7 on macOS with launchd, plus a single-instance lock.

`polyengine service install` writes a LaunchAgent that:
  * starts the engine at login and keeps it running (restarted ~30s after any crash);
  * wraps it in `caffeinate -i -s`, so the Mac doesn't idle-sleep while plugged in;
  * runs from the project directory, so `.env` and `data/` resolve as usual.

Network drops are handled inside the engine (it pauses and resumes); launchd covers crashes,
reboots and logins. All state lives in SQLite, so every restart picks up where it stopped.
"""

from __future__ import annotations

import fcntl
import os
import plistlib
import subprocess
import sys
from pathlib import Path

LABEL = "com.polyengine.agent"


# ---------------------------------------------------------------- single instance
class InstanceLock:
    """Exclusive lock on <db>.lock. Two engines on one database would double-trade."""

    def __init__(self, db_path: str):
        self.path = Path(db_path).with_suffix(".lock")
        self._fh = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+")
        try:
            fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._fh.close()
            self._fh = None
            return False
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(str(os.getpid()))
        self._fh.flush()
        return True  # released automatically by the OS when the process exits, even on a crash

    def holder(self) -> str:
        try:
            return self.path.read_text().strip() or "?"
        except OSError:
            return "?"

    def release(self) -> None:
        if self._fh:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None


# ---------------------------------------------------------------- launchd
def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def build_plist(project_dir: Path, live: bool = False, dashboard: bool = True, port: int = 8642) -> dict:
    exe = project_dir / ".venv" / "bin" / "polyengine"
    args = ["/usr/bin/caffeinate", "-i", "-s", str(exe), "run"]
    if dashboard:
        args += ["--dashboard", "--port", str(port)]
    if live:
        args.append("--live")
    log = str(project_dir / "data" / "service.log")
    return {
        "Label": LABEL,
        "ProgramArguments": args,
        "WorkingDirectory": str(project_dir),
        "RunAtLoad": True,
        # Restart only after a crash/non-zero exit; a clean stop (`service stop`) stays stopped.
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "StandardOutPath": log,
        "StandardErrorPath": log,
        "EnvironmentVariables": {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "PYTHONUNBUFFERED": "1"},
    }


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


def _domain() -> str:
    return f"gui/{os.getuid()}"


def install(project_dir: Path, live: bool, dashboard: bool, port: int) -> None:
    if sys.platform != "darwin":
        sys.exit("service install is macOS-only (launchd). On Linux use the Dockerfile or systemd.")
    exe = project_dir / ".venv" / "bin" / "polyengine"
    if not exe.exists():
        sys.exit(f"{exe} not found - create the venv and `pip install -e .` first")
    (project_dir / "data").mkdir(exist_ok=True)
    path = plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _launchctl("bootout", f"{_domain()}/{LABEL}")  # replace any previous version
    with open(path, "wb") as f:
        plistlib.dump(build_plist(project_dir, live, dashboard, port), f)
    r = _launchctl("bootstrap", _domain(), str(path))
    if r.returncode != 0:
        sys.exit(f"launchctl bootstrap failed: {r.stderr.strip() or r.stdout.strip()}")
    print(f"installed {path}\nthe engine is starting now and will start at every login"
          + (f"\ndashboard: http://127.0.0.1:{port}" if dashboard else ""))


def uninstall() -> None:
    _launchctl("bootout", f"{_domain()}/{LABEL}")
    path = plist_path()
    if path.exists():
        path.unlink()
    print("service stopped and removed")


def stop() -> None:
    r = _launchctl("bootout", f"{_domain()}/{LABEL}")
    print("stopped (until next login or `service start`)" if r.returncode == 0 else "service was not running")


def start() -> None:
    path = plist_path()
    if not path.exists():
        sys.exit("not installed - run `polyengine service install`")
    r = _launchctl("bootstrap", _domain(), str(path))
    print("started" if r.returncode == 0 else f"could not start: {r.stderr.strip()}")


def status() -> dict:
    r = _launchctl("print", f"{_domain()}/{LABEL}")
    info = {"installed": plist_path().exists(), "loaded": r.returncode == 0, "state": None, "pid": None,
            "last_exit": None, "runs": None}
    for line in r.stdout.splitlines():
        line = line.strip()
        for key, name in (("state = ", "state"), ("pid = ", "pid"), ("last exit code = ", "last_exit"),
                          ("runs = ", "runs")):
            if line.startswith(key):
                info[name] = line[len(key):]
    return info
