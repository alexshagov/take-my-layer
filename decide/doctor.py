"""doctor.py — one-command preflight before running anything heavy.

Checks the things that actually break a first run (docs/RUNNING.md):
Python >= 3.11 (mlx-lm 0.32.0 needs it), the anyjev==0.2.0 / mlx-lm==0.32.0
pins (anyjev 0.3.0 removed the L1/L2 machinery), the Hugging Face cache
configuration — an exported HF_HUB_CACHE silently overrides HF_HOME, so a
"cold" cache can quietly be warm — whether the models are cached or would
need to download, and whether the usual ports already have a server on them
(one MLX instance at a time on 32 GB). Stdlib only; never loads a model.

Exit 0 with at most warnings when running looks possible, exit 1 when a hard
requirement (Python version, a pin) is violated.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("mlx-community/Qwen3.5-4B-MLX-4bit",
          "mlx-community/Qwen3.5-9B-MLX-4bit")
PORTS = {8095: "decision endpoint (systemone.py)", 8080: "generation server",
         8090: "shadow proxy"}
GB = 1024 ** 3


def check_python() -> tuple[str, str]:
    ok = sys.version_info >= (3, 11)
    return ("python >= 3.11", f"{sys.version.split()[0]} — "
            + ("ok" if ok else "TOO OLD: install 3.11+ or use Nix")), ok


def check_pins() -> tuple[str, str, bool]:
    import importlib.metadata as md
    out, ok = [], True
    for name, want in (("anyjev", "0.2.0"), ("mlx-lm", "0.32.0")):
        try:
            got = md.version(name)
            good = got == want
            out.append(f"{name} {got}" + ("" if good else f" (want {want}: "
                        "anyjev 0.3.0 removed fit_head/observe)"))
            ok &= good
        except md.PackageNotFoundError:
            out.append(f"{name} MISSING (pip install -r requirements.txt)")
            ok = False
    return ("dependency pins", "; ".join(out)), ok


def check_hf_cache() -> tuple[str, list[str]]:
    home = os.environ.get("HF_HOME")
    hub = os.environ.get("HF_HUB_CACHE")
    notes = []
    if not home and not hub:
        notes.append("HF_HOME not set: the model lands in ~/.cache/huggingface "
                     "(about 2.9 GB for the 4B); export HF_HOME=$PWD/.hf to keep it in the project")
    if hub:
        notes.append(f"HF_HUB_CACHE={hub} — it OVERRIDES HF_HOME"
                     + (f" (HF_HOME={home} is ignored by the hub cache)" if home else ""))
    where = Path(hub or (Path(home) / "hub" if home
                         else Path.home() / ".cache/huggingface/hub"))
    return where, notes


def check_models(hub_dir: Path) -> list[str]:
    lines = []
    for m in MODELS:
        repo_dir = hub_dir / ("models--" + m.replace("/", "--"))
        snap = repo_dir / "snapshots"
        have = snap.is_dir() and any(snap.iterdir())
        size = ""
        if have:
            blobs = repo_dir / "blobs"
            try:
                size = f" ({sum(f.stat().st_size for f in blobs.iterdir()) / GB:.1f} GB cached)"
            except OSError:
                pass
        lines.append(f"{m}: " + ("cached" + size if have
                     else "NOT cached — first server start downloads it"))
    return lines


def check_ports() -> list[str]:
    lines = []
    for port, what in PORTS.items():
        with socket.socket(socket.AF_INET) as s:
            s.settimeout(0.3)
            busy = s.connect_ex(("127.0.0.1", port)) == 0
        if busy:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz",
                                            timeout=1) as r:
                    info = json.load(r)
                lines.append(f"port {port} ({what}): BUSY — serving "
                             f"{info.get('model', '?')}; stop it before model-loading benches")
            except Exception:
                lines.append(f"port {port} ({what}): BUSY (no /healthz answer)")
    return lines or [f"ports {sorted(PORTS)}: free"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="preflight: python/deps pins, HF cache config (the "
                    "HF_HUB_CACHE-overrides-HF_HOME trap), model cache, port "
                    "occupancy. Stdlib only, loads no model.")
    ap.parse_args()

    hard_ok = True
    (label, detail), ok = check_python()
    hard_ok &= ok
    print(("ok   " if ok else "FAIL ") + f"{label}: {detail}")
    (label, detail), ok = check_pins()
    hard_ok &= ok
    print(("ok   " if ok else "FAIL ") + f"{label}: {detail}")
    hub_dir, notes = check_hf_cache()
    print(f"ok   hf cache dir: {hub_dir}")
    for n in notes:
        print("warn " + n)
    for line in check_models(hub_dir):
        print("     " + line)
    for line in check_ports():
        print("     " + line)
    print("ok   preflight done" if hard_ok else "FAIL  fix the items above, then retry")
    return 0 if hard_ok else 1


if __name__ == "__main__":
    sys.exit(main())
