"""Output and preflight policy shared by the benchmark, fitting and authoring
scripts (docs/TODO.md PP-5, PP-17).

results/ holds frozen, quoted evidence and registry/ the shipped head artifacts.
A fresh run therefore writes to runs/ (gitignored) unless --out says otherwise,
and no script replaces an existing file under results/ or registry/ unless
--force is given. Scripts that call a server or an API check it BEFORE any work
starts, and a run in which every call failed exits non-zero without writing
anything.

To refresh frozen evidence on purpose: run, compare runs/X with results/X, then
`cp runs/X results/X` (or pass --out results/X --force).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
PROTECTED = (RESULTS, ROOT / "registry", ROOT / "data" / "eval")


def add_out(ap, name: str, what: str = "result JSON") -> None:
    """Add --out (default runs/<name>) and --force to a parser."""
    ap.add_argument("--out", default=str(RUNS / name),
                    help=f"where to write the {what} (default: runs/{name}, "
                         "gitignored; results/ is frozen evidence, copy a run "
                         "there deliberately)")
    ap.add_argument("--force", action="store_true",
                    help="allow --out to replace an existing file under "
                         "results/ (frozen evidence)")


def rel(path) -> str:
    """Repo-relative string for a path under ROOT (what reports should record:
    an absolute path would leak the machine's home directory into artifacts)."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(path)


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def check_out(ap, out, force: bool = False) -> Path:
    """Validate an output path before the run: refuse to replace an existing
    frozen artifact (results/, registry/, data/eval/) without --force, and
    create the parent directory."""
    p = Path(out)
    if p.exists() and not force and any(_inside(p, d) for d in PROTECTED):
        ap.error(f"{p} is a shipped artifact (results/, registry/ or "
                 "data/eval/); write to runs/ (the default) or pass --force "
                 "to replace it")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def require_endpoint(ap, base_url: str, what: str = "decision endpoint",
                     timeout: float = 5.0) -> dict:
    """Fail fast (argparse error, exit 2) unless GET <base_url>/healthz
    answers. Returns the health JSON ({"ok", "model", "head"})."""
    url = base_url.rstrip("/") + "/healthz"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            info = json.load(r)
    except Exception as e:                          # noqa: BLE001
        ap.error(f"{what} not reachable at {url} ({e!r}); start it first, "
                 "e.g. python decide/serve/systemone.py --port 8095")
    if not isinstance(info, dict) or not info.get("ok"):
        ap.error(f"{what} at {url} did not report ok: {info!r}")
    return info


def require_key(ap, key, what: str = "the Jev API leg", skip: str = "") -> None:
    """Fail fast unless an API key is present (`skip` names the flag, if any,
    that runs without it)."""
    if not key:
        ap.error(f"{what} needs an API key: set $OPENROUTER_API_KEY (or pass "
                 "--jev-key)" + (f"; {skip}" if skip else ""))


def abort_early(n_err: int, n_done: int, what: str, k: int = 3) -> None:
    """Stop at the k-th call when every call so far failed (a healthy
    /healthz does not make /v1/systemone work)."""
    if n_done == k and n_err == k:
        sys.exit(f"error: the first {k} {what} calls all failed; stopping. "
                 "Nothing written.")


def abort_if_all_failed(n_ok: int, n_total: int, what: str) -> None:
    """Exit 1 (nothing written) when a run produced no usable result at all."""
    if n_total and n_ok == 0:
        sys.exit(f"error: every {what} failed ({n_total}/{n_total}); "
                 "nothing written")


def warn_failures(n_failed: int, n_total: int, what: str) -> None:
    """Loud end-of-run line when some calls failed (they are scored as misses)."""
    if n_failed:
        print(f"WARNING: {n_failed} of {n_total} {what} failed; those rows "
              "count as misses in the numbers above", file=sys.stderr,
              flush=True)
