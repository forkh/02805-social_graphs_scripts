"""Run the whole chain in order, each step in a fresh Python process; outputs go to OUT_DIR = REPO_ROOT/reproduce_networks_outputs.

    python reproduce_networks/run_all.py                      # full run (OUT_DIR must be absent or empty)
    python reproduce_networks/run_all.py --from formal_roles  # resume (outputs of the earlier steps kept in OUT_DIR)

Before the run every file under REPO_ROOT outside OUT_DIR (and .git) is recorded (SHA-256, size, mtime); size and mtime are compared
after every step and all hashes at the end. Any new, missing or changed file outside OUT_DIR stops the run with an error.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as CFG  # noqa: E402

STEPS = [("voting", "voting_network.py"), ("temporal", "temporal_network.py"), ("speaking_order", "speaking_order.py"),
         ("formal_roles", "formal_roles.py"), ("alternation", "alternation.py"), ("fig1", "fig1.py")]


def snapshot(with_hash: bool) -> dict:
    out = {}
    for d, ds, fs in os.walk(CFG.REPO_ROOT):
        dp = Path(d)
        ds[:] = [x for x in ds if x != ".git" and (dp / x) != CFG.OUT_DIR]
        for f in fs:
            p = dp / f
            st = p.stat()
            out[p.relative_to(CFG.REPO_ROOT).as_posix()] = (st.st_size, st.st_mtime_ns,
                                                             hashlib.sha256(p.read_bytes()).hexdigest() if with_hash else None)
    return out


def diff(before: dict, after: dict, use_hash: bool) -> list[str]:
    idx = (0, 1, 2) if use_hash else (0, 1)
    return sorted(k for k in set(before) | set(after) if k not in before or k not in after or any(before[k][i] != after[k][i] for i in idx))


def main() -> int:
    names = [n for n, _ in STEPS]
    start = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else names[0]
    if start not in names:
        sys.exit(f"--from must be one of {names}")
    if start == names[0] and CFG.OUT_DIR.exists() and any(CFG.OUT_DIR.iterdir()):
        sys.exit(f"{CFG.OUT_DIR.name} is not empty; refusing to overwrite")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    env.setdefault("PYTHONHASHSEED", "0")
    before = snapshot(with_hash=True)
    (CFG.OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    info = {"PYTHONHASHSEED": env["PYTHONHASHSEED"], "python": sys.version.split()[0], "files_checked": len(before), "steps": {}}
    for name, script in STEPS[names.index(start):]:
        t0 = time.time()
        print(f"=== {name}", flush=True)
        with open(CFG.OUT_DIR / "logs" / f"{name}.log", "w", encoding="utf-8") as log:
            rc = subprocess.run([sys.executable, str(HERE / script)], cwd=CFG.REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        changed = diff(before, snapshot(with_hash=False), use_hash=False)
        info["steps"][name] = {"returncode": rc, "seconds": round(time.time() - t0, 1)}
        print(f"=== {name}: {'ok' if rc == 0 else 'FAILED'} ({info['steps'][name]['seconds']} s)", flush=True)
        if changed or rc:
            info["changed_outside_OUT_DIR"] = changed
            (CFG.OUT_DIR / "run_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
            sys.exit(f"STOP: files outside OUT_DIR changed during {name}: {changed[:10]}" if changed else f"STOP: step {name} failed; see logs/{name}.log")
    changed = diff(before, snapshot(with_hash=True), use_hash=True)
    info["repository_unchanged"] = not changed
    (CFG.OUT_DIR / "run_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    if changed:
        sys.exit(f"STOP: files outside OUT_DIR changed: {changed[:10]}")
    print("all steps done; nothing outside OUT_DIR changed (SHA-256, size, mtime)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
