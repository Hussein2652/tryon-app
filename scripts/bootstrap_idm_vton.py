#!/usr/bin/env python
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_DIR = Path("/third_party/idm_vton")
VENV_DIR = Path("/third_party/.venv_idm")
STAMP = Path("/third_party/.venv_idm_ok")


def main() -> None:
    try:
        if not REPO_DIR.exists():
            print("[idm_vton] repo missing; skipping venv bootstrap")
            return
        if STAMP.exists() and VENV_DIR.exists():
            print("[idm_vton] venv present (stamp)")
            return
        # Create venv
        VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
        print("[idm_vton] creating venv ...")
        subprocess.run(["python", "-m", "venv", "--system-site-packages", str(VENV_DIR)], check=True)
        pip = VENV_DIR / "bin" / "pip"
        req = REPO_DIR / "requirements.txt"
        if req.exists():
            print("[idm_vton] installing requirements ...")
            subprocess.run([str(pip), "install", "-r", str(req)], check=True)
        else:
            # Minimal fallback set often needed by demos. Pin gradio to avoid massive upgrades.
            print("[idm_vton] requirements.txt not found; installing minimal demo deps ...")
            subprocess.run([str(pip), "install", "pillow", "gradio==4.44.0"], check=True)
        STAMP.touch()
        print("[idm_vton] venv bootstrap complete")
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[idm_vton][warning] venv bootstrap failed: {exc}")


if __name__ == "__main__":
    main()
