#!/usr/bin/env python
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_DIR = Path("/third_party/idm_vton")
VENV_DIR = Path("/third_party/.venv_idm")
STAMP = Path("/third_party/.venv_idm_ok")

# Minimal packages needed so IDM-VTON sources import cleanly when the repo
# does not provide a requirements.txt. Keep this list conservative.
MIN_DEPS = [
    "einops",
]

# Optional full dependency set pinned to IDM-VTON's demo expectations.
FULL_DEPS = [
    # Core HF stack pins compatible with this repo
    "diffusers==0.25.0",
    "transformers==4.36.2",
    "accelerate==0.25.0",
    "huggingface_hub==0.23.2",
    # Numpy + consumers (keep numpy<2 for onnxruntime and cv2 wheels)
    "numpy<2.0",
    "onnxruntime==1.16.2",
    # OpenCV + plotting
    "opencv-python==4.8.1.78",
    "matplotlib==3.7.5",
    # Detectron2 ecosystem (pure-Python parts vendored)
    "fvcore==0.1.5.post20221221",
    "iopath==0.1.10",
    "yacs==0.1.8",
    "tabulate==0.9.0",
    "termcolor==3.1.0",
    "pycocotools==2.0.10",
    # Misc utilities used in the demo
    "cloudpickle==3.1.1",
    "omegaconf==2.3.0",
    "basicsr==1.4.2",
    "av==16.0.1",
    "torchmetrics==1.2.1",
    "tqdm==4.66.1",
]

# Behavior flags via env
SELF_HEAL = os.environ.get("IDM_VTON_SELF_HEAL", "0").lower() in {"1", "true", "yes"}
FULL_SETUP = os.environ.get("IDM_VTON_FULL_DEPS", "0").lower() in {"1", "true", "yes"}
ISOLATE = os.environ.get("IDM_VTON_ISOLATE", "0").lower() in {"1", "true", "yes"}
RECREATE = os.environ.get("IDM_VTON_RECREATE", "0").lower() in {"1", "true", "yes"}


def main() -> None:
    try:
        if not REPO_DIR.exists():
            print("[idm_vton] repo missing; skipping venv bootstrap")
            return
        if RECREATE and VENV_DIR.exists():
            try:
                print("[idm_vton] recreating venv on request ...")
                shutil.rmtree(VENV_DIR, ignore_errors=True)
                STAMP.unlink(missing_ok=True)
            except Exception as exc:
                print(f"[idm_vton][warning] failed to remove venv: {exc}")
        if STAMP.exists() and VENV_DIR.exists():
            print("[idm_vton] venv present (stamp)")
            if SELF_HEAL:
                # Best-effort: ensure critical deps exist even if the venv was stamped
                # by an older image that didn't include them (e.g., einops).
                vpython = VENV_DIR / "bin" / "python"
                try:
                    missing: list[str] = []
                    for mod, pkg in [("einops", "einops")]:
                        code = f"import importlib,sys; sys.exit(0 if importlib.util.find_spec('{mod}') else 1)"
                        ok = subprocess.run([str(vpython), "-c", code], check=False).returncode == 0
                        if not ok:
                            missing.append(pkg)
                    if FULL_SETUP:
                        # Ensure heavier deps as well if requested
                        for mod, pkg in [("onnxruntime", "onnxruntime==1.16.2"), ("cv2", "opencv-python==4.8.1.78"), ("matplotlib", "matplotlib==3.7.5")]:
                            code = f"import importlib,sys; sys.exit(0 if importlib.util.find_spec('{mod}') else 1)"
                            ok = subprocess.run([str(vpython), "-c", code], check=False).returncode == 0
                            if not ok:
                                missing.append(pkg)
                    if missing:
                        print(f"[idm_vton] installing missing deps: {', '.join(missing)} ...")
                        pip = VENV_DIR / "bin" / "pip"
                        # Install conservatively; avoid resolver upgrading numpy unless explicitly requested.
                        cmd = [str(pip), "install", *missing]
                        subprocess.run(cmd, check=True)
                except Exception as exc:  # pragma: no cover
                    print(f"[idm_vton][warning] dep check failed: {exc}")
            else:
                print("[idm_vton] self-heal disabled (IDM_VTON_SELF_HEAL=0); skipping dep check")
            return
        # Create venv
        VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
        print("[idm_vton] creating venv ...")
        venv_args = ["python", "-m", "venv"]
        if not ISOLATE:
            venv_args.append("--system-site-packages")
        venv_args.append(str(VENV_DIR))
        subprocess.run(venv_args, check=True)
        pip = VENV_DIR / "bin" / "pip"
        req = REPO_DIR / "requirements.txt"
        if req.exists():
            print("[idm_vton] installing requirements ...")
            subprocess.run([str(pip), "install", "-r", str(req)], check=True)
        else:
            if FULL_SETUP:
                print("[idm_vton] installing full pinned deps for IDM-VTON demo ...")
                subprocess.run([str(pip), "install", *FULL_DEPS], check=True)
            else:
                print("[idm_vton] requirements.txt not found; installing minimal demo deps ...")
                subprocess.run([str(pip), "install", *MIN_DEPS], check=True)
        STAMP.touch()
        print("[idm_vton] venv bootstrap complete")
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[idm_vton][warning] venv bootstrap failed: {exc}")


if __name__ == "__main__":
    main()
