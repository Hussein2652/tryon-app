#!/usr/bin/env python
"""Download StableVITON-related model assets if they are missing.

The script is designed to run during Docker build or container start. It
reads configuration from environment variables (matching the defaults in
`api/app/config.py`) and attempts to download any missing files.

Downloads are optional—if a URL requires authentication or fails, the
script will emit a warning and continue, leaving placeholders untouched so
the container can still run in placeholder mode.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover - optional dependency
    hf_hub_download = None  # type: ignore

try:
    import gdown  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    gdown = None  # type: ignore


BASE_MODELS_DIR = Path(os.environ.get("TRYON_MODELS_DIR", "/models"))

STABLEVITON_DIR = Path(os.environ.get("STABLEVITON_CKPT_DIR", BASE_MODELS_DIR / "stableviton"))
CONTROLNET_DIR = Path(os.environ.get("CONTROLNET_OPENPOSE_DIR", BASE_MODELS_DIR / "controlnet" / "openpose"))
SCHP_PATH = Path(os.environ.get("SCHP_WEIGHTS", BASE_MODELS_DIR / "schp" / "schp.pth"))
INSTANTID_DIR = Path(os.environ.get("INSTANTID_DIR", BASE_MODELS_DIR / "instantid"))

STABLEVITON_SHAREPOINT_URL = os.environ.get("STABLEVITON_SHAREPOINT_URL")
SCHP_DRIVE_URL = os.environ.get(
    "SCHP_DRIVE_URL",
    "https://drive.google.com/uc?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH",  # exp-schp-201908261155-lip.pth
)


@dataclass
class DownloadTask:
    description: str
    destination: Path
    action: Callable[[], None]

    def run(self) -> None:
        if self.destination.exists():
            print(f"[models] {self.description}: already present at {self.destination}")
            return
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.action()
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[models][warning] {self.description}: download failed ({exc})")
        else:
            print(f"[models] {self.description}: downloaded to {self.destination}")


def download_with_hf(repo_id: str, filename: str, destination: Path) -> None:
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is not installed")
    cached = hf_hub_download(repo_id=repo_id, filename=filename)
    shutil.copy2(cached, destination)


def download_controlnet_openpose() -> DownloadTask:
    dest = CONTROLNET_DIR / "control_v11p_sd15_openpose.safetensors"

    def action() -> None:
        download_with_hf("lllyasviel/control_v11p_sd15_openpose", "control_v11p_sd15_openpose.safetensors", dest)

    return DownloadTask("ControlNet OpenPose", dest, action)


def download_instantid_files() -> list[DownloadTask]:
    tasks: list[DownloadTask] = []
    for filename in [
        "ControlNetModel/config.json",
        "ControlNetModel/diffusion_pytorch_model.safetensors",
        "ip-adapter.bin",
    ]:
        dest = INSTANTID_DIR / filename

        def make_action(repo_file: str, destination: Path) -> Callable[[], None]:
            def action() -> None:
                download_with_hf("InstantX/InstantID", repo_file, destination)

            return action

        tasks.append(
            DownloadTask(
                description=f"InstantID asset {filename}",
                destination=dest,
                action=make_action(filename, dest),
            )
        )

    return tasks


def download_antelopev2() -> DownloadTask:
    zip_dest = INSTANTID_DIR / "antelopev2.zip"
    extract_dir = INSTANTID_DIR / "antelopev2"

    def action() -> None:
        if gdown is None:
            raise RuntimeError("gdown is not installed")
        url = os.environ.get(
            "INSTANTID_ANTELOPE_URL",
            "https://sourceforge.net/projects/insightface.mirror/files/v0.7/antelopev2.zip/download",
        )
        gdown.download(url, str(zip_dest), quiet=False)
        shutil.unpack_archive(zip_dest, extract_dir)
        zip_dest.unlink(missing_ok=True)

    return DownloadTask("InstantID antelopev2", extract_dir / "glintr100.onnx", action)


def download_schp() -> DownloadTask:
    def action() -> None:
        if gdown is None:
            raise RuntimeError("gdown is not installed")
        gdown.download(SCHP_DRIVE_URL, str(SCHP_PATH), quiet=False)

    return DownloadTask("SCHP weights", SCHP_PATH, action)


def warn_sharepoint() -> None:
    print(
        "[models][info] StableVITON checkpoints require manual download. "
        "Set STABLEVITON_SHAREPOINT_URL to an accessible link or mount the directory."
    )


def download_stableviton() -> Optional[DownloadTask]:
    if not STABLEVITON_SHAREPOINT_URL:
        warn_sharepoint()
        return None

    filename = os.environ.get("STABLEVITON_CHECKPOINT_FILENAME", "StableVITON.ckpt")
    dest = STABLEVITON_DIR / filename

    def action() -> None:
        if gdown is None:
            raise RuntimeError("gdown is not installed")
        gdown.download(STABLEVITON_SHAREPOINT_URL, str(dest), quiet=False)

    return DownloadTask("StableVITON checkpoint", dest, action)


def main() -> None:
    tasks: list[DownloadTask] = []

    stableviton_task = download_stableviton()
    if stableviton_task:
        tasks.append(stableviton_task)

    tasks.append(download_controlnet_openpose())
    tasks.extend(download_instantid_files())
    tasks.append(download_antelopev2())
    tasks.append(download_schp())

    success_count = 0
    for task in tasks:
        task.run()
        if task.destination.exists():
            success_count += 1

    print(f"[models] Completed downloads. {success_count}/{len(tasks)} assets present.")


if __name__ == "__main__":
    main()
