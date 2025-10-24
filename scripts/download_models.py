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
import urllib.request
import zipfile


try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:  # pragma: no cover - optional dependency
    hf_hub_download = None  # type: ignore
    snapshot_download = None  # type: ignore

try:
    import gdown  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    gdown = None  # type: ignore


BASE_MODELS_DIR = Path(os.environ.get("TRYON_MODELS_DIR", "/models"))
DOWNLOAD_STAMPS = BASE_MODELS_DIR / "_download_stamps"
DOWNLOAD_STAMPS.mkdir(parents=True, exist_ok=True)

STABLEVITON_DIR = Path(os.environ.get("STABLEVITON_CKPT_DIR", BASE_MODELS_DIR / "stableviton"))
CONTROLNET_DIR = Path(os.environ.get("CONTROLNET_OPENPOSE_DIR", BASE_MODELS_DIR / "controlnet" / "openpose"))
SCHP_PATH = Path(os.environ.get("SCHP_WEIGHTS", BASE_MODELS_DIR / "schp" / "schp.pth"))
INSTANTID_DIR = Path(os.environ.get("INSTANTID_DIR", BASE_MODELS_DIR / "instantid"))
SD15_MODEL_DIR = Path(os.environ.get("SD15_MODEL_DIR", BASE_MODELS_DIR / "sd15"))
SD15_MODEL_ID = os.environ.get("SD15_MODEL_ID", "runwayml/stable-diffusion-v1-5")

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


def download_repo_snapshot(repo_id: str, destination: Path) -> None:
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is not installed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=destination,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4,
    )


def stamp(name: str) -> Path:
    p = DOWNLOAD_STAMPS / f"{name}.ok"
    return p


def download_http(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(destination, "wb") as f:
        shutil.copyfileobj(resp, f)


def download_controlnet_openpose() -> DownloadTask:
    # Snapshot the full repo so config.json is present for from_pretrained
    dest = CONTROLNET_DIR / "config.json"

    def action() -> None:
        url = os.environ.get("CONTROLNET_OPENPOSE_URL", "")
        if url and url.endswith(".safetensors"):
            # If only a weights URL is provided, still snapshot the repo to bring configs
            weights = CONTROLNET_DIR / "control_v11p_sd15_openpose.safetensors"
            download_http(url, weights)
            try:
                download_repo_snapshot("lllyasviel/control_v11p_sd15_openpose", CONTROLNET_DIR)
            except Exception:
                # fallback: try direct config.json
                download_with_hf("lllyasviel/control_v11p_sd15_openpose", "config.json", dest)
        else:
            download_repo_snapshot("lllyasviel/control_v11p_sd15_openpose", CONTROLNET_DIR)

    return DownloadTask("ControlNet OpenPose", dest, action)


def _sd15_requirements_met(root: Path) -> bool:
    # Minimal set of files the pipeline expects
    required = [
        root / "model_index.json",
        root / "unet" / "diffusion_pytorch_model.safetensors",
        root / "vae" / "diffusion_pytorch_model.safetensors",
    ]
    return all(p.exists() for p in required)


def download_sd15_repo() -> DownloadTask:
    # Use a completion sentinel so we don't skip partial directories
    dest = SD15_MODEL_DIR / ".complete"

    def action() -> None:
        # Always call snapshot_download with resume; it will fetch only missing parts
        download_repo_snapshot(SD15_MODEL_ID, SD15_MODEL_DIR)
        if not _sd15_requirements_met(SD15_MODEL_DIR):
            raise RuntimeError("SD1.5 snapshot incomplete; required files missing")
        dest.touch()

    return DownloadTask("Stable Diffusion 1.5 (diffusers)", dest, action)


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
    extract_dir = INSTANTID_DIR / "antelopev2"
    # Use a completion sentinel so we don't repeat extraction when files exist
    complete_sentinel = extract_dir / ".complete"

    def action() -> None:
        # If an .onnx already exists in the folder, just write the sentinel
        try:
            if extract_dir.exists():
                for root, _, files in os.walk(extract_dir):
                    if any(f.lower().endswith(".onnx") for f in files):
                        complete_sentinel.parent.mkdir(parents=True, exist_ok=True)
                        complete_sentinel.touch()
                        return
        except Exception:
            pass

        # Try multiple sources until one succeeds
        url_env = os.environ.get(
            "INSTANTID_ANTELOPE_URL",
            "https://downloads.sourceforge.net/project/insightface/v0.7/antelopev2.zip",
        )
        candidates = [
            url_env,
            # GitHub release mirror
            "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
            # SourceForge legacy redirect
            "https://sourceforge.net/projects/insightface.mirror/files/v0.7/antelopev2.zip/download",
        ]
        zip_dest = INSTANTID_DIR / "antelopev2.zip"
        last_exc: Optional[Exception] = None
        for url in candidates:
            try:
                if "drive.google.com" in url:
                    if gdown is None:
                        raise RuntimeError("gdown is not installed for Google Drive URLs")
                    gdown.download(url, str(zip_dest), quiet=False)
                else:
                    download_http(url, zip_dest)
                if not zipfile.is_zipfile(zip_dest):
                    raise RuntimeError("downloaded file is not a valid zip")
                shutil.unpack_archive(zip_dest, extract_dir)
                zip_dest.unlink(missing_ok=True)
                complete_sentinel.touch()
                break
            except Exception as exc:  # pylint: disable=broad-except
                last_exc = exc
                zip_dest.unlink(missing_ok=True)
                continue
        else:
            raise RuntimeError(f"antelopev2 download failed: {last_exc}")

    return DownloadTask("InstantID antelopev2", complete_sentinel, action)


def download_schp() -> DownloadTask:
    def action() -> None:
        # Accept direct HTTP/HF URLs; fallback to gdown if Google Drive
        if SCHP_DRIVE_URL.startswith("http") and "drive.google.com" not in SCHP_DRIVE_URL:
            download_http(SCHP_DRIVE_URL, SCHP_PATH)
        else:
            if gdown is None:
                raise RuntimeError("gdown is not installed for Google Drive URLs")
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
    tasks.append(download_sd15_repo())
    tasks.extend(download_instantid_files())
    tasks.append(download_antelopev2())
    tasks.append(download_schp())

    success_count = 0
    for task in tasks:
        task.run()
        if task.destination.exists():
            success_count += 1

    print(f"[models] Completed downloads. {success_count}/{len(tasks)} assets present.")

    # ---- V2 models (IDM-VTON + SDXL) ----
    # Use stamps to avoid repeating snapshots
    try:
        # IDM-VTON Space ckpt subtree
        name = "idm_vton_ckpt"
        if not stamp(name).exists():
            space_dir = BASE_MODELS_DIR / "_tmp_idm_space"
            download_repo_snapshot("spaces/yisol/IDM-VTON", space_dir)
            src = space_dir / "ckpt"
            dst = BASE_MODELS_DIR / "idm_vton" / "ckpt"
            if src.exists():
                dst.mkdir(parents=True, exist_ok=True)
                for item in src.rglob("*"):
                    if item.is_file():
                        rel = item.relative_to(src)
                        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dst / rel)
            shutil.rmtree(space_dir, ignore_errors=True)
            stamp(name).touch()
            print("[models] IDM-VTON ckpt fetched")
        else:
            print("[models] IDM-VTON ckpt: already stamped")

        # SDXL inpaint and refiner
        for repo, subdir, mark in [
            ("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", BASE_MODELS_DIR / "sdxl-inpaint", "sdxl_inpaint"),
            ("stabilityai/stable-diffusion-xl-refiner-1.0", BASE_MODELS_DIR / "sdxl-refiner", "sdxl_refiner"),
            ("thibaud/controlnet-openpose-sdxl-1.0", BASE_MODELS_DIR / "controlnets/openpose-sdxl", "cn_openpose_sdxl"),
            ("SargeZT/controlnet-sd-xl-1.0-softedge-dexined", BASE_MODELS_DIR / "controlnets/softedge-sdxl", "cn_softedge_sdxl"),
            ("yzd-v/DWPose", BASE_MODELS_DIR / "dwpose", "dwpose_pack"),
            ("h94/IP-Adapter", BASE_MODELS_DIR / "ip_adapter", "ip_adapter"),
        ]:
            if not stamp(mark).exists():
                download_repo_snapshot(repo, subdir)
                stamp(mark).touch()
                print(f"[models] snapshot: {repo} -> {subdir}")
            else:
                print(f"[models] snapshot: {repo} already cached")

        # Optional InstantID
        enable_instantid = os.environ.get("ENABLE_INSTANTID", "0") in {"1","true","True"}
        if enable_instantid:
            if not stamp("instantid").exists():
                download_repo_snapshot("InstantX/InstantID", BASE_MODELS_DIR / "instantid")
                stamp("instantid").touch()
                print("[models] InstantID snapshot cached")
            else:
                print("[models] InstantID already cached")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[models][warning] V2 model snapshot failed: {exc}")


if __name__ == "__main__":
    main()
