from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from PIL import Image


def load_callable(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    try:
        mod_path = repo_root / "gradio_demo" / "app.py"
        if not mod_path.exists():
            raise RuntimeError("gradio_demo/app.py not found")
        import importlib.util

        spec = importlib.util.spec_from_file_location("idm_vton_gradio_app", str(mod_path))
        if spec is None or spec.loader is None:
            raise RuntimeError("unable to load gradio module")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        fn = None
        for name in ("process", "predict"):
            if hasattr(mod, name):
                fn = getattr(mod, name)
                break
        if fn is None:
            raise RuntimeError("no process/predict in gradio demo")
        return fn
    finally:
        try:
            sys.path.remove(str(repo_root))
        except ValueError:
            pass


def main(person_path: str, cloth_path: str, out_path: str, category: str, steps: int, guidance: float, seed: int) -> None:
    repo_root = Path("/third_party/idm_vton")
    fn = load_callable(repo_root)
    person = Image.open(person_path).convert("RGB")
    cloth = Image.open(cloth_path).convert("RGBA")
    # Try a few kwarg patterns
    variants = [
        dict(person_image=person, cloth_image=cloth, category=category, steps=steps, guidance=guidance, seed=seed),
        dict(human_img=person, cloth_img=cloth, category=category, steps=steps, cfg=guidance, seed=seed),
        dict(im=person, c=cloth, cate=category, num_steps=steps, cfg_scale=guidance, seed=seed),
    ]
    out_img: Optional[Image.Image] = None
    for kwargs in variants:
        try:
            allowed = {k: v for k, v in kwargs.items() if k in getattr(fn, "__code__").co_varnames}
            res = fn(**allowed)
            if isinstance(res, Image.Image):
                out_img = res
                break
            if isinstance(res, (list, tuple)) and res and isinstance(res[0], Image.Image):
                out_img = res[0]
                break
        except Exception:
            continue
    if out_img is None:
        raise RuntimeError("IDM-VTON callable invocation failed")
    out_img.save(out_path, format="PNG")


if __name__ == "__main__":
    # Args: person, cloth, out, category, steps, guidance, seed
    if len(sys.argv) < 8:
        raise SystemExit("usage: idm_runner.py <person> <cloth> <out> <category> <steps> <guidance> <seed>")
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]))

