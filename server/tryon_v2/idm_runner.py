from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from PIL import Image


def load_callable(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    # Also add gradio_demo dir so 'import utils_mask' and friends resolve
    gradio_dir = repo_root / "gradio_demo"
    added_gradio = False
    if gradio_dir.exists():
        sys.path.insert(0, str(gradio_dir))
        added_gradio = True
    # Provide a lightweight stub for gradio so importing the demo file does not try
    # to spin up a UI server. This avoids adding a heavy 'gradio' dependency to the
    # IDM-VTON venv and makes the import side-effect free.
    try:
        import types  # noqa: PLC0415
        class _Dummy:
            def __getattr__(self, _name):  # any attr -> self
                return self
            def __call__(self, *args, **kwargs):  # callable
                return self
            def queue(self, *args, **kwargs):
                return self
            def launch(self, *args, **kwargs):
                return None
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False

        dummy = _Dummy()
        # Create a real module object so import machinery is happy
        gr_stub = types.ModuleType("gradio")

        def _factory(*_a, **_k):
            return dummy

        # Common classes/widgets used by the demo map to a no-op factory
        for name in (
            "Blocks","Row","Column","Markdown","ImageEditor","Checkbox","Examples",
            "Image","Textbox","Button","Accordion","Number","Slider","CheckboxGroup",
        ):
            setattr(gr_stub, name, _factory)

        # PEP 562: allow dynamic fallback for any other attribute
        def __getattr__(name):  # type: ignore
            return dummy
        gr_stub.__getattr__ = __getattr__  # type: ignore[attr-defined]

        # Overwrite any existing gradio module with the stub to avoid stale imports
        sys.modules["gradio"] = gr_stub
    except Exception:
        pass
    try:
        mod_path = repo_root / "gradio_demo" / "app.py"
        if not mod_path.exists():
            raise RuntimeError("gradio_demo/app.py not found")
        import importlib.util

        # Ensure we always re-import a fresh module
        sys.modules.pop("idm_vton_gradio_app", None)
        spec = importlib.util.spec_from_file_location("idm_vton_gradio_app", str(mod_path))
        if spec is None or spec.loader is None:
            raise RuntimeError("unable to load gradio module")
        mod = importlib.util.module_from_spec(spec)
        # Monkeypatch torch.load to allow loading legacy pickled weights
        try:
            import torch  # type: ignore
            from torch import serialization as _ser  # type: ignore

            _orig_ser_load = _ser.load

            def _load_compat(*args, **kwargs):  # type: ignore
                kwargs.setdefault("weights_only", False)
                return _orig_ser_load(*args, **kwargs)

            # Patch both torch.load and torch.serialization.load
            _ser.load = _load_compat  # type: ignore
            torch.load = _load_compat  # type: ignore
        except Exception:
            pass
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        fn = None
        # Prefer explicit demo entrypoints; fall back to common names.
        for name in ("start_tryon", "process", "predict"):
            if hasattr(mod, name):
                fn = getattr(mod, name)
                break
        if fn is None:
            raise RuntimeError("no process/predict in gradio demo")
        return fn
    finally:
        try:
            sys.path.remove(str(repo_root))
            if added_gradio:
                try:
                    sys.path.remove(str(gradio_dir))
                except ValueError:
                    pass
        except ValueError:
            pass


def main(person_path: str, cloth_path: str, out_path: str, category: str, steps: int, guidance: float, seed: int) -> None:
    repo_root = Path("/third_party/idm_vton")
    fn = load_callable(repo_root)
    person = Image.open(person_path).convert("RGB")
    cloth = Image.open(cloth_path).convert("RGBA")
    out_img: Optional[Image.Image] = None
    # If the demo exposes start_tryon(dict, garm_img, garment_des, is_checked, is_checked_crop, denoise_steps, seed),
    # call it directly to avoid relying on the Gradio click wiring.
    try:
        if getattr(fn, "__name__", "") == "start_tryon":
            request_dict = {"background": person, "layers": [cloth], "composite": None}
            garm_img = cloth.convert("RGB")
            garment_des = "shirt"
            is_checked = True  # use auto-generated mask
            is_checked_crop = False
            res = fn(request_dict, garm_img, garment_des, is_checked, is_checked_crop, int(steps), int(seed))
            if isinstance(res, (list, tuple)) and res:
                if isinstance(res[0], Image.Image):
                    out_img = res[0]
            elif isinstance(res, Image.Image):
                out_img = res
        else:
            # Try a few kwarg patterns for other demos
            variants = [
                dict(person_image=person, cloth_image=cloth, category=category, steps=steps, guidance=guidance, seed=seed),
                dict(human_img=person, cloth_img=cloth, category=category, steps=steps, cfg=guidance, seed=seed),
                dict(im=person, c=cloth, cate=category, num_steps=steps, cfg_scale=guidance, seed=seed),
            ]
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
    except Exception:
        out_img = None
    if out_img is None:
        raise RuntimeError("IDM-VTON callable invocation failed")
    out_img.save(out_path, format="PNG")


if __name__ == "__main__":
    # Args: person, cloth, out, category, steps, guidance, seed
    if len(sys.argv) < 8:
        raise SystemExit("usage: idm_runner.py <person> <cloth> <out> <category> <steps> <guidance> <seed>")
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]))
