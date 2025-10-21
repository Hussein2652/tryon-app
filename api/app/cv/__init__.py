"""
Computer-vision helpers for the try-on pipeline.

These modules provide thin wrappers around segmentation (SCHP),
pose estimation (OpenPose / SMPL-X), and identity features.
Each helper attempts to import the corresponding heavy dependency
and falls back to lightweight placeholders when unavailable, so the
rest of the application can run in environments without the ML stack.
"""

from .preprocess import PreprocessArtifacts, PreprocessManager

__all__ = ["PreprocessArtifacts", "PreprocessManager"]
