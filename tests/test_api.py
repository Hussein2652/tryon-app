from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from api.app.main import app


client = TestClient(app)


def make_image(width: int = 512, height: int = 768, color=(128, 128, 128)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_size_recommend_direct_method():
    payload = {
        "brand_id": "ACME",
        "category": "upper_top",
        "fit": "regular",
        "stretch_pct": 0.05,
        "inputs": {"method": "direct", "chest_circ_cm": 96},
        "size_chart": [
            {"size": "S", "half_chest_cm": 49},
            {"size": "M", "half_chest_cm": 52},
            {"size": "L", "half_chest_cm": 55},
        ],
    }
    response = client.post("/size/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["recommended_size"] == "M"
    assert data["two_size_preview"] is True


def test_tryon_preview_placeholder(tmp_path):
    user_bytes = make_image()
    garment_bytes = make_image(color=(200, 50, 120))
    files = {
        "user_photo": ("user.png", user_bytes, "image/png"),
        "garment_front": ("garment.png", garment_bytes, "image/png"),
    }
    response = client.post("/tryon/preview", files=files, data={"sku": "SKU123", "size": "M"})
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["count"] == 5
    assert len(data["images"]) == 5
