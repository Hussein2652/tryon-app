from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

CATEGORY_CONFIG: Dict[str, Dict[str, str]] = {
    "upper_top": {"input_key": "chest_circ_cm", "chart_key": "half_chest_cm"},
    "top": {"input_key": "chest_circ_cm", "chart_key": "half_chest_cm"},
    "upper_outerwear": {"input_key": "chest_circ_cm", "chart_key": "half_chest_cm"},
    "bottoms": {"input_key": "waist_circ_cm", "chart_key": "half_waist_cm"},
    "pants": {"input_key": "waist_circ_cm", "chart_key": "half_waist_cm"},
    "dress": {"input_key": "bust_circ_cm", "chart_key": "half_bust_cm"},
}

DEFAULT_CATEGORY = "upper_top"

FIT_EASE_CM: Dict[str, Dict[str, float]] = {
    "upper_top": {"regular": 6.0, "slim": 4.5, "relaxed": 8.0},
    "upper_outerwear": {"regular": 8.0, "slim": 6.0, "relaxed": 10.0},
    "bottoms": {"regular": 4.0, "slim": 2.5, "relaxed": 5.5},
    "pants": {"regular": 4.0, "slim": 2.5, "relaxed": 5.5},
    "dress": {"regular": 6.0, "slim": 4.5, "relaxed": 7.5},
}

METHOD_BASE_CONFIDENCE: Dict[str, float] = {
    "direct": 0.9,
    "photo": 0.85,
    "smplx": 0.87,
    "manual": 0.8,
    "favorite_tee": 0.82,
}


class SizeChartEntry(BaseModel):
    size: str
    half_chest_cm: Optional[float] = None
    half_waist_cm: Optional[float] = None
    half_bust_cm: Optional[float] = None


class SizeInputs(BaseModel):
    method: Literal["direct", "photo", "smplx", "manual", "favorite_tee"]
    chest_circ_cm: Optional[float] = Field(default=None, ge=20, le=200)
    waist_circ_cm: Optional[float] = Field(default=None, ge=20, le=200)
    hip_circ_cm: Optional[float] = Field(default=None, ge=20, le=250)
    bust_circ_cm: Optional[float] = Field(default=None, ge=20, le=200)
    height_cm: Optional[float] = Field(default=None, ge=100, le=220)
    exif: Optional[Dict[str, float]] = None

    @model_validator(mode="after")
    def validate_measurement_presence(self) -> "SizeInputs":
        if self.method in {"direct", "manual", "photo", "smplx"}:
            if not any(
                getattr(self, field) is not None
                for field in ("chest_circ_cm", "waist_circ_cm", "bust_circ_cm")
            ):
                raise ValueError(
                    "At least one circumference measurement is required for this method."
                )
        if self.method == "favorite_tee" and getattr(self, "chest_circ_cm", None) is None:
            raise ValueError(
                "favorite_tee method expects chest_circ_cm converted from pit-to-pit."
            )
        return self


class SizeRecommendRequest(BaseModel):
    brand_id: str
    category: Optional[str] = Field(default=None)
    fit: Optional[str] = Field(default="regular")
    stretch_pct: float = Field(default=0.0, ge=0.0, le=0.4)
    inputs: SizeInputs
    size_chart: List[SizeChartEntry]

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, value: Optional[str]) -> str:
        if not value:
            return DEFAULT_CATEGORY
        return value.lower()

    @field_validator("fit", mode="before")
    @classmethod
    def normalize_fit(cls, value: Optional[str]) -> str:
        return (value or "regular").lower()

    @model_validator(mode="after")
    def validate_size_chart(self) -> "SizeRecommendRequest":
        config = resolve_category_config(self.category or DEFAULT_CATEGORY)
        chart_key = config["chart_key"]
        available = [getattr(entry, chart_key) for entry in self.size_chart]
        if not any(value is not None for value in available):
            raise ValueError(f"size_chart entries must include {chart_key}.")
        return self


class SizeRecommendResponse(BaseModel):
    ok: bool = True
    recommended_size: str
    target_half_cm: float
    chosen: Dict[str, Any]
    nearest_alt: Optional[Dict[str, Any]] = None
    two_size_preview: bool
    category: str
    fit: str
    stretch_pct: float
    method: str
    confidence: float
    confidence_factors: List[str]
    explanation: str


def resolve_category_config(category: str) -> Dict[str, str]:
    normalized = category.lower()
    if normalized in CATEGORY_CONFIG:
        return CATEGORY_CONFIG[normalized]
    # Default to upper_top if unknown to avoid runtime errors.
    return CATEGORY_CONFIG[DEFAULT_CATEGORY]


@dataclass
class Recommendation:
    recommended_size: str
    target_half_cm: float
    chosen_entry: SizeChartEntry
    nearest_alt: Optional[SizeChartEntry]
    two_size_preview: bool
    confidence: float
    confidence_factors: List[str]
    explanation: str


class SizingEngine:
    def recommend(self, request: SizeRecommendRequest) -> Recommendation:
        config = resolve_category_config(request.category or DEFAULT_CATEGORY)
        measurement, measurement_field = self._extract_measurement(
            request.inputs, config["input_key"]
        )
        target_half = self._compute_target_half(
            category=request.category or DEFAULT_CATEGORY,
            fit=request.fit,
            stretch_pct=request.stretch_pct,
            circumference_cm=measurement,
        )

        sorted_chart = self._sorted_chart(
            chart=request.size_chart, key=config["chart_key"]
        )
        chosen, alt = self._pick_size(sorted_chart, config["chart_key"], target_half)
        margin = max(0.0, getattr(chosen, config["chart_key"]) - target_half)
        step = (
            getattr(chosen, config["chart_key"])
            - getattr(alt, config["chart_key"])
            if alt
            else getattr(chosen, config["chart_key"])
        )
        confidence, factors = self._compute_confidence(
            base=request.inputs.method,
            margin=margin,
            step=step,
        )

        explanation = self._build_explanation(
            category=request.category or DEFAULT_CATEGORY,
            measurement_field=measurement_field,
            measurement_value=measurement,
            fit=request.fit,
            stretch_pct=request.stretch_pct,
        )

        two_size_preview = alt is not None and margin < max(2.0, step * 0.5)

        return Recommendation(
            recommended_size=chosen.size,
            target_half_cm=round(target_half, 2),
            chosen_entry=chosen,
            nearest_alt=alt,
            two_size_preview=two_size_preview,
            confidence=round(confidence, 2),
            confidence_factors=factors,
            explanation=explanation,
        )

    @staticmethod
    def _extract_measurement(
        inputs: SizeInputs, key: str
    ) -> Tuple[float, str]:
        value = getattr(inputs, key, None)
        if value is None:
            raise ValueError(f"inputs.{key} is required for the selected category.")
        return value, key

    def _compute_target_half(
        self,
        *,
        category: str,
        fit: str,
        stretch_pct: float,
        circumference_cm: float,
    ) -> float:
        category_key = category if category in FIT_EASE_CM else DEFAULT_CATEGORY
        fit_eases = FIT_EASE_CM[category_key]
        ease_cm = fit_eases.get(fit, fit_eases["regular"])
        ease_half = ease_cm / 2.0
        circumference_half = circumference_cm / 2.0

        # Stretch reduces the necessary garment width slightly as stretch increases.
        stretch_modifier = circumference_half * stretch_pct * 0.08
        target_half = circumference_half + ease_half - stretch_modifier

        return target_half

    @staticmethod
    def _sorted_chart(
        chart: List[SizeChartEntry],
        key: str,
    ) -> List[SizeChartEntry]:
        valid_entries = [
            entry for entry in chart if getattr(entry, key, None) is not None
        ]
        if not valid_entries:
            raise ValueError(f"size_chart entries must include {key}.")
        return sorted(valid_entries, key=lambda entry: getattr(entry, key))

    @staticmethod
    def _pick_size(
        chart: List[SizeChartEntry],
        key: str,
        target_half: float,
    ) -> Tuple[SizeChartEntry, Optional[SizeChartEntry]]:
        chosen = chart[-1]
        fallback = None
        for entry in chart:
            value = getattr(entry, key)
            if value is None:
                continue
            if value >= target_half:
                chosen = entry
                break
            fallback = entry
        return chosen, fallback

    def _compute_confidence(
        self,
        *,
        base: str,
        margin: float,
        step: float,
    ) -> Tuple[float, List[str]]:
        base_confidence = METHOD_BASE_CONFIDENCE.get(base, 0.75)
        step = max(step, 0.01)
        proximity_ratio = min(1.0, margin / step)
        penalty = (1.0 - proximity_ratio) * 0.2
        confidence = max(0.05, min(0.99, base_confidence - penalty))
        factors = [
            f"method={base} base={base_confidence:.2f}",
            f"boundary_margin_cm={margin:.1f} step≈{step:.1f}",
        ]
        return confidence, factors

    @staticmethod
    def _build_explanation(
        *,
        category: str,
        measurement_field: str,
        measurement_value: float,
        fit: str,
        stretch_pct: float,
    ) -> str:
        readable_field = measurement_field.replace("_circ_cm", "")
        category_label = category.replace("_", " ")
        return (
            f"{category_label}: {readable_field} {measurement_value:.1f}cm "
            f"+ ease({fit}) / stretch {int(stretch_pct * 100)}%"
        )


def build_response(
    request: SizeRecommendRequest, recommendation: Recommendation
) -> SizeRecommendResponse:
    config = resolve_category_config(request.category or DEFAULT_CATEGORY)
    chosen = recommendation.chosen_entry
    chosen_payload = {
        "size": chosen.size,
        config["chart_key"]: getattr(chosen, config["chart_key"]),
    }

    alt_payload = None
    if recommendation.nearest_alt is not None:
        alt_payload = {
            "size": recommendation.nearest_alt.size,
            config["chart_key"]: getattr(
                recommendation.nearest_alt, config["chart_key"]
            ),
        }

    return SizeRecommendResponse(
        recommended_size=recommendation.recommended_size,
        target_half_cm=recommendation.target_half_cm,
        chosen=chosen_payload,
        nearest_alt=alt_payload,
        two_size_preview=recommendation.two_size_preview,
        category=request.category or DEFAULT_CATEGORY,
        fit=request.fit,
        stretch_pct=request.stretch_pct,
        method=request.inputs.method,
        confidence=recommendation.confidence,
        confidence_factors=recommendation.confidence_factors,
        explanation=recommendation.explanation,
    )
