"""BPC pipeline configuration loader and dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class InputConfig:
    require_prepared_images: bool = True
    allowed_extensions: list[str] = field(default_factory=lambda: [".jpg", ".jpeg"])
    require_rgb: bool = True
    max_side: int = 768


@dataclass
class ModelStageConfig:
    workers: int = 1
    batch_size: int = 1
    devices: list[str] = field(default_factory=lambda: ["cuda:0"])
    model: str | None = None
    dtype: str | None = None
    empty_cache: bool = False


@dataclass
class VlmConfig:
    workers: int = 8
    # Classic deployment: one URL per model instance.
    endpoints: list[str] = field(default_factory=lambda: ["http://localhost:8001"])
    # Explicit target list with optional per-target concurrency overrides.
    # Each item may contain:
    #   - base_url: required
    #   - model: optional
    #   - max_concurrency: optional
    # When set, takes priority over endpoints/router_url/models.
    targets: list[dict[str, Any]] = field(default_factory=list)
    # Router deployment: ONE url, N model names selected via the "model" field
    # (e.g. gateway.frontierai.ru routing to "qwen3.5-4b-odie2", "...-odie6", ...).
    # When set, `models` (one entry per backend instance) takes priority over
    # `endpoints`/`model` for building the endpoint pool.
    router_url: str | None = None
    models: list[str] = field(default_factory=list)
    max_concurrency_per_endpoint: int = 2
    model: str | None = None
    timeout_s: int = 60


@dataclass
class JudgeConfig:
    enabled: bool = False
    tasks: list[str] = field(default_factory=lambda: ["bboxing", "pointing"])
    workers: int = 4
    endpoints: list[str] = field(default_factory=list)
    targets: list[dict[str, Any]] = field(default_factory=list)
    router_url: str | None = None
    models: list[str] = field(default_factory=list)
    max_concurrency_per_endpoint: int = 2
    allowed_edits: list[str] = field(default_factory=lambda: ["labels"])
    forbid_coordinate_changes: bool = True


@dataclass
class BboxValidationConfig:
    out_of_bounds_policy: str = "reject"
    min_box_area_px: int = 4
    max_boxes_per_image: int = 100


@dataclass
class PointingValidationConfig:
    out_of_bounds_policy: str = "reject"
    require_point_inside_related_bbox: bool = False
    max_points_per_image: int = 100


@dataclass
class CaptionValidationConfig:
    min_length: int = 5
    max_length: int = 1000
    forbidden_phrases: list[str] = field(default_factory=list)


@dataclass
class ValidationConfig:
    bbox: BboxValidationConfig = field(default_factory=BboxValidationConfig)
    pointing: PointingValidationConfig = field(default_factory=PointingValidationConfig)
    captions: CaptionValidationConfig = field(default_factory=CaptionValidationConfig)


@dataclass
class RetryConfig:
    max_attempts: int = 3
    backoff_seconds: float = 30.0


@dataclass
class StorageConfig:
    save_raw_vlm_responses: bool = False
    save_yolo_outputs: bool = False
    save_sam2_outputs: bool = False
    save_invalid_outputs: bool = False
    save_validated_copies: bool = False
    save_visualizations: bool = True


@dataclass
class BpcConfig:
    input: InputConfig = field(default_factory=InputConfig)
    yolo: ModelStageConfig = field(default_factory=ModelStageConfig)
    sam2: ModelStageConfig = field(default_factory=lambda: ModelStageConfig(batch_size=8))
    vlm: VlmConfig = field(default_factory=VlmConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


def _merge(defaults: Any, raw: Any) -> Any:
    """Recursively merge raw dict into a dataclass (shallow per-field)."""
    if not isinstance(raw, dict):
        return raw
    import dataclasses
    if not dataclasses.is_dataclass(defaults):
        return raw
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(defaults):
        val = raw.get(f.name, None)
        current = getattr(defaults, f.name)
        if val is None:
            kwargs[f.name] = current
        elif dataclasses.is_dataclass(current):
            kwargs[f.name] = _merge(current, val)
        else:
            kwargs[f.name] = val
    return type(defaults)(**kwargs)


def load_bpc_config(path: str | Path) -> BpcConfig:
    """Loads bpc.yaml and merges with defaults."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = BpcConfig()

    if "input" in raw:
        cfg.input = _merge(cfg.input, raw["input"])
    if "yolo" in raw:
        cfg.yolo = _merge(cfg.yolo, raw["yolo"])
    if "sam2" in raw:
        cfg.sam2 = _merge(cfg.sam2, raw["sam2"])
    if "vlm" in raw:
        cfg.vlm = _merge(cfg.vlm, raw["vlm"])
    if "judge" in raw:
        cfg.judge = _merge(cfg.judge, raw["judge"])
    if "validation" in raw:
        v = raw["validation"]
        if "bbox" in v:
            cfg.validation.bbox = _merge(cfg.validation.bbox, v["bbox"])
        if "pointing" in v:
            cfg.validation.pointing = _merge(cfg.validation.pointing, v["pointing"])
        if "captions" in v:
            cfg.validation.captions = _merge(cfg.validation.captions, v["captions"])
    if "retry" in raw:
        cfg.retry = _merge(cfg.retry, raw["retry"])
    if "storage" in raw:
        cfg.storage = _merge(cfg.storage, raw["storage"])

    return cfg
