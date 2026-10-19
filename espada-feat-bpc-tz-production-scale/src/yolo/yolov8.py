from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


class YoloV8:
    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cuda",
        batch_size: int = 1,
        empty_cache: bool = False,
    ):
        """Initializes a YOLOv8 detector with a chosen device and batch size."""

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        self.device = self._select_device(device)
        self.model = YOLO(str(model_path) if model_path is not None else "yolov8m.pt")
        self.batch_size = batch_size
        self.empty_cache = empty_cache

    def _maybe_empty_cache(self) -> None:
        """Frees CUDA cache after inference when enabled and on a CUDA device."""

        if self.empty_cache and str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, image: np.ndarray) -> pd.DataFrame:
        """Runs detection on a single image and returns detections as a DataFrame."""

        results = self.model.predict(
            source=[image], device=self.device, save=False, verbose=False, conf=0.22, iou=0.45
        )
        df = self._result_to_dataframe(results[0])
        self._maybe_empty_cache()
        return df

    def predict_on_batch(self, input_batch: list[np.ndarray]) -> list[pd.DataFrame]:
        """Runs detection on a list of images and returns one DataFrame per image."""

        if not input_batch:
            return []

        dfs: list[pd.DataFrame] = []
        for chunk in self._iter_chunks(input_batch, self.batch_size):
            results = self.model.predict(
                source=chunk,
                device=self.device,
                save=False,
                verbose=False,
                batch=len(chunk),
                conf=0.22,
                iou=0.45,
            )
            dfs.extend(self._result_to_dataframe(r) for r in results)

        self._maybe_empty_cache()
        return dfs

    @staticmethod
    def _select_device(device: str) -> str:
        """Chooses a valid runtime device string based on availability."""

        device = device.strip().lower()
        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        if device in {"cpu", "cuda"}:
            return "cpu"
        return device  # e.g. "cuda:0"

    @staticmethod
    def _iter_chunks(items: list[np.ndarray], chunk_size: int):
        """Yields successive chunks from a list."""

        for i in range(0, len(items), chunk_size):
            yield items[i : i + chunk_size]

    def _result_to_dataframe(self, result) -> pd.DataFrame:
        """Converts a single Ultralytics result to a standardized DataFrame."""

        rows: list[dict[str, object]] = []
        boxes = getattr(result, "boxes", None)

        if boxes is None or len(boxes) == 0:
            return pd.DataFrame(columns=["class", "class_id", "conf", "x1", "y1", "x2", "y2"])

        names = getattr(self.model, "names", {})
        for idx, b in enumerate(boxes):
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])

            rows.append(
                {
                    "ids": idx,
                    "class": names.get(cls_id, str(cls_id)),
                    "class_id": cls_id,
                    "conf": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

        return pd.DataFrame(rows)
