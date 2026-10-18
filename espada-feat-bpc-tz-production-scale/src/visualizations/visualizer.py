from __future__ import annotations

import cv2
import numpy as np

from ..utils import (
    _mask_inner_point,
    _normalize_masks,
    _to_bgr,
    build_extended_detections,
    format_bbox_dict,
)


class PipelineVisualizer:
    """Visualizer for the YOLO -> SAM2 -> VLM pipeline intermediate steps."""

    def __init__(
        self,
        image: np.ndarray,
        detections_df=None,
    ) -> None:
        self.image = image
        self.detections: dict[str, list] | None = None
        self.masks: np.ndarray | None = None

        if detections_df is not None:
            self.detections = build_extended_detections(detections_df)

    def set_masks(self, masks: np.ndarray) -> None:
        """Sets SAM2 masks for subsequent mask methods."""

        self.masks = masks

    def draw_detections_with_info(
        self,
        thickness: int = 2,
        font_scale: float = 0.5,
        font_thickness: int = 1,
    ) -> np.ndarray:
        """Draws boxes with class names and confidence scores on the image."""

        det = self._require_detections()
        output = self.image.copy()
        n = len(det["id"])
        colors = self._distinct_colors(n)

        for i in range(n):
            x1, y1, x2, y2 = map(int, det["bbox"][i])
            label = f"{det['class'][i]} {det['conf'][i]:.2f}"
            color = colors[i]

            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            (tw, th), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_thickness,
            )
            text_y = y1 - 6
            if text_y - th - baseline < 0:
                text_y = y1 + th + baseline + 6

            pad = 2
            cv2.rectangle(
                output,
                (x1 - pad, text_y - th - baseline - pad),
                (x1 + tw + pad, text_y + pad),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                output,
                label,
                (x1, text_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

        return output

    def draw_colored_masks(
        self,
        *,
        alpha: float = 0.45,
        boxes: bool = False,
        box_thickness: int = 2,
        draw_contours: bool = True,
        contour_thickness: int = 2,
        draw_centroids: bool = False,
        centroid_radius: int = 8,
        centroid_labels: bool = True,
        text_scale: float = 0.7,
        text_thickness: int = 2,
        start_index: int = 1,
        colors: list[tuple[int, int, int]] | None = None,
    ) -> tuple[np.ndarray, dict[str, list]]:
        """Overlays colored masks on the image; returns (vis, meta)."""

        masks_bool = _normalize_masks(self._require_masks())
        bgr = _to_bgr(self.image)
        n = len(masks_bool)

        if colors is None:
            colors = self._distinct_colors(n)

        box_list = self.detections["bbox"] if boxes and self.detections else None

        out = bgr.copy()
        meta: dict[str, list] = {"id": [], "points": []}

        for i, mask in enumerate(masks_bool):
            if not mask.any():
                continue

            idx = start_index + i
            color = np.array(colors[i % len(colors)], dtype=np.uint8)

            overlay = out.copy()
            overlay[mask] = (overlay[mask] * (1.0 - alpha) + color * alpha).astype(np.uint8)
            out = overlay

            if draw_contours:
                cnts, _ = cv2.findContours(
                    mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(out, cnts, -1, tuple(int(c) for c in color), contour_thickness)

            if box_list is not None and i < len(box_list):
                bx1, by1, bx2, by2 = map(int, box_list[i])
                cv2.rectangle(
                    out,
                    (bx1, by1),
                    (bx2, by2),
                    tuple(int(c) for c in color),
                    box_thickness,
                )

            centroid = _mask_inner_point(mask)
            if centroid is not None:
                cx, cy = centroid
                meta["id"].append(idx)
                meta["points"].append([cx, cy])

                if draw_centroids:
                    cv2.circle(out, (cx, cy), centroid_radius, (0, 0, 255), -1)
                    if centroid_labels:
                        label = str(idx)
                        (tw, th), _ = cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            text_scale,
                            text_thickness,
                        )
                        cv2.putText(
                            out,
                            label,
                            (int(cx - tw / 2), int(cy + th / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            text_scale,
                            (255, 255, 255),
                            text_thickness,
                            cv2.LINE_AA,
                        )

        return out, meta

    def draw_single_object_mask(
        self,
        object_index: int,
        *,
        alpha: float = 0.45,
        color: tuple[int, int, int] | None = None,
        draw_contour: bool = True,
        draw_centroid: bool = True,
        centroid_radius: int = 8,
    ) -> np.ndarray:
        """Draws one object mask by zero-based index."""

        masks_bool = _normalize_masks(self._require_masks())

        assert (
            0 <= object_index < len(masks_bool)
        ), f"object_index {object_index} out of range [0, {len(masks_bool)})"

        mask = masks_bool[object_index]
        bgr = _to_bgr(self.image)

        if color is None:
            palette = self._distinct_colors(len(masks_bool))
            color = palette[object_index]

        c = np.array(color, dtype=np.uint8)
        out = bgr.copy()
        overlay = out.copy()
        overlay[mask] = (overlay[mask] * (1.0 - alpha) + c * alpha).astype(np.uint8)
        out = overlay

        if draw_contour:
            cnts, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(out, cnts, -1, tuple(int(v) for v in color), 2)

        if draw_centroid:
            centroid = _mask_inner_point(mask)
            if centroid is not None:
                cx, cy = centroid
                cv2.circle(out, (cx, cy), centroid_radius, (0, 0, 255), -1)
                cv2.circle(out, (cx, cy), centroid_radius, (255, 255, 255), 2)

        return out

    def format_bbox_dict(self) -> dict[int, list[int]]:
        """Returns {id: [x1, y1, x2, y2]} mapping."""

        return format_bbox_dict(self._require_detections())

    def compute_box_statistics(self) -> dict:
        """Returns detection count, areas, and spatial quadrant distribution."""

        det = self._require_detections()
        h, w = self.image.shape[:2]
        image_area = h * w

        areas = []
        quadrants = {"top_left": 0, "top_right": 0, "bottom_left": 0, "bottom_right": 0}
        cx_mid, cy_mid = w / 2, h / 2

        for box in det["bbox"]:
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            areas.append(area)

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            h_key = "left" if cx < cx_mid else "right"
            v_key = "top" if cy < cy_mid else "bottom"
            quadrants[f"{v_key}_{h_key}"] += 1

        mean_pct = float(np.mean(areas) / image_area * 100) if areas else 0.0

        return {
            "count": len(det["bbox"]),
            "areas_px": areas,
            "mean_area_pct": round(mean_pct, 2),
            "quadrant_counts": quadrants,
        }

    def compute_mask_coverage(self) -> dict:
        """Returns mask-to-box coverage percentage per object."""

        det = self._require_detections()
        masks_bool = _normalize_masks(self._require_masks())
        boxes = det["bbox"]
        n = min(len(masks_bool), len(boxes))

        per_object = []
        for i in range(n):
            mask = masks_bool[i]
            x1, y1, x2, y2 = map(int, boxes[i])
            box_area = max((x2 - x1) * (y2 - y1), 1)
            mask_area = int(mask.sum())
            coverage = mask_area / box_area * 100

            per_object.append(
                {
                    "id": i + 1,
                    "mask_area_px": mask_area,
                    "box_area_px": box_area,
                    "coverage_pct": round(coverage, 1),
                }
            )

        mean_cov = float(np.mean([o["coverage_pct"] for o in per_object])) if per_object else 0.0

        return {
            "per_object": per_object,
            "mean_coverage_pct": round(mean_cov, 1),
        }

    @staticmethod
    def compute_caption_stats(text: str) -> dict:
        """Returns character, word, sentence, and approximate token counts."""

        words = text.split()
        sentences = [
            s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()
        ]

        return {
            "char_count": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "approx_token_count": int(len(words) * 1.3),
        }

    @staticmethod
    def _distinct_colors(n: int) -> list[tuple[int, int, int]]:
        """Returns N distinct BGR colors spaced evenly in HSV."""

        colors = []
        for i in range(n):
            hue = int(180 * i / max(n, 1))
            hsv = np.array([[[hue, 220, 240]]], dtype=np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            b, g, r = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
            colors.append((b, g, r))
        return colors

    def _require_detections(self) -> dict[str, list]:
        if self.detections is None:
            raise RuntimeError("Detections not set. Pass detections_df to constructor.")
        return self.detections

    def _require_masks(self) -> np.ndarray:
        if self.masks is None:
            raise RuntimeError("Masks not set. Call set_masks() first.")
        return self.masks
