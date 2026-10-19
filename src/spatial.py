"""Пространственные отношения из bbox'ов — без VLM, чисто геометрия.

Используется для:
- spatial_relation примеров
- qa_spatial
- object_state (on table, in hand, etc.)
- Валидация bbox-grounding

QAOA-аналогия: детерминированный, быстрый, проверяемый этап (как FastQAOA group mixer).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("factory.spatial")

@dataclass
class BBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    label: str
    score: float = 1.0

    @property
    def cx(self) -> float:
        return (self.x_min + self.x_max) / 2
    @property
    def cy(self) -> float:
        return (self.y_min + self.y_max) / 2
    @property
    def w(self) -> float:
        return self.x_max - self.x_min
    @property
    def h(self) -> float:
        return self.y_max - self.y_min
    @property
    def area(self) -> float:
        return max(0, self.w) * max(0, self.h)

    def iou(self, other: "BBox") -> float:
        inter_xmin = max(self.x_min, other.x_min)
        inter_ymin = max(self.y_min, other.y_min)
        inter_xmax = min(self.x_max, other.x_max)
        inter_ymax = min(self.y_max, other.y_max)
        inter_w = max(0, inter_xmax - inter_xmin)
        inter_h = max(0, inter_ymax - inter_ymin)
        inter = inter_w * inter_h
        union = self.area + other.area - inter
        return inter / union if union else 0.0


def describe_spatial_relations(bboxes: list[BBox], width: int, height: int) -> list[str]:
    """Генерирует человекочитаемые spatial описания.

    Примеры:
    - \"red block is to the left of bowl\"
    - \"gripper is above red block\"
    - \"cup is on the table, near the left edge\"
    """
    if not bboxes:
        return []
    relations: list[str] = []
    # Позиция относительно кадра
    for b in bboxes:
        cx_rel = b.cx / width
        cy_rel = b.cy / height
        pos = []
        if cx_rel < 0.33:
            pos.append("on the left side")
        elif cx_rel > 0.66:
            pos.append("on the right side")
        else:
            pos.append("in the center")
        if cy_rel < 0.33:
            pos.append("near the top")
        elif cy_rel > 0.66:
            pos.append("near the bottom")
        # Размер
        area_rel = b.area / (width * height)
        if area_rel > 0.15:
            size = "large"
        elif area_rel < 0.02:
            size = "small"
        else:
            size = ""
        # Не генерируем для каждого, только для крупных
        if b.area > (width*height*0.01):
            relations.append(f"{b.label} is {pos[0]} {pos[1] if len(pos)>1 else ''}".strip() + (f" ({size})" if size else ""))
    # Парные отношения
    for i, a in enumerate(bboxes):
        for j, b in enumerate(bboxes):
            if i >= j:
                continue
            # Горизонталь
            dx = b.cx - a.cx
            dy = b.cy - a.cy
            dist = (dx**2 + dy**2) ** 0.5
            # Игнор далёких
            if dist > max(width, height) * 0.5:
                continue
            if abs(dx) > abs(dy):
                if dx > 20:
                    relations.append(f"{a.label} is to the left of {b.label}")
                elif dx < -20:
                    relations.append(f"{a.label} is to the right of {b.label}")
            else:
                if dy > 20:
                    relations.append(f"{a.label} is above {b.label}")
                elif dy < -20:
                    relations.append(f"{a.label} is below {b.label}")
            # On top / inside via IoU + containment
            if a.iou(b) > 0.05:
                if a.area < b.area * 0.5 and b.x_min < a.cx < b.x_max and b.y_min < a.cy < b.y_max:
                    relations.append(f"{a.label} is inside {b.label}")
                elif abs(a.cy - b.cy) < 20 and abs(a.cx - b.cx) < 50:
                    relations.append(f"{a.label} is next to {b.label}")
            # Holding: gripper near object
            if "gripper" in a.label.lower() or "hand" in a.label.lower():
                if dist < 80:
                    relations.append(f"{a.label} is close to {b.label}")
            if "gripper" in b.label.lower() or "hand" in b.label.lower():
                if dist < 80:
                    relations.append(f"{b.label} is close to {a.label}")
    # Дедуп
    return sorted(set(r for r in relations if r))


def bbox_to_text(bboxes: list[BBox]) -> str:
    if not bboxes:
        return "No objects detected."
    return "; ".join(f"{b.label} at [{b.x_min},{b.y_min},{b.x_max},{b.y_max}]" for b in bboxes)


def heuristic_relations_for_qa(bboxes: list[BBox], width: int, height: int) -> list[tuple[str, str]]:
    """Генерирует QA-пары: (question, answer) для spatial bench (CrossPoint, EmbSpatial, BLINK)."""
    qas: list[tuple[str, str]] = []
    if len(bboxes) < 2:
        return qas
    rels = describe_spatial_relations(bboxes, width, height)
    for rel in rels[:6]:
        # Превращаем в QA
        # \"red block is to the left of bowl\" -> Q: \"Where is the red block relative to the bowl?\" A: \"To the left\"
        if " is " in rel:
            subj, pred = rel.split(" is ", 1)
            qas.append((f"Where is the {subj} relative to other objects? Describe its position.", rel))
            qas.append((f"Is the {subj} {pred}? Answer yes/no and explain.", f"Yes, {rel}."))
    # Counting
    from collections import Counter
    cnt = Counter(b.label for b in bboxes)
    for label, c in cnt.items():
        qas.append((f"How many {label}s are visible in the image? Count carefully.", f"There are {c} {label}(s) visible."))
    # Pointing-style
    for b in bboxes[:4]:
        qas.append((f"Point to the {b.label} in the image. Return coordinates as [x,y].", f"{{\"point_2d\": [{int(b.cx)}, {int(b.cy)}], \"label\": \"{b.label}\"}}"))
    return qas
