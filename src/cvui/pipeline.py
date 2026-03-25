"""Detection pipeline framework — context, stage ABC, and pipeline orchestrator.

This module contains only the structural pieces. Concrete stage implementations
live in cvui.stages.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class DetectionContext:
    """Shared state flowing through detection stages."""
    img: np.ndarray
    gray: np.ndarray | None = None
    binary: np.ndarray | None = None
    rects: list[tuple[int, int, int, int]] = field(default_factory=list)
    classifications: dict[int, str] = field(default_factory=dict)
    nested: dict[int, list[int]] = field(default_factory=dict)
    ui_states: dict[str, list[tuple[int, int, int, int]]] = field(default_factory=dict)
    quality_score: float = 0.0
    stage_log: list[str] = field(default_factory=list)
    scale: float = 1.0  # downscale factor, rects are in scaled coords until pipeline end

    @property
    def height(self) -> int:
        return self.img.shape[0]

    @property
    def width(self) -> int:
        return self.img.shape[1]

    def to_prompt(
        self,
        ocr_lines: list[tuple[int, int, int, int, str]] | None = None,
        zones: list | None = None,
    ) -> str:
        """Generate LLM-friendly structured description of the UI.

        Pipeline: deduplicate → spatial group into zones → sort top-to-bottom
        left-to-right → label with OCR → mark skeleton vs dynamic zones →
        mark truncated elements.

        Args:
            ocr_lines: OCR results as [(x1, y1, x2, y2, text), ...]
            zones: UIZone list for skeleton/dynamic grouping (optional)
        """
        w, h = self.width, self.height
        median = float(np.median(self.img.mean(axis=2))) if self.img is not None else 128
        theme = "dark" if median < 128 else "light"

        # Step 1: Deduplicate — remove big parents that have children inside
        rects = self._deduplicate(self.rects, w, h)

        # Step 2: Label with OCR
        labels = self._label_with_ocr(rects, ocr_lines) if ocr_lines else {}

        # Step 3: Classify (reindex after dedup)
        old_to_new = {}
        for new_i, r in enumerate(rects):
            if r in self.rects:
                old_i = self.rects.index(r)
                old_to_new[new_i] = self.classifications.get(old_i, "")

        # Step 4: Build output
        lines = []
        lines.append(f"Window: {w}x{h}, {theme} theme, {len(rects)} elements")
        if ocr_lines:
            lines.append(f"OCR: {len(ocr_lines)} words (approximate, may contain errors)")
        lines.append("")

        # Step 5: Group into zones or auto-detect columns
        if zones:
            self._format_with_zones(lines, rects, labels, old_to_new, zones, w, h)
        else:
            self._format_by_columns(lines, rects, labels, old_to_new, w, h)

        return "\n".join(lines)

    @staticmethod
    def _deduplicate(rects, img_w, img_h):
        """Remove parent rects that contain children.

        A large rect that has >=2 smaller rects inside it is a container
        artifact from NestedStage — drop it, keep the children.
        """
        if not rects:
            return rects

        rects = list(rects)
        to_remove = set()
        for i, r in enumerate(rects):
            area = (r[2] - r[0]) * (r[3] - r[1])
            if area < img_w * img_h * 0.02:
                continue  # small rects can't be containers
            children = 0
            for j, c in enumerate(rects):
                if i == j:
                    continue
                # c is inside r?
                if c[0] >= r[0] and c[1] >= r[1] and c[2] <= r[2] and c[3] <= r[3]:
                    children += 1
            if children >= 2:
                to_remove.add(i)

        return [r for i, r in enumerate(rects) if i not in to_remove]

    @staticmethod
    def _label_with_ocr(rects, ocr_lines):
        """Assign OCR text to each rect by center-point containment."""
        labels = {}
        if not ocr_lines:
            return labels
        for i, r in enumerate(rects):
            rx1, ry1, rx2, ry2 = r
            texts = []
            for ox1, oy1, ox2, oy2, text in ocr_lines:
                ocx, ocy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
                if rx1 <= ocx <= rx2 and ry1 <= ocy <= ry2:
                    texts.append(text)
            if texts:
                label = "".join(texts[:6])
                if len(texts) > 6:
                    label += "..."
                labels[i] = label
        return labels

    @staticmethod
    def _format_with_zones(lines, rects, labels, classifications, zones, img_w, img_h):
        """Format output grouped by zones (skeleton/dynamic)."""
        # Assign each rect to a zone
        assigned = set()
        for z in zones:
            zx1, zy1, zx2, zy2 = z.rect
            tag = "dynamic" if z.dynamic else "skeleton"
            zw, zh = zx2 - zx1, zy2 - zy1

            zone_elems = []
            for i, r in enumerate(rects):
                cx, cy = (r[0] + r[2]) // 2, (r[1] + r[3]) // 2
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    zone_elems.append(i)
                    assigned.add(i)

            if not zone_elems:
                continue

            # Sort top-to-bottom, left-to-right
            zone_elems.sort(key=lambda i: (rects[i][1], rects[i][0]))

            lines.append(f"[{tag}] ({zx1},{zy1})-({zx2},{zy2}) {zw}x{zh}")
            for i in zone_elems:
                r = rects[i]
                ew, eh = r[2] - r[0], r[3] - r[1]
                cls = classifications.get(i, "")
                label = labels.get(i, "")
                truncated = r[0] <= 5 or r[1] <= 5 or r[2] >= img_w - 5 or r[3] >= img_h - 5
                parts = [f"  [{i}]"]
                if cls:
                    parts.append(f"[{cls}]")
                if truncated:
                    parts.append("[truncated]")
                parts.append(f"({r[0]},{r[1]})")
                parts.append(f"{ew}x{eh}")
                if label:
                    parts.append(f'"{label}"')
                lines.append(" ".join(parts))
            lines.append("")

        # Unassigned rects
        unassigned = [i for i in range(len(rects)) if i not in assigned]
        if unassigned:
            lines.append("[unassigned]")
            for i in sorted(unassigned, key=lambda i: (rects[i][1], rects[i][0])):
                r = rects[i]
                label = labels.get(i, "")
                cls = classifications.get(i, "")
                parts = [f"  [{i}]"]
                if cls:
                    parts.append(f"[{cls}]")
                parts.append(f"({r[0]},{r[1]}) {r[2]-r[0]}x{r[3]-r[1]}")
                if label:
                    parts.append(f'"{label}"')
                lines.append(" ".join(parts))
            lines.append("")

    @staticmethod
    def _format_by_columns(lines, rects, labels, classifications, img_w, img_h):
        """Auto-detect columns and group elements spatially."""
        if not rects:
            lines.append("(no elements detected)")
            return

        # Auto-detect vertical columns from element X positions
        x_centers = [(r[0] + r[2]) // 2 for r in rects]
        # Bucket into columns (100px wide)
        bucket_size = max(80, img_w // 8)
        columns: dict[int, list[int]] = {}
        for i, xc in enumerate(x_centers):
            col = xc // bucket_size
            columns.setdefault(col, []).append(i)

        for col_id in sorted(columns.keys()):
            col_elems = columns[col_id]
            col_x1 = col_id * bucket_size
            col_x2 = min(col_x1 + bucket_size, img_w)

            # Sort by Y position
            col_elems.sort(key=lambda i: rects[i][1])

            lines.append(f"--- Column x={col_x1}-{col_x2} ({len(col_elems)} elements) ---")
            for i in col_elems:
                r = rects[i]
                ew, eh = r[2] - r[0], r[3] - r[1]
                cls = classifications.get(i, "")
                label = labels.get(i, "")
                truncated = r[0] <= 5 or r[1] <= 5 or r[2] >= img_w - 5 or r[3] >= img_h - 5

                parts = [f"  [{i}]"]
                if cls:
                    parts.append(f"[{cls}]")
                if truncated:
                    parts.append(f"[truncated]")
                parts.append(f"({r[0]},{r[1]})")
                parts.append(f"{ew}x{eh}")
                if label:
                    parts.append(f'"{label}"')
                lines.append(" ".join(parts))
            lines.append("")

    def to_report(self) -> dict:
        """Generate structured JSON-serializable report."""
        w, h = self.width, self.height
        median = float(np.median(self.img.mean(axis=2))) if self.img is not None else 128
        rects = self._deduplicate(self.rects, w, h)

        return {
            "window": {"width": w, "height": h, "theme": "dark" if median < 128 else "light"},
            "pipeline": {"stages": self.stage_log, "quality": round(self.quality_score, 3)},
            "elements": [
                {
                    "id": i,
                    "rect": list(r),
                    "type": self.classifications.get(self.rects.index(r), "") if r in self.rects else "",
                    "size": [r[2] - r[0], r[3] - r[1]],
                }
                for i, r in enumerate(rects)
            ],
            "ui_states": {k: [list(r) for r in v] for k, v in self.ui_states.items() if v},
        }


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class DetectionStage(ABC):
    """One step in the detection pipeline."""

    @abstractmethod
    def process(self, ctx: DetectionContext) -> DetectionContext:
        """Execute this stage, mutate and return ctx."""

    def should_continue(self, ctx: DetectionContext) -> bool:
        """After processing, should the pipeline continue? Default: yes."""
        return True


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class DetectionPipeline:
    """Run a sequence of DetectionStages with early-exit support."""

    def __init__(self, stages: list[DetectionStage] | None = None):
        self.stages = stages or []

    def run(self, img: np.ndarray) -> DetectionContext:
        ctx = DetectionContext(img=img)
        for stage in self.stages:
            ctx = stage.process(ctx)
            ctx.stage_log.append(type(stage).__name__)
            if not stage.should_continue(ctx):
                log.info("Pipeline: early exit after %s (quality=%.2f)",
                         type(stage).__name__, ctx.quality_score)
                break

        # Map rects back to original resolution if downscaled
        if ctx.scale != 1.0 and ctx.rects:
            s = 1.0 / ctx.scale
            ctx.rects = [
                (int(r[0] * s), int(r[1] * s), int(r[2] * s), int(r[3] * s))
                for r in ctx.rects
            ]
            # Also map ui_states
            for key in ctx.ui_states:
                ctx.ui_states[key] = [
                    (int(r[0] * s), int(r[1] * s), int(r[2] * s), int(r[3] * s))
                    for r in ctx.ui_states[key]
                ]
            ctx.scale = 1.0

        return ctx
