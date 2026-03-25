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

    def to_prompt(self, ocr_lines: list[tuple[int, int, int, int, str]] | None = None) -> str:
        """Generate LLM-friendly description of detected UI elements.

        Args:
            ocr_lines: optional OCR results as [(x1, y1, x2, y2, text), ...]
                       If None, element text will be empty.

        Returns:
            Multi-line string describing the UI layout for LLM consumption.
        """
        w, h = self.width, self.height
        median = float(np.median(self.img.mean(axis=2))) if self.img is not None else 128
        theme = "dark" if median < 128 else "light"
        n = len(self.rects)
        stages_str = " → ".join(self.stage_log) if self.stage_log else "none"

        # Build OCR lookup: for each rect, find overlapping text
        rect_labels: dict[int, str] = {}
        if ocr_lines:
            for i, r in enumerate(self.rects):
                rx1, ry1, rx2, ry2 = r
                texts = []
                for ox1, oy1, ox2, oy2, text in ocr_lines:
                    # OCR word must be mostly inside the element rect
                    # (center point inside, not just any overlap)
                    ocx = (ox1 + ox2) // 2
                    ocy = (oy1 + oy2) // 2
                    if rx1 <= ocx <= rx2 and ry1 <= ocy <= ry2:
                        texts.append(text)
                if texts:
                    # Cap at 8 words to avoid flooding large containers
                    label = " ".join(texts[:8])
                    if len(texts) > 8:
                        label += " ..."
                    rect_labels[i] = label

        lines = []
        lines.append(f"UI Screenshot Analysis ({w}x{h}, {theme} theme)")
        lines.append(f"Pipeline: {stages_str}")
        lines.append(f"Quality: {self.quality_score:.2f}")
        if ocr_lines:
            lines.append(f"OCR: {len(ocr_lines)} words (may contain errors, use for reference)")
        lines.append("")

        # Count by classification
        type_counts: dict[str, int] = {}
        for cls in self.classifications.values():
            type_counts[cls] = type_counts.get(cls, 0) + 1

        if type_counts:
            parts = [f"{count} {typ}" for typ, count in sorted(type_counts.items())]
            lines.append(f"Summary: {n} elements ({', '.join(parts)})")
        else:
            lines.append(f"Summary: {n} elements")
        lines.append("")

        # UI states
        for state_name, state_rects in self.ui_states.items():
            if state_rects and state_name in ("highlight", "badge", "link"):
                lines.append(f"State '{state_name}': {len(state_rects)} regions")

        if any(self.ui_states.get(k) for k in ("highlight", "badge", "link")):
            lines.append("")

        # Elements
        lines.append("Elements:")
        for i, r in enumerate(self.rects):
            x1, y1, x2, y2 = r
            w_r, h_r = x2 - x1, y2 - y1
            cls = self.classifications.get(i, "")
            label = rect_labels.get(i, "")
            cls_str = f" [{cls}]" if cls else ""
            label_str = f' "{label}"' if label else ""
            lines.append(f"  [{i:2d}]{cls_str} ({x1},{y1})-({x2},{y2}) {w_r}x{h_r}{label_str}")

        return "\n".join(lines)

    def to_report(self) -> dict:
        """Generate structured report as a dict (JSON-serializable)."""
        w, h = self.width, self.height
        median = float(np.median(self.img.mean(axis=2))) if self.img is not None else 128

        return {
            "window": {
                "width": w,
                "height": h,
                "theme": "dark" if median < 128 else "light",
            },
            "pipeline": {
                "stages": self.stage_log,
                "quality_score": round(self.quality_score, 3),
            },
            "elements": [
                {
                    "id": i,
                    "rect": list(r),
                    "type": self.classifications.get(i, ""),
                    "area": (r[2] - r[0]) * (r[3] - r[1]),
                }
                for i, r in enumerate(self.rects)
            ],
            "ui_states": {
                k: [list(r) for r in v]
                for k, v in self.ui_states.items()
                if v
            },
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
