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
