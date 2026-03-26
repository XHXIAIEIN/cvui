"""Morphology stages: Dilate, ConnectedComponent, RectFilter, Merge."""
from __future__ import annotations

import numpy as np

from cvui.pipeline import DetectionStage, DetectionContext


class DilateStage(DetectionStage):
    """Adaptive directional dilation — kernel size scales with content density.

    Dense UI (many small elements close together) → small kernel (don't merge)
    Sparse UI (few large elements far apart) → large kernel (connect icon+text)
    """

    def __init__(self, h_kernel: tuple[int, int] | None = None,
                 v_kernel: tuple[int, int] | None = None):
        self._h_kernel = h_kernel
        self._v_kernel = v_kernel

    def process(self, ctx):
        import cv2
        if ctx.binary is None:
            return ctx

        # Auto-adapt kernel if not explicitly set
        if self._h_kernel is None or self._v_kernel is None:
            h_k, v_k = self._auto_kernel(ctx.binary)
        else:
            h_k, v_k = self._h_kernel, self._v_kernel

        b = cv2.morphologyEx(ctx.binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        b = cv2.dilate(b, np.ones(h_k, np.uint8), iterations=1)
        b = cv2.dilate(b, np.ones(v_k, np.uint8), iterations=1)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        ctx.binary = b
        return ctx

    @staticmethod
    def _auto_kernel(binary: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
        """Compute kernel size from foreground density.

        High density (>15% foreground) = dense UI → small kernels
        Low density (<5% foreground) = sparse UI → large kernels
        """
        density = np.count_nonzero(binary) / max(binary.size, 1)

        if density > 0.10:
            # Dense UI (settings panels, toolbars, forms)
            return (2, 6), (3, 2)
        elif density > 0.05:
            # Medium density
            return (2, 10), (5, 2)
        else:
            # Sparse UI (chat apps, simple layouts)
            return (2, 15), (7, 2)


class ConnectedComponentStage(DetectionStage):
    """Extract bounding rects from binary mask + compute quality score."""

    def __init__(self, min_w: int = 15, min_h: int = 10):
        self.min_w = min_w
        self.min_h = min_h

    def process(self, ctx):
        import cv2
        if ctx.binary is None:
            return ctx

        h, w = ctx.height, ctx.width

        if ctx.zones:
            # Only detect within zones
            for zone in ctx.zones:
                zx1, zy1, zx2, zy2 = zone
                zone_binary = ctx.binary[zy1:zy2, zx1:zx2]
                contours, _ = cv2.findContours(
                    zone_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, rw, rh = cv2.boundingRect(cnt)
                    if rw < self.min_w or rh < self.min_h:
                        continue
                    ctx.rects.append((zx1 + x, zy1 + y, zx1 + x + rw, zy1 + y + rh))
        else:
            # Full image detection (original behavior)
            contours, _ = cv2.findContours(
                ctx.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, rw, rh = cv2.boundingRect(cnt)
                if rw < self.min_w or rh < self.min_h:
                    continue
                if rw > w * 0.95 and rh > h * 0.95:
                    continue
                ctx.rects.append((x, y, x + rw, y + rh))

        ctx.quality_score = self._compute_quality(ctx)
        return ctx

    def should_continue(self, ctx):
        return ctx.quality_score < 0.8

    @staticmethod
    def _compute_quality(ctx) -> float:
        if not ctx.rects:
            return 0.0
        total_area = ctx.height * ctx.width
        rect_area = sum((r[2] - r[0]) * (r[3] - r[1]) for r in ctx.rects)
        coverage = min(rect_area / max(total_area, 1), 1.0)
        n = len(ctx.rects)
        frag = 1.0 - min(n / 100.0, 1.0)
        return coverage * 0.5 + frag * 0.5


class RectFilterStage(DetectionStage):
    """Filter out rects that are too small, too thin, or window edge artifacts."""

    def __init__(self, min_area: int = 200, min_aspect: float = 0.1,
                 max_aspect: float = 15.0, edge_margin: int = 5):
        self.min_area = min_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.edge_margin = edge_margin

    def process(self, ctx):
        filtered = []
        img_h, img_w = ctx.height, ctx.width
        for r in ctx.rects:
            w, h = r[2] - r[0], r[3] - r[1]
            if w * h < self.min_area:
                continue
            aspect = w / max(h, 1)
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue
            if r[0] <= self.edge_margin and w < 20:
                continue
            if r[2] >= img_w - self.edge_margin and w < 20:
                continue
            if r[1] <= self.edge_margin and h < 20:
                continue
            if r[3] >= img_h - self.edge_margin and h < 20:
                continue
            filtered.append(r)
        ctx.rects = filtered
        return ctx


class MergeStage(DetectionStage):
    """Merge significantly overlapping bounding rects.

    Only merges when overlap area > min_overlap_ratio of the smaller rect.
    Prevents adjacent-but-separate elements from being swallowed.
    """

    def __init__(self, min_overlap_ratio: float = 0.3):
        self.min_overlap_ratio = min_overlap_ratio

    def process(self, ctx):
        ctx.rects = self._merge(ctx.rects, self.min_overlap_ratio)
        return ctx

    @staticmethod
    def _merge(boxes: list[tuple], min_overlap_ratio: float = 0.3) -> list[tuple]:
        if not boxes:
            return []
        result = [list(b) for b in boxes]
        merged = True
        while merged:
            merged = False
            new = []
            used = set()
            for i in range(len(result)):
                if i in used:
                    continue
                bx1, by1, bx2, by2 = result[i]
                for j in range(i + 1, len(result)):
                    if j in used:
                        continue
                    cx1, cy1, cx2, cy2 = result[j]

                    # Compute overlap
                    ox1 = max(bx1, cx1)
                    oy1 = max(by1, cy1)
                    ox2 = min(bx2, cx2)
                    oy2 = min(by2, cy2)

                    if ox1 >= ox2 or oy1 >= oy2:
                        continue  # no overlap

                    overlap_area = (ox2 - ox1) * (oy2 - oy1)
                    area_b = (bx2 - bx1) * (by2 - by1)
                    area_c = (cx2 - cx1) * (cy2 - cy1)
                    smaller_area = min(area_b, area_c)

                    # Only merge if overlap is significant relative to smaller rect
                    if smaller_area > 0 and overlap_area / smaller_area >= min_overlap_ratio:
                        bx1, by1 = min(bx1, cx1), min(by1, cy1)
                        bx2, by2 = max(bx2, cx2), max(by2, cy2)
                        used.add(j)
                        merged = True

                new.append((bx1, by1, bx2, by2))
                used.add(i)
            result = [list(b) for b in new]
        return [tuple(b) for b in result]
