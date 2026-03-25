"""Pluggable element detection pipeline.

Each processing step is a DetectionStage. Stages are chained in a
DetectionPipeline. A shared DetectionContext flows through stages,
accumulating results. Any stage can stop the pipeline early via
should_continue() returning False.

Preset pipelines:
- fast_pipeline(): Grayscale -> TopHat -> Otsu -> Dilate -> ConnectedComponent
- standard_pipeline(): fast + RectFilter + Merge
- full_pipeline(): standard + Nested + Classify + ChannelAnalysis
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


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

class DownscaleStage(DetectionStage):
    """Downscale image for faster processing. Coords mapped back by Pipeline.

    0.75x = 2x faster, loses ~2 elements on typical UI.
    0.5x  = 5x faster, loses ~30 elements (too aggressive for standard use).
    """

    def __init__(self, scale: float = 0.75, interpolation: str = "auto"):
        """
        Args:
            scale: target scale factor (0.75 = 2x faster, 0.5 = 5x faster)
            interpolation: "auto" (picks based on scale), "area" (best for shrink),
                          "linear" (fast), "nearest" (fastest, lossy)
        """
        self.target_scale = scale
        self.interpolation = interpolation

    def process(self, ctx):
        import cv2
        if self.target_scale >= 1.0:
            return ctx
        h, w = ctx.height, ctx.width
        new_h, new_w = int(h * self.target_scale), int(w * self.target_scale)

        interp_map = {
            "nearest": cv2.INTER_NEAREST,    # fastest, jaggy
            "linear": cv2.INTER_LINEAR,      # bilinear, fast, slight blur
            "cubic": cv2.INTER_CUBIC,        # bicubic (4x4 neighborhood), sharper than linear
            "area": cv2.INTER_AREA,          # area averaging — best for downscaling
            "lanczos": cv2.INTER_LANCZOS4,   # lanczos (8x8), sharpest, slowest
        }
        if self.interpolation == "auto":
            interp = cv2.INTER_AREA  # best for downscaling
        else:
            interp = interp_map.get(self.interpolation, cv2.INTER_AREA)

        ctx.img = cv2.resize(ctx.img, (new_w, new_h), interpolation=interp)
        ctx.scale = self.target_scale
        return ctx


class GrayscaleStage(DetectionStage):
    """Convert to grayscale with auto gamma from median brightness."""

    def process(self, ctx):
        import cv2
        gray = cv2.cvtColor(ctx.img, cv2.COLOR_BGR2GRAY)
        median = float(np.median(gray))
        if median > 0:
            gamma = np.log(100.0 / 255.0) / np.log(max(median, 1.0) / 255.0)
            gamma = float(np.clip(gamma, 0.4, 2.0))
            gray = np.clip(
                np.power(gray.astype(np.float32) / 255.0, gamma) * 255.0,
                0, 255,
            ).astype(np.uint8)
        ctx.gray = gray
        return ctx


class TopHatStage(DetectionStage):
    """Adaptive Top-Hat / Black-Hat with highlight-aware local inversion.

    Dark background (median < 128) -> TopHat (extract bright elements).
    Light background (median >= 128) -> BlackHat (extract dark elements).

    Additionally detects high-saturation colored regions (green highlights,
    colored badges) via HSV and applies the *opposite* transform there,
    so bright text on colored backgrounds is also captured.
    """

    def __init__(self, kernel_size: int = 80):
        self.kernel_size = kernel_size

    def process(self, ctx):
        import cv2
        if ctx.gray is None:
            ctx = GrayscaleStage().process(ctx)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
        median = float(np.median(ctx.gray))
        is_dark = median < 128

        # Primary transform based on overall brightness
        if is_dark:
            primary = cv2.morphologyEx(ctx.gray, cv2.MORPH_TOPHAT, kernel)
        else:
            primary = cv2.morphologyEx(ctx.gray, cv2.MORPH_BLACKHAT, kernel)

        # Detect colored highlight regions via HSV saturation
        hsv = cv2.cvtColor(ctx.img, cv2.COLOR_BGR2HSV)
        # High saturation = colored region (highlights, badges, colored buttons)
        color_mask = cv2.inRange(hsv, (0, 60, 50), (180, 255, 255))
        color_ratio = np.count_nonzero(color_mask) / max(color_mask.size, 1)

        if color_ratio > 0.01:  # at least 1% colored pixels
            # Apply opposite transform on colored regions
            if is_dark:
                opposite = cv2.morphologyEx(ctx.gray, cv2.MORPH_BLACKHAT, kernel)
            else:
                opposite = cv2.morphologyEx(ctx.gray, cv2.MORPH_TOPHAT, kernel)
            # Blend: use opposite result only in colored regions
            color_mask_3 = color_mask > 0
            primary[color_mask_3] = cv2.max(primary, opposite)[color_mask_3]

        ctx.gray = primary
        return ctx


class OtsuStage(DetectionStage):
    """Otsu auto-threshold — zero parameters."""

    def process(self, ctx):
        import cv2
        if ctx.gray is None:
            return ctx
        _, ctx.binary = cv2.threshold(
            ctx.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return ctx


class DilateStage(DetectionStage):
    """Directional dilation — connect icon+text on same row, title+subtitle."""

    def __init__(self, h_kernel: tuple[int, int] = (2, 15),
                 v_kernel: tuple[int, int] = (7, 2)):
        self.h_kernel = h_kernel
        self.v_kernel = v_kernel

    def process(self, ctx):
        import cv2
        if ctx.binary is None:
            return ctx
        b = cv2.morphologyEx(ctx.binary, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))
        b = cv2.dilate(b, np.ones(self.h_kernel, np.uint8), iterations=1)
        b = cv2.dilate(b, np.ones(self.v_kernel, np.uint8), iterations=1)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        ctx.binary = b
        return ctx


class ConnectedComponentStage(DetectionStage):
    """Extract bounding rects from binary mask + compute quality score."""

    def __init__(self, min_w: int = 15, min_h: int = 10):
        self.min_w = min_w
        self.min_h = min_h

    def process(self, ctx):
        import cv2
        if ctx.binary is None:
            return ctx
        contours, _ = cv2.findContours(
            ctx.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = ctx.height, ctx.width
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
    """Merge overlapping bounding rects."""

    def process(self, ctx):
        ctx.rects = self._merge(ctx.rects)
        return ctx

    @staticmethod
    def _merge(boxes: list[tuple]) -> list[tuple]:
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
                    if max(bx1, cx1) < min(bx2, cx2) and max(by1, cy1) < min(by2, cy2):
                        bx1, by1 = min(bx1, cx1), min(by1, cy1)
                        bx2, by2 = max(bx2, cx2), max(by2, cy2)
                        used.add(j)
                        merged = True
                new.append((bx1, by1, bx2, by2))
                used.add(i)
            result = [list(b) for b in new]
        return [tuple(b) for b in result]


class NestedStage(DetectionStage):
    """Recursively detect elements inside large containers."""

    def __init__(self, min_block_area_ratio: float = 0.05, max_depth: int = 2):
        self.min_block_area_ratio = min_block_area_ratio
        self.max_depth = max_depth

    def process(self, ctx):
        import cv2
        total_area = ctx.width * ctx.height
        min_area = total_area * self.min_block_area_ratio
        new_rects = list(ctx.rects)

        for r in ctx.rects:
            rw, rh = r[2] - r[0], r[3] - r[1]
            if rw * rh < min_area:
                continue
            children = self._detect_sub(ctx.img, r)
            new_rects.extend(children)

        title_h = min(60, ctx.height // 10)
        title_region = (0, 0, ctx.width, title_h)
        title_children = self._detect_sub(ctx.img, title_region, kernel_size=25)
        new_rects.extend(title_children)

        ctx.rects = new_rects
        return ctx

    def _detect_sub(self, img, region, kernel_size=40):
        """Run sub-pipeline on a region, return global-coord rects."""
        x1, y1, x2, y2 = region
        rw, rh = x2 - x1, y2 - y1
        sub_img = img[y1:y2, x1:x2]
        if sub_img.size == 0:
            return []
        sub_pipeline = DetectionPipeline([
            GrayscaleStage(), TopHatStage(kernel_size=kernel_size), OtsuStage(),
            DilateStage(h_kernel=(2, 8), v_kernel=(4, 2)),
            ConnectedComponentStage(min_w=10, min_h=8),
        ])
        sub_ctx = sub_pipeline.run(sub_img)
        results = []
        for sr in sub_ctx.rects:
            gx1, gy1 = sr[0] + x1, sr[1] + y1
            gx2, gy2 = sr[2] + x1, sr[3] + y1
            if (gx2 - gx1) > rw * 0.9 and (gy2 - gy1) > rh * 0.9:
                continue
            results.append((gx1, gy1, gx2, gy2))
        return results


class ClassifyStage(DetectionStage):
    """Heuristic classification by aspect ratio and area."""

    def process(self, ctx):
        for i, r in enumerate(ctx.rects):
            w, h = r[2] - r[0], r[3] - r[1]
            ctx.classifications[i] = self._classify(w, h, ctx.width, ctx.height)
        return ctx

    @staticmethod
    def _classify(w: int, h: int, img_w: int, img_h: int) -> str:
        area_ratio = (w * h) / max(img_w * img_h, 1)
        aspect = w / max(h, 1)
        if area_ratio > 0.1:
            return "image"
        if 0.7 < aspect < 1.4 and w < 80:
            return "icon"
        if aspect > 3:
            return "text"
        if area_ratio > 0.02:
            return "container"
        return "element"


class ChannelAnalysisStage(DetectionStage):
    """Extract UI states from color channel differences."""

    def process(self, ctx):
        import cv2
        b, g, r = cv2.split(ctx.img)
        ctx.ui_states["highlight"] = self._find_regions(
            cv2.subtract(g, r), threshold=30)
        ctx.ui_states["badge"] = self._find_regions(
            cv2.subtract(r, b), threshold=50, max_area=3000)
        ctx.ui_states["link"] = self._find_regions(
            cv2.subtract(b, r), threshold=30)
        return ctx

    @staticmethod
    def _find_regions(
        diff, threshold: int = 30, min_area: int = 50, max_area: int = 50000
    ) -> list[tuple[int, int, int, int]]:
        import cv2
        mask = (diff > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                rects.append((x, y, x + w, y + h))
        return rects


class DiffStage(DetectionStage):
    """Compare with previous frame, only flag changed regions."""

    def __init__(self, prev_img: np.ndarray | None = None,
                 change_threshold: int = 20, min_change_ratio: float = 0.02):
        self.prev_img = prev_img
        self.change_threshold = change_threshold
        self.min_change_ratio = min_change_ratio
        self._has_changes = True

    def process(self, ctx):
        import cv2
        if self.prev_img is None or self.prev_img.shape != ctx.img.shape:
            self._has_changes = True
            return ctx

        prev_gray = cv2.cvtColor(self.prev_img, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(ctx.img, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, mask = cv2.threshold(diff, self.change_threshold, 255, cv2.THRESH_BINARY)

        change_ratio = np.count_nonzero(mask) / max(mask.size, 1)
        if change_ratio < self.min_change_ratio:
            self._has_changes = False
            return ctx

        self._has_changes = True
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 20 and h > 20:
                ctx.rects.append((x, y, x + w, y + h))
        return ctx

    def should_continue(self, ctx):
        return self._has_changes


# ---------------------------------------------------------------------------
# Optional ML stages
# ---------------------------------------------------------------------------

class OmniParserStage(DetectionStage):
    """Optional: YOLO icon detection + Florence caption via OmniParser.

    Detects non-text UI elements (icons, buttons, avatars) that pure CV
    methods might miss. Requires OmniParser models or running server.

    Skips gracefully if models/server not available.
    """

    def __init__(self, model_path: str = "", server_url: str = "http://127.0.0.1:8000"):
        self.model_path = model_path
        self.server_url = server_url
        self._available: bool | None = None

    def process(self, ctx):
        if not self._check_available():
            log.debug("OmniParserStage: not available, skipping")
            return ctx
        try:
            new_rects = self._detect_via_server(ctx.img)
            ctx.rects.extend(new_rects)
        except Exception as e:
            log.warning("OmniParserStage: detection failed: %s", e)
        return ctx

    def _check_available(self) -> bool:
        if self._available is not None:
            return self._available
        # Try server
        try:
            import httpx
            r = httpx.get(f"{self.server_url}/probe/", timeout=2)
            self._available = r.status_code == 200
            if self._available:
                return True
        except Exception:
            pass
        # Try local model
        from pathlib import Path
        if self.model_path and Path(self.model_path).exists():
            self._available = True
            return True
        self._available = False
        return False

    def _detect_via_server(self, img) -> list[tuple[int, int, int, int]]:
        """Call OmniParser server API."""
        import base64
        import io
        import httpx
        from PIL import Image

        h, w = img.shape[:2]
        pil_img = Image.fromarray(img[:, :, ::-1])
        buf = io.BytesIO()
        pil_img.save(buf, format="WEBP", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()

        response = httpx.post(
            f"{self.server_url}/parse/",
            json={"base64_image": b64},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        rects = []
        for item in result.get("parsed_content_list", []):
            bbox = item.get("bbox", [0, 0, 0, 0])
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)
            if x2 - x1 > 5 and y2 - y1 > 5:
                rects.append((x1, y1, x2, y2))
        return rects


class GroundingDINOStage(DetectionStage):
    """Optional: text-guided zero-shot element detection via Grounding DINO.

    Given a text query (e.g. "search box"), finds matching UI elements.
    Requires transformers library with Grounding DINO support.

    Skips gracefully if transformers not installed.
    """

    def __init__(self, query: str = "", box_threshold: float = 0.3):
        self.query = query
        self.box_threshold = box_threshold
        self._model = None

    def process(self, ctx):
        if not self.query:
            return ctx
        try:
            from transformers import pipeline as hf_pipeline
            from PIL import Image

            if self._model is None:
                self._model = hf_pipeline(
                    "zero-shot-object-detection",
                    model="IDEA-Research/grounding-dino-tiny",
                )
            img_pil = Image.fromarray(ctx.img[:, :, ::-1])  # BGR -> RGB
            results = self._model(
                img_pil,
                candidate_labels=[self.query],
                threshold=self.box_threshold,
            )
            for r in results:
                box = r["box"]
                ctx.rects.append((
                    int(box["xmin"]), int(box["ymin"]),
                    int(box["xmax"]), int(box["ymax"]),
                ))
        except ImportError:
            log.debug("GroundingDINOStage: transformers not installed, skipping")
        except Exception as e:
            log.warning("GroundingDINOStage: %s", e)
        return ctx


class ListQuantizeStage(DetectionStage):
    """Detect list items by template propagation from a reference element.

    Strategy:
    1. Find a reference item (highlighted selection via HSV, or estimated from
       existing rects' spacing)
    2. Use its height as step
    3. Expand UP from reference -> stop at first empty slot
    4. Expand DOWN from reference -> stop at first empty slot
    5. Output: only items with content (no wasted checks on blank area)
    """

    def __init__(self, zone_rect: tuple[int,int,int,int] | None = None,
                 min_density: float = 0.02):
        self.zone_rect = zone_rect
        self.min_density = min_density

    def process(self, ctx):
        import cv2

        if self.zone_rect is None:
            return ctx

        zx1, zy1, zx2, zy2 = self.zone_rect
        zone_img = ctx.img[zy1:zy2, zx1:zx2]
        zh = zy2 - zy1

        # Step 1: find reference item
        ref = self._find_highlight(zone_img)
        if ref is None:
            ref = self._estimate_from_rects(ctx.rects, self.zone_rect)
        if ref is None:
            return ctx

        ref_h = ref[3] - ref[1]
        if ref_h < 20:
            return ctx

        # Prepare foreground map
        fg = self._get_foreground(zone_img)

        items = []

        # Step 2: expand UP
        y = ref[1] - ref_h
        while y >= 0:
            if not self._has_content(fg, y, y + ref_h):
                break
            items.insert(0, (zx1, zy1 + y, zx2, zy1 + y + ref_h))
            y -= ref_h

        # Reference item itself
        items.append((zx1, zy1 + ref[1], zx2, zy1 + ref[3]))

        # Step 3: expand DOWN
        y = ref[3]
        while y + ref_h <= zh:
            if not self._has_content(fg, y, y + ref_h):
                break
            items.append((zx1, zy1 + y, zx2, zy1 + y + ref_h))
            y += ref_h

        ctx.rects.extend(items)
        return ctx

    def _find_highlight(self, zone_img) -> tuple[int,int,int,int] | None:
        """Find highlighted/selected item via HSV saturation."""
        import cv2
        hsv = cv2.cvtColor(zone_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 50, 50), (180, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        zw = zone_img.shape[1]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if w > zw * 0.5 and h > 30 and area > best_area:
                best = (x, y, x + w, y + h)
                best_area = area
        return best

    def _estimate_from_rects(self, rects, zone_rect) -> tuple[int,int,int,int] | None:
        """Estimate item height from spacing of existing rects in zone."""
        zx1, zy1, zx2, zy2 = zone_rect
        zone_rects = [
            r for r in rects
            if r[0] >= zx1 and r[2] <= zx2 and r[1] >= zy1 and r[3] <= zy2
        ]
        if len(zone_rects) < 2:
            return None

        zone_rects.sort(key=lambda r: (r[1] + r[3]) // 2)

        gaps = []
        for i in range(len(zone_rects) - 1):
            cy1 = (zone_rects[i][1] + zone_rects[i][3]) // 2
            cy2 = (zone_rects[i+1][1] + zone_rects[i+1][3]) // 2
            gap = cy2 - cy1
            if gap > 20:
                gaps.append(gap)

        if not gaps:
            return None

        step = int(np.median(gaps))
        r = zone_rects[0]
        local_y = r[1] - zy1
        return (0, local_y, zone_rect[2] - zone_rect[0], local_y + step)

    @staticmethod
    def _get_foreground(zone_img):
        """Extract foreground using adaptive TopHat/BlackHat."""
        import cv2
        gray = cv2.cvtColor(zone_img, cv2.COLOR_BGR2GRAY)
        median = float(np.median(gray))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
        if median >= 128:
            return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        else:
            return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    def _has_content(self, fg, y_start: int, y_end: int) -> bool:
        """Check if a row region has enough foreground content."""
        if y_start < 0 or y_end > fg.shape[0]:
            return False
        region = fg[y_start:y_end, :]
        if region.size == 0:
            return False
        density = np.count_nonzero(region > 10) / region.size
        return density > self.min_density


# ---------------------------------------------------------------------------
# Preset pipelines
# ---------------------------------------------------------------------------

def fast_pipeline(scale: float = 0.75) -> DetectionPipeline:
    """Downscale -> Grayscale -> TopHat -> Otsu -> Dilate -> CC. ~17ms at 0.75x."""
    stages = []
    if scale < 1.0:
        stages.append(DownscaleStage(scale=scale))
    stages.extend([
        GrayscaleStage(), TopHatStage(), OtsuStage(),
        DilateStage(), ConnectedComponentStage(),
    ])
    return DetectionPipeline(stages)


def standard_pipeline(scale: float = 1.0) -> DetectionPipeline:
    """Fast + RectFilter + Merge. Full resolution by default. ~33ms."""
    stages = []
    if scale < 1.0:
        stages.append(DownscaleStage(scale=scale))
    stages.extend([
        GrayscaleStage(), TopHatStage(), OtsuStage(),
        DilateStage(), ConnectedComponentStage(),
        RectFilterStage(), MergeStage(),
    ])
    return DetectionPipeline(stages)


def full_pipeline(omniparser_path: str = "", grounding_query: str = "") -> DetectionPipeline:
    """Standard + Nested + Classify + ChannelAnalysis + optional ML stages. ~20ms+."""
    stages: list[DetectionStage] = [
        GrayscaleStage(), TopHatStage(), OtsuStage(),
        DilateStage(), ConnectedComponentStage(),
        RectFilterStage(), MergeStage(),
        NestedStage(), ClassifyStage(), ChannelAnalysisStage(),
    ]
    if omniparser_path:
        stages.append(OmniParserStage(model_path=omniparser_path))
    if grounding_query:
        stages.append(GroundingDINOStage(query=grounding_query))
    return DetectionPipeline(stages)


def grounding_pipeline(query: str, box_threshold: float = 0.3) -> DetectionPipeline:
    """Single-purpose: find one element by text description."""
    return DetectionPipeline([GroundingDINOStage(query=query, box_threshold=box_threshold)])
