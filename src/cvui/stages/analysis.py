"""Analysis stages: Nested, Classify, ChannelAnalysis, Diff, ListQuantize."""
from __future__ import annotations

import numpy as np

from cvui.pipeline import DetectionStage, DetectionContext, DetectionPipeline
from cvui.stages.preprocessing import GrayscaleStage, TopHatStage, OtsuStage
from cvui.stages.morphology import DilateStage, ConnectedComponentStage


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

    def _detect_sub(self, img, region, kernel_size=None):
        """Run sub-pipeline on a region, return global-coord rects.

        kernel_size auto-scales with region size: smaller regions get smaller
        kernels so small elements inside big containers aren't merged away.
        """
        x1, y1, x2, y2 = region
        rw, rh = x2 - x1, y2 - y1
        sub_img = img[y1:y2, x1:x2]
        if sub_img.size == 0:
            return []

        # Auto kernel: ~1/4 of the shorter side, clamped to [15, 60]
        if kernel_size is None:
            kernel_size = max(15, min(60, min(rw, rh) // 4))

        sub_pipeline = DetectionPipeline([
            GrayscaleStage(), TopHatStage(kernel_size=kernel_size), OtsuStage(),
            DilateStage(),  # auto-adapt kernel from content density
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

        # If sub-pipeline found nothing useful (0 children), try row splitting
        if len(results) == 0 and rh > 60:
            row_results = self._split_rows(img, region)
            if len(row_results) > 1:
                results = row_results

        return results

    @staticmethod
    def _split_rows(img, region):
        """Split a large rect into rows by detecting horizontal gaps."""
        import cv2
        x1, y1, x2, y2 = region
        sub = img[y1:y2, x1:x2]
        if sub.size == 0:
            return []

        gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        median = float(np.median(gray))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        if median >= 128:
            fg = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        else:
            fg = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # Row density profile
        row_density = np.mean(fg.astype(np.float32), axis=1)
        # Smooth slightly
        if len(row_density) > 10:
            row_density = np.convolve(row_density, np.ones(5)/5, mode='same')

        # Find content blocks (density > threshold)
        threshold = max(row_density.max() * 0.1, 2.0)
        blocks = []
        in_block = False
        start = 0
        for i in range(len(row_density)):
            if row_density[i] > threshold and not in_block:
                start = i
                in_block = True
            elif row_density[i] <= threshold and in_block:
                if i - start > 15:  # min row height
                    blocks.append((x1, y1 + start, x2, y1 + i))
                in_block = False
        if in_block and len(row_density) - start > 15:
            blocks.append((x1, y1 + start, x2, y1 + len(row_density)))

        return blocks


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
