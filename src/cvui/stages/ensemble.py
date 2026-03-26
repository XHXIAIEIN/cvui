"""Ensemble stage: multi-pass coarse-to-fine detection.

Five-pass strategy (from disco_ui experiments):
  Pass 0: Saturation separation — mask out colorful scene pixels
  Pass 1: Coarse detection — find panel-level regions (area > 1.5%)
  Pass 2: Fine detection — detect details inside each panel
  Pass 3: List quantize — auto-infer list zones, propagate items
  Pass 4: Merge + classify — deduplicate, assign tier labels
"""
from __future__ import annotations

import numpy as np

from cvui.pipeline import DetectionStage, DetectionContext


class EnsembleStage(DetectionStage):
    """Multi-pass ensemble detection: panels → details → lists → merge.

    Combines saturation filtering, two-level TopHat (coarse + fine),
    automatic list zone inference, and tiered classification into a
    single composable Stage.

    Args:
        coarse_kernel: TopHat kernel for panel detection (default 80).
        panel_area_pct: minimum panel area as % of image (default 1.5).
        fine_kernel_range: (min, max) for auto fine TopHat kernel (default 15-40).
        dedup_tolerance: pixels of tolerance for inside() dedup (default 5).
        list_min_density: minimum foreground density for list content (default 0.02).
    """

    def __init__(
        self,
        coarse_kernel: int = 80,
        panel_area_pct: float = 1.5,
        fine_kernel_range: tuple[int, int] = (15, 40),
        dedup_tolerance: int = 5,
        list_min_density: float = 0.02,
    ):
        self.coarse_kernel = coarse_kernel
        self.panel_area_pct = panel_area_pct
        self.fine_kernel_range = fine_kernel_range
        self.dedup_tolerance = dedup_tolerance
        self.list_min_density = list_min_density

    def process(self, ctx: DetectionContext) -> DetectionContext:
        import cv2

        img = ctx.img
        h, w = img.shape[:2]
        total_area = h * w
        min_panel_area = total_area * self.panel_area_pct / 100.0

        # Precompute shared layers (used by multiple strategies)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ctx.layers["gray"] = gray

        # ==================================================================
        # Phase 1: Detect panels
        #
        # Try multiple strategies. Each strategy returns candidate panels.
        # Filter out "whole image" panels (>80% area = not a real panel).
        # Use first strategy that produces >=2 meaningful panels.
        # ==================================================================
        panels = []
        for detect_fn in [
            lambda: self._detect_panels_by_brightness(gray, w, h, min_panel_area),
            lambda: self._detect_panels_by_variance(gray, w, h, min_panel_area),
            lambda: self._detect_panels_tophat(gray, w, h, min_panel_area),
        ]:
            candidates = detect_fn()
            # Drop panels that cover >80% of image (that's the whole screen)
            candidates = [
                p for p in candidates
                if (p[2] - p[0]) * (p[3] - p[1]) <= total_area * 0.80
            ]
            if len(candidates) >= 2:
                panels = candidates
                break
            elif len(candidates) == 1 and not panels:
                panels = candidates  # keep best single panel

        panels = self._merge_nearby_panels(panels, h)

        # Fallback: if nothing segmented, use whole image
        if not panels and w >= 100 and h >= 100:
            panels = [(0, 0, w, h)]

        ctx.zones = list(panels)

        # ==================================================================
        # Phase 2: Per-panel — probe → classify → dispatch strategy
        # ==================================================================
        details = []
        text_panels = []
        self._panel_metrics = {}
        for panel in panels:
            region_type = self._classify_region(img, gray, hsv, panel)

            if region_type == "text":
                text_panels.append(panel)
            elif region_type == "scene":
                pass  # pure game scene, no UI — skip
            elif region_type == "color-ui":
                details.extend(self._strategy_color_blocks(img, panel))
            elif region_type == "low-contrast-ui":
                details.extend(self._strategy_tophat(gray, hsv, panel))
            else:
                # fallback: try color blocks
                details.extend(self._strategy_color_blocks(img, panel))

        # ------------------------------------------------------------------
        # Pass 3: Auto-infer list zones + ListQuantize
        # ------------------------------------------------------------------
        list_items = []
        list_zones = self._infer_list_zones(panels, details, h)
        for zone in list_zones:
            items = self._quantize_list(img, zone)
            list_items.extend(items)

        # ------------------------------------------------------------------
        # Pass 4: Merge + classify
        # ------------------------------------------------------------------
        all_rects = self._merge_and_classify(
            panels, details, list_items, text_panels, ctx, w, h
        )

        ctx.rects = all_rects
        return ctx

    # ======================================================================
    # Pass 1: Panel detection
    # ======================================================================

    def _detect_panels_by_color(
        self, img: np.ndarray, w: int, h: int, min_area: float,
    ) -> list[tuple[int, int, int, int]]:
        """Segment UI into panels by quantized background color.

        1. Quantize image to N colors (reuses ColorQuantizeStage logic)
        2. For each quantized color, find large connected regions
        3. Large rectangular regions = panel backgrounds
        """
        import cv2
        from PIL import Image as PILImage

        n_colors = 8

        # --- Quantize (with fallback on error) ---
        try:
            pil = PILImage.fromarray(img[:, :, ::-1])
            quantized = pil.quantize(colors=n_colors, method=PILImage.Quantize.FASTOCTREE)
            q_indices = np.array(quantized)  # H×W index map (0..n_colors-1)
        except Exception:
            return []  # quantize failed, let TopHat fallback handle it

        palette_data = quantized.getpalette()
        palette = []
        if palette_data:
            for i in range(0, min(len(palette_data), n_colors * 3), 3):
                palette.append((palette_data[i], palette_data[i + 1], palette_data[i + 2]))

        # --- Find panel regions per color ---
        panels = []
        for color_idx in range(min(n_colors, len(palette))):
            mask = (q_indices == color_idx).astype(np.uint8) * 255

            # Close small gaps within panels (text holes in background)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                x, y, rw, rh = cv2.boundingRect(cnt)
                area = rw * rh
                if area < min_area:
                    continue
                # Skip if it's basically the whole image
                if area > w * h * 0.85:
                    continue
                # Rectangularity check: contour area vs bounding rect area
                cnt_area = cv2.contourArea(cnt)
                if cnt_area > 0 and cnt_area / area > 0.5:
                    panels.append((x, y, x + rw, y + rh))

        # Remove panels that are fully contained in a larger panel
        panels.sort(key=lambda p: (p[2] - p[0]) * (p[3] - p[1]), reverse=True)
        filtered = []
        for p in panels:
            contained = False
            for f in filtered:
                if p[0] >= f[0] and p[1] >= f[1] and p[2] <= f[2] and p[3] <= f[3]:
                    contained = True
                    break
            if not contained:
                filtered.append(p)

        filtered.sort(key=lambda p: (p[1], p[0]))
        return filtered

    def _detect_panels_by_brightness(
        self, gray: np.ndarray, w: int, h: int, min_area: float
    ) -> list[tuple[int, int, int, int]]:
        """Detect panels by brightness — dark regions are UI overlays.

        Uses block-wise median brightness on a grid. Blocks significantly
        darker than the image median are UI overlay candidates. Connected
        dark blocks form panels.

        This avoids Otsu's issue with skewed distributions (e.g., 76%
        dark pixels where Otsu threshold doesn't separate UI from scene).
        """
        import cv2

        # Downsample to a block grid for speed (~30x30 blocks)
        block_h = max(1, h // 30)
        block_w = max(1, w // 30)
        n_rows = h // block_h
        n_cols = w // block_w
        if n_rows < 3 or n_cols < 3:
            return []

        # Compute per-block median brightness
        block_medians = np.zeros((n_rows, n_cols), dtype=np.float32)
        for r in range(n_rows):
            for c in range(n_cols):
                block = gray[r*block_h:(r+1)*block_h, c*block_w:(c+1)*block_w]
                block_medians[r, c] = float(np.median(block))

        # Dark threshold: blocks below global P30 are "dark"
        threshold = float(np.percentile(block_medians, 30))
        # But only if there's meaningful contrast (bright areas exist)
        global_median = float(np.median(block_medians))
        if threshold > global_median * 0.8:
            return []  # no clear dark/bright separation

        dark_blocks = (block_medians <= threshold).astype(np.uint8) * 255

        # Morphology on block grid: close small gaps
        dark_blocks = cv2.morphologyEx(
            dark_blocks, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        )

        # Find connected dark regions
        contours, _ = cv2.findContours(
            dark_blocks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        panels = []
        for cnt in contours:
            x, y, rw, rh = cv2.boundingRect(cnt)
            # Map back to pixel coordinates
            px1, py1 = x * block_w, y * block_h
            px2, py2 = min((x + rw) * block_w, w), min((y + rh) * block_h, h)
            area = (px2 - px1) * (py2 - py1)
            if area < min_area:
                continue
            if area > w * h * 0.85:
                continue
            panels.append((px1, py1, px2, py2))

        panels.sort(key=lambda p: (p[1], p[0]))
        return panels

    def _detect_panels_by_variance(
        self, gray: np.ndarray, w: int, h: int, min_area: float
    ) -> list[tuple[int, int, int, int]]:
        """Detect panels by local brightness variance.

        Reuses GradientDetectorStage insight: UI panels are LOW variance
        (solid/flat backgrounds), scene textures are HIGH variance.

        1. Compute local variance map (same as GradientDetectorStage)
        2. Low-variance mask = potential UI regions
        3. Close gaps + TopHat on masked gray → panel content
        4. Dilate + CC → panel boundaries
        """
        import cv2

        # Local variance (from GradientDetectorStage, kernel=15)
        gray_f = gray.astype(np.float32)
        k = np.ones((15, 15), np.float32) / 225
        mean = cv2.filter2D(gray_f, -1, k)
        sq_mean = cv2.filter2D(gray_f ** 2, -1, k)
        local_var = np.clip(sq_mean - mean ** 2, 0, None)

        # Threshold: Otsu on variance map for adaptive split
        var_u8 = np.clip(local_var / max(local_var.max(), 1) * 255, 0, 255).astype(np.uint8)
        _, var_mask = cv2.threshold(var_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Close gaps in UI panels (text creates small high-var spots)
        var_mask = cv2.morphologyEx(var_mask, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))

        # Masked gray: only low-variance (UI) regions
        ui_gray = gray.copy()
        ui_gray[var_mask == 0] = 0

        # TopHat on UI-only gray to find content within panels
        return self._detect_panels_tophat(ui_gray, w, h, min_area)

    def _detect_panels_tophat(
        self, gray: np.ndarray, w: int, h: int, min_area: float
    ) -> list[tuple[int, int, int, int]]:
        """Fallback panel detection: TopHat → Otsu → dilate → CC."""
        import cv2

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.coarse_kernel, self.coarse_kernel)
        )
        median = float(np.median(gray[gray > 0])) if np.any(gray > 0) else 128.0
        if median >= 128:
            fg = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        else:
            fg = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        _, binary = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.dilate(binary, np.ones((3, 20), np.uint8), iterations=1)
        binary = cv2.dilate(binary, np.ones((10, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        panels = []
        for cnt in contours:
            x, y, rw, rh = cv2.boundingRect(cnt)
            if rw * rh >= min_area:
                panels.append((x, y, x + rw, y + rh))
        panels.sort(key=lambda p: (p[1], p[0]))
        return panels

    @staticmethod
    def _merge_nearby_panels(
        panels: list[tuple[int, int, int, int]], img_h: int
    ) -> list[tuple[int, int, int, int]]:
        """Merge vertically adjacent panels that overlap horizontally.

        A single window (terminal, editor) can be fragmented into multiple
        panels by Pass 1's dilate. Merge panels that are:
        - Vertically close (gap < 5% of image height)
        - Horizontally overlapping (shared X range > 50% of narrower panel)
        """
        if len(panels) <= 1:
            return panels

        max_gap = max(30, img_h // 20)  # 5% of image height
        merged = [list(p) for p in panels]
        changed = True

        while changed:
            changed = False
            new = []
            used = set()
            for i in range(len(merged)):
                if i in used:
                    continue
                ax1, ay1, ax2, ay2 = merged[i]
                for j in range(i + 1, len(merged)):
                    if j in used:
                        continue
                    bx1, by1, bx2, by2 = merged[j]
                    # Vertical gap
                    v_gap = max(0, max(by1 - ay2, ay1 - by2))
                    if v_gap > max_gap:
                        continue
                    # Horizontal overlap
                    ox = max(0, min(ax2, bx2) - max(ax1, bx1))
                    narrower_w = min(ax2 - ax1, bx2 - bx1)
                    if narrower_w > 0 and ox / narrower_w >= 0.5:
                        # Merge
                        ax1 = min(ax1, bx1)
                        ay1 = min(ay1, by1)
                        ax2 = max(ax2, bx2)
                        ay2 = max(ay2, by2)
                        used.add(j)
                        changed = True
                new.append((ax1, ay1, ax2, ay2))
                used.add(i)
            merged = [list(p) for p in new]

        result = [tuple(p) for p in merged]
        result.sort(key=lambda p: (p[1], p[0]))
        return result

    # ======================================================================
    # Phase 2: Region classification + strategy dispatch
    # ======================================================================

    def _classify_region(
        self, img: np.ndarray, gray: np.ndarray, hsv: np.ndarray,
        panel: tuple[int, int, int, int],
    ) -> str:
        """Probe a panel region and classify it for strategy selection.

        Computes quick features:
        - Saturation stats: high mean + high std → game scene
        - Color diversity: many distinct colors → scene; few → UI
        - Binary coverage from TopHat: high + many CCs → text area
        - Brightness uniformity: very uniform → solid panel / empty area

        Returns one of: "text", "scene", "color-ui", "low-contrast-ui"
        """
        import cv2

        x1, y1, x2, y2 = panel
        rw, rh = x2 - x1, y2 - y1
        if rw < 10 or rh < 10:
            return "scene"

        sub_gray = gray[y1:y2, x1:x2]
        sub_hsv = hsv[y1:y2, x1:x2]
        sub_s = sub_hsv[:, :, 1].astype(np.float32)

        # --- Feature 1: Saturation ---
        sat_mean = float(np.mean(sub_s))
        sat_std = float(np.std(sub_s))

        # --- Feature 2: Color diversity (quick quantize) ---
        from PIL import Image as PILImage
        try:
            sub_color = img[y1:y2, x1:x2]
            pil = PILImage.fromarray(sub_color[:, :, ::-1])
            q = pil.quantize(colors=6, method=PILImage.Quantize.FASTOCTREE)
            q_idx = np.array(q)
            color_counts = np.bincount(q_idx.ravel(), minlength=6)
            active_colors = int(np.sum(color_counts > q_idx.size * 0.02))
        except Exception:
            active_colors = 1

        # --- Feature 3: Text metrics (reuse existing method) ---
        k = max(15, min(40, min(rw, rh) // 4))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        border_px = np.concatenate([
            sub_gray[0, :], sub_gray[-1, :], sub_gray[:, 0], sub_gray[:, -1]
        ])
        bg = float(np.median(border_px))
        if bg >= 128:
            fg = cv2.morphologyEx(sub_gray, cv2.MORPH_BLACKHAT, kernel)
        else:
            fg = cv2.morphologyEx(sub_gray, cv2.MORPH_TOPHAT, kernel)
        _, binary = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        metrics = self._estimate_text_metrics(binary)
        if hasattr(self, "_panel_metrics"):
            self._panel_metrics[f"{x1},{y1},{x2},{y2}"] = metrics

        # --- Decision tree ---
        # Order matters: scene first (high sat + diverse = game/photo),
        # then text (only for low-sat, non-scene areas like terminals).
        def _decide():
            # Game scene: high saturation + many colors → skip
            if sat_mean > 40 and sat_std > 30 and active_colors >= 5:
                return "scene"
            # Text area: many regular lines in low-sat context (terminal/editor)
            if metrics["is_text_content"] and sat_mean < 40:
                return "text"
            # Color-distinct UI: moderate palette
            if 2 <= active_colors <= 5:
                return "color-ui"
            # Low-contrast UI: very few colors, dark background
            if active_colors <= 2 and bg < 80:
                return "low-contrast-ui"
            return "color-ui"

        result = _decide()
        # Store classification in metrics for debugging
        metrics["region_type"] = result
        metrics["_sat_mean"] = round(sat_mean, 1)
        metrics["_sat_std"] = round(sat_std, 1)
        metrics["_active_colors"] = active_colors
        metrics["_bg"] = round(bg, 1)
        return result

    def _strategy_color_blocks(
        self, img: np.ndarray, panel: tuple[int, int, int, int],
    ) -> list[tuple[int, int, int, int]]:
        """Detect UI elements as non-background color blocks."""
        import cv2
        from PIL import Image as PILImage

        x1, y1, x2, y2 = panel
        rw, rh = x2 - x1, y2 - y1
        sub_color = img[y1:y2, x1:x2]

        try:
            pil_sub = PILImage.fromarray(sub_color[:, :, ::-1])
            quantized = pil_sub.quantize(colors=6, method=PILImage.Quantize.FASTOCTREE)
            q_idx = np.array(quantized)
        except Exception:
            return []

        # Background = most common color at borders
        border_idx = np.concatenate([
            q_idx[0, :], q_idx[-1, :], q_idx[:, 0], q_idx[:, -1]
        ])
        bg_color_idx = int(np.bincount(border_idx).argmax())

        total_area = rw * rh
        min_block = max(total_area * 0.001, 100)
        max_block = total_area * 0.85

        details = []
        for color_idx in range(6):
            if color_idx == bg_color_idx:
                continue
            mask = (q_idx == color_idx).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                area = cw * ch
                if area < min_block or area > max_block:
                    continue
                if cw < 8 or ch < 8:
                    continue
                details.append((x1 + cx, y1 + cy, x1 + cx + cw, y1 + cy + ch))

        return details

    def _strategy_tophat(
        self, gray: np.ndarray, hsv: np.ndarray,
        panel: tuple[int, int, int, int],
    ) -> list[tuple[int, int, int, int]]:
        """Detect UI elements via saturation filter + TopHat on low-contrast panels."""
        import cv2

        x1, y1, x2, y2 = panel
        rw, rh = x2 - x1, y2 - y1
        sub_gray = gray[y1:y2, x1:x2]
        sub_s = hsv[y1:y2, x1:x2, 1]

        # Saturation filter: keep low-sat UI, mask high-sat scene
        _, ui_mask = cv2.threshold(sub_s, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ui_mask = cv2.morphologyEx(ui_mask, cv2.MORPH_CLOSE, np.ones((8, 8), np.uint8))
        filtered = sub_gray.copy()
        filtered[ui_mask == 0] = 0

        # TopHat on filtered
        k = max(15, min(40, min(rw, rh) // 4))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        border_px = np.concatenate([
            filtered[0, :], filtered[-1, :], filtered[:, 0], filtered[:, -1]
        ])
        bg = float(np.median(border_px))
        if bg >= 128:
            fg = cv2.morphologyEx(filtered, cv2.MORPH_BLACKHAT, kernel)
        else:
            fg = cv2.morphologyEx(filtered, cv2.MORPH_TOPHAT, kernel)

        _, binary = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Dilate to connect nearby elements
        scale = max(1.0, min(rw, rh) / 400.0)
        binary = cv2.dilate(binary, np.ones((max(2, int(2*scale)), max(8, int(10*scale))), np.uint8))
        binary = cv2.dilate(binary, np.ones((max(4, int(5*scale)), max(2, int(2*scale))), np.uint8))

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        total_area = rw * rh
        min_block = max(total_area * 0.001, 80)

        details = []
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area < min_block:
                continue
            if cw < 8 or ch < 8:
                continue
            if cw > rw * 0.95 and ch > rh * 0.95:
                continue
            details.append((x1 + cx, y1 + cy, x1 + cx + cw, y1 + cy + ch))

        return details

    def _detect_details_in_panel(
        self, img: np.ndarray, panel: tuple[int, int, int, int]
    ) -> tuple[bool, list[tuple[int, int, int, int]]]:
        """Per-panel: color quantize → find color blocks that differ from bg.

        Strategy shift: detect COLOR BLOCKS, not text edges.
        1. Crop panel from the FULL COLOR image (not gray)
        2. Identify background color (most common color at border)
        3. Quantize panel to N colors
        4. Each non-background color region = a UI element (button, slider,
           icon, highlighted area)
        5. Text is intentionally ignored — that's OCR's job

        Also estimates text metrics from the gray channel for text-content
        gating (many small CCs in regular rows = text area, skip it).

        Returns (is_text_content, details).
        """
        import cv2
        from PIL import Image as PILImage

        x1, y1, x2, y2 = panel
        rw, rh = x2 - x1, y2 - y1
        if rw < 10 or rh < 10:
            return False, []

        sub_color = img[y1:y2, x1:x2]
        if sub_color.size == 0:
            return False, []

        # --- Text-content gate (still uses gray + CC analysis) ---
        sub_gray = cv2.cvtColor(sub_color, cv2.COLOR_BGR2GRAY)
        short_side = min(rw, rh)
        k = max(15, min(40, short_side // 4))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        border_px = np.concatenate([
            sub_gray[0, :], sub_gray[-1, :], sub_gray[:, 0], sub_gray[:, -1]
        ])
        bg_brightness = float(np.median(border_px))
        if bg_brightness >= 128:
            fg = cv2.morphologyEx(sub_gray, cv2.MORPH_BLACKHAT, kernel)
        else:
            fg = cv2.morphologyEx(sub_gray, cv2.MORPH_TOPHAT, kernel)
        _, text_binary = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        metrics = self._estimate_text_metrics(text_binary)
        if hasattr(self, "_panel_metrics"):
            self._panel_metrics[f"{x1},{y1},{x2},{y2}"] = metrics
        if metrics["is_text_content"]:
            return True, []

        # --- Color block detection ---
        try:
            pil_sub = PILImage.fromarray(sub_color[:, :, ::-1])
            n_colors = 6
            quantized = pil_sub.quantize(colors=n_colors, method=PILImage.Quantize.FASTOCTREE)
            q_idx = np.array(quantized)
        except Exception:
            return False, []

        # Background color = most common color at the 4 borders
        border_indices = np.concatenate([
            q_idx[0, :], q_idx[-1, :], q_idx[:, 0], q_idx[:, -1]
        ])
        bg_color_idx = int(np.bincount(border_indices).argmax())

        # Find non-background color blocks
        total_area = rw * rh
        min_block = max(total_area * 0.001, 100)  # 0.1% of panel or 100px²
        max_block = total_area * 0.85

        details = []
        for color_idx in range(n_colors):
            if color_idx == bg_color_idx:
                continue
            mask = (q_idx == color_idx).astype(np.uint8) * 255
            # Light close to merge adjacent same-color pixels
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                area = cw * ch
                if area < min_block or area > max_block:
                    continue
                if cw < 8 or ch < 8:
                    continue
                details.append((x1 + cx, y1 + cy, x1 + cx + cw, y1 + cy + ch))

        return False, details

    @staticmethod
    def _estimate_text_metrics(binary: np.ndarray) -> dict:
        """Estimate text metrics from binary via connected-component heights.

        Strategy: run CC on the raw binary (no dilate), measure the small
        bounding-rect heights. The median height of small CCs = character
        height. Vertical center-to-center spacing = line pitch.

        This is more robust than row-density autocorrelation because it
        works directly on detected shapes, regardless of overall coverage.

        Also determines if the panel is a text-content area:
        - Dense coverage (>30% filled, >60% active rows), OR
        - >= 8 lines of similar-height elements with regular spacing

        Returns dict with: line_height, line_pitch, line_gap,
                           n_lines, is_text_content
        """
        import cv2

        if binary.size == 0:
            return {
                "line_height": 0, "char_width": 0, "line_pitch": 0,
                "line_gap": 0, "n_lines": 0, "is_text_content": False,
            }
        h, w = binary.shape[:2]
        result = {
            "line_height": 0, "char_width": 0, "line_pitch": 0,
            "line_gap": 0, "n_lines": 0, "is_text_content": False,
        }
        if h < 30 or w < 10:
            return result

        # --- Quick check: high coverage = dense text ---
        # Dense text (terminals, editors) has high binary coverage AND
        # many small CCs (characters). UI panels with a few large blocks
        # can also have high coverage but far fewer CCs.
        row_density = np.mean(binary > 0, axis=1).astype(np.float32)
        coverage = float(np.mean(row_density))
        active_rows = float(np.mean(row_density > 0.05))

        # --- CC-based measurement ---
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) < 3:
            return result

        # High coverage + many CCs = dense text (characters are numerous)
        # A panel with only 5-10 large CCs is UI, not text.
        if coverage > 0.30 and active_rows > 0.60 and len(contours) >= 30:
            result["is_text_content"] = True

        # Collect bounding rects of CCs
        rects = []
        for cnt in contours:
            x, y, rw, rh = cv2.boundingRect(cnt)
            if rw >= 3 and rh >= 3:
                rects.append((x, y, x + rw, y + rh))

        if len(rects) < 3:
            return result

        # Character height/width = median of all CCs
        heights = np.array([r[3] - r[1] for r in rects])
        widths = np.array([r[2] - r[0] for r in rects])
        char_h = int(np.median(heights))
        char_w = int(np.median(widths))
        result["line_height"] = char_h
        result["char_width"] = char_w

        # --- Estimate line pitch from Y-center clustering ---
        # Group rects into "lines" by similar Y center (within char_h/2)
        centers_y = sorted([(r[1] + r[3]) / 2.0 for r in rects])
        lines: list[float] = []  # Y center of each line
        current_line_y = centers_y[0]
        current_count = 1
        merge_dist = max(char_h * 0.6, 5)

        for cy in centers_y[1:]:
            if cy - current_line_y <= merge_dist:
                # Same line: update running average
                current_line_y = (current_line_y * current_count + cy) / (
                    current_count + 1
                )
                current_count += 1
            else:
                lines.append(current_line_y)
                current_line_y = cy
                current_count = 1
        lines.append(current_line_y)

        result["n_lines"] = len(lines)

        if len(lines) >= 2:
            gaps = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
            pitch = int(np.median(gaps))
            result["line_pitch"] = pitch
            result["line_gap"] = max(0, pitch - char_h)

        # --- Text-content: many lines, most with regular spacing ---
        # Real text areas (terminals, editors) have some irregular gaps
        # (paragraph breaks, section headers). Instead of requiring ALL
        # gaps to be uniform (low CV), check if MOST gaps are near the
        # median — tolerates blank lines and section breaks.
        if not result["is_text_content"] and len(lines) >= 6:
            if len(lines) >= 2:
                gaps_arr = np.array(
                    [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
                )
                med_gap = float(np.median(gaps_arr))
                if med_gap > 3:
                    # "Regular" = within 1.5x of median gap
                    regular = np.sum(gaps_arr <= med_gap * 1.5)
                    regular_ratio = float(regular) / len(gaps_arr)
                    if regular_ratio >= 0.5:
                        result["is_text_content"] = True

        return result

    # ======================================================================
    # Pass 3: List zone inference + quantization
    # ======================================================================

    def _infer_list_zones(
        self,
        panels: list[tuple[int, int, int, int]],
        details: list[tuple[int, int, int, int]],
        img_h: int,
    ) -> list[tuple[int, int, int, int]]:
        """Infer list regions from vertical alignment of detail elements.

        A list zone is a sub-region of a panel where >=3 detail elements
        are vertically stacked with regular spacing.
        """
        zones = []
        for panel in panels:
            px1, py1, px2, py2 = panel
            # Details inside this panel
            panel_details = [
                d for d in details
                if d[0] >= px1 and d[2] <= px2 and d[1] >= py1 and d[3] <= py2
            ]
            if len(panel_details) < 3:
                continue

            # Sort by vertical center
            panel_details.sort(key=lambda d: (d[1] + d[3]) / 2)

            # Find groups with regular vertical spacing
            zone = self._find_regular_group(panel_details, panel)
            if zone is not None:
                zones.append(zone)

        return zones

    @staticmethod
    def _find_regular_group(
        details: list[tuple[int, int, int, int]],
        panel: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int] | None:
        """Find the largest group of vertically-regular details.

        Returns a tight zone rect around the group, or None.
        """
        if len(details) < 3:
            return None

        centers_y = [(d[1] + d[3]) / 2 for d in details]
        gaps = [centers_y[i + 1] - centers_y[i] for i in range(len(centers_y) - 1)]

        if not gaps:
            return None

        median_gap = float(np.median(gaps))
        if median_gap < 15:
            return None

        # Find the longest run of roughly-equal gaps
        tolerance = median_gap * 0.4
        best_start, best_len = 0, 0
        run_start, run_len = 0, 1

        for i, gap in enumerate(gaps):
            if abs(gap - median_gap) <= tolerance:
                run_len += 1
            else:
                if run_len > best_len:
                    best_start, best_len = run_start, run_len
                run_start = i + 1
                run_len = 1
        if run_len > best_len:
            best_start, best_len = run_start, run_len

        if best_len < 3:
            return None

        # Build zone from the regular group
        group = details[best_start : best_start + best_len]
        px1, py1, px2, py2 = panel
        zy1 = min(d[1] for d in group)
        zy2 = max(d[3] for d in group)
        # Use panel width for zone
        return (px1, zy1, px2, zy2)

    def _quantize_list(
        self,
        img: np.ndarray,
        zone: tuple[int, int, int, int],
    ) -> list[tuple[int, int, int, int]]:
        """Run ListQuantize logic on an auto-inferred zone."""
        from cvui.stages.analysis import ListQuantizeStage

        lq = ListQuantizeStage(zone_rect=zone, min_density=self.list_min_density)
        # Build a minimal context just for ListQuantize
        ctx = DetectionContext(img=img)
        ctx = lq.process(ctx)
        return ctx.rects

    # ======================================================================
    # Pass 4: Merge + classify
    # ======================================================================

    def _merge_and_classify(
        self,
        panels: list[tuple[int, int, int, int]],
        details: list[tuple[int, int, int, int]],
        list_items: list[tuple[int, int, int, int]],
        text_panels: list[tuple[int, int, int, int]],
        ctx: DetectionContext,
        w: int,
        h: int,
    ) -> list[tuple[int, int, int, int]]:
        """Deduplicate and merge all detected rects.

        Priority: list_items > details > panels.
        Dedup: if a detail is inside a list item, drop the detail.
        Text-content panels get "text-content" classification instead
        of individual detail elements.
        """
        tol = self.dedup_tolerance
        all_rects: list[tuple[int, int, int, int]] = []
        classifications: dict[int, str] = {}
        text_panel_set = set(text_panels)

        def _inside(inner, outer):
            return (
                inner[0] >= outer[0] - tol
                and inner[1] >= outer[1] - tol
                and inner[2] <= outer[2] + tol
                and inner[3] <= outer[3] + tol
            )

        # Collect list items (highest priority)
        list_set = set()
        for item in list_items:
            all_rects.append(item)
            classifications[len(all_rects) - 1] = "list-item"
            list_set.add(item)

        # Add details not inside any list item
        # Tier thresholds scale with image area (resolution-independent)
        total_area = max(w * h, 1)
        primary_threshold = total_area * 0.005    # ~0.5% of image
        secondary_threshold = total_area * 0.0005  # ~0.05% of image
        for detail in details:
            if any(_inside(detail, li) for li in list_set):
                continue
            all_rects.append(detail)
            area = (detail[2] - detail[0]) * (detail[3] - detail[1])
            if area > primary_threshold:
                tier = "primary"
            elif area > secondary_threshold:
                tier = "secondary"
            else:
                tier = "auxiliary"
            classifications[len(all_rects) - 1] = tier

        # Add panels
        for panel in panels:
            # Skip if panel is basically just one list zone
            if any(_inside(panel, li) for li in list_set):
                continue
            all_rects.append(panel)
            if panel in text_panel_set:
                classifications[len(all_rects) - 1] = "text-content"
            else:
                classifications[len(all_rects) - 1] = "panel"

        # Store classifications
        ctx.classifications = classifications

        # Store ensemble metadata
        ctx.ui_states["ensemble"] = {
            "panels": len(panels),
            "text_panels": len(text_panels),
            "details": len(details),
            "list_items": len(list_items),
            "total": len(all_rects),
        }
        # Store per-panel text metrics (set during Pass 2)
        if hasattr(self, "_panel_metrics"):
            ctx.ui_states["text_metrics"] = self._panel_metrics

        return all_rects
