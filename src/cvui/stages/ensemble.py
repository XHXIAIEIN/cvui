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

        # ------------------------------------------------------------------
        # Pass 0: Saturation separation
        # ------------------------------------------------------------------
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        _, ui_mask = cv2.threshold(
            s_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        ui_mask = cv2.morphologyEx(
            ui_mask, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8)
        )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtered_gray = gray.copy()
        filtered_gray[ui_mask == 0] = 0

        ctx.layers["ui_mask"] = ui_mask
        ctx.layers["filtered_gray"] = filtered_gray

        # ------------------------------------------------------------------
        # Pass 1: Coarse detection → panels
        # ------------------------------------------------------------------
        panels = self._detect_panels(filtered_gray, w, h, min_panel_area)
        ctx.zones = list(panels)

        # ------------------------------------------------------------------
        # Pass 2: Fine detection within each panel
        # ------------------------------------------------------------------
        details = []
        text_panels = []  # panels classified as text-content areas
        self._panel_metrics = {}
        for i, panel in enumerate(panels):
            is_text, panel_details = self._detect_details_in_panel(
                filtered_gray, panel
            )
            if is_text:
                text_panels.append(panel)
            details.extend(panel_details)

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

    def _detect_panels(
        self, gray: np.ndarray, w: int, h: int, min_area: float
    ) -> list[tuple[int, int, int, int]]:
        """Coarse TopHat → Otsu → medium dilate → area filter."""
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

        # Medium dilate: connect text lines horizontally, titles vertically
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

    # ======================================================================
    # Pass 2: Fine detection inside panels
    # ======================================================================

    def _detect_details_in_panel(
        self, gray: np.ndarray, panel: tuple[int, int, int, int]
    ) -> tuple[bool, list[tuple[int, int, int, int]]]:
        """Per-panel: TopHat → Otsu → estimate metrics → slice into rows → CC.

        Architecture: coarse-to-fine with row slicing.
        1. TopHat + Otsu on the whole panel → binary
        2. Estimate text metrics (line_height, line_pitch) from CCs
        3. If text-content → skip
        4. Slice binary into horizontal strips at line_pitch intervals
        5. Per-strip: horizontal-only dilate → CC → collect elements

        Row slicing prevents cross-line merging entirely — each strip
        is one logical "line" of the UI.

        Returns (is_text_content, details).
        """
        import cv2

        x1, y1, x2, y2 = panel
        sub = gray[y1:y2, x1:x2]
        if sub.size == 0:
            return False, []

        rw, rh = x2 - x1, y2 - y1
        short_side = min(rw, rh)
        k_min, k_max = self.fine_kernel_range
        k = max(k_min, min(k_max, short_side // 4))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        border = np.concatenate([
            sub[0, :], sub[-1, :], sub[:, 0], sub[:, -1]
        ])
        bg_brightness = float(np.median(border))
        if bg_brightness >= 128:
            fg = cv2.morphologyEx(sub, cv2.MORPH_BLACKHAT, kernel)
        else:
            fg = cv2.morphologyEx(sub, cv2.MORPH_TOPHAT, kernel)

        _, binary = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # --- Step 2: Estimate text metrics ---
        metrics = self._estimate_text_metrics(binary)
        if hasattr(self, "_panel_metrics"):
            panel_key = f"{x1},{y1},{x2},{y2}"
            self._panel_metrics[panel_key] = metrics

        # --- Step 3: Text-content gate ---
        if metrics["is_text_content"]:
            return True, []

        line_h = metrics["line_height"]
        char_w = metrics["char_width"]
        line_pitch = metrics["line_pitch"]

        # --- Step 4+5: Slice into rows, detect per-row ---
        if line_pitch > 0 and line_h > 0:
            details = self._detect_per_row(
                binary, x1, y1, rw, rh, line_h, line_pitch, char_w
            )
        else:
            # No metrics: fall back to whole-panel detection
            details = self._detect_whole_panel(
                binary, x1, y1, rw, rh, short_side
            )

        return False, details

    @staticmethod
    def _detect_per_row(
        binary: np.ndarray,
        panel_x: int, panel_y: int,
        rw: int, rh: int,
        line_h: int, line_pitch: int, char_w: int,
    ) -> list[tuple[int, int, int, int]]:
        """Slice panel binary into row strips, detect elements per-row.

        Each strip height = line_pitch (one logical row).
        Only horizontal dilate within each strip — no vertical merging.
        """
        import cv2

        # Horizontal dilate kernel: connect chars within words.
        # Cap to ~15% of panel width.
        max_h_dilate = max(8, rw // 7)
        h_dilate_w = max(8, min(max_h_dilate, int(char_w * 1.2)))
        h_dilate_h = max(2, min(6, line_h // 6))
        h_kernel = np.ones((h_dilate_h, h_dilate_w), np.uint8)

        min_elem_h = max(4, line_h // 3)
        details = []

        # Slice into rows
        y = 0
        while y < rh:
            strip_end = min(y + line_pitch, rh)
            strip = binary[y:strip_end, :]
            if strip.size == 0:
                y += line_pitch
                continue

            # Horizontal-only dilate within this strip
            dilated = cv2.dilate(strip, h_kernel, iterations=1)

            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                if cw < 6 or ch < min_elem_h:
                    continue
                # Skip if it spans the entire strip (whole-row blob)
                if cw > rw * 0.95 and ch > (strip_end - y) * 0.95:
                    continue
                details.append((
                    panel_x + cx, panel_y + y + cy,
                    panel_x + cx + cw, panel_y + y + cy + ch,
                ))

            y += line_pitch

        return details

    @staticmethod
    def _detect_whole_panel(
        binary: np.ndarray,
        panel_x: int, panel_y: int,
        rw: int, rh: int, short_side: int,
    ) -> list[tuple[int, int, int, int]]:
        """Fallback: detect on the whole panel when no text metrics available."""
        import cv2

        scale_factor = max(1.0, short_side / 400.0)
        h_k = (max(2, int(2 * scale_factor)), max(8, int(8 * scale_factor)))
        v_k = (max(4, int(4 * scale_factor)), max(2, int(2 * scale_factor)))

        dilated = cv2.dilate(binary, np.ones(h_k, np.uint8), iterations=1)
        dilated = cv2.dilate(dilated, np.ones(v_k, np.uint8), iterations=1)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        details = []
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            if cw < 10 or ch < 8:
                continue
            if cw > rw * 0.95 and ch > rh * 0.95:
                continue
            details.append((
                panel_x + cx, panel_y + cy,
                panel_x + cx + cw, panel_y + cy + ch,
            ))

        return details

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

        h, w = binary.shape[:2]
        result = {
            "line_height": 0, "char_width": 0, "line_pitch": 0,
            "line_gap": 0, "n_lines": 0, "is_text_content": False,
        }
        if h < 30:
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
        if not result["is_text_content"] and len(lines) >= 8:
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
        for detail in details:
            if any(_inside(detail, li) for li in list_set):
                continue
            all_rects.append(detail)
            area = (detail[2] - detail[0]) * (detail[3] - detail[1])
            if area > 30000:
                tier = "primary"
            elif area > 3000:
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
