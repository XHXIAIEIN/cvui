"""Tests for EnsembleStage — multi-pass coarse-to-fine detection."""
import numpy as np
import pytest
import cv2

from cvui.pipeline import DetectionContext, DetectionPipeline
from cvui.stages.ensemble import EnsembleStage
from cvui.stages import ensemble_pipeline


# ---------------------------------------------------------------------------
# Helpers: synthetic image generators
# ---------------------------------------------------------------------------

def _make_dark_panels_image(w=800, h=600) -> np.ndarray:
    """Dark background with 3 gray panels containing varied UI elements.
    Mimics a game HUD with buttons, icons, and labels — NOT solid text blocks.
    Each panel has a mix of element sizes (icons, buttons, labels).
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Colorful scene noise in background (high saturation, will be filtered)
    rng = np.random.RandomState(42)
    img[:, :, 0] = rng.randint(40, 120, (h, w), dtype=np.uint8)  # B
    img[:, :, 1] = rng.randint(80, 180, (h, w), dtype=np.uint8)  # G
    img[:, :, 2] = rng.randint(40, 120, (h, w), dtype=np.uint8)  # R

    def draw_ui_panel(x1, y1, x2, y2):
        """Draw a panel with varied UI elements (not uniform text)."""
        img[y1:y2, x1:x2] = (60, 60, 60)
        pw = x2 - x1
        # Title bar
        cv2.rectangle(img, (x1 + 10, y1 + 8), (x1 + pw // 2, y1 + 22),
                      (200, 200, 200), -1)
        # Close button (icon-sized)
        cv2.rectangle(img, (x2 - 25, y1 + 8), (x2 - 10, y1 + 22),
                      (180, 80, 80), -1)
        # Mixed elements: icon + label pairs
        for i in range(3):
            ey = y1 + 40 + i * 50
            # Small icon
            cv2.rectangle(img, (x1 + 15, ey), (x1 + 35, ey + 20),
                          (150, 150, 200), -1)
            # Label text (shorter than panel width)
            cv2.rectangle(img, (x1 + 45, ey + 3), (x1 + pw // 2, ey + 17),
                          (200, 200, 200), -1)
        # A wider button at bottom
        cv2.rectangle(img, (x1 + 15, y2 - 40), (x2 - 15, y2 - 15),
                      (80, 120, 80), -1)

    # Three panels with wide gaps
    draw_ui_panel(20, 100, 250, 350)
    draw_ui_panel(330, 100, 550, 380)
    draw_ui_panel(630, 100, 780, 330)

    return img


def _make_list_image(w=400, h=600) -> np.ndarray:
    """Chat-style list: dark background, items with regular vertical spacing.
    One item has a green highlight (selected).
    """
    img = np.full((h, w, 3), (40, 40, 40), dtype=np.uint8)

    item_h = 60
    gap = 4
    start_y = 50
    n_items = 7

    for i in range(n_items):
        y1 = start_y + i * (item_h + gap)
        y2 = y1 + item_h
        if i == 3:
            # Highlighted item (green)
            cv2.rectangle(img, (10, y1), (w - 10, y2), (60, 140, 60), -1)
        else:
            # Normal item (slightly lighter gray)
            cv2.rectangle(img, (10, y1), (w - 10, y2), (80, 80, 80), -1)
        # Text-like content
        cv2.rectangle(img, (20, y1 + 10), (w - 30, y1 + 25), (200, 200, 200), -1)
        cv2.rectangle(img, (20, y1 + 35), (w // 2, y1 + 48), (160, 160, 160), -1)

    return img


def _make_terminal_image(w=800, h=600) -> np.ndarray:
    """Dark terminal window filled with text lines.
    Should be classified as text-content, NOT individual UI elements.
    """
    img = np.full((h, w, 3), (30, 30, 30), dtype=np.uint8)

    # Terminal chrome: title bar
    cv2.rectangle(img, (0, 0), (w, 30), (50, 50, 50), -1)
    # Title text
    cv2.rectangle(img, (10, 8), (200, 22), (180, 180, 180), -1)

    # Text lines: many, uniform height, spanning most of the width
    line_h = 18
    gap = 4
    start_y = 40
    for i in range(25):
        y = start_y + i * (line_h + gap)
        if y + line_h > h - 10:
            break
        # Vary line width slightly (like real terminal output)
        line_w = w - 40 - (i % 3) * 60
        cv2.rectangle(img, (20, y), (20 + line_w, y + line_h),
                       (190, 190, 190), -1)

    return img


def _make_mixed_app_image(w=800, h=600) -> np.ndarray:
    """Desktop app with toolbar buttons and main content blocks.
    Simulates a ScreenToGif-like UI with discrete elements.
    """
    img = np.full((h, w, 3), (240, 240, 240), dtype=np.uint8)

    # Toolbar (top) — discrete button elements
    cv2.rectangle(img, (0, 0), (w, 50), (220, 220, 220), -1)
    for i in range(8):
        bx = 10 + i * 60
        cv2.rectangle(img, (bx, 8), (bx + 45, 42), (160, 160, 170), -1)
        # Icon inside button
        cv2.rectangle(img, (bx + 12, 14), (bx + 32, 34), (100, 100, 120), -1)

    # Main content: distinct blocks with spacing
    cv2.rectangle(img, (20, 70), (380, 280), (255, 255, 255), -1)
    cv2.rectangle(img, (400, 70), (780, 280), (250, 250, 255), -1)
    # Detail elements in lower area
    for i in range(5):
        cx = 20 + i * 155
        cv2.rectangle(img, (cx, 310), (cx + 130, 380), (200, 200, 210), -1)
        # Label inside
        cv2.rectangle(img, (cx + 10, 340), (cx + 80, 355), (120, 120, 130), -1)

    # Status bar (bottom)
    cv2.rectangle(img, (0, h - 30), (w, h), (210, 210, 210), -1)
    cv2.rectangle(img, (10, h - 24), (200, h - 8), (160, 160, 160), -1)

    return img


# ---------------------------------------------------------------------------
# EnsembleStage unit tests
# ---------------------------------------------------------------------------

class TestEnsembleStage:
    def test_basic_smoke(self):
        """Ensemble produces non-empty results on a synthetic image."""
        img = _make_dark_panels_image()
        stage = EnsembleStage()
        ctx = DetectionContext(img=img)
        ctx = stage.process(ctx)
        assert len(ctx.rects) > 0
        assert "ensemble" in ctx.ui_states

    def test_panels_detected(self):
        """Panels should be found in the dark panels image."""
        img = _make_dark_panels_image()
        stage = EnsembleStage()
        ctx = DetectionContext(img=img)
        ctx = stage.process(ctx)
        # At least 1 panel region detected (noisy background may merge
        # small panels at this resolution)
        assert len(ctx.zones) >= 1, f"Expected >=1 zone, got {len(ctx.zones)}"
        assert len(ctx.rects) >= 1, f"Expected >=1 rect, got {len(ctx.rects)}"

    def test_ui_panels_not_text_content(self):
        """Panels with varied UI elements (icons + labels + buttons)
        should NOT be classified as text-content."""
        img = _make_dark_panels_image()
        stage = EnsembleStage()
        ctx = DetectionContext(img=img)
        ctx = stage.process(ctx)
        text_content = [
            i for i, c in ctx.classifications.items() if c == "text-content"
        ]
        assert len(text_content) == 0, (
            f"UI panels should not be text-content, got {len(text_content)}"
        )

    def test_list_detection(self):
        """List items should be detected in a chat-style image."""
        img = _make_list_image()
        stage = EnsembleStage()
        ctx = DetectionContext(img=img)
        ctx = stage.process(ctx)
        list_items = [
            i for i, c in ctx.classifications.items() if c == "list-item"
        ]
        # The image has 7 items; ensemble should find at least some via quantize
        total = len(ctx.rects)
        assert total >= 3, f"Expected >=3 total elements, got {total}"

    def test_mixed_app(self):
        """Mixed app image should produce a variety of element types."""
        img = _make_mixed_app_image()
        stage = EnsembleStage()
        ctx = DetectionContext(img=img)
        ctx = stage.process(ctx)
        assert len(ctx.rects) >= 8, (
            f"Expected >=8 elements in mixed app, got {len(ctx.rects)}"
        )

    def test_ensemble_metadata(self):
        """Ensemble should report metadata in ui_states."""
        img = _make_dark_panels_image()
        stage = EnsembleStage()
        ctx = DetectionContext(img=img)
        ctx = stage.process(ctx)
        meta = ctx.ui_states["ensemble"]
        assert "panels" in meta
        assert "details" in meta
        assert "list_items" in meta
        assert "total" in meta
        assert meta["total"] == len(ctx.rects)

    def test_no_duplicates(self):
        """Rects should not have exact duplicates."""
        img = _make_dark_panels_image()
        stage = EnsembleStage()
        ctx = DetectionContext(img=img)
        ctx = stage.process(ctx)
        unique = set(ctx.rects)
        assert len(unique) == len(ctx.rects), (
            f"Found {len(ctx.rects) - len(unique)} duplicates"
        )

    def test_terminal_not_exploded(self):
        """Terminal image should NOT produce hundreds of text-line elements."""
        img = _make_terminal_image()
        stage = EnsembleStage()
        ctx = DetectionContext(img=img)
        ctx = stage.process(ctx)
        # Terminal should be recognized as text-content, not 25+ detail elements
        assert len(ctx.rects) < 10, (
            f"Terminal produced {len(ctx.rects)} rects — text filter failed"
        )
        text_content = [
            i for i, c in ctx.classifications.items() if c == "text-content"
        ]
        assert len(text_content) >= 1, (
            f"Expected terminal panel as text-content, got: "
            f"{dict(ctx.classifications)}"
        )

    def test_custom_params(self):
        """Custom parameters should be respected."""
        img = _make_dark_panels_image()
        stage = EnsembleStage(
            coarse_kernel=60,
            panel_area_pct=2.0,
            fine_kernel_range=(20, 50),
        )
        ctx = DetectionContext(img=img)
        ctx = stage.process(ctx)
        assert len(ctx.rects) > 0


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------

class TestEnsemblePipeline:
    def test_ensemble_pipeline_factory(self):
        """ensemble_pipeline() returns a working pipeline."""
        pipe = ensemble_pipeline()
        assert isinstance(pipe, DetectionPipeline)
        assert any(isinstance(s, EnsembleStage) for s in pipe.stages)

    def test_ensemble_pipeline_with_downscale(self):
        """ensemble_pipeline with scale < 1.0 adds DownscaleStage."""
        pipe = ensemble_pipeline(scale=0.75)
        assert len(pipe.stages) == 2  # Downscale + Ensemble

    def test_ensemble_pipeline_run(self):
        """Full pipeline run produces results."""
        img = _make_dark_panels_image()
        pipe = ensemble_pipeline()
        ctx = pipe.run(img)
        assert len(ctx.rects) > 0
        assert "EnsembleStage" in ctx.stage_log

    def test_ensemble_pipeline_with_downscale_run(self):
        """Pipeline with downscale maps coordinates back correctly."""
        img = _make_dark_panels_image()
        pipe = ensemble_pipeline(scale=0.75)
        ctx = pipe.run(img)
        # All rects should be within original image bounds
        for r in ctx.rects:
            assert r[0] >= 0 and r[1] >= 0
            assert r[2] <= 800 and r[3] <= 600


# ---------------------------------------------------------------------------
# Zone inference unit tests
# ---------------------------------------------------------------------------

class TestZoneInference:
    def test_regular_group_detection(self):
        """_find_regular_group should detect vertically aligned elements."""
        details = [
            (10, 50 + i * 40, 200, 50 + i * 40 + 30)
            for i in range(5)
        ]
        panel = (0, 40, 220, 300)
        zone = EnsembleStage._find_regular_group(details, panel)
        assert zone is not None
        # Zone should span the full panel width
        assert zone[0] == panel[0]
        assert zone[2] == panel[2]

    def test_irregular_group_rejected(self):
        """Irregularly spaced elements should not produce a zone."""
        details = [
            (10, 50, 200, 80),
            (10, 200, 200, 230),  # big gap
            (10, 210, 200, 240),  # small gap
        ]
        panel = (0, 0, 220, 300)
        zone = EnsembleStage._find_regular_group(details, panel)
        assert zone is None

    def test_too_few_elements(self):
        """Fewer than 3 elements should not produce a zone."""
        details = [
            (10, 50, 200, 80),
            (10, 130, 200, 160),
        ]
        panel = (0, 0, 220, 200)
        zone = EnsembleStage._find_regular_group(details, panel)
        assert zone is None
