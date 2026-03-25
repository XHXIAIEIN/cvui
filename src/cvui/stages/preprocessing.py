"""Preprocessing stages: Downscale, Grayscale, TopHat, Otsu."""
from __future__ import annotations

import numpy as np

from cvui.pipeline import DetectionStage, DetectionContext


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
