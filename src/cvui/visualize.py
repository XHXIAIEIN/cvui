"""Blueprint visualization — skeleton overlay and element annotation.

Two output modes:
- skeleton: semi-transparent color blocks showing major zones
- annotated: green bounding boxes on every detected UI component

Element detection uses grayscale + gamma correction + adaptive threshold
+ directional dilation + connected components. Pure CV, no OCR dependency,
~40ms.
"""
from __future__ import annotations

import io
import logging

from PIL import Image, ImageDraw

from .types import UIBlueprint, OCRWord

log = logging.getLogger(__name__)

# Zone overlay colors (RGBA) — distinct, semi-transparent
ZONE_COLORS = [
    (80, 80, 80, 90),      # gray — icon bars, toolbars
    (60, 60, 200, 70),     # blue — lists, navigation
    (180, 60, 180, 70),    # purple — headers, panels
    (60, 180, 60, 70),     # green — content areas
    (200, 140, 50, 70),    # brown/gold — input areas
    (200, 60, 60, 70),     # red — alerts, special
    (60, 180, 180, 70),    # cyan
    (180, 180, 60, 70),    # yellow
]

ELEMENT_COLOR = (0, 255, 0)  # green for element boxes
ELEMENT_WIDTH = 2


def render_skeleton(
    screenshot: Image.Image | bytes,
    blueprint: UIBlueprint,
) -> Image.Image:
    """Render skeleton overlay — semi-transparent color blocks per zone."""
    if isinstance(screenshot, bytes):
        screenshot = Image.open(io.BytesIO(screenshot))

    base = screenshot.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for i, zone in enumerate(blueprint.zones):
        color = ZONE_COLORS[i % len(ZONE_COLORS)]
        r = zone.rect
        draw.rectangle([r[0], r[1], r[2], r[3]], fill=color)

        tag = "dynamic" if zone.dynamic else "skeleton"
        label = f"{zone.name} [{tag}]"
        draw.text((r[0] + 8, r[1] + 8), label, fill=(255, 255, 255, 220))

    result = Image.alpha_composite(base, overlay)
    return result.convert("RGB")


def detect_elements(
    png_bytes: bytes,
    mode: str = "standard",
) -> list[tuple[int, int, int, int]]:
    """Detect UI elements via pluggable pipeline.

    Args:
        png_bytes: screenshot as PNG bytes
        mode: "fast", "standard", or "full"

    Returns:
        list of (x1, y1, x2, y2) bounding rectangles
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        log.warning("detect_elements: opencv not installed")
        return []

    nparr = np.frombuffer(png_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    from .detection import fast_pipeline, standard_pipeline, full_pipeline
    pipelines = {
        "fast": fast_pipeline,
        "standard": standard_pipeline,
        "full": full_pipeline,
    }
    factory = pipelines.get(mode, standard_pipeline)
    ctx = factory().run(img)
    return ctx.rects


# Colors by element classification
CLASS_COLORS = {
    "icon": (0, 200, 255),       # cyan
    "text": (0, 255, 0),         # green
    "button": (255, 200, 0),     # yellow
    "image": (200, 100, 255),    # purple
    "container": (255, 150, 0),  # orange
    "element": (0, 255, 0),      # green (default)
}
TRUNCATED_COLOR = (255, 0, 0)    # red for edge-truncated elements
EDGE_MARGIN = 5                  # pixels from window edge = truncated


def render_annotated(
    screenshot: Image.Image | bytes,
    element_rects: list[tuple[int, int, int, int]] | None = None,
    ctx: object = None,
    mode: str = "standard",
) -> Image.Image:
    """Render element annotation with color-coded types and truncation markers.

    Args:
        screenshot: PIL Image or PNG bytes
        element_rects: pre-computed bounding boxes (simple mode)
        ctx: DetectionContext with classifications (rich mode, overrides element_rects)
        mode: detection pipeline mode if auto-detecting

    Colors:
        cyan = icon, green = text/element, yellow = button,
        purple = image, orange = container, red = truncated (touching edge)
    """
    if isinstance(screenshot, bytes):
        png_bytes = screenshot
        img = Image.open(io.BytesIO(screenshot))
    else:
        img = screenshot.copy()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

    img = img.convert("RGB")
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)

    # Get rects and classifications
    if ctx is not None:
        rects = ctx.rects
        classifications = getattr(ctx, "classifications", {})
    elif element_rects is not None:
        rects = element_rects
        classifications = {}
    else:
        rects = detect_elements(png_bytes, mode=mode)
        classifications = {}

    for i, (x1, y1, x2, y2) in enumerate(rects):
        ew, eh = x2 - x1, y2 - y1

        # Truncated = touching edge AND looks cut off
        # (element extends to edge but isn't a full-width/full-height bar)
        touches_edge = (
            x1 <= EDGE_MARGIN or y1 <= EDGE_MARGIN
            or x2 >= img_w - EDGE_MARGIN or y2 >= img_h - EDGE_MARGIN
        )
        # A full-width toolbar or header touching edge is normal, not truncated
        # Truncated = touches edge but NOT spanning the full dimension
        is_full_width = ew > img_w * 0.85
        is_full_height = eh > img_h * 0.85
        truncated = touches_edge and not is_full_width and not is_full_height

        if truncated:
            color = TRUNCATED_COLOR
        else:
            cls = classifications.get(i, "element")
            color = CLASS_COLORS.get(cls, ELEMENT_COLOR)

        # Draw with 1px padding so frame doesn't overlap content
        pad = 1
        dx1 = max(0, x1 - pad)
        dy1 = max(0, y1 - pad)
        dx2 = min(img_w, x2 + pad)
        dy2 = min(img_h, y2 + pad)
        draw.rectangle([dx1, dy1, dx2, dy2], outline=color, width=ELEMENT_WIDTH)

    return img


def render_grayscale(
    screenshot: Image.Image | bytes,
    gamma: float = 0.6,
    contrast: float = 1.3,
    brightness: float = 10,
) -> Image.Image:
    """Render adjusted grayscale — the intermediate used for element detection.

    Useful for debugging and visual inspection.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        if isinstance(screenshot, bytes):
            return Image.open(io.BytesIO(screenshot)).convert("L")
        return screenshot.convert("L")

    if isinstance(screenshot, bytes):
        nparr = np.frombuffer(screenshot, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        img = np.array(screenshot.convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_gamma = np.power(gray / 255.0, gamma) * 255.0
    gray_adj = np.clip(contrast * gray_gamma + brightness, 0, 255).astype(np.uint8)

    return Image.fromarray(gray_adj, mode="L")
