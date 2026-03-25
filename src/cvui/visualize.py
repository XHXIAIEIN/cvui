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


def render_annotated(
    screenshot: Image.Image | bytes,
    element_rects: list[tuple[int, int, int, int]] | None = None,
    mode: str = "standard",
) -> Image.Image:
    """Render element annotation — green boxes around detected UI components.

    If element_rects is not provided, runs detect_elements() automatically.

    Args:
        screenshot: PIL Image or PNG bytes
        element_rects: pre-computed element bounding boxes (optional)
        mode: detection pipeline mode ("fast", "standard", "full")

    Returns:
        PIL Image with green bounding boxes
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

    if element_rects is None:
        element_rects = detect_elements(png_bytes, mode=mode)

    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2 in element_rects:
        draw.rectangle([x1, y1, x2, y2], outline=ELEMENT_COLOR, width=ELEMENT_WIDTH)

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
