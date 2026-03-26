"""Advanced stages for complex scenes: games, video players, dynamic apps."""
from __future__ import annotations

import numpy as np

from cvui.pipeline import DetectionStage, DetectionContext


class MultiFrameAccumulatorStage(DetectionStage):
    """Accumulate N frames, extract static regions (HUD) vs dynamic (scene).

    Static pixels (low variance across frames) = UI elements.
    Dynamic pixels (high variance) = game scene / video content.

    Usage: call add_frame() before process() for each new frame.
    """

    def __init__(self, n_frames: int = 10, static_threshold: float = 50.0):
        self.n_frames = n_frames
        self.static_threshold = static_threshold
        self.buffer: list[np.ndarray] = []

    def add_frame(self, frame: np.ndarray):
        """Add a frame to the buffer. Call this before process()."""
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        self.buffer.append(gray.astype(np.float32))
        if len(self.buffer) > self.n_frames:
            self.buffer.pop(0)

    def process(self, ctx):
        import cv2
        if len(self.buffer) < 3:
            # Not enough frames yet — add current and pass through
            self.add_frame(ctx.img)
            return ctx

        # Compute per-pixel variance across frames
        stack = np.stack(self.buffer, axis=0)
        variance = np.var(stack, axis=0)

        # Low variance = static UI, high variance = dynamic scene
        static_mask = (variance < self.static_threshold).astype(np.uint8) * 255

        # Store as binary for downstream stages
        ctx.binary = static_mask
        ctx.ui_states["static_mask"] = [(0, 0, ctx.width, ctx.height)]
        return ctx

    def should_continue(self, ctx):
        return len(self.buffer) >= 3


class ColorQuantizeStage(DetectionStage):
    """Quantize image to limited palette, separate UI from scene.

    UI uses a designed palette (5-8 colors). Game scenes use continuous colors.
    After quantization, UI elements have clean boundaries.

    Uses PIL's FASTOCTREE quantizer (~5ms).
    """

    def __init__(self, n_colors: int = 8, method: str = "octree"):
        self.n_colors = n_colors
        self.method = method

    def process(self, ctx):
        from PIL import Image
        import cv2

        # BGR numpy → RGB PIL
        pil = Image.fromarray(ctx.img[:, :, ::-1])

        # Quantize
        method_map = {
            "octree": Image.Quantize.FASTOCTREE,
            "median_cut": Image.Quantize.MEDIANCUT,
            "max_coverage": Image.Quantize.MAXCOVERAGE,
        }
        q_method = method_map.get(self.method, Image.Quantize.FASTOCTREE)
        quantized = pil.quantize(colors=self.n_colors, method=q_method)

        # Extract palette
        palette_data = quantized.getpalette()
        if palette_data:
            colors = []
            for i in range(0, min(len(palette_data), self.n_colors * 3), 3):
                colors.append((palette_data[i], palette_data[i+1], palette_data[i+2]))
            ctx.ui_states["palette"] = colors

        # Convert back to BGR numpy for downstream
        q_rgb = np.array(quantized.convert("RGB"))
        ctx.img = q_rgb[:, :, ::-1]  # RGB → BGR
        return ctx


class MultiColorSpaceStage(DetectionStage):
    """Analyze image in multiple color spaces for richer features.

    LAB L-channel: perceptually uniform brightness (better than RGB grayscale).
    HSV S-channel: highlights colorful UI elements (icons, badges).
    """

    def process(self, ctx):
        import cv2

        # LAB — L channel is perceptually uniform brightness
        lab = cv2.cvtColor(ctx.img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # HSV — S channel highlights colorful elements
        hsv = cv2.cvtColor(ctx.img, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]

        # Use L channel as gray (better than simple grayscale for UI)
        ctx.gray = l_channel

        # Store saturation map for classification
        ctx.ui_states["saturation_map"] = s_channel.tolist()  # store as serializable
        return ctx


class GradientDetectorStage(DetectionStage):
    """Detect and mask gradient regions (game backgrounds, not UI).

    Uniform regions = solid UI panels.
    Gradient regions = decorative backgrounds.
    High-variance regions = textures / scene content.

    Masks out non-UI regions in the binary mask.
    """

    def __init__(self, scene_threshold: float = 500.0):
        self.scene_threshold = scene_threshold

    def process(self, ctx):
        import cv2
        if ctx.gray is None:
            ctx.gray = cv2.cvtColor(ctx.img, cv2.COLOR_BGR2GRAY)

        # Compute local variance in small patches
        gray_f = ctx.gray.astype(np.float32)
        kernel = np.ones((15, 15), np.float32) / 225
        mean = cv2.filter2D(gray_f, -1, kernel)
        sq_mean = cv2.filter2D(gray_f ** 2, -1, kernel)
        local_var = np.clip(sq_mean - mean ** 2, 0, None)

        # High local variance = texture/scene → not UI
        scene_mask = (local_var > self.scene_threshold).astype(np.uint8) * 255

        # Apply: zero out scene regions in binary
        if ctx.binary is not None:
            ctx.binary[scene_mask > 0] = 0

        ctx.ui_states["scene_mask_area"] = int(np.count_nonzero(scene_mask))
        return ctx


class TrackingStage(DetectionStage):
    """Track known elements across frames instead of re-detecting.

    Given previous frame's rects, search for each element in a small
    neighborhood of the current frame using template matching.
    Much faster than full re-detection for real-time use.
    """

    def __init__(self, prev_rects: list[tuple[int,int,int,int]] | None = None,
                 prev_img: np.ndarray | None = None,
                 search_radius: int = 30):
        self.prev_rects = prev_rects or []
        self.prev_img = prev_img
        self.search_radius = search_radius

    def process(self, ctx):
        import cv2
        if not self.prev_rects or self.prev_img is None:
            return ctx  # first frame, need full detection

        prev_gray = cv2.cvtColor(self.prev_img, cv2.COLOR_BGR2GRAY) if len(self.prev_img.shape) == 3 else self.prev_img
        curr_gray = cv2.cvtColor(ctx.img, cv2.COLOR_BGR2GRAY) if len(ctx.img.shape) == 3 else ctx.img

        tracked = []
        for r in self.prev_rects:
            x1, y1, x2, y2 = r
            rw, rh = x2 - x1, y2 - y1
            if rw < 5 or rh < 5:
                continue

            # Extract template from previous frame
            template = prev_gray[y1:y2, x1:x2]
            if template.size == 0:
                continue

            # Search region in current frame
            sx1 = max(0, x1 - self.search_radius)
            sy1 = max(0, y1 - self.search_radius)
            sx2 = min(ctx.width, x2 + self.search_radius)
            sy2 = min(ctx.height, y2 + self.search_radius)
            search = curr_gray[sy1:sy2, sx1:sx2]

            if search.shape[0] < rh or search.shape[1] < rw:
                continue

            # Template matching
            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > 0.7:  # good match
                nx1 = sx1 + max_loc[0]
                ny1 = sy1 + max_loc[1]
                tracked.append((nx1, ny1, nx1 + rw, ny1 + rh))

        if tracked:
            ctx.rects.extend(tracked)

        return ctx

    def should_continue(self, ctx):
        # If tracking found elements, skip full detection
        return len(ctx.rects) == 0


class SaturationFilterStage(DetectionStage):
    """Filter out high-saturation scene regions, keep low-saturation UI.

    Game UIs typically use desaturated colors (gray/black semi-transparent
    panels, white text). Game scenes have rich, saturated colors. By masking
    out high-saturation pixels before detection, scene textures (stairs,
    walls, foliage) are excluded while UI elements are preserved.

    Uses Otsu auto-threshold on the HSV saturation channel.
    """

    def process(self, ctx):
        import cv2

        hsv = cv2.cvtColor(ctx.img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]  # saturation channel

        # Otsu on saturation: BINARY_INV so low sat (UI) = 255, high sat (scene) = 0
        _, ui_mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Close gaps in UI panels
        ui_mask = cv2.morphologyEx(ui_mask, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))

        # Store as layer
        ctx.layers["ui_mask"] = ui_mask

        # Create filtered gray: scene pixels zeroed
        gray = ctx.gray
        if gray is None:
            gray = cv2.cvtColor(ctx.img, cv2.COLOR_BGR2GRAY)
        filtered = gray.copy()
        filtered[ui_mask == 0] = 0
        ctx.gray = filtered

        # Store mask info
        total = ui_mask.size
        ui_pct = np.count_nonzero(ui_mask) * 100 / max(total, 1)
        ctx.ui_states["saturation_filter"] = f"{ui_pct:.0f}% UI, {100-ui_pct:.0f}% scene"
        return ctx


class ZoneDetectorStage(DetectionStage):
    """Detect UI panel zones from content distribution in ui_mask.

    Uses content heat map: gaussian blur on foreground pixels,
    threshold to find dense clusters = UI panels.
    Writes results to ctx.zones.
    """

    def __init__(self, blur_size: int = 81, threshold: int = 25,
                 min_width: int = 100, min_height: int = 50):
        self.blur_size = blur_size
        self.threshold = threshold
        self.min_width = min_width
        self.min_height = min_height

    def process(self, ctx):
        import cv2

        # Use ui_mask if available, otherwise use gray/foreground
        source = ctx.layers.get("ui_mask")
        if source is None:
            source = ctx.layers.get("foreground")
        if source is None:
            source = ctx.gray
        if source is None:
            return ctx

        # Create heat map from content distribution
        heat = cv2.GaussianBlur(source.astype(np.float32),
                                (self.blur_size, self.blur_size), 0)
        if heat.max() > 0:
            heat_norm = (heat / heat.max() * 255).astype(np.uint8)
        else:
            return ctx

        # Threshold to find panels
        _, panel_mask = cv2.threshold(heat_norm, self.threshold, 255, cv2.THRESH_BINARY)
        panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_CLOSE,
                                       np.ones((30, 30), np.uint8))

        # Find contours = zones
        contours, _ = cv2.findContours(panel_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= self.min_width and h >= self.min_height:
                ctx.zones.append((x, y, x + w, y + h))

        ctx.zones.sort(key=lambda z: (z[1], z[0]))
        ctx.layers["zones_mask"] = panel_mask
        return ctx
