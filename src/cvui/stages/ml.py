"""ML-based stages: OmniParser, GroundingDINO."""
from __future__ import annotations

import logging

from cvui.pipeline import DetectionStage, DetectionContext

log = logging.getLogger(__name__)


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
