# cvui Steal Sheet Stages — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 6 patterns from Orchestrator's steal-sheet research as new cvui Stages.

**Architecture:** All new stages follow the existing `DetectionStage` ABC (`process(ctx) -> ctx`). Each stage goes in the appropriate module, gets exported from `stages/__init__.py`, and has tests.

**Source patterns:** From `D:/Users/Administrator/Documents/GitHub/orchestrator/docs/architecture/PATTERNS.md`

---

## Overview

| # | Pattern ID | Name | Source | Module | Effort |
|---|-----------|------|--------|--------|--------|
| 1 | V1 | VLMZoneStage | Gemini/OmniParser (R7) | stages/ml.py | Medium |
| 2 | V2 | CNNClassifyStage | UIED (R7) | stages/ml.py | Medium |
| 3 | V4 | FormatConverterStage | labelU (R7) | stages/analysis.py | Low |
| 4 | V7 | TilingStage | DarkHelp (R7) | stages/advanced.py | Medium |
| 5 | P11 | AdaptiveDownscaleStage | Carbonyl (R9) | stages/preprocessing.py | Low |
| 6 | V5 | PreAnnotationExporter | labelU (R7) | new: stages/export.py | Low |
| 7 | V3 | DetectionFeatureRetrieval | PaddleX PP-ShiTuV2 (R7) | stages/ml.py | High |
| 8 | V6 | Pix2EmbStage | NExT-Chat (R7) | research/ | High |
| 9 | V8 | DOTSOCRStage | R7 supplement | stages/ml.py | Medium |
| 10 | V9 | SyntheticDataGenerator | DocLayout-YOLO (R7) | research/ | High |
| 11 | V14 | TextFirstLayeredStrategy | Carbonyl (R9) | stages/text.py | Medium |

---

## Shelved / Research Patterns (from Orchestrator PATTERNS.md migration 2026-04-05)

These were moved from Orchestrator's PATTERNS.md to cvui scope:

- **V3 DetectionFeatureRetrieval** — FAISS vector cache for similar windows. Current exact cache sufficient. ⏸️ shelved
- **V6 Pix2EmbStage** — Position embedding → decoder → bbox/mask. Long-term research. ⏸️ shelved
- **V8 DOTSOCRStage** — 1.7B VLM, prompt_layout_only outputs bbox + category. Alternative VLMZoneStage backend. 📐 designed
- **V9 SyntheticDataGenerator** — Bin-packing synthetic UI pages for YOLO training. Long-term research. ⏸️ shelved
- **V14 TextFirstLayeredStrategy** — DOM text / Win32 control text → trust first; OCR only as fallback. 📐 designed

---

## Task 1: AdaptiveDownscaleStage (P11)

**Files:**
- Modify: `src/cvui/stages/preprocessing.py`
- Modify: `src/cvui/stages/__init__.py`
- Test: `tests/test_detection.py`

**What:** Replace fixed downscale ratio with resolution-target-driven scaling. Instead of "scale to 75%", specify "target width=800" and compute scale factor dynamically.

- [ ] **Step 1: Write failing test**

```python
class TestAdaptiveDownscaleStage:
    def test_downscale_to_target(self):
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ctx = AdaptiveDownscaleStage(target_width=960).process(DetectionContext(img=img))
        assert ctx.width == 960
        assert ctx.height == 540
        assert ctx.scale == 0.5

    def test_no_upscale(self):
        """Small images should not be upscaled."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        ctx = AdaptiveDownscaleStage(target_width=960).process(DetectionContext(img=img))
        assert ctx.width == 640  # unchanged
        assert ctx.scale == 1.0

    def test_max_scale(self):
        """Scale should not exceed max_scale."""
        img = np.zeros((4320, 7680, 3), dtype=np.uint8)  # 8K
        ctx = AdaptiveDownscaleStage(target_width=960, max_scale=0.25).process(DetectionContext(img=img))
        assert ctx.scale >= 0.25
```

- [ ] **Step 2: Run test, verify fail**

```bash
cd D:/Users/Administrator/Documents/GitHub/cvui
python -m pytest tests/test_detection.py::TestAdaptiveDownscaleStage -v
```

- [ ] **Step 3: Implement AdaptiveDownscaleStage**

Add to `src/cvui/stages/preprocessing.py`:

```python
class AdaptiveDownscaleStage(DetectionStage):
    """Resolution-target-driven downscaling.

    Instead of a fixed ratio, specify a target width (or height).
    Scale factor is computed dynamically. Never upscales.

    Args:
        target_width: desired output width in pixels (default 960)
        target_height: if set, picks the smaller scale factor
        min_scale: floor for scale factor (default 0.1)
        max_scale: ceiling for scale factor (default 1.0, no upscale)
    """

    def __init__(
        self,
        target_width: int = 960,
        target_height: int | None = None,
        min_scale: float = 0.1,
        max_scale: float = 1.0,
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.min_scale = min_scale
        self.max_scale = max_scale

    def process(self, ctx: DetectionContext) -> DetectionContext:
        import cv2

        h, w = ctx.img.shape[:2]
        scale_w = self.target_width / w

        if self.target_height:
            scale_h = self.target_height / h
            scale = min(scale_w, scale_h)
        else:
            scale = scale_w

        # Clamp to [min_scale, max_scale]
        scale = max(self.min_scale, min(self.max_scale, scale))

        if scale >= 1.0:
            ctx.scale = 1.0
            return ctx

        new_w = int(w * scale)
        new_h = int(h * scale)
        ctx.img = cv2.resize(ctx.img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ctx.scale = scale
        return ctx
```

- [ ] **Step 4: Run test, verify pass**

```bash
python -m pytest tests/test_detection.py::TestAdaptiveDownscaleStage -v
```

- [ ] **Step 5: Export from `stages/__init__.py`**

Add to imports:
```python
from .preprocessing import AdaptiveDownscaleStage
```
Add to `__all__`.

- [ ] **Step 6: Commit**

```bash
git add src/cvui/stages/preprocessing.py src/cvui/stages/__init__.py tests/test_detection.py
git commit -m "feat: AdaptiveDownscaleStage — resolution-target-driven scaling (P11)"
```

---

## Task 2: VLMZoneStage (V1)

**Files:**
- Modify: `src/cvui/stages/ml.py`
- Modify: `src/cvui/stages/__init__.py`
- Test: `tests/test_detection.py`

**What:** Send screenshot to a VLM (Gemini, local model, etc.) asking "which rectangular regions contain UI elements?" The VLM returns bbox zones. Downstream CV stages only run inside these zones, reducing noise.

- [ ] **Step 1: Write failing test**

```python
class TestVLMZoneStage:
    def test_skip_when_unavailable(self):
        """Should passthrough when no VLM server is available."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        ctx = VLMZoneStage(server_url="http://localhost:99999").process(DetectionContext(img=img))
        assert ctx.zones == []  # passthrough, no crash

    def test_zones_from_mock(self):
        """With a mock VLM response, should populate ctx.zones."""
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        stage = VLMZoneStage()
        # Mock the VLM call
        stage._call_vlm = lambda img: [
            {"bbox": [100, 100, 500, 500], "label": "toolbar"},
            {"bbox": [100, 600, 900, 900], "label": "content"},
        ]
        ctx = stage.process(DetectionContext(img=img))
        assert len(ctx.zones) == 2
```

- [ ] **Step 2: Run test, verify fail**

- [ ] **Step 3: Implement VLMZoneStage**

```python
class VLMZoneStage(DetectionStage):
    """Use a VLM to identify UI regions before CV processing.

    Sends screenshot to VLM with prompt: "Identify rectangular UI regions."
    VLM returns bounding boxes. These become ctx.zones for downstream stages.

    Supports: Gemini API, local Ollama vision models, OmniParser server.
    Gracefully skips if no VLM is available.

    Gemini bbox format: [y_min, x_min, y_max, x_max] normalized 0-1000.
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:11434",
        model: str = "gemma3:4b",
        prompt: str = "Identify all rectangular UI regions in this screenshot. "
                      "Return as JSON array of {bbox: [x1, y1, x2, y2], label: string}.",
        timeout: float = 10.0,
    ):
        self.server_url = server_url
        self.model = model
        self.prompt = prompt
        self.timeout = timeout
        self._available: bool | None = None

    def process(self, ctx):
        try:
            zones = self._call_vlm(ctx.img)
            for z in zones:
                bbox = z.get("bbox", [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    ctx.zones.append((x1, y1, x2, y2))
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"VLMZoneStage: skipped ({e})")
        return ctx

    def _call_vlm(self, img):
        """Call VLM API. Override in tests."""
        import base64
        import json
        import cv2

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])
        b64 = base64.b64encode(buf).decode()

        try:
            import httpx
        except ImportError:
            return []

        resp = httpx.post(
            f"{self.server_url}/api/generate",
            json={
                "model": self.model,
                "prompt": self.prompt,
                "images": [b64],
                "stream": False,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "")

        # Parse JSON from response
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []
```

- [ ] **Step 4: Run test, verify pass**
- [ ] **Step 5: Export + commit**

```bash
git commit -m "feat: VLMZoneStage — VLM-guided UI region detection (V1)"
```

---

## Task 3: CNNClassifyStage (V2)

**Files:**
- Modify: `src/cvui/stages/ml.py`
- Test: `tests/test_detection.py`

**What:** Replace heuristic ClassifyStage with a lightweight CNN (MobileNetV3-Small, ~2M params) for element type classification: button, text, slider, icon, checkbox, dropdown.

- [ ] **Step 1: Write failing test**

```python
class TestCNNClassifyStage:
    def test_skip_without_model(self):
        """Should fall back to heuristic when model unavailable."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        ctx = DetectionContext(img=img)
        ctx.rects = [(10, 10, 50, 30), (60, 10, 190, 90)]
        ctx = CNNClassifyStage(model_path="nonexistent").process(ctx)
        # Should still have classifications (heuristic fallback)
        assert len(ctx.classifications) == 2

    def test_classifies_rects(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        ctx = DetectionContext(img=img)
        ctx.rects = [(10, 10, 50, 30)]
        stage = CNNClassifyStage()
        stage._classify_crop = lambda crop: ("button", 0.9)  # mock
        ctx = stage.process(ctx)
        assert ctx.classifications[0] == "button"
```

- [ ] **Step 2: Implement CNNClassifyStage**

```python
class CNNClassifyStage(DetectionStage):
    """Classify detected elements using CNN or heuristic fallback.

    Uses MobileNetV3-Small if available, falls back to aspect-ratio heuristics.
    Categories: button, text, icon, checkbox, slider, dropdown, container, unknown.

    Args:
        model_path: path to ONNX model file. Empty = heuristic only.
        confidence_threshold: minimum confidence to accept CNN prediction.
    """

    CATEGORIES = ["button", "text", "icon", "checkbox", "slider", "dropdown", "container", "unknown"]

    def __init__(self, model_path: str = "", confidence_threshold: float = 0.6):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self._session = None

    def process(self, ctx):
        for i, rect in enumerate(ctx.rects):
            x1, y1, x2, y2 = rect
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                ctx.classifications[i] = "unknown"
                continue

            crop = ctx.img[y1:y2, x1:x2]
            label, conf = self._classify_crop(crop)

            if conf >= self.confidence_threshold:
                ctx.classifications[i] = label
            else:
                ctx.classifications[i] = self._heuristic_classify(w, h)

        return ctx

    def _classify_crop(self, crop):
        """Classify a cropped element. Override for testing."""
        if self._session is None and self.model_path:
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(self.model_path)
            except Exception:
                pass

        if self._session:
            import cv2
            import numpy as np
            resized = cv2.resize(crop, (64, 64))
            blob = resized.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
            outputs = self._session.run(None, {"input": blob})
            probs = outputs[0][0]
            idx = int(np.argmax(probs))
            return self.CATEGORIES[min(idx, len(self.CATEGORIES)-1)], float(probs[idx])

        # No model — use heuristic
        h, w = crop.shape[:2]
        return self._heuristic_classify(w, h), 0.5

    @staticmethod
    def _heuristic_classify(w, h):
        aspect = w / max(h, 1)
        area = w * h
        if area < 400:
            return "icon"
        if aspect > 4:
            return "text"
        if 0.8 < aspect < 1.2 and area < 2000:
            return "checkbox"
        if aspect > 2.5:
            return "button"
        return "container"
```

- [ ] **Step 3: Run test, verify pass**
- [ ] **Step 4: Export + commit**

```bash
git commit -m "feat: CNNClassifyStage — CNN + heuristic element classification (V2)"
```

---

## Task 4: TilingStage (V7)

**Files:**
- Modify: `src/cvui/stages/advanced.py`
- Test: `tests/test_detection.py`

**What:** Split large screenshots into NxN tiles, run a sub-pipeline per tile, merge results back with coordinate offset + deduplication.

- [ ] **Step 1: Write failing test**

```python
class TestTilingStage:
    def test_small_image_passthrough(self):
        """Images below tile threshold should pass through unchanged."""
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        ctx = TilingStage(tile_size=512).process(DetectionContext(img=img))
        assert ctx.rects == []  # no tiling needed

    def test_tiles_large_image(self):
        """Large image should be tiled and rects offset-corrected."""
        img = np.zeros((2000, 3000, 3), dtype=np.uint8)
        # Draw some rectangles in different quadrants
        img[100:200, 100:300] = 255
        img[1500:1600, 2500:2700] = 255

        from cvui.stages import fast_pipeline
        stage = TilingStage(tile_size=1024, sub_pipeline=fast_pipeline(0.5))
        ctx = stage.process(DetectionContext(img=img))
        # Should find rects in different tiles, coordinates mapped to full image
        assert len(ctx.rects) >= 2
```

- [ ] **Step 2: Implement TilingStage**

```python
class TilingStage(DetectionStage):
    """Split large images into tiles, process each, merge results.

    For images larger than tile_size, split into NxN grid with overlap.
    Run sub_pipeline on each tile. Merge rects with coordinate offset.
    Deduplicate overlapping rects from tile boundaries.

    Args:
        tile_size: max tile dimension in pixels (default 1024)
        overlap: pixel overlap between tiles (default 64, for boundary elements)
        sub_pipeline: pipeline to run per tile (default: fast_pipeline)
        merge_iou: IoU threshold for deduplication (default 0.5)
    """

    def __init__(
        self,
        tile_size: int = 1024,
        overlap: int = 64,
        sub_pipeline=None,
        merge_iou: float = 0.5,
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.sub_pipeline = sub_pipeline
        self.merge_iou = merge_iou

    def process(self, ctx):
        h, w = ctx.img.shape[:2]

        # Skip tiling for small images
        if h <= self.tile_size and w <= self.tile_size:
            return ctx

        # Generate tile coordinates
        tiles = self._generate_tiles(w, h)

        # Run sub-pipeline on each tile
        pipeline = self.sub_pipeline
        if pipeline is None:
            from cvui.stages import fast_pipeline
            pipeline = fast_pipeline(1.0)

        all_rects = []
        for tx, ty, tw, th in tiles:
            tile_img = ctx.img[ty:ty+th, tx:tx+tw]
            tile_ctx = DetectionContext(img=tile_img)
            tile_ctx = pipeline.run_ctx(tile_ctx) if hasattr(pipeline, 'run_ctx') else pipeline.run(tile_img)

            # Offset rects to full-image coordinates
            for rx1, ry1, rx2, ry2 in tile_ctx.rects:
                all_rects.append((rx1 + tx, ry1 + ty, rx2 + tx, ry2 + ty))

        # Deduplicate overlapping rects
        ctx.rects = self._nms(all_rects, self.merge_iou)
        return ctx

    def _generate_tiles(self, w, h):
        """Generate tile coordinates with overlap."""
        tiles = []
        step = self.tile_size - self.overlap
        for y in range(0, h, step):
            for x in range(0, w, step):
                tw = min(self.tile_size, w - x)
                th = min(self.tile_size, h - y)
                tiles.append((x, y, tw, th))
        return tiles

    @staticmethod
    def _nms(rects, iou_threshold):
        """Non-maximum suppression for deduplication."""
        if not rects:
            return []

        import numpy as np
        boxes = np.array(rects)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = areas.argsort()[::-1]  # sort by area descending

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / np.maximum(union, 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return [rects[i] for i in keep]
```

- [ ] **Step 3: Run test, verify pass**
- [ ] **Step 4: Export + commit**

```bash
git commit -m "feat: TilingStage — NxN tile split + per-tile pipeline + NMS merge (V7)"
```

---

## Task 5: FormatConverterStage (V4)

**Files:**
- Modify: `src/cvui/stages/analysis.py` (or new `stages/export.py`)
- Test: `tests/test_detection.py`

**What:** Export detection results to standard annotation formats: COCO, YOLO, LabelMe. Makes cvui usable as a pre-annotation generator for training data.

- [ ] **Step 1: Write failing test**

```python
class TestFormatConverterStage:
    def test_to_coco(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        ctx = DetectionContext(img=img)
        ctx.rects = [(10, 10, 50, 30), (60, 60, 150, 90)]
        ctx.classifications = {0: "button", 1: "text"}

        stage = FormatConverterStage(format="coco")
        ctx = stage.process(ctx)
        coco = ctx.ui_states["export_coco"]
        assert "annotations" in coco
        assert len(coco["annotations"]) == 2

    def test_to_yolo(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        ctx = DetectionContext(img=img)
        ctx.rects = [(10, 10, 50, 30)]
        ctx.classifications = {0: "icon"}

        stage = FormatConverterStage(format="yolo")
        ctx = stage.process(ctx)
        yolo = ctx.ui_states["export_yolo"]
        assert len(yolo) == 1
        # YOLO format: "class_id cx cy w h" (normalized)
        assert yolo[0].startswith("2 ")  # icon = index 2
```

- [ ] **Step 2: Implement FormatConverterStage**

```python
class FormatConverterStage(DetectionStage):
    """Export detection results to standard annotation formats.

    Stores exported data in ctx.ui_states["export_{format}"].

    Supported formats:
    - coco: COCO JSON format (annotations list)
    - yolo: YOLO txt format (class_id cx cy w h, normalized)
    - labelme: LabelMe JSON format (shapes list)
    """

    CATEGORY_MAP = {
        "button": 0, "text": 1, "icon": 2, "checkbox": 3,
        "slider": 4, "dropdown": 5, "container": 6, "unknown": 7,
    }

    def __init__(self, format: str = "coco"):
        self.format = format.lower()

    def process(self, ctx):
        if self.format == "coco":
            ctx.ui_states["export_coco"] = self._to_coco(ctx)
        elif self.format == "yolo":
            ctx.ui_states["export_yolo"] = self._to_yolo(ctx)
        elif self.format == "labelme":
            ctx.ui_states["export_labelme"] = self._to_labelme(ctx)
        return ctx

    def _to_coco(self, ctx):
        h, w = ctx.height, ctx.width
        annotations = []
        for i, (x1, y1, x2, y2) in enumerate(ctx.rects):
            label = ctx.classifications.get(i, "unknown")
            annotations.append({
                "id": i,
                "category_id": self.CATEGORY_MAP.get(label, 7),
                "category_name": label,
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO: [x, y, w, h]
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            })
        return {
            "images": [{"id": 0, "width": w, "height": h}],
            "annotations": annotations,
            "categories": [{"id": v, "name": k} for k, v in self.CATEGORY_MAP.items()],
        }

    def _to_yolo(self, ctx):
        h, w = ctx.height, ctx.width
        lines = []
        for i, (x1, y1, x2, y2) in enumerate(ctx.rects):
            label = ctx.classifications.get(i, "unknown")
            cat_id = self.CATEGORY_MAP.get(label, 7)
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return lines

    def _to_labelme(self, ctx):
        shapes = []
        for i, (x1, y1, x2, y2) in enumerate(ctx.rects):
            label = ctx.classifications.get(i, "unknown")
            shapes.append({
                "label": label,
                "points": [[x1, y1], [x2, y2]],
                "shape_type": "rectangle",
            })
        return {
            "version": "5.0.0",
            "shapes": shapes,
            "imageHeight": ctx.height,
            "imageWidth": ctx.width,
        }
```

- [ ] **Step 3: Run test, verify pass**
- [ ] **Step 4: Export + commit**

```bash
git commit -m "feat: FormatConverterStage — export to COCO/YOLO/LabelMe (V4)"
```

---

## Task 6: PreAnnotationExporter (V5)

**Files:**
- Create: `src/cvui/stages/export.py`
- Test: `tests/test_detection.py`

**What:** Run detection pipeline → export as annotation file → ready for human correction on a labeling platform. This is the "flywheel" — AI annotate → human correct → retrain.

- [ ] **Step 1: Write test**

```python
class TestPreAnnotationExporter:
    def test_export_to_file(self, tmp_path):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        ctx = DetectionContext(img=img)
        ctx.rects = [(10, 10, 50, 30)]
        ctx.classifications = {0: "button"}

        output = tmp_path / "annotations.json"
        stage = PreAnnotationExporter(output_path=str(output), format="coco")
        ctx = stage.process(ctx)

        assert output.exists()
        import json
        data = json.loads(output.read_text())
        assert len(data["annotations"]) == 1
```

- [ ] **Step 2: Implement PreAnnotationExporter**

```python
class PreAnnotationExporter(DetectionStage):
    """Export detection results as pre-annotation files for human review.

    Writes annotation files that can be loaded into labeling tools
    (CVAT, labelU, Label Studio) for human correction.

    Part of the Detection → Correction → Training flywheel.
    """

    def __init__(
        self,
        output_path: str = "",
        format: str = "coco",
        include_image: bool = False,
        confidence_threshold: float = 0.3,
    ):
        self.output_path = output_path
        self.format = format
        self.include_image = include_image
        self.confidence_threshold = confidence_threshold

    def process(self, ctx):
        # Use FormatConverterStage internally
        converter = FormatConverterStage(format=self.format)
        ctx = converter.process(ctx)

        key = f"export_{self.format}"
        data = ctx.ui_states.get(key, {})

        if self.output_path and data:
            import json
            from pathlib import Path
            path = Path(self.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            if self.format == "yolo":
                path.write_text("\n".join(data))
            else:
                path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

        return ctx
```

- [ ] **Step 3: Run test, verify pass**
- [ ] **Step 4: Export + commit**

```bash
git commit -m "feat: PreAnnotationExporter — detection→annotation flywheel (V5)"
```

---

## Final: Update preset pipelines

- [ ] **Step 1: Add new preset to `stages/__init__.py`**

```python
def vlm_pipeline(server_url="http://127.0.0.1:11434", model="gemma3:4b"):
    """VLM-guided pipeline: zone detection → per-zone standard pipeline."""
    return DetectionPipeline([
        VLMZoneStage(server_url=server_url, model=model),
        GrayscaleStage(),
        OtsuStage(),
        ConnectedComponentStage(),
        RectFilterStage(),
        MergeStage(),
        CNNClassifyStage(),
    ])
```

- [ ] **Step 2: Commit**

```bash
git commit -m "feat: add vlm_pipeline preset + export all new stages"
```

---

## Verification

After all tasks:
- [ ] All 6 new stages importable: `from cvui.stages import AdaptiveDownscaleStage, VLMZoneStage, CNNClassifyStage, TilingStage, FormatConverterStage, PreAnnotationExporter`
- [ ] All tests pass: `python -m pytest tests/ -v`
- [ ] `vlm_pipeline()` preset available
- [ ] Update orchestrator's `docs/architecture/PATTERNS.md`: V1, V2, V4, V5, V7, P11 → ✅
