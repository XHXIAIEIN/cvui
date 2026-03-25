# cvui

Pure CV UI element detection from screenshots. No ML models, no GPU, no training data — just OpenCV.

Detect buttons, icons, text regions, list items, and UI zones from any desktop application screenshot in **4-30ms**.

## What it does

Given a screenshot (PNG bytes or numpy array), cvui detects UI elements and returns bounding boxes:

```python
from cvui.stages import standard_pipeline

pipeline = standard_pipeline()
ctx = pipeline.run(img)  # numpy BGR array

for rect in ctx.rects:
    x1, y1, x2, y2 = rect
    print(f"Element at ({x1},{y1}) size {x2-x1}x{y2-y1}")
```

## Why this exists

Existing UI detection tools require ML models (OmniParser needs YOLO + Florence, ScreenAI needs a 5B VLM). They're accurate but slow (500ms+) and need GPU.

cvui takes a different approach: **pure computer vision**. Adaptive TopHat/BlackHat separates foreground from background regardless of theme, Otsu auto-thresholds with zero parameters, morphological operations group elements into components. The result: UI element detection in **4ms** on CPU.

This is designed as the **fast first pass** in a multi-layer perception stack:

```
Layer -1: DOM / Accessibility Tree  (browser, 0ms, perfect)
Layer  0: Win32 UI Automation       (standard apps, 0ms, precise)
Layer  1: cvui                      (self-drawing apps, 4-30ms, good)  <-- this
Layer  2: OCR                       (text supplement, 100ms)
Layer  3: ML models                 (semantic understanding, 500ms+)
```

Most apps have structural data (DOM or accessibility tree) and don't need cvui at all. cvui is the fallback for **self-drawing applications** that expose no control tree: WeChat (Qt), games (DirectX), some Electron apps, native rendering engines.

## Performance

Benchmarked on WeChat (1021x1453 screenshot, Windows 11):

| Pipeline | Time | Elements | Description |
|----------|------|----------|-------------|
| `fast_pipeline()` | **4ms** | ~70 | Bounding boxes only |
| `fast_pipeline(scale=0.75)` | **17ms** | ~52 | Downscaled, 2x faster |
| `standard_pipeline()` | **30ms** | ~55 | + filter + merge |
| `full_pipeline()` | **40ms** | ~60 | + nested + classify + color analysis |

Zero parameters to tune. Works on both dark and light themes automatically.

## Installation

```bash
pip install cvui
```

Or from source:

```bash
git clone https://github.com/anthropics/cvui.git
cd cvui
pip install -e .
```

### Optional dependencies

```bash
pip install cvui[ocr]      # Windows OCR (WinRT) — text extraction
pip install cvui[ml]       # OmniParser + Grounding DINO stages
pip install cvui[window]   # Win32 window capture + input simulation
```

## Quick start

### Detect elements from a screenshot file

```python
import cv2
from cvui.stages import standard_pipeline

img = cv2.imread("screenshot.png")
ctx = standard_pipeline().run(img)

print(f"Found {len(ctx.rects)} elements")
for x1, y1, x2, y2 in ctx.rects:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite("annotated.png", img)
```

### Capture and analyze a live window (Windows)

```python
from cvui.window import Win32WindowManager
from cvui.stages import standard_pipeline

wm = Win32WindowManager()
wm.lock(title_contains="WeChat")
png = wm.capture_window()

import cv2, numpy as np
img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
ctx = standard_pipeline().run(img)
print(f"Detected {len(ctx.rects)} UI elements")
```

### Visualize results

```python
from cvui.visualize import render_annotated, render_skeleton, render_grayscale

# Green bounding boxes on every detected element
annotated = render_annotated(png_bytes, mode="standard")
annotated.save("annotated.png")

# Skeleton: colored zone overlay showing major UI regions
from cvui.types import UIBlueprint, UIZone
bp = UIBlueprint("MyApp", (1024, 768), zones=[...])
skeleton = render_skeleton(png_bytes, bp)

# Grayscale: the preprocessed image used for detection (debug)
gray = render_grayscale(png_bytes)
```

### Detect list items by template propagation

```python
from cvui.stages.analysis import ListQuantizeStage
from cvui.pipeline import DetectionContext

ctx = DetectionContext(img=img)
# zone_rect = the list area (x1, y1, x2, y2)
ctx = ListQuantizeStage(zone_rect=(100, 90, 460, 1400)).process(ctx)
# Finds highlighted item → uses its height as step → expands up/down until empty
print(f"List has {len(ctx.rects)} items")
```

### Build a custom pipeline

```python
from cvui.pipeline import DetectionPipeline
from cvui.stages import *

# Only the stages you need
pipeline = DetectionPipeline([
    DownscaleStage(scale=0.5),
    GrayscaleStage(),
    TopHatStage(kernel_size=60),
    OtsuStage(),
    ConnectedComponentStage(min_w=20, min_h=20),
])

ctx = pipeline.run(img)
# Pipeline stops early if ConnectedComponentStage.quality_score > 0.8
```

## Architecture

### Pluggable pipeline

Every processing step is a `DetectionStage`. Stages chain in a `DetectionPipeline`. A shared `DetectionContext` flows through, accumulating results. Any stage can stop the pipeline early via `should_continue()`.

```
DetectionContext (img, gray, binary, rects, quality_score, ...)
     │
     ▼
┌─ GrayscaleStage ─┐  auto gamma from median brightness
├─ TopHatStage ─────┤  dark theme → TopHat, light theme → BlackHat (auto)
├─ OtsuStage ───────┤  zero-parameter threshold
├─ DilateStage ─────┤  connect icon + text on same row
├─ ConnectedComp ───┤  bounding boxes + quality_score → early exit if good enough
├─ RectFilterStage ─┤  remove edge artifacts, strip shapes
├─ MergeStage ──────┤  merge overlapping boxes
├─ NestedStage ─────┤  recurse into large containers for child elements
├─ ClassifyStage ───┤  icon / text / image / container by aspect ratio
├─ ChannelAnalysis ─┤  G-R = selected, R-B = badge, B-R = link
├─ DiffStage ───────┤  frame comparison, skip if unchanged
├─ ListQuantize ────┤  template propagation for list items
├─ OmniParserStage ─┤  optional: YOLO icon detection
└─ GroundingDINO ───┘  optional: text-guided zero-shot detection
```

### File structure

```
src/cvui/
├── pipeline.py              # DetectionContext, DetectionStage ABC, DetectionPipeline
├── stages/
│   ├── preprocessing.py     # Downscale, Grayscale, TopHat, Otsu
│   ├── morphology.py        # Dilate, ConnectedComponent, RectFilter, Merge
│   ├── analysis.py          # Nested, Classify, ChannelAnalysis, Diff, ListQuantize
│   └── ml.py                # OmniParser, GroundingDINO (optional deps)
├── visualize.py             # render_skeleton, render_annotated, render_grayscale
├── types.py                 # UIElement, UIZone, UIBlueprint, OCRWord
├── ocr.py                   # OCREngine ABC + WinOCREngine (Windows native)
├── window/                  # Win32 window management
│   ├── discovery.py         # find_windows, focus, is_alive
│   ├── capture.py           # PrintWindow screenshot
│   ├── input.py             # send_text, send_click, send_hotkey
│   └── _vk_map.py           # virtual key codes
└── screen.py                # Multi-monitor DPI-aware screenshot (mss)
```

### Key design decisions

**TopHat + BlackHat auto-selection**: Median brightness < 128 → TopHat (extract bright elements from dark bg). >= 128 → BlackHat (extract dark elements from light bg). High-saturation regions (green highlights, colored badges) get the opposite transform locally. Works on any theme without configuration.

**Quality-based early exit**: `ConnectedComponentStage` computes a quality score (coverage x fragmentation). If score > 0.8 after the basic stages, the pipeline skips Nested, Classify, and other expensive stages.

**List item template propagation**: `ListQuantizeStage` finds a reference item (highlighted selection or spacing pattern), uses its height as step, and expands up/down stopping at the first empty slot. 3ms for a 6-item list — no wasted checks on blank areas.

**Downscale for speed**: `DownscaleStage(scale=0.75)` cuts processing time in half (33ms → 17ms) while losing only ~2 elements. Coordinates auto-map back to original resolution.

## Preset pipelines

| Preset | Stages | Speed | Use case |
|--------|--------|-------|----------|
| `fast_pipeline()` | 6 (with 0.75x downscale) | ~17ms | Real-time, boxes only |
| `standard_pipeline()` | 7 | ~30ms | General purpose |
| `full_pipeline()` | 10+ | ~40ms | Full analysis + optional ML |
| `grounding_pipeline("search box")` | 1 | ~2s | Find one element by description |

## Supported applications

| Application type | Recommended approach | cvui needed? |
|-----------------|---------------------|-------------|
| Browser pages | Chrome DevTools Protocol → DOM | No |
| WPF / WinForms / MFC | UI Automation tree | No |
| Standard Win32 controls | EnumChildWindows | No |
| **WeChat (Qt self-draw)** | **cvui** | **Yes** |
| **Games (DirectX)** | **cvui + multi-frame accumulation** | **Yes** |
| **Custom rendering engines** | **cvui** | **Yes** |
| Complex unknown UI | cvui + OmniParser | Yes (with ML) |

## Writing custom stages

```python
from cvui.pipeline import DetectionStage, DetectionContext

class MyStage(DetectionStage):
    def process(self, ctx: DetectionContext) -> DetectionContext:
        # Access ctx.img (BGR numpy), ctx.gray, ctx.binary, ctx.rects
        # Modify and return ctx
        return ctx

    def should_continue(self, ctx: DetectionContext) -> bool:
        # Return False to stop the pipeline after this stage
        return True
```

## Research references

This project draws from:

- **UIED** (ESEC/FSE 2020) — Top-down coarse-to-fine GUI element detection, hybrid CV+DL outperforms pure DL
- **REMAUI** (ASE 2015) — CV edge detection + OCR fusion for UI hierarchy reconstruction
- **OmniParser** (Microsoft) — YOLO + Florence for icon detection and semantic description
- **Grounding DINO** (ECCV 2023) — Zero-shot text-guided object detection, COCO 52.5 AP
- **Rico** (UIST 2017) — 66K mobile UI screenshots, 25 component categories
- **ScreenAI** (Google 2024) — DETR-based UI element annotation

## License

MIT
