"""Tests for blueprint visualization."""
import io
import pytest
from PIL import Image

from cvui.types import UIBlueprint, UIElement, UIZone, OCRWord
from cvui.visualize import (
    render_skeleton, render_annotated, render_grayscale, detect_elements,
)


def _make_image(w=800, h=600):
    return Image.new("RGB", (w, h), (40, 40, 40))


def _make_png(w=800, h=600):
    img = _make_image(w, h)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_ui_png(w=800, h=600):
    """Create a synthetic UI image with visible elements."""
    import numpy as np
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    img[50:80, 20:50] = (180, 180, 180)
    img[100:130, 20:50] = (180, 180, 180)
    img[50:70, 100:250] = (200, 200, 200)
    img[75:90, 100:200] = (120, 120, 120)
    img[120:140, 100:250] = (200, 200, 200)
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


class TestRenderSkeleton:
    def test_returns_image(self):
        bp = UIBlueprint("Test", (800, 600),
                         zones=[UIZone("z", (0, 0, 400, 600), "panel", False)])
        result = render_skeleton(_make_image(), bp)
        assert isinstance(result, Image.Image)
        assert result.size == (800, 600)

    def test_accepts_bytes(self):
        bp = UIBlueprint("Test", (800, 600),
                         zones=[UIZone("z", (0, 0, 400, 600), "panel", False)])
        result = render_skeleton(_make_png(), bp)
        assert isinstance(result, Image.Image)

    def test_multiple_zones(self):
        bp = UIBlueprint("Test", (800, 600), zones=[
            UIZone("a", (0, 0, 200, 600), "panel", False),
            UIZone("b", (200, 0, 800, 300), "content", True),
            UIZone("c", (200, 300, 800, 600), "panel", False),
        ])
        result = render_skeleton(_make_image(), bp)
        assert result.size == (800, 600)


class TestDetectElements:
    def test_returns_list(self):
        result = detect_elements(_make_png())
        assert isinstance(result, list)

    def test_detects_visible_elements(self):
        png = _make_ui_png()
        rects = detect_elements(png)
        assert len(rects) > 0

    def test_rects_are_tuples(self):
        png = _make_ui_png()
        rects = detect_elements(png)
        for r in rects:
            assert len(r) == 4


class TestRenderAnnotated:
    def test_returns_image(self):
        result = render_annotated(_make_png())
        assert isinstance(result, Image.Image)

    def test_accepts_pil_image(self):
        result = render_annotated(_make_image())
        assert isinstance(result, Image.Image)

    def test_with_precomputed_rects(self):
        rects = [(10, 10, 100, 50), (200, 200, 300, 250)]
        result = render_annotated(_make_image(), element_rects=rects)
        assert result.size == (800, 600)

    def test_auto_detects_when_no_rects(self):
        result = render_annotated(_make_ui_png())
        assert isinstance(result, Image.Image)


class TestRenderGrayscale:
    def test_returns_grayscale_image(self):
        result = render_grayscale(_make_png())
        assert isinstance(result, Image.Image)
        assert result.mode == "L"

    def test_accepts_pil_image(self):
        result = render_grayscale(_make_image())
        assert isinstance(result, Image.Image)


class TestBlueprintVisualize:
    def test_skeleton_mode(self):
        bp = UIBlueprint("T", (800, 600),
                         zones=[UIZone("z", (0, 0, 800, 600), "panel", False)])
        result = bp.visualize(_make_image(), mode="skeleton")
        assert isinstance(result, Image.Image)

    def test_annotated_mode(self):
        bp = UIBlueprint("T", (800, 600))
        result = bp.visualize(_make_png(), mode="annotated")
        assert isinstance(result, Image.Image)

    def test_grayscale_mode(self):
        bp = UIBlueprint("T", (800, 600))
        result = bp.visualize(_make_png(), mode="grayscale")
        assert isinstance(result, Image.Image)

    def test_save_path(self, tmp_path):
        bp = UIBlueprint("T", (100, 100),
                         zones=[UIZone("z", (0, 0, 100, 100), "panel", False)])
        out = str(tmp_path / "test.png")
        result = bp.visualize(_make_image(100, 100), mode="skeleton", save_path=out)
        assert result is not None
        loaded = Image.open(out)
        assert loaded.size == (100, 100)

    def test_invalid_mode_raises(self):
        bp = UIBlueprint("T", (100, 100))
        with pytest.raises(ValueError, match="Unknown mode"):
            bp.visualize(_make_image(100, 100), mode="invalid")
