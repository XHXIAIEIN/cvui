"""Tests for pluggable detection pipeline."""
import numpy as np
import pytest

from cvui.detection import (
    DetectionContext,
    DetectionStage,
    DetectionPipeline,
    GrayscaleStage,
    TopHatStage,
    OtsuStage,
    DilateStage,
    ConnectedComponentStage,
    RectFilterStage,
    MergeStage,
)


# ---------------------------------------------------------------------------
# Framework tests
# ---------------------------------------------------------------------------

class TestDetectionContext:
    def test_create_from_image(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        ctx = DetectionContext(img=img)
        assert ctx.rects == []
        assert ctx.quality_score == 0.0
        assert ctx.stage_log == []

    def test_dimensions(self):
        ctx = DetectionContext(img=np.zeros((600, 800, 3), dtype=np.uint8))
        assert ctx.height == 600
        assert ctx.width == 800


class TestDetectionStage:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            DetectionStage()


class TestDetectionPipeline:
    def test_runs_in_order(self):
        class A(DetectionStage):
            def process(self, ctx):
                ctx.rects.append((0, 0, 10, 10))
                return ctx

        class B(DetectionStage):
            def process(self, ctx):
                ctx.rects.append((20, 20, 30, 30))
                return ctx

        ctx = DetectionPipeline([A(), B()]).run(np.zeros((100, 100, 3), dtype=np.uint8))
        assert len(ctx.rects) == 2
        assert len(ctx.stage_log) == 2

    def test_early_exit(self):
        class StopEarly(DetectionStage):
            def process(self, ctx):
                ctx.quality_score = 0.9
                return ctx
            def should_continue(self, ctx):
                return ctx.quality_score < 0.8

        class NeverReached(DetectionStage):
            def process(self, ctx):
                ctx.rects.append((99, 99, 99, 99))
                return ctx

        ctx = DetectionPipeline([StopEarly(), NeverReached()]).run(
            np.zeros((100, 100, 3), dtype=np.uint8))
        assert len(ctx.rects) == 0
        assert len(ctx.stage_log) == 1

    def test_empty_pipeline(self):
        ctx = DetectionPipeline([]).run(np.zeros((50, 50, 3), dtype=np.uint8))
        assert ctx.rects == []


# ---------------------------------------------------------------------------
# Stage tests
# ---------------------------------------------------------------------------

class TestGrayscaleStage:
    def test_produces_gray(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (40, 80, 120)
        ctx = GrayscaleStage().process(DetectionContext(img=img))
        assert ctx.gray is not None
        assert ctx.gray.shape == (100, 100)

    def test_dark_image_brightened(self):
        img = np.full((100, 100, 3), 30, dtype=np.uint8)
        ctx = GrayscaleStage().process(DetectionContext(img=img))
        assert np.median(ctx.gray) > 30

    def test_light_image_not_overexposed(self):
        img = np.full((100, 100, 3), 220, dtype=np.uint8)
        ctx = GrayscaleStage().process(DetectionContext(img=img))
        assert np.median(ctx.gray) < 255


class TestTopHatStage:
    def test_extracts_bright_elements(self):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        img[50:80, 50:150] = (180, 180, 180)
        ctx = DetectionContext(img=img)
        ctx = GrayscaleStage().process(ctx)
        ctx = TopHatStage().process(ctx)
        # TopHat result should have high values where bright element is
        assert ctx.gray[65, 100] > ctx.gray[10, 10]

    def test_needs_gray(self):
        """Auto-generates gray if missing."""
        img = np.full((50, 50, 3), 100, dtype=np.uint8)
        ctx = TopHatStage().process(DetectionContext(img=img))
        assert ctx.gray is not None


class TestOtsuStage:
    def test_produces_binary(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        img[30:60, 30:80] = (180, 180, 180)
        ctx = DetectionContext(img=img)
        ctx = GrayscaleStage().process(ctx)
        ctx = TopHatStage().process(ctx)
        ctx = OtsuStage().process(ctx)
        assert ctx.binary is not None
        assert set(np.unique(ctx.binary)).issubset({0, 255})

    def test_skips_without_gray(self):
        ctx = OtsuStage().process(DetectionContext(img=np.zeros((10, 10, 3), dtype=np.uint8)))
        assert ctx.binary is None


class TestDilateStage:
    def test_connects_horizontal_gap(self):
        binary = np.zeros((100, 200), dtype=np.uint8)
        binary[40:50, 20:40] = 255
        binary[40:50, 50:70] = 255  # 10px gap
        ctx = DetectionContext(img=np.zeros((100, 200, 3), dtype=np.uint8))
        ctx.binary = binary
        ctx = DilateStage(h_kernel=(2, 15), v_kernel=(7, 2)).process(ctx)
        assert ctx.binary[45, 45] == 255

    def test_skips_without_binary(self):
        ctx = DilateStage().process(
            DetectionContext(img=np.zeros((10, 10, 3), dtype=np.uint8)))
        assert ctx.binary is None


class TestConnectedComponentStage:
    def test_produces_rects(self):
        binary = np.zeros((200, 300), dtype=np.uint8)
        binary[20:60, 20:100] = 255
        binary[100:130, 150:250] = 255
        ctx = DetectionContext(img=np.zeros((200, 300, 3), dtype=np.uint8))
        ctx.binary = binary
        ctx = ConnectedComponentStage().process(ctx)
        assert len(ctx.rects) == 2

    def test_filters_tiny(self):
        binary = np.zeros((200, 300), dtype=np.uint8)
        binary[10:13, 10:13] = 255  # too small
        binary[50:90, 50:150] = 255
        ctx = DetectionContext(img=np.zeros((200, 300, 3), dtype=np.uint8))
        ctx.binary = binary
        ctx = ConnectedComponentStage(min_w=15, min_h=10).process(ctx)
        assert len(ctx.rects) == 1

    def test_computes_quality(self):
        binary = np.zeros((200, 300), dtype=np.uint8)
        binary[20:60, 20:100] = 255
        ctx = DetectionContext(img=np.zeros((200, 300, 3), dtype=np.uint8))
        ctx.binary = binary
        ctx = ConnectedComponentStage().process(ctx)
        assert ctx.quality_score > 0

    def test_should_continue_when_low_quality(self):
        stage = ConnectedComponentStage()
        ctx = DetectionContext(img=np.zeros((10, 10, 3), dtype=np.uint8))
        ctx.quality_score = 0.3
        assert stage.should_continue(ctx) is True

    def test_should_stop_when_high_quality(self):
        stage = ConnectedComponentStage()
        ctx = DetectionContext(img=np.zeros((10, 10, 3), dtype=np.uint8))
        ctx.quality_score = 0.9
        assert stage.should_continue(ctx) is False


class TestRectFilterStage:
    def test_removes_small(self):
        ctx = DetectionContext(img=np.zeros((200, 300, 3), dtype=np.uint8))
        ctx.rects = [(10, 10, 100, 50), (200, 200, 205, 203)]  # big + tiny
        ctx = RectFilterStage(min_area=500).process(ctx)
        assert len(ctx.rects) == 1

    def test_keeps_large(self):
        ctx = DetectionContext(img=np.zeros((200, 300, 3), dtype=np.uint8))
        ctx.rects = [(0, 0, 100, 100)]
        ctx = RectFilterStage(min_area=200).process(ctx)
        assert len(ctx.rects) == 1


class TestMergeStage:
    def test_merges_significantly_overlapping(self):
        """Two rects with >30% overlap → merge."""
        ctx = DetectionContext(img=np.zeros((200, 300, 3), dtype=np.uint8))
        # 40x40 and 40x40 with 30x30 overlap = 56% → merge
        ctx.rects = [(10, 10, 50, 50), (20, 20, 60, 60)]
        ctx = MergeStage().process(ctx)
        assert len(ctx.rects) == 1

    def test_keeps_slightly_overlapping(self):
        """Two rects with <30% overlap → keep separate."""
        ctx = DetectionContext(img=np.zeros((200, 300, 3), dtype=np.uint8))
        # 40x40 and 40x40 with 20x20 overlap = 25% → no merge
        ctx.rects = [(10, 10, 50, 50), (30, 30, 70, 70)]
        ctx = MergeStage().process(ctx)
        assert len(ctx.rects) == 2

    def test_keeps_separate(self):
        ctx = DetectionContext(img=np.zeros((200, 300, 3), dtype=np.uint8))
        ctx.rects = [(10, 10, 50, 50), (200, 200, 250, 250)]
        ctx = MergeStage().process(ctx)
        assert len(ctx.rects) == 2

    def test_chain_merge(self):
        """Significantly overlapping chain: A→B→C all merge."""
        ctx = DetectionContext(img=np.zeros((200, 300, 3), dtype=np.uint8))
        # Each 30x30 with 20x20 overlap = 44% → merge chain
        ctx.rects = [(0, 0, 30, 30), (10, 10, 40, 40), (20, 20, 50, 50)]
        ctx = MergeStage().process(ctx)
        assert len(ctx.rects) == 1

    def test_chain_no_merge_small_overlap(self):
        """Slightly overlapping chain: A→B→C stay separate."""
        ctx = DetectionContext(img=np.zeros((200, 300, 3), dtype=np.uint8))
        ctx.rects = [(0, 0, 30, 30), (20, 20, 50, 50), (40, 40, 70, 70)]
        ctx = MergeStage().process(ctx)
        assert len(ctx.rects) == 3

    def test_empty(self):
        ctx = DetectionContext(img=np.zeros((10, 10, 3), dtype=np.uint8))
        ctx.rects = []
        ctx = MergeStage().process(ctx)
        assert ctx.rects == []


# ---------------------------------------------------------------------------
# Integration: mini pipeline
# ---------------------------------------------------------------------------

class TestMiniPipeline:
    def _make_ui_image(self):
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        img[30:60, 30:80] = (180, 180, 180)
        img[30:55, 100:250] = (200, 200, 200)
        img[100:130, 30:80] = (160, 160, 160)
        img[100:120, 100:200] = (180, 180, 180)
        return img

    def test_fast_pipeline(self):
        pipeline = DetectionPipeline([
            GrayscaleStage(), TopHatStage(), OtsuStage(),
            DilateStage(), ConnectedComponentStage(),
        ])
        ctx = pipeline.run(self._make_ui_image())
        assert len(ctx.rects) >= 2
        assert len(ctx.stage_log) <= 5

    def test_standard_pipeline(self):
        pipeline = DetectionPipeline([
            GrayscaleStage(), TopHatStage(), OtsuStage(),
            DilateStage(), ConnectedComponentStage(),
            RectFilterStage(), MergeStage(),
        ])
        ctx = pipeline.run(self._make_ui_image())
        assert len(ctx.rects) >= 1


from cvui.detection import (
    NestedStage, ClassifyStage, ChannelAnalysisStage, DiffStage,
)


class TestNestedStage:
    def test_detects_children(self):
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)
        img[50:350, 50:550] = (60, 60, 60)
        img[80:110, 80:200] = (180, 180, 180)
        img[150:190, 80:120] = (160, 160, 160)
        ctx = DetectionContext(img=img)
        ctx.rects = [(50, 50, 550, 350)]
        ctx = NestedStage().process(ctx)
        assert len(ctx.rects) > 1

    def test_skips_small_rects(self):
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        ctx = DetectionContext(img=img)
        ctx.rects = [(10, 10, 30, 30)]  # too small for nesting
        original_count = len(ctx.rects)
        ctx = NestedStage().process(ctx)
        assert len(ctx.rects) == original_count


class TestClassifyStage:
    def test_icon(self):
        ctx = DetectionContext(img=np.zeros((600, 800, 3), dtype=np.uint8))
        ctx.rects = [(10, 10, 40, 40)]  # 30x30 square
        ctx = ClassifyStage().process(ctx)
        assert ctx.classifications[0] == "icon"

    def test_text(self):
        ctx = DetectionContext(img=np.zeros((600, 800, 3), dtype=np.uint8))
        ctx.rects = [(10, 10, 210, 30)]  # 200x20 wide
        ctx = ClassifyStage().process(ctx)
        assert ctx.classifications[0] == "text"

    def test_image(self):
        ctx = DetectionContext(img=np.zeros((600, 800, 3), dtype=np.uint8))
        ctx.rects = [(0, 0, 500, 400)]  # large
        ctx = ClassifyStage().process(ctx)
        assert ctx.classifications[0] == "image"

    def test_container(self):
        ctx = DetectionContext(img=np.zeros((600, 800, 3), dtype=np.uint8))
        ctx.rects = [(10, 10, 200, 100)]  # medium, not square
        ctx = ClassifyStage().process(ctx)
        assert ctx.classifications[0] in ("container", "element")


class TestChannelAnalysisStage:
    def test_detects_green_highlight(self):
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        img[20:50, 20:200, 1] = 150  # G channel high (BGR: index 1 = G)
        ctx = ChannelAnalysisStage().process(DetectionContext(img=img))
        assert len(ctx.ui_states.get("highlight", [])) >= 1

    def test_detects_red_badge(self):
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        img[10:25, 350:370, 2] = 200  # R channel high (BGR: index 2 = R)
        ctx = ChannelAnalysisStage().process(DetectionContext(img=img))
        assert len(ctx.ui_states.get("badge", [])) >= 1

    def test_no_false_positives_on_gray(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        ctx = ChannelAnalysisStage().process(DetectionContext(img=img))
        assert len(ctx.ui_states.get("highlight", [])) == 0
        assert len(ctx.ui_states.get("badge", [])) == 0


class TestDiffStage:
    def test_detects_change(self):
        prev = np.full((200, 300, 3), 40, dtype=np.uint8)
        curr = prev.copy()
        curr[100:150, 100:200] = 180
        ctx = DiffStage(prev_img=prev).process(DetectionContext(img=curr))
        assert len(ctx.rects) >= 1

    def test_no_change(self):
        img = np.full((200, 300, 3), 40, dtype=np.uint8)
        stage = DiffStage(prev_img=img.copy())
        ctx = stage.process(DetectionContext(img=img))
        assert len(ctx.rects) == 0
        assert stage.should_continue(ctx) is False

    def test_no_prev_continues(self):
        stage = DiffStage(prev_img=None)
        ctx = stage.process(DetectionContext(img=np.zeros((100, 100, 3), dtype=np.uint8)))
        assert stage.should_continue(ctx) is True

    def test_different_shape_continues(self):
        prev = np.zeros((100, 100, 3), dtype=np.uint8)
        curr = np.zeros((200, 200, 3), dtype=np.uint8)
        stage = DiffStage(prev_img=prev)
        ctx = stage.process(DetectionContext(img=curr))
        assert stage.should_continue(ctx) is True


from cvui.detection import (
    OmniParserStage, GroundingDINOStage, DownscaleStage,
    fast_pipeline, standard_pipeline, full_pipeline, grounding_pipeline,
)


class TestOmniParserStage:
    def test_skips_when_not_available(self):
        ctx = DetectionContext(img=np.zeros((100, 100, 3), dtype=np.uint8))
        ctx.rects = [(10, 10, 50, 50)]
        stage = OmniParserStage(model_path="/nonexistent", server_url="http://127.0.0.1:99999")
        ctx = stage.process(ctx)
        assert len(ctx.rects) >= 1  # original preserved

    def test_is_detection_stage(self):
        assert issubclass(OmniParserStage, DetectionStage)


class TestGroundingDINOStage:
    def test_skips_without_query(self):
        ctx = DetectionContext(img=np.zeros((100, 100, 3), dtype=np.uint8))
        ctx = GroundingDINOStage(query="").process(ctx)
        assert ctx.rects == []

    def test_skips_without_transformers(self):
        """Should not crash even if transformers is not installed."""
        ctx = DetectionContext(img=np.zeros((100, 100, 3), dtype=np.uint8))
        ctx = GroundingDINOStage(query="test element").process(ctx)
        # Either finds something or gracefully returns empty
        assert isinstance(ctx.rects, list)

    def test_is_detection_stage(self):
        assert issubclass(GroundingDINOStage, DetectionStage)


class TestPresetPipelines:
    def test_fast_pipeline(self):
        p = fast_pipeline()
        assert len(p.stages) == 6  # DownscaleStage + 5 core stages
        assert isinstance(p.stages[0], DownscaleStage)

    def test_fast_pipeline_no_scale(self):
        p = fast_pipeline(scale=1.0)
        assert len(p.stages) == 5  # no DownscaleStage

    def test_standard_pipeline(self):
        p = standard_pipeline()
        assert len(p.stages) == 7  # no downscale by default

    def test_standard_pipeline_with_scale(self):
        p = standard_pipeline(scale=0.75)
        assert len(p.stages) == 8
        assert isinstance(p.stages[0], DownscaleStage)

    def test_full_pipeline_default(self):
        p = full_pipeline()
        assert len(p.stages) == 11

    def test_full_pipeline_with_omniparser(self):
        p = full_pipeline(omniparser_path="/some/path")
        assert len(p.stages) == 12
        assert any(isinstance(s, OmniParserStage) for s in p.stages)

    def test_full_pipeline_with_grounding(self):
        p = full_pipeline(grounding_query="search box")
        assert len(p.stages) == 12
        assert any(isinstance(s, GroundingDINOStage) for s in p.stages)

    def test_grounding_pipeline(self):
        p = grounding_pipeline("button")
        assert len(p.stages) == 1
        assert isinstance(p.stages[0], GroundingDINOStage)


from cvui.detection import ListQuantizeStage


class TestListQuantizeStage:
    def test_finds_items_from_highlight(self):
        """6 equal-height items with one highlighted -> detect all 6."""
        img = np.full((600, 400, 3), 240, dtype=np.uint8)
        for i in range(6):
            y = 50 + i * 80
            img[y+10:y+30, 20:60] = (100, 100, 100)    # avatar
            img[y+10:y+25, 70:200] = (60, 60, 60)       # name
            img[y+30:y+42, 70:250] = (120, 120, 120)    # subtitle
        # Highlight item 2 (y=210-290) with green bg
        img[210:290, 0:400] = (50, 180, 80)
        img[220:240, 20:60] = (255, 255, 255)
        img[220:235, 70:200] = (255, 255, 255)

        ctx = DetectionContext(img=img)
        ctx = ListQuantizeStage(zone_rect=(0, 50, 400, 530)).process(ctx)
        assert len(ctx.rects) >= 5

    def test_stops_at_empty(self):
        """3 items then blank -> no rects in blank area."""
        img = np.full((600, 400, 3), 240, dtype=np.uint8)
        for i in range(3):
            y = 50 + i * 80
            img[y+10:y+30, 20:200] = (60, 60, 60)
            img[y+30:y+42, 20:250] = (120, 120, 120)
        # Highlight item 1
        img[130:210, 0:400] = (50, 180, 80)
        img[140:160, 20:200] = (255, 255, 255)

        ctx = DetectionContext(img=img)
        ctx = ListQuantizeStage(zone_rect=(0, 50, 400, 530)).process(ctx)
        # Should find 3, not extend into blank area
        assert len(ctx.rects) <= 4

    def test_no_highlight_uses_rects(self):
        """No highlight -> estimate from existing rects spacing."""
        img = np.full((400, 300, 3), 240, dtype=np.uint8)
        for i in range(4):
            y = 30 + i * 70
            img[y+5:y+20, 10:100] = (60, 60, 60)

        ctx = DetectionContext(img=img)
        # Pre-populate rects (as if ConnectedComponentStage ran first)
        ctx.rects = [
            (10, 35, 100, 50),
            (10, 105, 100, 120),
            (10, 175, 100, 190),
            (10, 245, 100, 260),
        ]
        ctx = ListQuantizeStage(zone_rect=(0, 30, 300, 310)).process(ctx)
        assert len(ctx.rects) >= 4  # original rects + list items

    def test_no_zone_returns_unchanged(self):
        ctx = DetectionContext(img=np.zeros((100, 100, 3), dtype=np.uint8))
        ctx.rects = [(10, 10, 50, 50)]
        ctx = ListQuantizeStage(zone_rect=None).process(ctx)
        assert len(ctx.rects) == 1

    def test_is_detection_stage(self):
        assert issubclass(ListQuantizeStage, DetectionStage)


class TestToPrompt:
    def test_basic_output(self):
        ctx = DetectionContext(img=np.zeros((600, 800, 3), dtype=np.uint8))
        ctx.rects = [(10, 10, 100, 50), (200, 200, 300, 250)]
        prompt = ctx.to_prompt()
        assert "800x600" in prompt
        assert "2 elements" in prompt

    def test_with_classifications(self):
        ctx = DetectionContext(img=np.zeros((600, 800, 3), dtype=np.uint8))
        ctx.rects = [(10, 10, 40, 40), (100, 100, 300, 120)]
        ctx.classifications = {0: "icon", 1: "text"}
        prompt = ctx.to_prompt()
        assert "[icon]" in prompt
        assert "[text]" in prompt

    def test_with_ocr(self):
        ctx = DetectionContext(img=np.zeros((600, 800, 3), dtype=np.uint8))
        ctx.rects = [(10, 10, 200, 50)]
        ocr = [(20, 15, 80, 40, "Hello"), (90, 15, 180, 40, "World")]
        prompt = ctx.to_prompt(ocr_lines=ocr)
        assert "HelloWorld" in prompt
        assert "OCR" in prompt

    def test_deduplication(self):
        """Parent with children inside should be removed."""
        ctx = DetectionContext(img=np.zeros((600, 800, 3), dtype=np.uint8))
        # Parent (0,0)-(400,300) contains two children
        ctx.rects = [(0, 0, 400, 300), (10, 10, 100, 50), (200, 200, 300, 250)]
        prompt = ctx.to_prompt()
        # Parent should be deduplicated, only 2 children remain
        assert "2 elements" in prompt

    def test_empty(self):
        ctx = DetectionContext(img=np.zeros((100, 100, 3), dtype=np.uint8))
        prompt = ctx.to_prompt()
        assert "0 elements" in prompt


class TestToReport:
    def test_returns_dict(self):
        ctx = DetectionContext(img=np.zeros((600, 800, 3), dtype=np.uint8))
        ctx.rects = [(10, 10, 100, 50)]
        report = ctx.to_report()
        assert isinstance(report, dict)
        assert report["window"]["width"] == 800
        assert report["window"]["height"] == 600

    def test_elements_in_report(self):
        ctx = DetectionContext(img=np.zeros((600, 800, 3), dtype=np.uint8))
        ctx.rects = [(10, 10, 100, 50), (200, 200, 300, 250)]
        report = ctx.to_report()
        # Deduplicated count
        assert len(report["elements"]) <= 2

    def test_theme_detection(self):
        # Dark image
        dark = DetectionContext(img=np.full((100, 100, 3), 30, dtype=np.uint8))
        assert dark.to_report()["window"]["theme"] == "dark"
        # Light image
        light = DetectionContext(img=np.full((100, 100, 3), 200, dtype=np.uint8))
        assert light.to_report()["window"]["theme"] == "light"


from cvui.stages.advanced import (
    MultiFrameAccumulatorStage, ColorQuantizeStage,
    MultiColorSpaceStage, GradientDetectorStage, TrackingStage,
)


class TestMultiFrameAccumulatorStage:
    def test_not_enough_frames(self):
        stage = MultiFrameAccumulatorStage(n_frames=5)
        ctx = DetectionContext(img=np.zeros((100, 100, 3), dtype=np.uint8))
        ctx = stage.process(ctx)
        assert ctx.binary is None  # not enough frames yet

    def test_static_regions_detected(self):
        stage = MultiFrameAccumulatorStage(n_frames=3, static_threshold=50)
        # Add 3 identical frames → everything is static
        frame = np.full((100, 200, 3), 128, dtype=np.uint8)
        for _ in range(3):
            stage.add_frame(frame)
        ctx = DetectionContext(img=frame)
        ctx = stage.process(ctx)
        assert ctx.binary is not None
        assert np.count_nonzero(ctx.binary) > 0  # static regions found

    def test_dynamic_regions_masked(self):
        stage = MultiFrameAccumulatorStage(n_frames=3, static_threshold=50)
        # Add frames with one changing region
        for i in range(3):
            frame = np.full((100, 200, 3), 128, dtype=np.uint8)
            frame[40:60, 40:60] = i * 80  # this region changes
            stage.add_frame(frame)
        ctx = DetectionContext(img=frame)
        ctx = stage.process(ctx)
        # The changing region should NOT be in static mask
        assert ctx.binary[50, 50] == 0


class TestColorQuantizeStage:
    def test_reduces_colors(self):
        img = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        ctx = DetectionContext(img=img)
        ctx = ColorQuantizeStage(n_colors=4).process(ctx)
        unique_colors = len(np.unique(ctx.img.reshape(-1, 3), axis=0))
        assert unique_colors <= 8  # some tolerance

    def test_palette_extracted(self):
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        ctx = DetectionContext(img=img)
        ctx = ColorQuantizeStage(n_colors=4).process(ctx)
        assert "palette" in ctx.ui_states

    def test_is_stage(self):
        assert issubclass(ColorQuantizeStage, DetectionStage)


class TestMultiColorSpaceStage:
    def test_produces_gray(self):
        ctx = DetectionContext(img=np.full((50, 50, 3), 128, dtype=np.uint8))
        ctx = MultiColorSpaceStage().process(ctx)
        assert ctx.gray is not None
        assert ctx.gray.shape == (50, 50)

    def test_saturation_map(self):
        ctx = DetectionContext(img=np.full((50, 50, 3), 128, dtype=np.uint8))
        ctx = MultiColorSpaceStage().process(ctx)
        assert "saturation_map" in ctx.ui_states


class TestGradientDetectorStage:
    def test_masks_high_variance(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:] = 128
        # Add noisy region (scene-like)
        img[20:80, 100:180] = np.random.randint(0, 256, (60, 80, 3), dtype=np.uint8)
        ctx = DetectionContext(img=img)
        ctx.binary = np.ones((100, 200), dtype=np.uint8) * 255
        ctx = GradientDetectorStage(scene_threshold=200).process(ctx)
        # Noisy region should be zeroed out in binary
        assert ctx.binary[50, 140] == 0

    def test_preserves_uniform(self):
        img = np.full((100, 200, 3), 128, dtype=np.uint8)
        ctx = DetectionContext(img=img)
        ctx.binary = np.ones((100, 200), dtype=np.uint8) * 255
        ctx = GradientDetectorStage().process(ctx)
        # Uniform image should keep binary intact
        assert np.count_nonzero(ctx.binary) > 0


class TestTrackingStage:
    def test_tracks_moved_element(self):
        # Use a dark background with a textured patch so TM_CCOEFF_NORMED works
        rng = np.random.default_rng(42)
        # Textured patch (high variance so correlation is discriminative)
        patch = rng.integers(100, 256, (30, 50, 3), dtype=np.uint8)
        prev = np.zeros((200, 300, 3), dtype=np.uint8)
        curr = np.zeros((200, 300, 3), dtype=np.uint8)
        prev[50:80, 50:100] = patch
        # Same patch shifted right by 10px
        curr[50:80, 60:110] = patch

        stage = TrackingStage(
            prev_rects=[(50, 50, 100, 80)],
            prev_img=prev,
            search_radius=30,
        )
        ctx = DetectionContext(img=curr)
        ctx = stage.process(ctx)
        assert len(ctx.rects) >= 1
        # Should find element near x=60
        found = ctx.rects[0]
        assert abs(found[0] - 60) < 15

    def test_no_prev_passthrough(self):
        stage = TrackingStage()
        ctx = DetectionContext(img=np.zeros((100, 100, 3), dtype=np.uint8))
        ctx = stage.process(ctx)
        assert ctx.rects == []

    def test_should_continue_when_no_tracks(self):
        stage = TrackingStage()
        ctx = DetectionContext(img=np.zeros((100, 100, 3), dtype=np.uint8))
        ctx = stage.process(ctx)
        assert stage.should_continue(ctx) is True

    def test_should_stop_when_tracked(self):
        prev = np.zeros((100, 100, 3), dtype=np.uint8)
        prev[30:60, 30:60] = 200
        stage = TrackingStage(prev_rects=[(30, 30, 60, 60)], prev_img=prev)
        ctx = DetectionContext(img=prev.copy())
        ctx = stage.process(ctx)
        assert stage.should_continue(ctx) is False  # found tracks, skip full detection


from cvui.stages.analysis import LayoutPatternStage


class TestLayoutPatternStage:
    def test_detects_header_sidebar_content(self):
        ctx = DetectionContext(img=np.zeros((600, 800, 3), dtype=np.uint8))
        # Header elements (top)
        ctx.rects = [
            (10, 10, 100, 40), (200, 10, 300, 40), (400, 10, 500, 40),  # header
            (10, 100, 150, 200), (10, 250, 150, 350), (10, 400, 150, 500),  # sidebar
            (200, 100, 700, 200), (200, 250, 700, 350),  # content
        ]
        ctx = LayoutPatternStage().process(ctx)
        assert ctx.ui_states["layout_pattern"] == "header+sidebar+content"

    def test_detects_single_column(self):
        ctx = DetectionContext(img=np.zeros((600, 800, 3), dtype=np.uint8))
        # Vertically stacked, narrow
        ctx.rects = [(300, y, 500, y+30) for y in range(50, 500, 50)]
        ctx = LayoutPatternStage().process(ctx)
        assert ctx.ui_states["layout_pattern"] == "single-column"

    def test_detects_grid(self):
        ctx = DetectionContext(img=np.zeros((600, 800, 3), dtype=np.uint8))
        # Grid of similar-sized elements
        ctx.rects = [(x, y, x+80, y+80) for x in range(50, 700, 100) for y in range(50, 500, 100)]
        ctx = LayoutPatternStage().process(ctx)
        assert ctx.ui_states["layout_pattern"] == "grid"

    def test_empty(self):
        ctx = DetectionContext(img=np.zeros((100, 100, 3), dtype=np.uint8))
        ctx = LayoutPatternStage().process(ctx)
        assert ctx.ui_states["layout_pattern"] == "empty"

    def test_is_stage(self):
        assert issubclass(LayoutPatternStage, DetectionStage)


# ---------------------------------------------------------------------------
# SaturationFilterStage tests
# ---------------------------------------------------------------------------

import cv2
from cvui.stages.advanced import SaturationFilterStage


class TestSaturationFilterStage:
    def test_masks_saturated_regions(self):
        """High-saturation pixels should be zeroed in gray."""
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        # Left: desaturated UI (gray)
        img[:, :200] = (128, 128, 128)
        # Right: saturated scene (bright green)
        img[:, 200:] = (0, 200, 0)
        ctx = DetectionContext(img=img)
        ctx.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ctx = SaturationFilterStage().process(ctx)
        # Right side (scene) should be zeroed
        assert ctx.gray[100, 300] == 0
        # Left side (UI) should be preserved
        assert ctx.gray[100, 100] > 0

    def test_preserves_ui_elements(self):
        """Low-saturation text on dark background should survive."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)  # dark bg
        img[30:50, 30:170] = (200, 200, 200)  # white text (low sat)
        ctx = DetectionContext(img=img)
        ctx = SaturationFilterStage().process(ctx)
        assert ctx.gray is not None
        assert ctx.gray[40, 100] > 0  # text preserved

    def test_stores_stats(self):
        ctx = DetectionContext(img=np.full((100, 100, 3), 128, dtype=np.uint8))
        ctx = SaturationFilterStage().process(ctx)
        assert "saturation_filter" in ctx.ui_states

    def test_is_stage(self):
        assert issubclass(SaturationFilterStage, DetectionStage)

    def test_writes_ui_mask_layer(self):
        """SaturationFilterStage should store ui_mask in ctx.layers."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:] = (128, 128, 128)
        ctx = DetectionContext(img=img)
        ctx = SaturationFilterStage().process(ctx)
        assert "ui_mask" in ctx.layers
        assert ctx.layers["ui_mask"].shape == (100, 200)


# ---------------------------------------------------------------------------
# ZoneDetectorStage tests
# ---------------------------------------------------------------------------

from cvui.stages.advanced import ZoneDetectorStage


class TestZoneDetectorStage:
    def test_finds_zones_from_ui_mask(self):
        """UI mask with two bright clusters -> 2 zones."""
        mask = np.zeros((600, 800), dtype=np.uint8)
        mask[50:200, 50:300] = 255   # left panel
        mask[50:200, 400:750] = 255  # right panel
        ctx = DetectionContext(img=np.zeros((600, 800, 3), dtype=np.uint8))
        ctx.layers["ui_mask"] = mask
        ctx = ZoneDetectorStage().process(ctx)
        assert len(ctx.zones) == 2

    def test_no_mask_passthrough(self):
        ctx = DetectionContext(img=np.zeros((100, 100, 3), dtype=np.uint8))
        ctx = ZoneDetectorStage().process(ctx)
        assert ctx.zones == []

    def test_is_stage(self):
        assert issubclass(ZoneDetectorStage, DetectionStage)


# ---------------------------------------------------------------------------
# Zone-aware ConnectedComponentStage tests
# ---------------------------------------------------------------------------


class TestZoneAwareConnectedComponent:
    def test_only_detects_in_zones(self):
        """Elements outside zones should be ignored."""
        binary = np.zeros((400, 600), dtype=np.uint8)
        binary[50:80, 50:150] = 255   # inside zone
        binary[300:330, 400:500] = 255  # outside zone
        ctx = DetectionContext(img=np.zeros((400, 600, 3), dtype=np.uint8))
        ctx.binary = binary
        ctx.zones = [(0, 0, 300, 200)]  # only left-top zone
        ctx = ConnectedComponentStage().process(ctx)
        # Should only find the element inside the zone
        assert len(ctx.rects) == 1
        assert ctx.rects[0][0] >= 0 and ctx.rects[0][2] <= 300

    def test_full_image_when_no_zones(self):
        """Without zones, detect in full image (backward compat)."""
        binary = np.zeros((400, 600), dtype=np.uint8)
        binary[50:80, 50:150] = 255
        binary[300:330, 400:500] = 255
        ctx = DetectionContext(img=np.zeros((400, 600, 3), dtype=np.uint8))
        ctx.binary = binary
        ctx = ConnectedComponentStage().process(ctx)
        assert len(ctx.rects) == 2
