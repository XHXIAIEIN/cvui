"""Detection stages and preset pipelines."""
from .preprocessing import DownscaleStage, GrayscaleStage, TopHatStage, OtsuStage
from .morphology import DilateStage, ConnectedComponentStage, RectFilterStage, MergeStage
from .analysis import NestedStage, ClassifyStage, ChannelAnalysisStage, DiffStage, ListQuantizeStage, LayoutPatternStage
from .ml import OmniParserStage, GroundingDINOStage
from .advanced import (
    MultiFrameAccumulatorStage, ColorQuantizeStage,
    MultiColorSpaceStage, GradientDetectorStage, TrackingStage,
    SaturationFilterStage, ZoneDetectorStage,
)
from cvui.pipeline import DetectionPipeline, DetectionStage


# ---------------------------------------------------------------------------
# Preset pipelines
# ---------------------------------------------------------------------------

def fast_pipeline(scale: float = 0.75) -> DetectionPipeline:
    """Downscale -> Grayscale -> TopHat -> Otsu -> Dilate -> CC. ~17ms at 0.75x."""
    stages = []
    if scale < 1.0:
        stages.append(DownscaleStage(scale=scale))
    stages.extend([
        GrayscaleStage(), TopHatStage(), OtsuStage(),
        DilateStage(), ConnectedComponentStage(),
    ])
    return DetectionPipeline(stages)


def standard_pipeline(scale: float = 1.0) -> DetectionPipeline:
    """Fast + RectFilter + Merge. Full resolution by default. ~33ms."""
    stages = []
    if scale < 1.0:
        stages.append(DownscaleStage(scale=scale))
    stages.extend([
        GrayscaleStage(), TopHatStage(), OtsuStage(),
        DilateStage(), ConnectedComponentStage(),
        RectFilterStage(), MergeStage(),
    ])
    return DetectionPipeline(stages)


def full_pipeline(omniparser_path: str = "", grounding_query: str = "") -> DetectionPipeline:
    """Standard + Nested + Classify + ChannelAnalysis + optional ML stages. ~20ms+."""
    stages: list[DetectionStage] = [
        GrayscaleStage(), TopHatStage(), OtsuStage(),
        DilateStage(), ConnectedComponentStage(),
        RectFilterStage(), MergeStage(),
        NestedStage(), ClassifyStage(), LayoutPatternStage(), ChannelAnalysisStage(),
    ]
    if omniparser_path:
        stages.append(OmniParserStage(model_path=omniparser_path))
    if grounding_query:
        stages.append(GroundingDINOStage(query=grounding_query))
    return DetectionPipeline(stages)


def grounding_pipeline(query: str, box_threshold: float = 0.3) -> DetectionPipeline:
    """Single-purpose: find one element by text description."""
    return DetectionPipeline([GroundingDINOStage(query=query, box_threshold=box_threshold)])


def game_pipeline() -> DetectionPipeline:
    """For games: saturation filter → large dilate → strict filter.

    Games have noisy backgrounds. Uses:
    - SaturationFilter to mask out colorful scene regions
    - Large dilate kernels to merge UI text blocks into panels
    - High min_area filter to remove scene debris
    """
    return DetectionPipeline([
        GrayscaleStage(),
        SaturationFilterStage(),
        TopHatStage(), OtsuStage(),
        DilateStage(h_kernel=(3, 20), v_kernel=(10, 3)),  # medium-large for game UI
        ConnectedComponentStage(),
        RectFilterStage(min_area=800),  # filter small scene debris
        MergeStage(),
        ClassifyStage(), LayoutPatternStage(),
    ])


__all__ = [
    "DownscaleStage", "GrayscaleStage", "TopHatStage", "OtsuStage",
    "DilateStage", "ConnectedComponentStage", "RectFilterStage", "MergeStage",
    "NestedStage", "ClassifyStage", "ChannelAnalysisStage", "DiffStage", "ListQuantizeStage", "LayoutPatternStage",
    "OmniParserStage", "GroundingDINOStage",
    "DetectionPipeline", "DetectionStage",
    "MultiFrameAccumulatorStage", "ColorQuantizeStage",
    "MultiColorSpaceStage", "GradientDetectorStage", "TrackingStage",
    "SaturationFilterStage", "ZoneDetectorStage",
    "fast_pipeline", "standard_pipeline", "full_pipeline", "grounding_pipeline",
    "game_pipeline",
]
