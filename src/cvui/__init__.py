"""cvui — Pure CV UI element detection library.

Pluggable detection pipeline with 20+ stages:
- GrayscaleStage, TopHatStage, OtsuStage, DilateStage
- ConnectedComponentStage, RectFilterStage, MergeStage
- NestedStage, ClassifyStage, ChannelAnalysisStage, DiffStage
- ListQuantizeStage, EnsembleStage
- OmniParserStage, GroundingDINOStage
- MultiFrameAccumulatorStage, ColorQuantizeStage, MultiColorSpaceStage
- GradientDetectorStage, TrackingStage

Preset pipelines: fast_pipeline, standard_pipeline, full_pipeline,
                   ensemble_pipeline, game_pipeline
"""

__version__ = "0.1.0"

from .types import (
    OCRWord, UIElement, UIZone, UIBlueprint,
    MonitorInfo, WindowInfo,
)
from .detection import (
    DetectionContext, DetectionStage, DetectionPipeline,
    DownscaleStage, GrayscaleStage, TopHatStage, OtsuStage,
    DilateStage, ConnectedComponentStage, RectFilterStage, MergeStage,
    NestedStage, ClassifyStage, ChannelAnalysisStage, DiffStage,
    ListQuantizeStage, LayoutPatternStage, OmniParserStage, GroundingDINOStage,
    fast_pipeline, standard_pipeline, full_pipeline, grounding_pipeline,
)
from .stages.advanced import (
    MultiFrameAccumulatorStage, ColorQuantizeStage,
    MultiColorSpaceStage, GradientDetectorStage, TrackingStage,
    SaturationFilterStage, ZoneDetectorStage,
)
from .stages.ensemble import EnsembleStage
from .stages import ensemble_pipeline, game_pipeline
from .visualize import (
    render_skeleton, render_annotated, render_grayscale, detect_elements,
)
