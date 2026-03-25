"""Backward compatibility — re-exports everything from pipeline + stages.

This module exists so that existing code importing from cvui.detection
continues to work without modification.
"""
from cvui.pipeline import DetectionContext, DetectionStage, DetectionPipeline
from cvui.stages import (
    DownscaleStage, GrayscaleStage, TopHatStage, OtsuStage,
    DilateStage, ConnectedComponentStage, RectFilterStage, MergeStage,
    NestedStage, ClassifyStage, ChannelAnalysisStage, DiffStage,
    ListQuantizeStage, OmniParserStage, GroundingDINOStage,
    fast_pipeline, standard_pipeline, full_pipeline, grounding_pipeline,
)

__all__ = [
    "DetectionContext", "DetectionStage", "DetectionPipeline",
    "DownscaleStage", "GrayscaleStage", "TopHatStage", "OtsuStage",
    "DilateStage", "ConnectedComponentStage", "RectFilterStage", "MergeStage",
    "NestedStage", "ClassifyStage", "ChannelAnalysisStage", "DiffStage",
    "ListQuantizeStage", "OmniParserStage", "GroundingDINOStage",
    "fast_pipeline", "standard_pipeline", "full_pipeline", "grounding_pipeline",
]
