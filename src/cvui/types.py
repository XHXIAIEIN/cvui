"""Shared data models for cvui."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OCRWord:
    """A single word detected by OCR with bounding box."""
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float
    line_num: int
    word_num: int


@dataclass
class UIElement:
    """A UI element — button, input, label, icon, etc."""
    name: str
    rect: tuple[int, int, int, int]  # (left, top, right, bottom)
    element_type: str
    action: str
    text: str
    source: str
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        return (self.rect[0] + self.rect[2]) // 2, (self.rect[1] + self.rect[3]) // 2

    @property
    def width(self) -> int:
        return self.rect[2] - self.rect[0]

    @property
    def height(self) -> int:
        return self.rect[3] - self.rect[1]


@dataclass
class UIZone:
    """A rectangular region — static (skeleton) or dynamic."""
    name: str
    rect: tuple[int, int, int, int]
    zone_type: str
    dynamic: bool

    @property
    def width(self) -> int:
        return self.rect[2] - self.rect[0]

    @property
    def height(self) -> int:
        return self.rect[3] - self.rect[1]


@dataclass
class UIBlueprint:
    """Structural analysis of a window."""
    window_class: str
    window_size: tuple[int, int]
    elements: list[UIElement] = field(default_factory=list)
    zones: list[UIZone] = field(default_factory=list)
    perception_layers: list[str] = field(default_factory=list)
    created_at: float = 0.0

    def find(self, text: str) -> UIElement | None:
        for e in self.elements:
            if e.text == text:
                return e
        for e in self.elements:
            if text in e.text or e.text in text:
                return e
        return None

    def zone(self, name: str) -> UIZone | None:
        for z in self.zones:
            if z.name == name:
                return z
        return None

    def visualize(self, screenshot, mode="skeleton", save_path=""):
        from cvui.visualize import render_skeleton, render_annotated, render_grayscale
        if mode == "skeleton":
            result = render_skeleton(screenshot, self)
        elif mode == "annotated":
            result = render_annotated(screenshot)
        elif mode == "grayscale":
            result = render_grayscale(screenshot)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
        if save_path:
            result.save(save_path)
        return result


@dataclass
class MonitorInfo:
    id: int
    x_offset: int
    y_offset: int
    width: int
    height: int
    width_logical: int
    height_logical: int
    scale_factor: int


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    pid: int
    class_name: str
    rect: tuple[int, int, int, int]

    @property
    def width(self) -> int:
        return self.rect[2] - self.rect[0]

    @property
    def height(self) -> int:
        return self.rect[3] - self.rect[1]
