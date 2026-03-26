"""Windows.Graphics.Capture — background window capture via WinRT.

Uses the windows-capture library (pip install windows-capture).
Supports DirectX/Vulkan games, background capture without focus.
Requires Python 3.12 (3.14 has compatibility issues with torch/winsdk).

Falls back to PrintWindow if windows-capture is not installed.
"""
from __future__ import annotations

import logging
import os
import tempfile

log = logging.getLogger(__name__)

_WINDOWS_CAPTURE_AVAILABLE = False
try:
    from windows_capture import WindowsCapture, Frame, InternalCaptureControl
    _WINDOWS_CAPTURE_AVAILABLE = True
except ImportError:
    pass


def capture_window_wgc(window_name: str, timeout: float = 5.0) -> bytes | None:
    """Capture a window using Windows.Graphics.Capture API.

    Args:
        window_name: exact or partial window title
        timeout: max seconds to wait for a frame

    Returns:
        PNG bytes, or None if capture failed
    """
    if not _WINDOWS_CAPTURE_AVAILABLE:
        log.debug("wgc_capture: windows-capture not installed")
        return None

    tmp = os.path.join(tempfile.gettempdir(), "cvui_wgc_frame.png")
    result = [False]

    try:
        capture = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            window_name=window_name,
        )

        @capture.event
        def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
            frame.save_as_image(tmp)
            result[0] = True
            capture_control.stop()

        @capture.event
        def on_closed():
            pass

        capture.start()

    except Exception as e:
        log.warning("wgc_capture failed: %s", e)
        return None

    if result[0] and os.path.exists(tmp):
        with open(tmp, "rb") as f:
            return f.read()
    return None
