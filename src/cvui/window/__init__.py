"""Window management interface + Win32 default implementation.

Two operating modes:
- Foreground: focus the window, then use pyautogui
- Background: use Win32 PostMessage / SendMessage to interact without focusing
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from ..types import WindowInfo

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class WindowManager(ABC):
    """Pluggable window management backend."""

    @abstractmethod
    def find_windows(self, title_contains: str = "", process_name: str = "",
                     class_name: str = "") -> list[WindowInfo]:
        """Find all visible windows matching the given criteria."""

    @abstractmethod
    def lock(self, title_contains: str = "", process_name: str = "",
             hwnd: int = 0) -> WindowInfo | None:
        """Lock onto a target window. Returns WindowInfo or None if not found."""

    @abstractmethod
    def unlock(self) -> None:
        """Release window lock."""

    @abstractmethod
    def focus(self) -> bool:
        """Bring target window to foreground."""

    @abstractmethod
    def capture_window(self) -> bytes | None:
        """Capture the target window as PNG bytes."""

    @abstractmethod
    def send_text(self, text: str) -> bool:
        """Send text to target window (background, no focus required)."""

    @abstractmethod
    def send_click(self, x: int, y: int, button: str = "left") -> bool:
        """Send a click at window-local coords (background, no focus required)."""

    @abstractmethod
    def send_hotkey(self, *keys: str) -> bool:
        """Send a hotkey combination (background, no focus required)."""

    @abstractmethod
    def is_alive(self) -> bool:
        """Check if the target window still exists."""

    @property
    @abstractmethod
    def target(self) -> WindowInfo | None:
        """Currently locked target window."""


# ---------------------------------------------------------------------------
# Default implementation: Win32 API via ctypes
# ---------------------------------------------------------------------------

class Win32WindowManager(WindowManager):
    """Find and interact with specific windows using Win32 API."""

    def __init__(self):
        self._target: WindowInfo | None = None

    @property
    def target(self) -> WindowInfo | None:
        return self._target

    # ------------------------------------------------------------------
    # Window discovery
    # ------------------------------------------------------------------

    def find_windows(self, title_contains: str = "", process_name: str = "",
                     class_name: str = "") -> list[WindowInfo]:
        from .discovery import find_windows
        return find_windows(title_contains=title_contains,
                            process_name=process_name,
                            class_name=class_name)

    def lock(self, title_contains: str = "", process_name: str = "",
             hwnd: int = 0) -> WindowInfo | None:
        from .discovery import info_from_hwnd
        if hwnd:
            self._target = info_from_hwnd(hwnd)
            if self._target:
                log.info("Win32WindowManager: locked onto hwnd=%d '%s'",
                         hwnd, self._target.title)
            return self._target

        wins = self.find_windows(title_contains=title_contains,
                                 process_name=process_name)
        if not wins:
            log.warning("Win32WindowManager: no window found "
                        "(title='%s', process='%s')", title_contains, process_name)
            return None

        self._target = wins[0]
        log.info("Win32WindowManager: locked onto '%s' (hwnd=%d, pid=%d)",
                 self._target.title, self._target.hwnd, self._target.pid)
        return self._target

    def unlock(self) -> None:
        self._target = None

    def refresh(self) -> WindowInfo | None:
        """Refresh the target window's rect (in case it moved/resized)."""
        if not self._target:
            return None
        from .discovery import info_from_hwnd
        self._target = info_from_hwnd(self._target.hwnd)
        return self._target

    def is_alive(self) -> bool:
        if not self._target:
            return False
        from .discovery import is_window_alive
        return is_window_alive(self._target.hwnd)

    # ------------------------------------------------------------------
    # Foreground control
    # ------------------------------------------------------------------

    def focus(self) -> bool:
        if not self._target:
            return False
        from .discovery import focus_window
        return focus_window(self._target.hwnd)

    # ------------------------------------------------------------------
    # Window-specific screenshot (works even if window is behind others)
    # ------------------------------------------------------------------

    def capture_window(self) -> bytes | None:
        """Capture the target window using Win32 PrintWindow.

        Works even if the window is behind others. If minimized, restores
        without stealing focus (saves/restores previous foreground window).
        """
        if not self._target:
            return None

        import ctypes
        import ctypes.wintypes
        import time
        user32 = ctypes.windll.user32

        hwnd = self._target.hwnd

        # If minimized, restore without stealing focus
        if user32.IsIconic(hwnd):
            prev_fg = user32.GetForegroundWindow()
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            time.sleep(0.15)
            # Restore previous foreground window
            if prev_fg and prev_fg != hwnd:
                user32.SetForegroundWindow(prev_fg)

        self.refresh()
        if not self._target:
            return None

        from .capture import capture_window_png
        return capture_window_png(self._target.hwnd,
                                  self._target.width,
                                  self._target.height)

    def capture_window_wgc(self) -> bytes | None:
        """Capture using Windows.Graphics.Capture (background, DirectX-compatible).

        Falls back to PrintWindow if windows-capture library not available.
        """
        if not self._target:
            return None
        from .wgc_capture import capture_window_wgc
        png = capture_window_wgc(self._target.title)
        if png:
            return png
        # Fallback to PrintWindow
        return self.capture_window()

    # ------------------------------------------------------------------
    # Background input (no foreground focus needed)
    # ------------------------------------------------------------------

    def send_text(self, text: str) -> bool:
        """Send text via clipboard + WM_PASTE. No focus required."""
        if not self._target:
            return False
        from .input import send_text_to
        return send_text_to(self._target.hwnd, text)

    def send_click(self, x: int, y: int, button: str = "left") -> bool:
        """Send a click at window-local coords via PostMessage. No focus required."""
        if not self._target:
            return False
        from .input import send_click_to
        return send_click_to(self._target.hwnd, x, y, button)

    def send_hotkey(self, *keys: str) -> bool:
        """Send a hotkey combination via PostMessage. No focus required."""
        if not self._target:
            return False
        from .input import send_hotkey_to
        return send_hotkey_to(self._target.hwnd, *keys)


__all__ = ["WindowManager", "Win32WindowManager"]
