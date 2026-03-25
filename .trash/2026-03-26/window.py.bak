"""Window management interface + Win32 default implementation.

Two operating modes:
- Foreground: focus the window, then use pyautogui
- Background: use Win32 PostMessage / SendMessage to interact without focusing
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import subprocess
import time
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Optional

from .types import WindowInfo

try:
    from PIL import Image
except ImportError:
    Image = None

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

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32


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
        results: list[WindowInfo] = []
        title_lower = title_contains.lower()
        proc_lower = process_name.lower()

        # Get PID->process name mapping if needed
        pid_map: dict[int, str] = {}
        if proc_lower:
            pid_map = self._build_pid_map()

        def enum_callback(hwnd, _lparam):
            if not user32.IsWindowVisible(hwnd):
                return True

            # Get title
            buf = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(hwnd, buf, 512)
            win_title = buf.value
            if not win_title:
                return True

            # Title filter
            if title_lower and title_lower not in win_title.lower():
                return True

            # Get PID
            pid = ctypes.wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            pid_val = pid.value

            # Process name filter
            if proc_lower:
                pname = pid_map.get(pid_val, "").lower()
                if proc_lower not in pname:
                    return True

            # Class name filter
            cls_buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, cls_buf, 256)
            cls = cls_buf.value
            if class_name and class_name.lower() not in cls.lower():
                return True

            # Get window rect
            rect = ctypes.wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))

            results.append(WindowInfo(
                hwnd=hwnd,
                title=win_title,
                pid=pid_val,
                class_name=cls,
                rect=(rect.left, rect.top, rect.right, rect.bottom),
            ))
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_ulong, ctypes.c_ulong)
        user32.EnumWindows(WNDENUMPROC(enum_callback), 0)
        return results

    def lock(self, title_contains: str = "", process_name: str = "",
             hwnd: int = 0) -> WindowInfo | None:
        if hwnd:
            self._target = self._info_from_hwnd(hwnd)
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
        self._target = self._info_from_hwnd(self._target.hwnd)
        return self._target

    def is_alive(self) -> bool:
        if not self._target:
            return False
        return bool(user32.IsWindow(self._target.hwnd))

    # ------------------------------------------------------------------
    # Foreground control
    # ------------------------------------------------------------------

    def focus(self) -> bool:
        if not self._target:
            return False
        hwnd = self._target.hwnd

        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            time.sleep(0.2)

        current_thread = ctypes.windll.kernel32.GetCurrentThreadId()
        target_thread = user32.GetWindowThreadProcessId(hwnd, None)

        if current_thread != target_thread:
            user32.AttachThreadInput(current_thread, target_thread, True)

        result = user32.SetForegroundWindow(hwnd)

        if current_thread != target_thread:
            user32.AttachThreadInput(current_thread, target_thread, False)

        if result:
            time.sleep(0.15)
        return bool(result)

    # ------------------------------------------------------------------
    # Window-specific screenshot (works even if window is behind others)
    # ------------------------------------------------------------------

    def capture_window(self) -> bytes | None:
        """Capture the target window using Win32 PrintWindow.
        Works even if the window is partially or fully occluded."""
        if not self._target or Image is None:
            return None

        hwnd = self._target.hwnd
        self.refresh()
        if not self._target:
            return None

        w = self._target.width
        h = self._target.height
        if w <= 0 or h <= 0:
            return None

        # Create compatible DC and bitmap
        hwnd_dc = user32.GetWindowDC(hwnd)
        mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
        old_bmp = gdi32.SelectObject(mem_dc, bitmap)

        # PrintWindow with PW_RENDERFULLCONTENT (flag 2) for modern apps
        PW_RENDERFULLCONTENT = 2
        user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT)

        # Read bitmap bits
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", ctypes.c_uint32),
                ("biWidth", ctypes.c_int32),
                ("biHeight", ctypes.c_int32),
                ("biPlanes", ctypes.c_uint16),
                ("biBitCount", ctypes.c_uint16),
                ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32),
                ("biXPelsPerMeter", ctypes.c_int32),
                ("biYPelsPerMeter", ctypes.c_int32),
                ("biClrUsed", ctypes.c_uint32),
                ("biClrImportant", ctypes.c_uint32),
            ]

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = w
        bmi.biHeight = -h  # top-down
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0  # BI_RGB

        buf_size = w * h * 4
        buf = ctypes.create_string_buffer(buf_size)
        gdi32.GetDIBits(mem_dc, bitmap, 0, h, buf, ctypes.byref(bmi), 0)

        # Cleanup GDI resources
        gdi32.SelectObject(mem_dc, old_bmp)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hwnd, hwnd_dc)

        # Convert BGRA -> RGB PIL Image -> PNG bytes
        img = Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1)
        img = img.convert("RGB")
        out = BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()

    # ------------------------------------------------------------------
    # Background input (no foreground focus needed)
    # ------------------------------------------------------------------

    def send_text(self, text: str) -> bool:
        """Send text via clipboard + WM_PASTE. No focus required."""
        if not self._target:
            return False

        edit_hwnd = self._find_edit_control(self._target.hwnd)
        target_hwnd = edit_hwnd or self._target.hwnd
        if edit_hwnd:
            log.debug("Win32WindowManager: send_text to edit control hwnd=%d", edit_hwnd)

        if not self._set_clipboard(text):
            return False

        WM_PASTE = 0x0302
        user32.SendMessageW(target_hwnd, WM_PASTE, 0, 0)
        time.sleep(0.05)
        return True

    @staticmethod
    def _find_edit_control(parent_hwnd: int) -> int | None:
        """Recursively find the deepest editable child control."""
        EDIT_CLASSES = {"edit", "richedit", "richedit20w", "richeditd2dpt",
                        "notepadtextbox", "scintilla", "textbox"}
        found = []

        def callback(child_hwnd, _):
            cls_buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(child_hwnd, cls_buf, 256)
            cls = cls_buf.value.lower()
            if cls in EDIT_CLASSES:
                found.append((child_hwnd, cls))
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_ulong, ctypes.c_ulong)
        user32.EnumChildWindows(parent_hwnd, WNDENUMPROC(callback), 0)

        if not found:
            return None

        priority = ["richeditd2dpt", "richedit20w", "richedit", "edit",
                     "scintilla", "notepadtextbox", "textbox"]
        for pclass in priority:
            for hwnd, cls in found:
                if cls == pclass:
                    return hwnd
        return found[0][0]

    @staticmethod
    def _set_clipboard(text: str) -> bool:
        """Set clipboard content via PowerShell Set-Clipboard."""
        escaped = text.replace("'", "''")
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"Set-Clipboard -Value '{escaped}'"],
                capture_output=True, timeout=3,
            )
            return r.returncode == 0
        except Exception as e:
            log.error("Win32WindowManager: Set-Clipboard failed: %s", e)
            return False

    def send_click(self, x: int, y: int, button: str = "left") -> bool:
        """Send a click at window-local coords via PostMessage. No focus required."""
        if not self._target:
            return False
        hwnd = self._target.hwnd
        lparam = y << 16 | (x & 0xFFFF)

        if button == "left":
            WM_DOWN, WM_UP = 0x0201, 0x0202
        elif button == "right":
            WM_DOWN, WM_UP = 0x0204, 0x0205
        else:
            WM_DOWN, WM_UP = 0x0201, 0x0202

        user32.PostMessageW(hwnd, WM_DOWN, 1, lparam)
        time.sleep(0.02)
        user32.PostMessageW(hwnd, WM_UP, 0, lparam)
        return True

    def send_hotkey(self, *keys: str) -> bool:
        """Send a hotkey combination via PostMessage. No focus required."""
        if not self._target:
            return False
        hwnd = self._target.hwnd
        WM_KEYDOWN = 0x0100
        WM_KEYUP = 0x0101

        vk_map = self._vk_map()
        vk_codes = []
        for key in keys:
            vk = vk_map.get(key.lower())
            if vk is None:
                vk = ord(key.upper()) if len(key) == 1 else 0
            vk_codes.append(vk)

        for vk in vk_codes:
            user32.PostMessageW(hwnd, WM_KEYDOWN, vk, 0)
            time.sleep(0.01)
        for vk in reversed(vk_codes):
            user32.PostMessageW(hwnd, WM_KEYUP, vk, 0)
            time.sleep(0.01)
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _info_from_hwnd(hwnd: int) -> WindowInfo | None:
        """Build WindowInfo from an HWND. Returns None if window is gone."""
        if not user32.IsWindow(hwnd):
            return None
        buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, buf, 512)
        title = buf.value

        pid = ctypes.wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

        cls_buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls_buf, 256)

        rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))

        return WindowInfo(
            hwnd=hwnd, title=title, pid=pid.value,
            class_name=cls_buf.value,
            rect=(rect.left, rect.top, rect.right, rect.bottom),
        )

    @staticmethod
    def _build_pid_map() -> dict[int, str]:
        """Build PID -> process name mapping via tasklist."""
        try:
            out = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            mapping = {}
            for line in out.strip().split("\n"):
                parts = line.strip().strip('"').split('","')
                if len(parts) >= 2:
                    try:
                        mapping[int(parts[1])] = parts[0]
                    except ValueError:
                        pass
            return mapping
        except Exception:
            return {}

    @staticmethod
    def _vk_map() -> dict[str, int]:
        """Common key name -> Win32 virtual key code."""
        return {
            "ctrl": 0x11, "control": 0x11,
            "alt": 0x12, "menu": 0x12,
            "shift": 0x10,
            "win": 0x5B, "lwin": 0x5B,
            "enter": 0x0D, "return": 0x0D,
            "tab": 0x09,
            "esc": 0x1B, "escape": 0x1B,
            "backspace": 0x08, "back": 0x08,
            "delete": 0x2E, "del": 0x2E,
            "space": 0x20,
            "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
            "home": 0x24, "end": 0x23,
            "pageup": 0x21, "pagedown": 0x22,
            "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
            "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77,
            "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
            "a": 0x41, "b": 0x42, "c": 0x43, "d": 0x44,
            "e": 0x45, "f": 0x46, "g": 0x47, "h": 0x48,
            "i": 0x49, "j": 0x4A, "k": 0x4B, "l": 0x4C,
            "m": 0x4D, "n": 0x4E, "o": 0x4F, "p": 0x50,
            "q": 0x51, "r": 0x52, "s": 0x53, "t": 0x54,
            "u": 0x55, "v": 0x56, "w": 0x57, "x": 0x58,
            "y": 0x59, "z": 0x5A,
        }
