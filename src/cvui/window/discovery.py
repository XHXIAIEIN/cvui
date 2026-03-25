"""Window discovery and management functions for Win32."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import subprocess
from typing import Optional

from ..types import WindowInfo

user32 = ctypes.windll.user32


def find_windows(
    title_contains: str = "",
    process_name: str = "",
    class_name: str = "",
) -> list[WindowInfo]:
    """Find all visible windows matching the given criteria."""
    results: list[WindowInfo] = []
    title_lower = title_contains.lower()
    proc_lower = process_name.lower()

    pid_map: dict[int, str] = {}
    if proc_lower:
        pid_map = build_pid_map()

    def enum_callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True

        buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, buf, 512)
        win_title = buf.value
        if not win_title:
            return True

        if title_lower and title_lower not in win_title.lower():
            return True

        pid = ctypes.wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        pid_val = pid.value

        if proc_lower:
            pname = pid_map.get(pid_val, "").lower()
            if proc_lower not in pname:
                return True

        cls_buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls_buf, 256)
        cls = cls_buf.value
        if class_name and class_name.lower() not in cls.lower():
            return True

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


def build_pid_map() -> dict[int, str]:
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


def info_from_hwnd(hwnd: int) -> WindowInfo | None:
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


def focus_window(hwnd: int) -> bool:
    """Bring a window to the foreground."""
    import time
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


def is_window_alive(hwnd: int) -> bool:
    """Check if a window handle is still valid."""
    return bool(user32.IsWindow(hwnd))
