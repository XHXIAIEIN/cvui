"""Win32 background input: text, click, hotkey."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import subprocess
import time

user32 = ctypes.windll.user32
log = logging.getLogger(__name__)


def find_edit_control(parent_hwnd: int) -> int | None:
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


def set_clipboard(text: str) -> bool:
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
        log.error("set_clipboard failed: %s", e)
        return False


def send_text_to(hwnd: int, text: str) -> bool:
    """Send text via clipboard + WM_PASTE. No focus required."""
    edit_hwnd = find_edit_control(hwnd)
    target_hwnd = edit_hwnd or hwnd
    if edit_hwnd:
        log.debug("send_text_to: using edit control hwnd=%d", edit_hwnd)

    if not set_clipboard(text):
        return False

    WM_PASTE = 0x0302
    user32.SendMessageW(target_hwnd, WM_PASTE, 0, 0)
    time.sleep(0.05)
    return True


def send_click_to(hwnd: int, x: int, y: int, button: str = "left") -> bool:
    """Send a click at window-local coords via PostMessage. No focus required."""
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


def send_hotkey_to(hwnd: int, *keys: str) -> bool:
    """Send a hotkey combination via PostMessage. No focus required."""
    from ._vk_map import vk_map as _vk_map
    WM_KEYDOWN = 0x0100
    WM_KEYUP = 0x0101

    key_map = _vk_map()
    vk_codes = []
    for key in keys:
        vk = key_map.get(key.lower())
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
