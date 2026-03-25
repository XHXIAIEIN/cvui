"""Win32 window capture: PrintWindow + GDI -> PNG bytes."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
from io import BytesIO

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

try:
    from PIL import Image
except ImportError:
    Image = None


def capture_window_png(hwnd: int, width: int, height: int) -> bytes | None:
    """Capture a window by HWND using PrintWindow + GDI, returns PNG bytes.

    Works even if the window is partially or fully occluded.
    width/height must be pre-fetched by the caller (from WindowInfo).
    """
    if Image is None:
        return None
    if width <= 0 or height <= 0:
        return None

    hwnd_dc = user32.GetWindowDC(hwnd)
    mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
    old_bmp = gdi32.SelectObject(mem_dc, bitmap)

    PW_RENDERFULLCONTENT = 2
    user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT)

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
    bmi.biWidth = width
    bmi.biHeight = -height  # top-down
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0  # BI_RGB

    buf_size = width * height * 4
    buf = ctypes.create_string_buffer(buf_size)
    gdi32.GetDIBits(mem_dc, bitmap, 0, height, buf, ctypes.byref(bmi), 0)

    gdi32.SelectObject(mem_dc, old_bmp)
    gdi32.DeleteObject(bitmap)
    gdi32.DeleteDC(mem_dc)
    user32.ReleaseDC(hwnd, hwnd_dc)

    img = Image.frombuffer("RGBA", (width, height), buf, "raw", "BGRA", 0, 1)
    img = img.convert("RGB")
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()
