"""OCR engine interface + WinRT default implementation."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

from PIL import Image

from .types import OCRWord

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class OCREngine(ABC):
    """Pluggable OCR backend. Implement extract_words to add a new engine."""

    @abstractmethod
    def extract_words(self, img: Image.Image, lang: str) -> list[OCRWord]:
        """Extract words with bounding boxes from an image."""


# ---------------------------------------------------------------------------
# Default implementation: WinRT (winocr)
# ---------------------------------------------------------------------------

_WINOCR_AVAILABLE = False
try:
    import winocr as _winocr
    _WINOCR_AVAILABLE = True
except ImportError:
    _winocr = None  # type: ignore[assignment]


class WinOCREngine(OCREngine):
    """Windows-native OCR via WinRT (winocr). ~0.1s, good CJK support."""

    def extract_words(self, img: Image.Image, lang: str) -> list[OCRWord]:
        if not _WINOCR_AVAILABLE:
            log.error("WinOCREngine: winocr not installed (pip install winocr)")
            return []

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self._recognize, img, lang)
                return future.result(timeout=10)
        else:
            return self._recognize(img, lang)

    def _recognize(self, img: Image.Image, lang: str) -> list[OCRWord]:
        async def _run():
            return await _winocr.recognize_pil(img, lang=lang)

        result = asyncio.run(_run())
        words: list[OCRWord] = []

        for line_idx, line in enumerate(result.lines):
            for word_idx, word in enumerate(line.words):
                text = word.text.strip()
                if not text:
                    continue
                br = word.bounding_rect
                words.append(OCRWord(
                    text=text,
                    left=int(br.x),
                    top=int(br.y),
                    width=int(br.width),
                    height=int(br.height),
                    conf=95.0,
                    line_num=line_idx,
                    word_num=word_idx,
                ))

        return words
