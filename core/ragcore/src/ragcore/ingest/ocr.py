"""Text from page images, for PDF pages that have no text layer.

RapidOCR ships its detection and recognition models inside the wheel, so this
never downloads anything. The engine (onnxruntime + OpenCV) is heavy to import
and load, so it is built on the first scanned page, once per process.
"""

from __future__ import annotations

from functools import cache

# 216 dpi: small print stays legible to the recogniser; RapidOCR caps the long side at 2000 px.
RENDER_SCALE = 3


@cache
def _engine():
    from rapidocr import RapidOCR

    # Detection runs on the page shrunk to 960 px; recognition still reads crops of the
    # full render. At the default (shortest side >= 736 px, i.e. the full 2000 px page)
    # the detector misses a short line on an otherwise blank page.
    return RapidOCR(
        params={
            "Global.log_level": "warning",
            "Det.limit_type": "max",
            "Det.limit_side_len": 960,
        }
    )


def read_page(page) -> str:
    """OCR one pypdfium2 page; lines joined in the order the recogniser reports them.

    ponytail: the bundled recogniser is the default PP-OCRv6 one; if accented Latin
    text (à, è, ß) comes back mangled, switch ``Rec.lang_type`` to the latin model.
    """
    bitmap = page.render(scale=RENDER_SCALE)
    try:
        result = _engine()(bitmap.to_numpy())
    finally:
        bitmap.close()
    return "\n".join(result.txts or ())
