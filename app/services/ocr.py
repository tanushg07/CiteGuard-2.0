"""OCR fallback for scanned (image-only) PDFs.

Pages are rendered with pypdfium2 (already a pdfplumber dependency) and recognised by the first available engine:

- rapidocr_onnxruntime (pip-installable, CPU, no system binary)   -> engine 'rapidocr'
- pytesseract + a `tesseract` binary on PATH                     -> engine 'tesseract'

No engine is bundled in requirements-lock.txt; see docs/MODELS.md ("OCR"). When none is installed, callers get
OCRUnavailable and must report EXTRACTION_FAILED with cause OCR_UNAVAILABLE instead of pretending the PDF is empty.
OCR text is lower fidelity than embedded text, so every passage it yields is tagged extraction_method='ocr:<engine>'.
"""
from __future__ import annotations

import shutil
from collections.abc import Callable


class OCRUnavailable(RuntimeError):
    pass


_ENGINE: tuple[str, Callable] | None = None


def _rapidocr():
    from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]
    engine = RapidOCR()

    def run(image) -> str:
        import numpy as np
        result, _ = engine(np.asarray(image))
        # result rows: [box, text, score]; keep reading order as returned (top-to-bottom, left-to-right)
        return '\n'.join(row[1] for row in result or [])
    return run


def _tesseract():
    import pytesseract  # type: ignore[import-not-found]
    if not shutil.which('tesseract'):
        raise ImportError('tesseract binary not on PATH')

    def run(image) -> str:
        return pytesseract.image_to_string(image)
    return run


def engine() -> tuple[str, Callable]:
    """(name, recognise(image) -> text). Raises OCRUnavailable listing what was tried."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    errors = []
    for name, factory in (('rapidocr', _rapidocr), ('tesseract', _tesseract)):
        try:
            _ENGINE = (name, factory())
            return _ENGINE
        except Exception as exc:  # noqa: BLE001 - any import/initialisation failure means "engine not usable"
            errors.append(f'{name}: {type(exc).__name__}: {exc}'[:160])
    raise OCRUnavailable('No OCR engine is installed (' + '; '.join(errors) + '). '
                         'Install one with: pip install rapidocr_onnxruntime')


def ocr_pdf(path: str, max_pages: int = 30, recogniser: tuple[str, Callable] | None = None, scale: float = 2.0) -> tuple[str, list[str]]:
    """Return (engine_name, page_texts). `recogniser` is injectable for tests."""
    name, recognise = recogniser or engine()
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(path)
    try:
        pages = []
        for index in range(min(len(pdf), max_pages)):
            page = pdf[index]
            image = page.render(scale=scale).to_pil()
            pages.append(recognise(image) or '')
            page.close()
        return name, pages
    finally:
        pdf.close()


def reset_for_tests():
    global _ENGINE
    _ENGINE = None
