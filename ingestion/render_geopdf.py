"""
DPI-Aware GeoPDF Renderer and Page Thumbnail Generator.
Uses PyMuPDF (fitz) to render high-resolution raster buffers and UI preview thumbnails.
"""

import os
from typing import Tuple, Optional, List, Dict, Any
import pymupdf
import numpy as np
from PIL import Image


class PDFRenderer:
    """Renders PDF pages to numpy image arrays with custom DPI and viewport bounds."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    def render_page(
        self,
        page_index: int = 0,
        dpi: int = 300,
        clip_rect_pts: Optional[Tuple[float, float, float, float]] = None,
        layer_visibility: Optional[Dict[int, bool]] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Renders a PDF page to an RGB numpy array (H, W, 3) at the requested DPI.
        
        Args:
            page_index: 0-indexed page number.
            dpi: Rendering DPI (default 300).
            clip_rect_pts: (x0, y0, x1, y1) bounding box in points to crop/render.
            layer_visibility: Dict of OCG layer ID to boolean visibility.
            
        Returns:
            rgb_array: uint8 numpy array of shape (H, W, 3).
            scale_factor: points-to-pixels scale factor.
        """
        doc = pymupdf.open(self.pdf_path)
        if page_index >= len(doc):
            raise IndexError(f"Page index {page_index} out of range (total pages: {len(doc)})")

        page = doc[page_index]

        # Apply layer visibility if specified
        if layer_visibility:
            for ocg_id, is_visible in layer_visibility.items():
                try:
                    doc.set_ocg_state(ocg_id, is_visible)
                except Exception:
                    pass

        # Standard PDF 1 pt = 1/72 inch => scale = dpi / 72.0
        scale = dpi / 72.0
        matrix = pymupdf.Matrix(scale, scale)

        clip = None
        if clip_rect_pts:
            clip = pymupdf.Rect(clip_rect_pts)

        pix = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
        
        # Convert PyMuPDF pixmap to numpy RGB array
        img_bytes = pix.samples
        img_np = np.frombuffer(img_bytes, dtype=np.uint8).reshape((pix.height, pix.width, 3))

        doc.close()
        return img_np, scale

    def generate_thumbnails(
        self,
        max_size: int = 256,
        target_pages: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generates fast UI thumbnails for each page in the PDF.
        
        Returns:
            List of dicts: [{"page_index": i, "width": w, "height": h, "image": PIL.Image}]
        """
        doc = pymupdf.open(self.pdf_path)
        thumbnails = []
        page_indices = target_pages if target_pages is not None else list(range(len(doc)))

        for idx in page_indices:
            if idx >= len(doc):
                continue
            page = doc[idx]
            rect = page.rect
            scale = max_size / max(rect.width, rect.height)
            matrix = pymupdf.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
            pil_img = Image.fromarray(img_data)

            thumbnails.append({
                "page_index": idx,
                "page_number": idx + 1,
                "width": pix.width,
                "height": pix.height,
                "image": pil_img,
                "array": img_data
            })

        doc.close()
        return thumbnails
