from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import List

import fitz
from PIL import Image as PILImage
from PIL import UnidentifiedImageError

from src.config import settings


class ImageExtractor:
    """Extract and manage images from PDFs."""

    def __init__(self) -> None:
        self.images_dir = Path(settings.chroma_dir).parent / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.images_dir / "metadata.jsonl"

    def extract_images(self, pdf_path: Path, source_name: str) -> List[dict]:
        """Extract all images from PDF and save to disk.
        
        Returns list of dicts: {"id", "filename", "page", "path", "format"}
        """
        images_data = []
        pdf = fitz.open(pdf_path)
        
        try:
            for page_idx, page in enumerate(pdf, start=1):
                images = page.get_images(full=True)
                for img_idx, img in enumerate(images, start=1):
                    try:
                        xref = img[0]
                        image_data = pdf.extract_image(xref)
                        image_bytes = image_data["image"]
                        
                        # Validate image
                        pil_image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
                        
                        # Generate unique ID
                        img_hash = hashlib.md5(image_bytes).hexdigest()[:8]
                        img_id = f"{Path(source_name).stem}_p{page_idx}_img{img_idx}_{img_hash}"
                        
                        # Save image
                        image_format = image_data.get("ext", "png").lower()
                        if image_format not in ["png", "jpg", "jpeg"]:
                            image_format = "png"
                        
                        img_path = self.images_dir / f"{img_id}.{image_format}"
                        pil_image.save(img_path, format=image_format.upper())
                        
                        images_data.append({
                            "id": img_id,
                            "filename": source_name,
                            "page": page_idx,
                            "path": str(img_path.relative_to(self.images_dir.parent)),
                            "format": image_format,
                            "size": pil_image.size,  # (width, height)
                        })
                    except (KeyError, RuntimeError, ValueError, OSError, UnidentifiedImageError):
                        continue
        finally:
            pdf.close()
        
        return images_data

    def get_image_by_id(self, image_id: str) -> bytes | None:
        """Retrieve image binary data by ID."""
        # Search for the image file
        for img_file in self.images_dir.glob(f"{image_id}.*"):
            if img_file.is_file():
                return img_file.read_bytes()
        return None

    def list_images(self, filename: str | None = None) -> List[dict]:
        """List all extracted images, optionally filtered by filename."""
        images = []
        for img_file in self.images_dir.glob("*.*"):
            if img_file.is_file() and img_file.suffix in [".png", ".jpg", ".jpeg"]:
                img_id = img_file.stem
                # Extract metadata from filename
                try:
                    parts = img_id.split("_")
                    source = "_".join(parts[0:-3])  # Everything except p{page}_img{idx}_{hash}
                    page = int(parts[-3][1:]) if parts[-3].startswith("p") else None
                    
                    if filename and source != Path(filename).stem:
                        continue
                    
                    images.append({
                        "id": img_id,
                        "filename": f"{source}.pdf",
                        "page": page,
                        "format": img_file.suffix[1:].lower(),
                    })
                except (ValueError, IndexError):
                    continue
        
        return sorted(images, key=lambda x: (x.get("filename", ""), x.get("page", 0)))
