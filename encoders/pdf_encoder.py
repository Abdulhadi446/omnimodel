"""
PDF Encoder - Extracts text and images, encodes them together.
"""

from typing import List, Dict, Any, Optional

try:
    from PyPDF2 import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False


class PDFEncoder:
    """
    Encodes PDF documents by extracting text and rendering images.
    """
    
    def __init__(self, image_encoder=None, max_pages: int = 10):
        """
        Args:
            image_encoder: ImageEncoder instance
            max_pages: Maximum pages to encode
        """
        self.image_encoder = image_encoder
        self.max_pages = max_pages
        
        print(f"✓ Created PDFEncoder (max {max_pages} pages)")
    
    def encode(self, pdf_path: str) -> Dict[str, Any]:
        """
        Encode a PDF file.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Dict with "content" list of text and image tokens
        """
        if not PYPDF_AVAILABLE:
            print("⚠ PyPDF2 not available, install with: pip install PyPDF2")
            return {"content": []}
        
        try:
            pdf = PdfReader(pdf_path)
            num_pages = min(len(pdf.pages), self.max_pages)
            
            content = []
            
            # Extract text from each page
            for page_idx in range(num_pages):
                page = pdf.pages[page_idx]
                text = page.extract_text()
                
                if text:
                    content.append({
                        "type": "text",
                        "value": text,
                        "page": page_idx,
                    })
            
            # Try to extract images if PyMuPDF available
            if FITZ_AVAILABLE:
                try:
                    doc = fitz.open(pdf_path)
                    
                    for page_idx, page in enumerate(doc):
                        if page_idx >= self.max_pages:
                            break
                        
                        images = page.get_images()
                        for img_idx, img in enumerate(images):
                            try:
                                # Extract and encode image
                                xref = img[0]
                                pix = fitz.Pixmap(doc, xref)
                                
                                if pix.n - pix.alpha < 4:  # GRAY or RGB
                                    # Encode image
                                    if self.image_encoder:
                                        # Save temporarily
                                        import tempfile
                                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                                            pix.save(f.name)
                                            tokens = self.image_encoder.encode_image_file(f.name)
                                            
                                            content.append({
                                                "type": "image",
                                                "value": tokens,
                                                "page": page_idx,
                                            })
                                            
                                            import os
                                            os.unlink(f.name)
                            except Exception as e:
                                pass  # Skip problematic images
                
                except Exception as e:
                    print(f"⚠ Could not extract images: {e}")
            
            print(f"✓ Encoded PDF: {num_pages} pages, {len(content)} content items")
            return {"content": content}
        
        except Exception as e:
            print(f"✗ Error encoding PDF: {e}")
            return {"content": []}
    
    def encode_to_tokens(self, pdf_path: str) -> List[int]:
        """
        Encode PDF to a flat list of tokens.
        Interleaves text and image tokens.
        """
        content_dict = self.encode(pdf_path)
        tokens = []
        
        for item in content_dict.get("content", []):
            if item["type"] == "text":
                # Encode text to character tokens (simplified)
                text_tokens = [ord(c) for c in item["value"][:200]]  # Limit text
                tokens.extend(text_tokens)
            elif item["type"] == "image":
                # Image tokens
                tokens.extend(item["value"][:256])  # Limit tokens per image
        
        return tokens


def create_pdf_encoder(image_encoder=None) -> Optional[PDFEncoder]:
    """Create a PDF encoder"""
    return PDFEncoder(image_encoder=image_encoder)


if __name__ == "__main__":
    encoder = create_pdf_encoder()
    if encoder:
        print("✓ PDF encoder created")
