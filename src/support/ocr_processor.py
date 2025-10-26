"""
OCR and ICR processing module for scanned documents and images.
Provides text extraction from image-based PDFs, scanned documents, and image files.
"""

import logging
import os
import tempfile
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)


class OCRMethod(Enum):
    """Available OCR processing methods."""

    TESSERACT = "tesseract"
    TESSERACT_ENHANCED = "tesseract_enhanced"


class ImageProcessingMethod(Enum):
    """Image preprocessing methods."""

    NONE = "none"
    GRAYSCALE = "grayscale"
    THRESHOLD = "threshold"
    DENOISE = "denoise"
    MORPHOLOGICAL = "morphological"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"


class OCRResult:
    """Result of OCR processing operation."""

    def __init__(
        self,
        success: bool,
        text: str = "",
        confidence: float = 0.0,
        metadata: Dict = None,
        error: str = "",
    ):
        self.success = success
        self.text = text
        self.confidence = confidence
        self.metadata = metadata or {}
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "success": self.success,
            "text": self.text,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "error": self.error,
        }


class OCRProcessor:
    """OCR and ICR processor for extracting text from images and scanned documents."""

    def __init__(self):
        """Initialize OCR processor."""
        self.tesseract_available = self._check_tesseract()
        self.pdf2image_available = self._check_pdf2image()

        # Default OCR configuration
        self.default_tesseract_config = "--oem 3 --psm 6"
        self.enhanced_tesseract_config = "--oem 1 --psm 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?;:()-[]"

        logger.info(
            f"OCR Processor initialized - Tesseract: {self.tesseract_available}, PDF2Image: {self.pdf2image_available}"
        )

    def _check_tesseract(self) -> bool:
        """Check if Tesseract OCR is available."""
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            return False

    def _check_pdf2image(self) -> bool:
        """Check if pdf2image is available."""
        try:
            import pdf2image

            return True
        except ImportError:
            logger.warning("pdf2image not available")
            return False

    def is_image_based_pdf(self, pdf_path: str) -> bool:
        """
        Determine if a PDF is image-based (scanned) by checking text content.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            True if PDF appears to be image-based, False otherwise
        """
        try:
            # Try to extract text using standard methods
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                total_chars = 0
                total_pages = len(pdf.pages)

                # Sample first few pages
                pages_to_check = min(3, total_pages)

                for i in range(pages_to_check):
                    page_text = pdf.pages[i].extract_text()
                    if page_text:
                        total_chars += len(page_text.strip())

                # If very little text found relative to page count, likely image-based
                avg_chars_per_page = (
                    total_chars / pages_to_check if pages_to_check > 0 else 0
                )

                # Threshold: less than 50 characters per page suggests scanned PDF
                if avg_chars_per_page < 50:
                    logger.info(
                        f"PDF appears to be image-based: {avg_chars_per_page:.1f} chars/page"
                    )
                    return True
                else:
                    logger.info(
                        f"PDF appears to be text-based: {avg_chars_per_page:.1f} chars/page"
                    )
                    return False

        except Exception as e:
            logger.warning(f"Error checking PDF type: {e}")
            # If we can't determine, assume it might be image-based
            return True

    def preprocess_image(
        self,
        image: np.ndarray,
        method: ImageProcessingMethod = ImageProcessingMethod.ADAPTIVE_THRESHOLD,
    ) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy.

        Args:
            image: Input image as numpy array
            method: Preprocessing method to apply

        Returns:
            Preprocessed image
        """
        try:
            if method == ImageProcessingMethod.NONE:
                return image

            # Convert to grayscale if not already
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            if method == ImageProcessingMethod.GRAYSCALE:
                return gray

            elif method == ImageProcessingMethod.THRESHOLD:
                _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                return thresh

            elif method == ImageProcessingMethod.ADAPTIVE_THRESHOLD:
                # Adaptive threshold works better for varying lighting
                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                return thresh

            elif method == ImageProcessingMethod.DENOISE:
                # Denoise the image
                denoised = cv2.fastNlMeansDenoising(gray)
                return denoised

            elif method == ImageProcessingMethod.MORPHOLOGICAL:
                # Apply morphological operations
                kernel = np.ones((2, 2), np.uint8)
                processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
                return processed

            return gray

        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image

    def extract_text_from_image(
        self,
        image_path: str,
        ocr_method: OCRMethod = OCRMethod.TESSERACT_ENHANCED,
        preprocessing: ImageProcessingMethod = ImageProcessingMethod.ADAPTIVE_THRESHOLD,
        language_codes: Optional[List[str]] = None,
    ) -> OCRResult:
        """
        Extract text from a single image file.

        Args:
            image_path: Path to the image file
            ocr_method: OCR method to use
            preprocessing: Image preprocessing method
            language_codes: List of language codes for OCR (e.g., ['en', 'es', 'fr'])

        Returns:
            OCRResult with extracted text and metadata
        """
        if not self.tesseract_available:
            return OCRResult(
                success=False,
                error="Tesseract OCR not available. Please install tesseract-ocr.",
                metadata={"ocr_method": ocr_method.value},
            )

        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return OCRResult(
                    success=False,
                    error=f"Could not load image: {image_path}",
                    metadata={"ocr_method": ocr_method.value},
                )

            # Preprocess image
            processed_image = self.preprocess_image(image, preprocessing)

            # Convert to PIL Image for Tesseract
            if len(processed_image.shape) == 3:
                pil_image = Image.fromarray(
                    cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                )
            else:
                pil_image = Image.fromarray(processed_image)

            # Configure Tesseract with language support
            if ocr_method == OCRMethod.TESSERACT_ENHANCED:
                config = self.enhanced_tesseract_config
            else:
                config = self.default_tesseract_config

            # Add language parameter if specified
            lang_param = None
            if language_codes:
                try:
                    from language_support import language_manager

                    lang_param = language_manager.get_tesseract_languages(
                        language_codes
                    )
                except ImportError:
                    logger.warning(
                        "Language support not available, using default language"
                    )

            # Extract text with language support
            if lang_param:
                extracted_text = pytesseract.image_to_string(
                    pil_image, lang=lang_param, config=config
                ).strip()
            else:
                extracted_text = pytesseract.image_to_string(
                    pil_image, config=config
                ).strip()

            # Get confidence data
            try:
                data = pytesseract.image_to_data(
                    pil_image, config=config, output_type=pytesseract.Output.DICT
                )
                confidences = [int(conf) for conf in data["conf"] if int(conf) > 0]
                avg_confidence = (
                    sum(confidences) / len(confidences) if confidences else 0
                )
            except:
                avg_confidence = 0

            # Create metadata
            metadata = {
                "ocr_method": ocr_method.value,
                "preprocessing": preprocessing.value,
                "image_dimensions": f"{image.shape[1]}x{image.shape[0]}",
                "character_count": len(extracted_text),
                "word_count": len(extracted_text.split()),
                "avg_confidence": round(avg_confidence, 2),
                "languages_used": language_codes if language_codes else ["eng"],
                "tesseract_lang_param": lang_param if lang_param else "eng",
            }

            if not extracted_text:
                return OCRResult(
                    success=False,
                    error="No text could be extracted from the image",
                    metadata=metadata,
                )

            return OCRResult(
                success=True,
                text=extracted_text,
                confidence=avg_confidence / 100.0,  # Convert to 0-1 scale
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"OCR processing failed for {image_path}: {e}")
            return OCRResult(
                success=False,
                error=f"OCR processing error: {str(e)}",
                metadata={"ocr_method": ocr_method.value},
            )

    def extract_text_from_pdf_images(
        self,
        pdf_path: str,
        ocr_method: OCRMethod = OCRMethod.TESSERACT_ENHANCED,
        preprocessing: ImageProcessingMethod = ImageProcessingMethod.ADAPTIVE_THRESHOLD,
        dpi: int = 300,
        language_codes: Optional[List[str]] = None,
    ) -> OCRResult:
        """
        Extract text from scanned PDF by converting pages to images and applying OCR.

        Args:
            pdf_path: Path to the PDF file
            ocr_method: OCR method to use
            preprocessing: Image preprocessing method
            dpi: DPI for PDF to image conversion
            language_codes: List of language codes for OCR

        Returns:
            OCRResult with extracted text from all pages
        """
        if not self.pdf2image_available:
            return OCRResult(
                success=False,
                error="pdf2image not available. Cannot process scanned PDFs.",
                metadata={"ocr_method": ocr_method.value},
            )

        if not self.tesseract_available:
            return OCRResult(
                success=False,
                error="Tesseract OCR not available. Please install tesseract-ocr.",
                metadata={"ocr_method": ocr_method.value},
            )

        try:
            from pdf2image import convert_from_path

            # Convert PDF pages to images
            logger.info(f"Converting PDF to images: {pdf_path}")
            images = convert_from_path(pdf_path, dpi=dpi)

            if not images:
                return OCRResult(
                    success=False,
                    error="Could not convert PDF pages to images",
                    metadata={"ocr_method": ocr_method.value},
                )

            # Process each page
            page_texts = []
            page_confidences = []
            total_chars = 0

            for page_num, pil_image in enumerate(images):
                try:
                    # Convert PIL to OpenCV format for preprocessing
                    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                    # Preprocess image
                    processed_image = self.preprocess_image(opencv_image, preprocessing)

                    # Convert back to PIL for OCR
                    if len(processed_image.shape) == 3:
                        processed_pil = Image.fromarray(
                            cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                        )
                    else:
                        processed_pil = Image.fromarray(processed_image)

                    # Configure Tesseract with language support
                    if ocr_method == OCRMethod.TESSERACT_ENHANCED:
                        config = self.enhanced_tesseract_config
                    else:
                        config = self.default_tesseract_config

                    # Add language parameter if specified
                    lang_param = None
                    if language_codes:
                        try:
                            from language_support import language_manager

                            lang_param = language_manager.get_tesseract_languages(
                                language_codes
                            )
                        except ImportError:
                            logger.warning(
                                "Language support not available, using default language"
                            )

                    # Extract text from page with language support
                    if lang_param:
                        page_text = pytesseract.image_to_string(
                            processed_pil, lang=lang_param, config=config
                        ).strip()
                    else:
                        page_text = pytesseract.image_to_string(
                            processed_pil, config=config
                        ).strip()

                    if page_text:
                        page_texts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                        total_chars += len(page_text)

                        # Get confidence for this page
                        try:
                            data = pytesseract.image_to_data(
                                processed_pil,
                                config=config,
                                output_type=pytesseract.Output.DICT,
                            )
                            confidences = [
                                int(conf) for conf in data["conf"] if int(conf) > 0
                            ]
                            page_confidence = (
                                sum(confidences) / len(confidences)
                                if confidences
                                else 0
                            )
                            page_confidences.append(page_confidence)
                        except:
                            page_confidences.append(0)

                    logger.info(
                        f"Processed page {page_num + 1}/{len(images)}: {len(page_text)} characters"
                    )

                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1}: {e}")
                    page_texts.append(
                        f"--- Page {page_num + 1} ---\n[Error processing page]"
                    )
                    page_confidences.append(0)

            # Combine all pages
            full_text = "\n\n".join(page_texts)
            avg_confidence = (
                sum(page_confidences) / len(page_confidences) if page_confidences else 0
            )

            # Create metadata
            metadata = {
                "ocr_method": ocr_method.value,
                "preprocessing": preprocessing.value,
                "total_pages": len(images),
                "pages_with_text": len(
                    [t for t in page_texts if "Error processing" not in t]
                ),
                "total_characters": total_chars,
                "word_count": len(full_text.split()),
                "avg_confidence": round(avg_confidence, 2),
                "dpi": dpi,
                "languages_used": language_codes if language_codes else ["eng"],
                "tesseract_lang_param": lang_param if lang_param else "eng",
            }

            if not full_text.strip():
                return OCRResult(
                    success=False,
                    error="No text could be extracted from any pages",
                    metadata=metadata,
                )

            return OCRResult(
                success=True,
                text=full_text,
                confidence=avg_confidence / 100.0,  # Convert to 0-1 scale
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"PDF OCR processing failed for {pdf_path}: {e}")
            return OCRResult(
                success=False,
                error=f"PDF OCR processing error: {str(e)}",
                metadata={"ocr_method": ocr_method.value},
            )

    def enhance_ocr_with_multiple_methods(
        self, image_path: str, language_codes: Optional[List[str]] = None
    ) -> OCRResult:
        """
        Try multiple OCR methods and preprocessing techniques to get the best result.

        Args:
            image_path: Path to the image file
            language_codes: List of language codes for OCR

        Returns:
            Best OCRResult from multiple attempts
        """
        methods_to_try = [
            (OCRMethod.TESSERACT_ENHANCED, ImageProcessingMethod.ADAPTIVE_THRESHOLD),
            (OCRMethod.TESSERACT, ImageProcessingMethod.ADAPTIVE_THRESHOLD),
            (OCRMethod.TESSERACT_ENHANCED, ImageProcessingMethod.THRESHOLD),
            (OCRMethod.TESSERACT, ImageProcessingMethod.DENOISE),
            (OCRMethod.TESSERACT_ENHANCED, ImageProcessingMethod.MORPHOLOGICAL),
        ]

        best_result = None
        best_score = 0

        for ocr_method, preprocessing in methods_to_try:
            try:
                result = self.extract_text_from_image(
                    image_path, ocr_method, preprocessing, language_codes
                )

                if result.success:
                    # Score based on confidence and text length
                    score = result.confidence * 0.7 + (len(result.text) / 1000) * 0.3

                    if score > best_score:
                        best_score = score
                        best_result = result
                        best_result.metadata["enhancement_score"] = round(score, 3)
                        best_result.metadata["method_used"] = (
                            f"{ocr_method.value}+{preprocessing.value}"
                        )

            except Exception as e:
                logger.warning(
                    f"Method {ocr_method.value}+{preprocessing.value} failed: {e}"
                )
                continue

        if best_result is None:
            return OCRResult(
                success=False,
                error="All OCR enhancement methods failed",
                metadata={"enhancement_attempted": True},
            )

        return best_result


# Global OCR processor instance
ocr_processor = OCRProcessor()


def extract_text_from_scanned_pdf(
    pdf_path: str, enhanced: bool = True, language_codes: Optional[List[str]] = None
) -> OCRResult:
    """
    Extract text from scanned PDF using OCR.

    Args:
        pdf_path: Path to the PDF file
        enhanced: Whether to use enhanced OCR methods
        language_codes: List of language codes for OCR

    Returns:
        OCRResult with extracted text
    """
    method = OCRMethod.TESSERACT_ENHANCED if enhanced else OCRMethod.TESSERACT
    return ocr_processor.extract_text_from_pdf_images(
        pdf_path, method, language_codes=language_codes
    )


def extract_text_from_image_file(
    image_path: str, enhanced: bool = True, language_codes: Optional[List[str]] = None
) -> OCRResult:
    """
    Extract text from image file using OCR.

    Args:
        image_path: Path to the image file
        enhanced: Whether to use enhanced OCR methods
        language_codes: List of language codes for OCR

    Returns:
        OCRResult with extracted text
    """
    if enhanced:
        return ocr_processor.enhance_ocr_with_multiple_methods(
            image_path, language_codes
        )
    else:
        return ocr_processor.extract_text_from_image(
            image_path, language_codes=language_codes
        )


def is_scanned_pdf(pdf_path: str) -> bool:
    """
    Check if a PDF is scanned/image-based.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        True if PDF appears to be scanned
    """
    return ocr_processor.is_image_based_pdf(pdf_path)


if __name__ == "__main__":
    # Test OCR functionality
    processor = OCRProcessor()

    print("OCR Processor Test Results:")
    print("=" * 50)
    print(f"Tesseract available: {processor.tesseract_available}")
    print(f"PDF2Image available: {processor.pdf2image_available}")

    if processor.tesseract_available:
        try:
            version = pytesseract.get_tesseract_version()
            print(f"Tesseract version: {version}")
        except:
            print("Could not get Tesseract version")

    print("\nOCR Processor ready for scanned document processing!")
