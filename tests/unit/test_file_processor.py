"""
Unit tests for FileProcessor class.

Tests file processing, text extraction, and format handling.
"""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import tempfile

from support.file_processor import FileProcessor


class TestFileProcessor:
    """Test cases for FileProcessor."""

    def test_init(self):
        """Test file processor initialization."""
        processor = FileProcessor()
        assert hasattr(processor, 'supported_formats')
        assert hasattr(processor, 'text_extractors')

    @pytest.mark.parametrize("filename,expected_type", [
        ("document.pdf", "pdf"),
        ("document.docx", "docx"),
        ("document.doc", "doc"),
        ("document.txt", "txt"),
        ("document.md", "md"),
        ("image.png", "png"),
        ("image.jpg", "jpg"),
        ("image.jpeg", "jpeg"),
        ("unknown.xyz", "unknown")
    ])
    def test_get_file_type(self, file_processor, filename, expected_type):
        """Test file type detection."""
        file_type = file_processor.get_file_type(filename)
        assert file_type == expected_type

    @pytest.mark.parametrize("filename,expected_valid", [
        ("document.pdf", True),
        ("document.docx", True),
        ("document.txt", True),
        ("document.md", True),
        ("image.png", True),
        ("unknown.xyz", False),
        ("file_without_extension", False),
        ("", False)
    ])
    def test_is_valid_file(self, file_processor, filename, expected_valid):
        """Test file validation."""
        is_valid = file_processor.is_valid_file(filename)
        assert is_valid == expected_valid

    def test_extract_text_from_txt_file(self, file_processor, temp_files):
        """Test text extraction from plain text file."""
        content = file_processor.extract_text_content(str(temp_files["math"]))

        assert isinstance(content, str)
        assert len(content) > 0
        assert "John Doe" in content
        assert "Mathematics" in content

    def test_extract_text_from_empty_file(self, file_processor, temp_files):
        """Test text extraction from empty file."""
        content = file_processor.extract_text_content(str(temp_files["empty"]))

        assert isinstance(content, str)
        assert len(content) == 0

    def test_extract_text_from_nonexistent_file(self, file_processor):
        """Test text extraction from non-existent file."""
        with pytest.raises((FileNotFoundError, OSError)):
            file_processor.extract_text_content("nonexistent_file.txt")

    @patch('support.file_processor.PyPDF2')
    def test_extract_text_from_pdf(self, mock_pypdf2, file_processor, temp_dir):
        """Test PDF text extraction."""
        # Mock PDF reader
        mock_reader = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample PDF content"
        mock_reader.pages = [mock_page]
        mock_pypdf2.PdfReader.return_value = mock_reader

        # Create a mock PDF file
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"Mock PDF content")

        content = file_processor.extract_text_content(str(pdf_file))

        assert content == "Sample PDF content"

    @patch('support.file_processor.docx')
    def test_extract_text_from_docx(self, mock_docx, file_processor, temp_dir):
        """Test DOCX text extraction."""
        # Mock docx document
        mock_doc = Mock()
        mock_paragraph = Mock()
        mock_paragraph.text = "Sample DOCX content"
        mock_doc.paragraphs = [mock_paragraph]
        mock_docx.Document.return_value = mock_doc

        # Create a mock DOCX file
        docx_file = temp_dir / "test.docx"
        docx_file.write_bytes(b"Mock DOCX content")

        content = file_processor.extract_text_content(str(docx_file))

        assert content == "Sample DOCX content"

    @patch('support.file_processor.mammoth')
    def test_extract_text_from_doc(self, mock_mammoth, file_processor, temp_dir):
        """Test DOC text extraction."""
        # Mock mammoth result
        mock_result = Mock()
        mock_result.value = "Sample DOC content"
        mock_mammoth.extract_text.return_value = mock_result

        # Create a mock DOC file
        doc_file = temp_dir / "test.doc"
        doc_file.write_bytes(b"Mock DOC content")

        content = file_processor.extract_text_content(str(doc_file))

        assert content == "Sample DOC content"

    def test_extract_metadata_from_text(self, file_processor):
        """Test metadata extraction from text content."""
        text_with_metadata = """
        Name: John Doe
        Date: 2025-01-15
        Class: Algebra II
        Subject: Mathematics

        Assignment content here...
        """

        metadata = file_processor.extract_metadata(text_with_metadata)

        assert isinstance(metadata, dict)
        assert metadata.get("name") == "John Doe"
        assert metadata.get("date") == "2025-01-15"
        assert metadata.get("class") == "Algebra II"
        assert metadata.get("subject") == "Mathematics"

    def test_extract_metadata_partial(self, file_processor):
        """Test metadata extraction with partial information."""
        text_partial_metadata = """
        Name: Jane Smith
        Subject: Spanish

        Some assignment content...
        """

        metadata = file_processor.extract_metadata(text_partial_metadata)

        assert isinstance(metadata, dict)
        assert metadata.get("name") == "Jane Smith"
        assert metadata.get("subject") == "Spanish"
        assert "date" not in metadata or metadata["date"] is None

    def test_extract_metadata_no_metadata(self, file_processor):
        """Test metadata extraction with no metadata."""
        text_no_metadata = "Just some regular text without metadata headers."

        metadata = file_processor.extract_metadata(text_no_metadata)

        assert isinstance(metadata, dict)
        assert len(metadata) == 0 or all(v is None for v in metadata.values())

    def test_clean_text_content(self, file_processor):
        """Test text cleaning functionality."""
        dirty_text = """


        This    has     extra    spaces

        And multiple


        blank lines


        """

        cleaned_text = file_processor.clean_text_content(dirty_text)

        assert isinstance(cleaned_text, str)
        assert "extra    spaces" not in cleaned_text
        assert cleaned_text.count('\n\n\n') == 0  # No triple newlines

    def test_validate_file_content(self, file_processor):
        """Test file content validation."""
        valid_content = "This is valid assignment content with sufficient length."
        invalid_content = "x"  # Too short
        empty_content = ""

        assert file_processor.validate_file_content(valid_content) == True
        assert file_processor.validate_file_content(invalid_content) == False
        assert file_processor.validate_file_content(empty_content) == False

    def test_estimate_processing_time(self, file_processor):
        """Test processing time estimation."""
        short_text = "Short text"
        long_text = "word " * 10000

        short_time = file_processor.estimate_processing_time(short_text)
        long_time = file_processor.estimate_processing_time(long_text)

        assert isinstance(short_time, (int, float))
        assert isinstance(long_time, (int, float))
        assert long_time > short_time

    def test_get_file_stats(self, file_processor, temp_files):
        """Test file statistics gathering."""
        stats = file_processor.get_file_stats(str(temp_files["math"]))

        assert isinstance(stats, dict)
        assert 'file_size' in stats
        assert 'file_type' in stats
        assert 'character_count' in stats
        assert 'word_count' in stats
        assert 'line_count' in stats

    def test_batch_file_processing(self, file_processor, temp_files):
        """Test batch processing of multiple files."""
        file_paths = [
            str(temp_files["math"]),
            str(temp_files["spanish"]),
            str(temp_files["science"])
        ]

        results = file_processor.process_batch(file_paths)

        assert isinstance(results, list)
        assert len(results) == 3

        for result in results:
            assert 'file_path' in result
            assert 'content' in result
            assert 'metadata' in result
            assert 'success' in result

    def test_error_handling_corrupted_file(self, file_processor, temp_dir):
        """Test error handling with corrupted files."""
        # Create a file with invalid content for its extension
        corrupted_pdf = temp_dir / "corrupted.pdf"
        corrupted_pdf.write_text("This is not valid PDF content")

        # Should handle gracefully without crashing
        content = file_processor.extract_text_content(str(corrupted_pdf))
        assert isinstance(content, str)  # Should return empty string or error message

    def test_large_file_handling(self, file_processor, temp_dir):
        """Test handling of large files."""
        # Create a large text file
        large_content = "This is a large file. " * 100000  # ~2MB of text
        large_file = temp_dir / "large.txt"
        large_file.write_text(large_content)

        content = file_processor.extract_text_content(str(large_file))

        assert isinstance(content, str)
        assert len(content) > 0

    def test_unicode_handling(self, file_processor, temp_dir):
        """Test handling of Unicode characters."""
        unicode_content = "Mathématiques: résoudre l'équation x² + 2x - 3 = 0 🧮"
        unicode_file = temp_dir / "unicode.txt"
        unicode_file.write_text(unicode_content, encoding='utf-8')

        content = file_processor.extract_text_content(str(unicode_file))

        assert isinstance(content, str)
        assert "Mathématiques" in content
        assert "🧮" in content

    @patch('support.file_processor.OCRProcessor')
    def test_ocr_integration(self, mock_ocr, file_processor, temp_dir):
        """Test OCR integration for image files."""
        # Mock OCR processor
        mock_ocr_instance = Mock()
        mock_ocr_instance.extract_text_from_image.return_value = "OCR extracted text"
        mock_ocr.return_value = mock_ocr_instance

        # Create a mock image file
        image_file = temp_dir / "test.png"
        image_file.write_bytes(b"Mock image content")

        content = file_processor.extract_text_content(str(image_file))

        assert content == "OCR extracted text"

    def test_content_type_detection(self, file_processor):
        """Test detection of content types within text."""
        math_content = "Solve for x: 2x + 5 = 13"
        spanish_content = "Escriba un ensayo sobre la familia"
        science_content = "Hypothesis: Plants need sunlight for photosynthesis"

        math_type = file_processor.detect_content_type(math_content)
        spanish_type = file_processor.detect_content_type(spanish_content)
        science_type = file_processor.detect_content_type(science_content)

        assert "math" in math_type.lower() or "equation" in math_type.lower()
        assert "spanish" in spanish_type.lower() or "language" in spanish_type.lower()
        assert "science" in science_type.lower() or "hypothesis" in science_type.lower()

    def test_file_security_validation(self, file_processor, temp_dir):
        """Test file security validation."""
        # Test with normal file
        normal_file = temp_dir / "normal.txt"
        normal_file.write_text("Normal content")

        assert file_processor.validate_file_security(str(normal_file)) == True

        # Test with suspicious filename patterns
        suspicious_patterns = [
            "../../../etc/passwd",
            "..\\windows\\system32\\file.txt",
            "file_with_null_byte\x00.txt"
        ]

        for pattern in suspicious_patterns:
            assert file_processor.validate_file_security(pattern) == False

    def test_processing_progress_tracking(self, file_processor, temp_files):
        """Test processing progress tracking."""
        file_paths = [
            str(temp_files["math"]),
            str(temp_files["spanish"])
        ]

        progress_updates = []

        def progress_callback(current, total, file_path):
            progress_updates.append((current, total, file_path))

        file_processor.process_batch(file_paths, progress_callback=progress_callback)

        assert len(progress_updates) > 0
        assert all(isinstance(update, tuple) and len(update) == 3 for update in progress_updates)