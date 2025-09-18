# 🎓 RT-Capstone Gradio Web Interface

A comprehensive web interface for the RT-Capstone Assignment Grading System built with Gradio.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Virtual environment activated
- All dependencies installed

### Installation
```bash
# Install Gradio dependencies (if not already installed)
pip install gradio pandas

# Or install all requirements
pip install -r requirements.txt
```

### Launch the Interface
```bash
# Simple launch
python launch_gradio.py

# Or run directly
python src/gradio_app.py
```

The interface will automatically open in your browser at `http://localhost:7860`

## 🌟 Features

### 📄 Single Assignment Processing
- Upload individual assignment files (PDF, DOCX, TXT, MD)
- Real-time processing with progress updates
- Comprehensive grading and analysis
- Downloadable results (JSON + summary)

### 📚 Batch Processing
- Upload multiple assignments simultaneously
- Batch processing with progress tracking
- Summary table of all results
- Bulk download capabilities

### 🔧 System Monitoring
- Real-time system status display
- LLM provider health monitoring
- Processing component status

## 📋 Processing Options

### Core Features
- **Grammar Analysis**: Multi-language grammar checking with error detection
- **Plagiarism Detection**: AI-powered plagiarism analysis with confidence scoring
- **Source Relevance**: Content alignment and factual consistency checking
- **Automated Grading**: 10-point scale with letter grades
- **Summary Generation**: Subject-aware AI summaries
- **Subject-Specific Analysis**: Specialized processing for different academic subjects

### Subject-Specific Processing
- **📐 Mathematics**: Equation solving, step-by-step analysis, notation clarity
- **🇪🇸 Spanish**: Grammar checking, cultural references, fluency assessment
- **🔬 Science**: Scientific method evaluation, formula recognition, hypothesis analysis
- **📚 History**: Chronological analysis, source evaluation, bias detection
- **📝 English**: Literary analysis, writing quality, argumentation assessment

## 🎯 Supported File Formats

| Format | Extension | OCR Support | Notes |
|--------|-----------|-------------|-------|
| PDF | `.pdf` | ✅ | Text and scanned documents |
| Microsoft Word | `.docx`, `.doc` | ❌ | Native text extraction |
| Plain Text | `.txt` | ❌ | Direct processing |
| Markdown | `.md` | ❌ | Formatted text support |

## 🖥️ Interface Components

### Main Tabs

#### 📄 Single Assignment Tab
1. **Upload Section**: Drag & drop or click to upload files
2. **Processing Options**: Checkboxes for enabling/disabling features
3. **Results Panel**: Real-time status, summary, and detailed results
4. **Download**: ZIP file with JSON results and summary

#### 📚 Batch Processing Tab
1. **Multi-Upload**: Select multiple files for batch processing
2. **Batch Options**: Apply same settings to all files
3. **Results Table**: HTML table with processing summary
4. **Batch Download**: CSV file with batch results

#### ❓ Help & Documentation Tab
- Comprehensive usage instructions
- Feature explanations
- Troubleshooting guide
- System architecture overview

### Status Indicators
- 🔗 **LLM Providers**: Shows healthy vs total providers
- 📄 **File Processor**: Processing engine status
- 🔧 **Workflow Engine**: Agentic workflow status

## 🔧 Technical Details

### Architecture
- **Frontend**: Gradio web interface
- **Backend**: RT-Capstone agentic workflow system
- **Processing**: Multi-LLM provider system with automatic failover
- **Storage**: Temporary file handling with automatic cleanup

### Performance
- **Single File**: ~30-60 seconds per assignment
- **Batch Processing**: Parallel processing with rate limit handling
- **File Size**: Supports files up to 100MB
- **Concurrent Users**: Designed for multiple simultaneous users

### Security
- **File Validation**: Automatic format and content validation
- **Temporary Storage**: Secure temporary file handling
- **Error Handling**: Comprehensive error catching and reporting
- **Data Privacy**: No permanent storage of user data

## 🛠️ Development

### Project Structure
```
src/
├── gradio_app.py          # Main Gradio application
├── workflows/             # Agentic workflow system
├── core/                  # Core processing components
├── processors/            # Subject-specific processors
└── support/               # Support utilities

launch_gradio.py           # Simple launcher script
GRADIO_README.md          # This documentation
```

### Key Components
- **GradioAssignmentGrader**: Main interface class
- **File Processing**: Multi-format file handling
- **Async Processing**: Agentic workflow integration
- **Results Formatting**: Human-readable output generation
- **Download Management**: File creation and cleanup

### Customization
The interface can be customized by modifying:
- **Processing Options**: Add/remove feature checkboxes
- **Styling**: Update CSS in `custom_css` variable
- **File Types**: Modify `file_types` in upload components
- **Result Display**: Customize `_format_results()` method

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure you're in the correct directory
cd /path/to/RT-Capstone

# Verify Python path
python -c "import sys; print(sys.path)"

# Reinstall dependencies
pip install -r requirements.txt
```

#### File Upload Issues
- **Large Files**: Check file size limits
- **Unsupported Formats**: Verify file extension
- **Corrupted Files**: Try re-saving the document

#### Processing Errors
- **LLM Failures**: System automatically switches providers
- **Rate Limits**: Built-in retry logic handles this
- **Content Issues**: Check file has readable text

#### Performance Issues
- **Slow Processing**: Normal for complex assignments
- **Memory Usage**: Large batches may require more RAM
- **Network Issues**: LLM providers require internet connection

### Debug Mode
Enable debug mode by modifying `launch()` call in `gradio_app.py`:
```python
interface.launch(debug=True, show_error=True)
```

## 📞 Support

For issues specific to the Gradio interface:
1. Check this documentation
2. Review error messages in the interface
3. Check console output for detailed error logs
4. Verify all dependencies are installed correctly

For system-level issues, refer to the main project documentation.

## 🔄 Updates

The Gradio interface automatically reflects updates to:
- Core processing system
- Subject-specific processors
- LLM provider configurations
- File processing capabilities

No additional configuration needed when updating the backend system.

---

**🎯 Ready to grade assignments with AI? Launch the interface and start processing!**