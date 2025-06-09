# AI Visual Assistant

A Python-based AI assistant that:
1. Detects objects in webcam frames using YOLOv8
2. Reads text in the image using Tesseract OCR
3. Converts scene summary to speech using gTTS
4. Provides API endpoints using FastAPI

## Requirements
- Python 3.8+
- Tesseract OCR
- OpenAI API key

## Installation

1. Clone the repository
```bash
git clone https://github.com/PalakKhem16/AI_Visual_Assistant.git
cd AI_Visual_Assistant
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
- Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- Add Tesseract to PATH

4. Set up environment variables:
Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key
```

## Usage

1. Start the server:
```bash
python main.py
```

2. Test image processing:
```bash
python test_request.py
```

## API Endpoints

### POST /process-frame/
Process an image and get object detection results with audio description.

### POST /voice-command/
Process voice commands and get spoken responses.

## Project Structure
- `main.py`: FastAPI application
- `utils/`:
  - `vision.py`: YOLO and OCR processing
  - `audio.py`: Text-to-speech and speech-to-text
  - `nlp.py`: GPT-4 integration
- `static/`: Generated audio files
- `test_request.py`: Test script for image processing
