from gtts import gTTS
import openai
import os
import tempfile
from pathlib import Path
from datetime import datetime


class AudioProcessor:
    def __init__(self):
        # Static directory: project_root/static
        self.static_dir = Path(__file__).parent.parent / "static"
        self.static_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = None

    def text_to_speech(self, text: str, language: str = "en") -> str | None:
        """Convert text to speech using gTTS and return the file path."""
        if not text:
            print("Error: Empty text provided")
            return None

        try:
            # Generate a unique filename using timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path = self.static_dir / f"output_{timestamp}.mp3"

            # Generate and save the audio
            tts = gTTS(text=text, lang=language)
            tts.save(str(output_path))

            self.output_path = output_path
            return str(output_path)
        except Exception as exc:
            print(f"text_to_speech error: {exc}")
            return None

    def transcribe_audio(self, audio_file: str) -> str | None:
        """Transcribe a local audio file (wav/mp3) to text with OpenAI Whisper."""
        try:
            with open(audio_file, "rb") as f:
                resp = openai.Audio.transcribe("whisper-1", f)
            return resp.get("text", "")
        except Exception as exc:
            print(f"transcribe_audio error: {exc}")
            return None

    def save_uploaded_file(self, file) -> str | None:
        """
        Save an uploaded FileStorage (e.g., from Flask) to a temp .wav file
        and return its path.
        """
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(file.read())
            tmp.close()
            return tmp.name
        except Exception as exc:
            print(f"save_uploaded_file error: {exc}")
            return None
