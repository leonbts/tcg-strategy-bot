from pathlib import Path
from typing import Literal

import whisper


ModelSize = Literal["tiny", "base", "small", "medium", "large"]


class WhisperTranscriber:
    """
    Simple wrapper around open-source Whisper for speech-to-text.
    Loads the model once and reuses it.
    """

    def __init__(self, model_size: ModelSize = "base", device: str | None = None):
        """
        model_size: one of "tiny", "base", "small", "medium", "large"
        device: "cpu" or "cuda". If None, Whisper auto-selects.
        """
        print(f"Loading Whisper model: {model_size} (this may take a moment the first time)...")
        self.model = whisper.load_model(model_size, device=device)

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file to text.
        Supports common formats: .wav, .mp3, .m4a, .flac, .ogg, etc.
        """
        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Transcribing audio: {audio_path}")
        result = self.model.transcribe(str(p))
        text = (result.get("text") or "").strip()
        print(f"Transcription: {text!r}")
        return text


def main():
    print("Whisper speech-to-text demo")
    audio_path = input("Path to audio file: ").strip()
    if not audio_path:
        print("No path provided, exiting.")
        return

    transcriber = WhisperTranscriber(model_size="small")  # tweak size if needed
    text = transcriber.transcribe(audio_path)
    print("\n=== FINAL TRANSCRIPT ===")
    print(text)


if __name__ == "__main__":
    main()