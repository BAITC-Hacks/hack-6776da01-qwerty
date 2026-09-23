from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = "Участник"


def transcribe_audio(audio_path: str | Path, model_size: str = "small") -> list[Segment]:
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), language=None, vad_filter=True, beam_size=5)
    return [Segment(float(s.start), float(s.end), s.text.strip()) for s in segments]


def segments_to_text(segments: list[Segment]) -> str:
    return "\n".join(f"[{s.start:06.1f}-{s.end:06.1f}] {s.speaker}: {s.text}" for s in segments)
