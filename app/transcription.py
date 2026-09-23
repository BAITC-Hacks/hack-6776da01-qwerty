"""Локальная мультиязычная транскрибация и определение говорящих.

Совместимость с текущим интерфейсом сохраняется: ``app.main`` по-прежнему
вызывает ``transcribe_audio`` и ``segments_to_text``. Аудио никогда не
отправляется во внешний API.

Два режима определения говорящих:
1. pyannote.audio, если он установлен и доступен локальный checkpoint;
2. безопасный fallback по паузам между репликами (SPEAKER_00/01...).
Fallback не претендует на полноценную биометрическую диаризацию, но лучше
отделяет очереди реплик, чем общий ярлык «Участник», и работает без токенов.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel


LOGGER = logging.getLogger(__name__)


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = "Участник"


def _language(value: str | None) -> str | None:
    if not value or value.lower() == "auto":
        return None
    return {"rus": "ru", "kaz": "kk", "kz": "kk"}.get(value.lower(), value.lower())


@lru_cache(maxsize=3)
def _get_model(model_size: str, device: str, compute_type: str, model_dir: str) -> WhisperModel:
    """Кэшируем модель, чтобы Streamlit не загружал её на каждый клик."""
    return WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=model_dir,
    )


def _pyannote_annotation(audio_path: Path, device: str) -> Any | None:
    """Запускает локальную pyannote-диаризацию, если она настроена."""
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        return None

    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    model_name = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")
    local_model = Path(model_name).exists()
    if not token and not local_model:
        LOGGER.info("pyannote установлен, но локальный checkpoint/HF_TOKEN не задан; используем fallback")
        return None

    try:
        pipeline = Pipeline.from_pretrained(
            model_name,
            use_auth_token=token,
        )
        if device == "cuda":
            import torch

            pipeline.to(torch.device("cuda"))
        return pipeline(str(audio_path))
    except Exception as exc:
        LOGGER.warning("Локальная pyannote-диаризация недоступна: %s", exc)
        return None


def _speaker_from_annotation(segment: Segment, annotation: Any) -> str | None:
    if annotation is None:
        return None
    overlaps: list[tuple[float, str]] = []
    try:
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            overlap = max(0.0, min(segment.end, turn.end) - max(segment.start, turn.start))
            if overlap:
                overlaps.append((overlap, str(speaker)))
    except Exception as exc:
        LOGGER.debug("Не удалось сопоставить реплику и speaker turn: %s", exc)
        return None
    return max(overlaps, key=lambda item: item[0])[1] if overlaps else None


def _heuristic_speakers(segments: list[Segment], pause_seconds: float = 0.85) -> None:
    """Fallback без сторонних моделей: смена очереди реплик по паузам."""
    if not segments:
        return
    speaker_index = 0
    segments[0].speaker = "SPEAKER_00"
    for previous, current in zip(segments, segments[1:]):
        pause = max(0.0, current.start - previous.end)
        if pause >= pause_seconds:
            speaker_index = (speaker_index + 1) % 2
        current.speaker = f"SPEAKER_{speaker_index:02d}"


def transcribe_audio(
    audio_path: str | Path,
    model_size: str = "small",
    diarization: bool = True,
    language: str | None = None,
) -> list[Segment]:
    """Распознать MP3/WAV/M4A локально и вернуть совместимые сегменты."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(path)

    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8" if device == "cpu" else "float16")
    model_dir = os.getenv("WHISPER_MODEL_DIR", "models")
    model = _get_model(model_size, device, compute_type, model_dir)
    options: dict[str, Any] = {
        "language": _language(language),
        "vad_filter": True,
        "beam_size": 5,
        "condition_on_previous_text": True,
        "temperature": 0.0,
    }
    if options["language"] is None:
        options.pop("language")

    raw_segments, info = model.transcribe(str(path), **options)
    segments = [
        Segment(float(item.start), float(item.end), item.text.strip())
        for item in raw_segments
        if item.text and item.text.strip()
    ]
    LOGGER.info(
        "Распознано %s сегментов, язык=%s (%.2f)",
        len(segments),
        getattr(info, "language", "unknown"),
        float(getattr(info, "language_probability", 0.0)),
    )

    _heuristic_speakers(segments)
    if diarization:
        annotation = _pyannote_annotation(path, device)
        if annotation is not None:
            for segment in segments:
                segment.speaker = _speaker_from_annotation(segment, annotation) or segment.speaker
    return segments


def segments_to_text(segments: list[Segment]) -> str:
    """Стабильный текстовый формат для analyzer.py и экспорта в DOCX."""
    return "\n".join(
        f"[{segment.start:06.1f}-{segment.end:06.1f}] "
        f"{segment.speaker}: {segment.text}"
        for segment in segments
    )
