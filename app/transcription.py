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
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel


LOGGER = logging.getLogger(__name__)

NAME_PATTERNS = (
    ("Асхат Ерланович", re.compile(r"\bАсхат\s+Ерланович\b", re.IGNORECASE)),
    ("Гульмира Сериковна", re.compile(r"\bГульмир[аеы]?\s+Сер[ие]ковн[аы]\b", re.IGNORECASE)),
    ("Тимур Булатович", re.compile(r"\bТимур\s+Б[ау]латович\b", re.IGNORECASE)),
    ("Айнур Каировна", re.compile(r"\bАйн[уо]р\s+Каировн[аы]\b", re.IGNORECASE)),
    ("Нурлан Сагатович", re.compile(r"\bНурлан\s+С[ао]гатович\b", re.IGNORECASE)),
)
RESPONSE_CUES = re.compile(
    r"вам слово|что у вас|что предлагаете|как вы можете|ситуация такая же|"
    r"когда последний|подскажите|можно добавить|а по обучению|у нас же",
    re.IGNORECASE,
)
CHAIR_CUES = re.compile(
    r"^\s*(понятно|согласен|согласна|так,|это недопустимо|хороший вопрос|"
    r"отлично|фиксируем|значит так|переходим|подытожим|итого)",
    re.IGNORECASE,
)


class VoiceMemory:
    """Локальная долговременная память голосовых профилей."""

    def __init__(self, path: Path):
        self.path = path
        self.profiles: dict[str, dict[str, Any]] = {}
        if path.exists():
            try:
                import json

                loaded = json.loads(path.read_text(encoding="utf-8"))
                # Старый формат с одним усреднённым embedding был неточным;
                # не используем его после обновления памяти голосов.
                self.profiles = {
                    name: profile for name, profile in loaded.items()
                    if isinstance(profile, dict) and profile.get("embeddings")
                }
            except Exception as exc:
                LOGGER.warning("Не удалось прочитать память голосов: %s", exc)

    @staticmethod
    def _normalise(vector):
        import numpy as np

        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def enroll(self, name: str, vector) -> None:
        vector = self._normalise(vector)
        old = self.profiles.get(name)
        if old and old.get("embeddings"):
            embeddings = old["embeddings"]
        elif old and old.get("embedding"):
            embeddings = [old["embedding"]]
        else:
            embeddings = []
        embeddings.append(vector.tolist())
        # Не раздуваем локальный файл; храним несколько реальных эталонов,
        # чтобы один неудачный или короткий фрагмент не испортил профиль.
        self.profiles[name] = {"embeddings": embeddings[-8:], "samples": len(embeddings[-8:])}

    def match(self, vector, threshold: float = 0.62) -> str | None:
        import numpy as np

        vector = self._normalise(vector)
        candidates = []
        for name, profile in self.profiles.items():
            raw_embeddings = profile.get("embeddings") or [profile.get("embedding")]
            scores = [float(np.dot(vector, self._normalise(item))) for item in raw_embeddings if item]
            if scores:
                candidates.append((max(scores), name))
        if not candidates:
            return None
        score, name = max(candidates)
        threshold = float(os.getenv("VOICE_MATCH_THRESHOLD", str(threshold)))
        LOGGER.debug("Голос сравнен с %s: %.3f", name, score)
        return name if score >= threshold else None

    def save(self) -> None:
        import json

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.profiles, ensure_ascii=False, indent=2), encoding="utf-8")


@lru_cache(maxsize=1)
def _voice_encoder():
    """Загрузить локальный ECAPA encoder; модель скачивается только при первом запуске."""
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        try:
            from speechbrain.pretrained import EncoderClassifier
        except ImportError:
            LOGGER.info("SpeechBrain не установлен; память голосов отключена")
            return None
    try:
        device = os.getenv("WHISPER_DEVICE", "cpu")
        return EncoderClassifier.from_hparams(
            source=os.getenv("VOICE_MODEL", "speechbrain/spkrec-ecapa-voxceleb"),
            savedir=os.getenv("VOICE_MODEL_DIR", "models/voice-ecapa"),
            run_opts={"device": device},
        )
    except Exception as exc:
        LOGGER.warning("Не удалось загрузить модель памяти голосов: %s", exc)
        return None


def _voice_embedding(samples, start: float, end: float, encoder, sample_rate: int = 16000):
    if encoder is None:
        return None
    try:
        import numpy as np
        import torch

        clip = samples[int(start * sample_rate):int(end * sample_rate)]
        if len(clip) < int(sample_rate * 1.2):
            return None
        waveform = torch.from_numpy(np.asarray(clip, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            embedding = encoder.encode_batch(waveform).squeeze().detach().cpu().numpy()
        return VoiceMemory._normalise(embedding)
    except Exception as exc:
        LOGGER.debug("Не удалось получить voice embedding: %s", exc)
        return None


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


def _audio_samples(audio_path: Path, sample_rate: int = 16000):
    """Decode audio locally through PyAV; no ffmpeg process or cloud API."""
    import av
    import numpy as np

    container = av.open(str(audio_path))
    stream = container.streams.audio[0]
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    chunks = []
    for frame in container.decode(stream):
        converted = resampler.resample(frame)
        if not isinstance(converted, list):
            converted = [converted]
        for item in converted:
            chunks.append(item.to_ndarray().reshape(-1))
    container.close()
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _acoustic_embedding(samples, start: float, end: float, sample_rate: int = 16000):
    """Compact voice-timbre features for a local, dependency-light fallback."""
    import numpy as np

    clip = samples[int(start * sample_rate):int(end * sample_rate)]
    if len(clip) < sample_rate // 5:
        return None
    frame_size = int(sample_rate * 0.025)
    hop = int(sample_rate * 0.010)
    frames = []
    window = np.hanning(frame_size)
    for pos in range(0, max(1, len(clip) - frame_size), hop):
        frame = clip[pos:pos + frame_size]
        if len(frame) == frame_size:
            spectrum = np.abs(np.fft.rfft(frame * window)) + 1e-7
            power = spectrum ** 2
            freqs = np.fft.rfftfreq(frame_size, 1 / sample_rate)
            bands = [(80, 300), (300, 1000), (1000, 3000), (3000, 7000)]
            band_energy = [float(power[(freqs >= low) & (freqs < high)].mean()) for low, high in bands]
            zcr = float(np.mean(np.abs(np.diff(np.signbit(frame)))))
            centroid = float((freqs * power).sum() / power.sum())
            frames.append([float(np.log(power.mean())), centroid / 4000.0, zcr, *np.log1p(band_energy)])
    if not frames:
        return None
    return np.asarray(frames, dtype=np.float32).mean(axis=0)


def _acoustic_speakers(audio_path: Path, segments: list[Segment]) -> bool:
    """Cluster segment voice features into two local speaker tracks.

    This is a fallback, not speaker identification. It is useful when pyannote
    is unavailable and avoids the old all-``SPEAKER_00`` output.
    """
    if len(segments) < 4:
        return False
    try:
        import numpy as np
    except ImportError:
        LOGGER.warning("numpy unavailable; speaker labels remain generic")
        return False
    try:
        samples = _audio_samples(audio_path)
        vectors = [_acoustic_embedding(samples, item.start, item.end) for item in segments]
        valid = [(index, vector) for index, vector in enumerate(vectors) if vector is not None]
        if len(valid) < 4:
            return False
        matrix = np.asarray([vector for _, vector in valid])
        matrix = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-6)
        # Deterministic two-cluster k-means keeps the fallback lightweight.
        centers = matrix[[0, int(np.argmax(np.sum((matrix - matrix[0]) ** 2, axis=1)))]]
        for _ in range(30):
            distances = ((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            new_centers = np.asarray(
                [matrix[labels == cluster].mean(axis=0) if np.any(labels == cluster) else centers[cluster]
                 for cluster in range(2)]
            )
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        counts = np.bincount(labels, minlength=2)
        if min(counts) < 2:
            return False
        first_seen = {int(label): position for position, label in enumerate(labels)}
        order = sorted(first_seen, key=first_seen.get)
        remap = {old: new for new, old in enumerate(order)}
        for (index, _), label in zip(valid, labels):
            segments[index].speaker = f"SPEAKER_{remap[int(label)]:02d}"
        return True
    except Exception as exc:
        LOGGER.warning("Акустическая диаризация недоступна: %s", exc)
        return False


def _name_in_text(text: str) -> tuple[str, re.Match[str]] | None:
    for canonical, pattern in NAME_PATTERNS:
        match = pattern.search(text)
        if match:
            before = text[max(0, match.start() - 28):match.start()].lower()
            if "ответственн" in before:
                continue
            return canonical, match
    # Для новых участников, которых нет в словаре: имя + отчество.
    generic = re.search(
        r"\b([А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{4,}(?:ович|евич|овна|евна|ична))\b",
        text,
    )
    if generic:
        before = text[max(0, generic.start() - 28):generic.start()].lower()
        if "ответственн" not in before:
            return generic.group(1), generic
    return None


def _apply_name_hints(segments: list[Segment]) -> None:
    """Привязывает ответную реплику к имени после прямого обращения."""
    pending_name: str | None = None
    pending_turns = 0
    for index, segment in enumerate(segments):
        text = segment.text
        found = _name_in_text(text)
        is_response_request = bool(found and RESPONSE_CUES.search(text))
        is_chair_line = bool(CHAIR_CUES.search(text))

        if pending_name and index > 0 and not is_chair_line:
            segment.speaker = pending_name
            pending_turns -= 1
            if pending_turns <= 0:
                pending_name = None

        if is_response_request and found:
            pending_name = found[0]
            pending_turns = 2
            if segment.speaker.startswith("SPEAKER_") or segment.speaker == "Участник":
                segment.speaker = "Председатель"

        if pending_name and index + 1 < len(segments):
            if CHAIR_CUES.search(segments[index + 1].text):
                pending_name = None
                pending_turns = 0


def _apply_voice_memory(audio_path: Path, segments: list[Segment]) -> bool:
    """Обучить профили по именованным репликам и распознать последующие."""
    encoder = _voice_encoder()
    if encoder is None:
        # Не подменяем voice embeddings грубыми спектральными признаками:
        # это может ошибочно подписать все реплики одним человеком.
        LOGGER.warning("Память голосов не активна: установите speechbrain")
        return False
    try:
        samples = _audio_samples(audio_path)
    except Exception as exc:
        LOGGER.warning("Не удалось подготовить аудио для памяти голосов: %s", exc)
        return False

    memory = VoiceMemory(Path(os.getenv("VOICE_PROFILES_PATH", "data/voice_profiles.json")))
    embeddings = [_voice_embedding(samples, item.start, item.end, encoder) for item in segments]
    named = {item.speaker for item in segments if item.speaker not in {"SPEAKER_00", "SPEAKER_01", "Председатель"}}
    enrolled = 0
    for item, embedding in zip(segments, embeddings):
        if embedding is not None and item.speaker in named:
            memory.enroll(item.speaker, embedding)
            enrolled += 1

    matched = 0
    for item, embedding in zip(segments, embeddings):
        if embedding is None or item.speaker in named or item.speaker == "Председатель":
            continue
        name = memory.match(embedding)
        if name:
            item.speaker = name
            matched += 1
    if enrolled or matched:
        memory.save()
    LOGGER.info("Память голосов: профилей=%s, обучено=%s, сопоставлено=%s", len(memory.profiles), enrolled, matched)
    return bool(memory.profiles)


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
        "vad_parameters": {"min_silence_duration_ms": 750, "speech_pad_ms": 300},
        "beam_size": 5,
        # Сохраняем контекст: это важно для поручений, сроков и имён.
        "condition_on_previous_text": True,
        "temperature": 0.0,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.35,
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

    diarization_used = False
    if diarization:
        annotation = _pyannote_annotation(path, device)
        if annotation is not None:
            for segment in segments:
                segment.speaker = _speaker_from_annotation(segment, annotation) or "SPEAKER_00"
            diarization_used = True
        elif _voice_encoder() is not None:
            # Кластеризация используется только вместе с voice encoder;
            # без него она давала ложные смены говорящих и ухудшала протокол.
            diarization_used = _acoustic_speakers(path, segments)
        else:
            LOGGER.warning("SpeechBrain не загружен: отключена эвристическая диаризация")
    if not diarization_used:
        for segment in segments:
            segment.speaker = "SPEAKER_00"
        LOGGER.warning("Настоящая диаризация не активна; все реплики помечены SPEAKER_00")
    _apply_name_hints(segments)
    _apply_voice_memory(path, segments)
    return segments


def segments_to_text(segments: list[Segment]) -> str:
    """Стабильный текстовый формат для analyzer.py и экспорта в DOCX."""
    return "\n".join(
        f"[{segment.start:06.1f}-{segment.end:06.1f}] "
        f"{segment.speaker}: {segment.text}"
        for segment in segments
    )
