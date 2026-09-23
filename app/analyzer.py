"""Локальное извлечение поручений из транскрипта совещания."""

from __future__ import annotations

import re


_DEADLINE_RE = re.compile(
    r"(?:до|к|на|через)\s+(?:конца\s+недели|конца\s+месяца|пятницы|среды|"
    r"следующей\s+неделе|этой\s+неделе|две\s+недели|"
    r"\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|"
    r"сентября|октября|ноября|декабря)|пятнадцатого\s+октября|"
    r"двадцатого\s+октября)", re.IGNORECASE)
_EXPLICIT_OWNER_RE = re.compile(
    r"(?:ответственный|ответственная)\s+(.+?)(?:,?\s+срок\s+)(.+?)(?:[.]|$)", re.IGNORECASE)
_SPEAKER_RE = re.compile(r"^\[[^]]+\]\s*([^:]+):\s*(.*)$")
_NAME_RE = re.compile(r"^((?:[А-ЯЁ][а-яё]+\s+){1,2}[А-ЯЁ][а-яё]+),\s*(.+)$")
_ACTION_RE = re.compile(
    r"\b(разработать|провести|подготовить|организовать|найти|проверить|"
    r"согласовать|разобраться|предоставить|представить|зафиксировать|"
    r"запросить|свяжитесь|проведите|проверьте|подготовьте|организуйте|"
    r"найдите|согласуйте|разберитесь)\b", re.IGNORECASE)


def _text_and_speaker(line: str) -> tuple[str, str]:
    match = _SPEAKER_RE.match(line.strip())
    if match:
        return match.group(2).strip(), match.group(1).strip()
    return line.strip(), "Не определён"


def _deadline(text: str) -> str:
    match = _DEADLINE_RE.search(text)
    return match.group(0).strip() if match else "Не указан"


def _clean_task(text: str) -> str:
    text = re.sub(r"^[0-9]+[.]\s*", "", text.strip())
    text = re.sub(r"^(?:первое|второе|третье|четвёртое|четвертое|пятое)\s*[.:,-]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r",?\s*(?:ответственный|ответственная)\s+.+?(?:,?\s+срок\s+).+?[.]?$", "", text, flags=re.IGNORECASE)
    text = re.sub(r",?\s*(?:срок|до|к)\s+(?:.+)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .,-")
    return text[:1].upper() + text[1:] if text and text[0].islower() else text


def _owner_and_task(text: str, speaker: str) -> tuple[str, str]:
    explicit = _EXPLICIT_OWNER_RE.search(text)
    if explicit:
        return explicit.group(1).strip(" ,."), _clean_task(text[: explicit.start()])
    generic = re.search(r",\s*([^,]+?)\s*,?\s*срок\s+", text, re.IGNORECASE)
    if generic:
        return generic.group(1).strip(" ,."), _clean_task(text[: generic.start()])
    addressed = _NAME_RE.match(text)
    if addressed:
        return addressed.group(1).strip(), _clean_task(addressed.group(2))
    return speaker, _clean_task(text)


def extract_tasks(transcript: str) -> list[dict[str, str]]:
    """Извлекает вероятные поручения и удаляет повторы."""
    tasks: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_line in transcript.splitlines():
        if not raw_line.strip():
            continue
        text, speaker = _text_and_speaker(raw_line)
        has_explicit_owner = bool(_EXPLICIT_OWNER_RE.search(text))
        has_deadline = bool(_DEADLINE_RE.search(text))
        if not _ACTION_RE.search(text) or (not has_deadline and not has_explicit_owner):
            continue
        owner, task_text = _owner_and_task(text, speaker)
        if len(task_text) < 12:
            continue
        key = (re.sub(r"\W", "", task_text.lower()), owner.lower())
        if key in seen:
            continue
        seen.add(key)
        tasks.append({"task": task_text, "responsible": owner or "Не определён", "deadline": _deadline(text)})
    return tasks


def make_summary(transcript: str, tasks: list[dict[str, str]]) -> str:
    lines = [line.strip() for line in transcript.splitlines() if line.strip()]
    if not lines:
        return "Текст совещания отсутствует."
    speakers = sorted({_text_and_speaker(line)[1] for line in lines})
    return (
        f"В совещании распознано {len(lines)} фрагментов речи. "
        f"Участники: {', '.join(speakers[:5]) or 'не определены'}. "
        f"Выделено поручений: {len(tasks)}. "
        "Проверьте сроки и ответственных перед отправкой протокола."
    )
