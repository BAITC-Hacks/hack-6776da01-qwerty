"""Проверка результата протоколирования без внешних сервисов."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _speaker_count(transcript: str) -> int:
    speakers = set()
    for line in transcript.splitlines():
        if ":" in line:
            prefix = line.split(":", 1)[0]
            if "]" in prefix:
                speakers.add(prefix.rsplit("]", 1)[-1].strip())
    return len(speakers)


def validate_protocol(transcript: str, tasks: Iterable[dict[str, Any]], summary: str = "") -> dict[str, Any]:
    """Возвращает понятный отчёт качества для UI, тестов и демо."""
    task_list = list(tasks)
    with_responsible = sum(bool(item.get("responsible") and item["responsible"] != "Не определён") for item in task_list)
    with_deadline = sum(bool(item.get("deadline_iso") or (item.get("deadline") and item["deadline"] != "Не указан")) for item in task_list)
    warnings: list[str] = []
    if not transcript.strip():
        warnings.append("Транскрипт пустой")
    if not summary.strip():
        warnings.append("Саммари отсутствует")
    if task_list and with_responsible < len(task_list):
        warnings.append("Есть поручения без ответственного")
    if task_list and with_deadline < len(task_list):
        warnings.append("Есть поручения без срока")
    if _speaker_count(transcript) <= 1:
        warnings.append("Определён только один говорящий или диаризация не сработала")
    score_parts = [bool(transcript.strip()), bool(summary.strip()), bool(task_list or "поруч" not in transcript.lower()), with_responsible == len(task_list) if task_list else True, with_deadline == len(task_list) if task_list else True]
    return {
        "transcript_chars": len(transcript.strip()),
        "speech_segments": sum(bool(line.strip()) for line in transcript.splitlines()),
        "speakers": _speaker_count(transcript),
        "tasks": len(task_list),
        "tasks_with_responsible": with_responsible,
        "tasks_with_deadline": with_deadline,
        "responsible_coverage": round(with_responsible / len(task_list), 2) if task_list else 1.0,
        "deadline_coverage": round(with_deadline / len(task_list), 2) if task_list else 1.0,
        "quality_score": round(sum(score_parts) / len(score_parts) * 100),
        "warnings": warnings,
        "status": "Требует проверки" if warnings else "Готово к экспорту",
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [
        "Отчёт качества протокола",
        f"Статус: {report['status']}",
        f"Оценка: {report['quality_score']}/100",
        f"Фрагментов речи: {report['speech_segments']}",
        f"Говорящих: {report['speakers']}",
        f"Поручений: {report['tasks']}",
        f"С ответственным: {report['tasks_with_responsible']}/{report['tasks']}",
        f"Со сроком: {report['tasks_with_deadline']}/{report['tasks']}",
    ]
    if report["warnings"]:
        lines.append("Предупреждения:")
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines)
