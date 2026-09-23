import re


DEADLINE_PATTERNS = (r"до конца недели", r"до конца месяца", r"до пятницы", r"на следующей неделе", r"на этой неделе", r"текущей недели", r"через две недели", r"к?\s*\d{1,2}\s*(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)")


def _deadline(text: str) -> str:
    for pattern in DEADLINE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return "Не указан"


def extract_tasks(transcript: str) -> list[dict[str, str]]:
    tasks = []
    action_words = ("подготов", "провест", "организ", "найд", "провер", "согласов", "разобрат", "предостав", "представ", "разработ", "зафиксир", "запрос", "сдел")
    for line in (line.strip() for line in transcript.splitlines() if line.strip()):
        if not any(word in line.lower() for word in action_words):
            continue
        speaker_match = re.search(r"\]\s*([^:]+):", line)
        speaker = speaker_match.group(1).strip() if speaker_match else "Не определён"
        clean = re.sub(r"^\[[^]]+\]\s*[^:]+:\s*", "", line)
        tasks.append({"task": clean, "responsible": speaker, "deadline": _deadline(line)})
    return tasks


def make_summary(transcript: str, tasks: list[dict[str, str]]) -> str:
    lines = [line.strip() for line in transcript.splitlines() if line.strip()]
    if not lines:
        return "Текст совещания отсутствует."
    return f"Совещание содержит {len(lines)} фрагментов речи и {len(tasks)} потенциальных поручений. Краткое начало: {' '.join(lines[:3])[:700]}"
