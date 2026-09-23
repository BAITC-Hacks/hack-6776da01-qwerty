"""Анонимизирует текстовый транскрипт для безопасной демонстрации."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.privacy.anonymizer import anonymize_text  # noqa: E402


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("Использование: py tools/anonymize_transcript.py input.txt [output.txt]")
        return 2
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2]) if len(sys.argv) == 3 else source.with_name(f"{source.stem}.anonymous{source.suffix}")
    anonymized, counts = anonymize_text(source.read_text(encoding="utf-8"))
    destination.write_text(anonymized, encoding="utf-8")
    print(f"Сохранено: {destination}")
    print("Заменено: " + ", ".join(f"{key}={value}" for key, value in counts.items() if value) or "ничего")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
