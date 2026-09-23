"""CLI-проверка протокола из JSON-файла.

Пример:
    py tools/evaluate_demo.py samples/expected/demo_protocol.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.quality.validator import format_report, validate_protocol  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("Использование: py tools/evaluate_demo.py path/to/protocol.json")
        return 2
    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    report = validate_protocol(data.get("transcript", ""), data.get("tasks", []), data.get("summary", ""))
    print(format_report(report))
    return 0 if report["status"] == "Готово к экспорту" else 1


if __name__ == "__main__":
    raise SystemExit(main())
