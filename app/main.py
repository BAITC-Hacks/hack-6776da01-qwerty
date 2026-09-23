import json
import re
import tempfile
from datetime import date
from pathlib import Path

import streamlit as st

from analyzer import extract_tasks, make_summary
from exporter import make_docx
from pdf_exporter import make_pdf
from quality import validate_protocol
from transcription import segments_to_text, transcribe_audio

st.set_page_config(page_title="MeetingMind", page_icon="📝", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>.block-container{max-width:1180px;padding-top:2rem}.hero{padding:1.4rem 1.6rem;border-radius:18px;background:linear-gradient(135deg,#172554,#2563eb);color:white;margin-bottom:1.2rem}.hero h1{margin:0;font-size:2.35rem}</style><div class="hero"><h1>MeetingMind</h1><p>Автоматический протокол совещания: речь, саммари и поручения в одном месте.</p></div>""", unsafe_allow_html=True)
st.info("🔔 Внимание: ведётся запись и ИИ-транскрибация совещания. Участники должны быть уведомлены. Обработка данных происходит строго локально (On-Premise) без передачи во внешние облачные API.")


def _speakers(transcript: str) -> list[str]:
    return sorted(set(re.findall(r"\]\s*([^:]+):", transcript)))


def _rename_speakers(transcript: str, mapping: dict[str, str]) -> str:
    for source, target in mapping.items():
        if target.strip():
            transcript = re.sub(rf"(?<=\])\s*{re.escape(source)}(?=:)", f" {target.strip()}", transcript)
    return transcript


def _demo_split(transcript: str) -> str:
    lines = [line for line in transcript.splitlines() if line.strip()]
    return "\n".join(re.sub(r"(\]\s*)[^:]+(?=:)", rf"\1SPEAKER_{index % 2:02d}", line) for index, line in enumerate(lines))


def _overdue(deadline: str) -> bool:
    matches = re.findall(r"(?<!\d)(\d{1,2})[./-](\d{1,2})(?:[./-](\d{2,4}))?(?!\d)", deadline)
    if not matches:
        return False
    day, month, year = matches[0]
    year = int(year or date.today().year)
    if year < 100:
        year += 2000
    try:
        return date(year, int(month), int(day)) < date.today()
    except ValueError:
        return False


with st.sidebar:
    st.subheader("Настройки")
    model_size = st.selectbox("Размер локальной модели", ["small", "medium"], index=0)
    demo_mode = st.checkbox("🎭 Демо-режим: разделить диалог на 2 спикеров по паузам", help="Только визуальная демонстрация. Не является настоящей диаризацией.")
    st.divider()
    st.markdown("**Как это работает**")
    st.markdown("1. Уведомьте участников и загрузите запись\n2. Создайте протокол\n3. Назначьте имена и проверьте поручения\n4. Сформируйте уведомления и скачайте протокол")

audio = st.file_uploader("Загрузите запись совещания", type=["mp3", "wav", "m4a", "mp4"], help="Поддерживаются MP3, WAV, M4A и MP4")
consent = st.checkbox("Участники уведомлены о записи и локальной ИИ-транскрибации")
if audio:
    st.audio(audio)
    st.info(f"Файл готов к обработке: **{audio.name}** · {audio.size / 1024 / 1024:.1f} МБ")

if audio and st.button("Создать протокол", type="primary"):
    if not consent:
        st.error("Подтвердите, что участники уведомлены о записи.")
        st.stop()
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio.name).suffix) as temp:
        temp.write(audio.getbuffer())
        audio_path = temp.name
    try:
        with st.spinner("Распознаю аудио локально. Первый запуск скачает модель..."):
            segments = transcribe_audio(audio_path, model_size)
        raw_transcript = segments_to_text(segments)
        if demo_mode and len(_speakers(raw_transcript)) <= 1:
            raw_transcript = _demo_split(raw_transcript)
        st.session_state["result"] = {"raw_transcript": raw_transcript}
    except Exception as exc:
        st.error(f"Не удалось обработать запись: {exc}")
    finally:
        Path(audio_path).unlink(missing_ok=True)

result = st.session_state.get("result")
if result:
    raw_transcript = result["raw_transcript"]
    raw_speakers = _speakers(raw_transcript)
    with st.sidebar:
        st.subheader("Сопоставление спикеров")
        st.caption("Введите ФИО, чтобы заменить технические метки во всём протоколе.")
        mapping = {speaker: st.text_input(speaker, key=f"mapping_{speaker}", placeholder="Например: Ахметова Айжан") for speaker in raw_speakers}

    transcript = _rename_speakers(raw_transcript, mapping)
    tasks = extract_tasks(transcript)
    summary = make_summary(transcript, tasks)
    quality = validate_protocol(transcript, tasks, summary)
    speakers = _speakers(transcript)
    st.success("Протокол сформирован. Проверьте имена, сроки и поручения перед отправкой коллегам.")
    left, middle, right = st.columns(3)
    left.metric("Фрагментов речи", len([x for x in transcript.splitlines() if x.strip()]))
    middle.metric("Поручений найдено", len(tasks))
    right.metric("Форматы экспорта", "DOCX · PDF · JSON")
    if len(raw_speakers) <= 1 and not demo_mode:
        st.warning("Обнаружен только один спикер. Для демонстрации можно включить демо-режим, но он не заменяет настоящую диаризацию.")
    if demo_mode and len(raw_speakers) > 1:
        st.warning("Включён демо-режим разделения по очереди сегментов; результат нельзя считать акустической диаризацией.")

    result_tab, transcript_tab, export_tab = st.tabs(["📌 Результат", "🎙️ Транскрипт", "📤 Экспорт"])
    with result_tab:
        st.subheader("Саммари")
        st.markdown(summary.replace("\n", "  \n"))
        st.subheader("Поручения и контроль сроков")
        if tasks:
            display_tasks = []
            for item in tasks:
                row = dict(item)
                row["status"] = "⚠️ Просрочено" if _overdue(item.get("deadline", "")) else "Не проверено"
                display_tasks.append(row)
            edited_tasks = st.data_editor(display_tasks, use_container_width=True, hide_index=True, column_config={
                "task": st.column_config.TextColumn("Поручение", width="large"),
                "responsible": st.column_config.TextColumn("Ответственный"),
                "deadline": st.column_config.TextColumn("Срок"),
                "status": st.column_config.SelectboxColumn("Статус", options=["Не проверено", "В работе", "Выполнено", "⚠️ Просрочено"]),
            })
        else:
            edited_tasks = []
            st.warning("Поручения не найдены. Проверьте транскрипт вручную.")
        st.subheader("Контроль качества")
        st.metric("Оценка протокола", f"{quality['quality_score']}/100")
        for warning in quality["warnings"]:
            st.warning(warning)
        if tasks and st.button("🔔 Сформировать уведомления ответственным"):
            st.subheader("Готовые напоминания")
            for item in edited_tasks:
                st.info(f"Уважаемый(ая) {item.get('responsible', 'коллега')}, напоминаем о поручении «{item.get('task', '')}» со сроком {item.get('deadline', 'не указан')}.")

    with transcript_tab:
        st.subheader("Полный транскрипт с именами")
        edited_transcript = st.text_area("Текст совещания", transcript, height=480, label_visibility="collapsed")
        if st.button("🔄 Пересчитать поручения из исправленного текста"):
            st.session_state["result"] = {"raw_transcript": edited_transcript}
            st.rerun()

    structured = {"summary": summary, "tasks": edited_tasks, "transcript": transcript, "speakers": speakers, "quality": quality}
    with export_tab:
        st.subheader("Скачать результат")
        export_col, pdf_col, json_col = st.columns(3)
        with export_col:
            st.download_button("⬇️ Скачать DOCX", data=make_docx(transcript, summary, edited_tasks), file_name="protocol.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", type="primary")
        with pdf_col:
            st.download_button("📄 Скачать PDF", data=make_pdf(transcript, summary, edited_tasks), file_name="protocol.pdf", mime="application/pdf")
        with json_col:
            st.download_button("↗️ Скачать JSON", data=json.dumps(structured, ensure_ascii=False, indent=2), file_name="protocol.json", mime="application/json")
