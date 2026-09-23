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
st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 1.6rem; padding-bottom: 3rem;}
.hero {position: relative; overflow: hidden; padding: 2rem 2.2rem; border-radius: 24px; background: radial-gradient(circle at 90% 15%, rgba(96,165,250,.42), transparent 32%), linear-gradient(135deg,#111827 0%,#172554 48%,#2563eb 100%); color: white; margin-bottom: 1.4rem; box-shadow: 0 18px 45px rgba(15,23,42,.28);}
.hero h1 {margin: 0; font-size: 2.55rem; letter-spacing: -0.045em;}
.hero p {max-width: 650px; margin: .6rem 0 0; color: #dbeafe; font-size: 1.05rem; line-height: 1.55;}
.eyebrow {font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; color: #bfdbfe; font-weight: 700; margin-bottom: .65rem;}
.privacy {display: inline-block; margin-top: 1.15rem; padding: .42rem .7rem; border: 1px solid rgba(191,219,254,.35); border-radius: 999px; color: #eff6ff; font-size: .82rem; background: rgba(15,23,42,.2);}
.upload-card {padding: 1.25rem 1.35rem .7rem; margin: 1rem 0 1.4rem; border: 1px solid rgba(148,163,184,.25); border-radius: 18px; background: rgba(30,41,59,.3);}
.empty-title {font-size: 1.45rem; font-weight: 700; margin: .35rem 0 .25rem;}
.empty-subtitle {color: #94a3b8; margin-bottom: 1.15rem;}
.step-card {padding: 1rem; min-height: 105px; border: 1px solid rgba(148,163,184,.22); border-radius: 14px; background: rgba(30,41,59,.32);}
.step-number {color: #93c5fd; font-size: .78rem; font-weight: 800; letter-spacing: .08em;}
.step-card strong {display: block; margin: .35rem 0; color: #f8fafc;}
.step-card span {color: #94a3b8; font-size: .88rem;}
[data-testid="stMetric"] {padding: 1rem 1.1rem; border: 1px solid rgba(148,163,184,.2); border-radius: 16px; background: rgba(30,41,59,.32);}
[data-testid="stMetricLabel"] {color: #94a3b8;}
[data-testid="stMetricValue"] {color: #f8fafc;}
.stButton > button, .stDownloadButton > button {border-radius: 10px; font-weight: 650; min-height: 2.6rem;}
</style>
<div class="hero"><div class="eyebrow">AI meeting intelligence</div><h1>MeetingMind</h1><p>Автоматический протокол совещания: речь, саммари и поручения в одном месте.</p><div class="privacy">🔒 Локальная обработка · данные не покидают ваш компьютер</div></div>
""", unsafe_allow_html=True)
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
    st.markdown("## 📝 MeetingMind")
    st.caption("Автопротоколирование совещаний")
    st.divider()
    st.subheader("Настройки")
    model_size = st.selectbox("Размер локальной модели", ["small", "medium"], index=0)
    demo_mode = st.checkbox("🎭 Демо-режим: разделить диалог на 2 спикеров по паузам", help="Только визуальная демонстрация. Не является настоящей диаризацией.")
    st.divider()
    st.markdown("**Как это работает**")
    st.markdown("1. Уведомьте участников и загрузите запись\n2. Создайте протокол\n3. Назначьте имена и проверьте поручения\n4. Сформируйте уведомления и скачайте протокол")

st.markdown('<div class="upload-card">', unsafe_allow_html=True)
st.markdown('<div class="empty-title">Загрузите запись совещания</div><div class="empty-subtitle">MP3, WAV, M4A или MP4 · обработка выполняется локально</div>', unsafe_allow_html=True)
audio = st.file_uploader("Выберите аудиофайл", type=["mp3", "wav", "m4a", "mp4"], help="Поддерживаются MP3, WAV, M4A и MP4", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)
consent = st.checkbox("Участники уведомлены о записи и локальной ИИ-транскрибации")
if audio:
    st.audio(audio)
    st.info(f"Файл готов к обработке: **{audio.name}** · {audio.size / 1024 / 1024:.1f} МБ")
elif not st.session_state.get("result"):
    st.markdown("<div class='empty-title'>От аудио к готовому протоколу</div><div class='empty-subtitle'>Три шага до результата</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, number, title, text in [(c1, "01", "Загрузите запись", "MP3, WAV, M4A или MP4"), (c2, "02", "Проверьте результат", "Транскрипт, саммари и поручения"), (c3, "03", "Скачайте протокол", "DOCX, PDF или JSON")]:
        with col:
            st.markdown(f"<div class='step-card'><div class='step-number'>{number}</div><strong>{title}</strong><span>{text}</span></div>", unsafe_allow_html=True)

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
