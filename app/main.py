import json
import re
import tempfile
from pathlib import Path

import streamlit as st

from analyzer import extract_tasks, make_summary
from exporter import make_docx
from pdf_exporter import make_pdf
from transcription import segments_to_text, transcribe_audio


st.set_page_config(page_title="MeetingMind", page_icon="📝", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2.2rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(135deg,#172554 0%,#1e3a8a 55%,#2563eb 100%); color: white; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2.35rem; letter-spacing: -0.03em;}
.hero p {margin: .45rem 0 0; color: #dbeafe; font-size: 1.02rem;}
.section {margin-top: 1.2rem; padding: .25rem 0; border: 0; background: transparent;}
</style>
<div class="hero"><h1>MeetingMind</h1><p>Автоматический протокол совещания: речь, саммари и поручения в одном месте.</p></div>
""", unsafe_allow_html=True)
st.caption("🔒 Локальная обработка: аудио и текст не отправляются во внешние облачные API.")

with st.sidebar:
    st.subheader("Настройки")
    model_size = st.selectbox("Размер локальной модели", ["small", "medium"], index=0)
    st.divider()
    st.markdown("**Как это работает**")
    st.markdown("1. Загрузите запись\n2. Нажмите «Создать протокол»\n3. Проверьте поручения\n4. Скачайте DOCX")

audio = st.file_uploader("Загрузите запись совещания", type=["mp3", "wav", "m4a", "mp4"], help="Поддерживаются MP3, WAV, M4A и MP4")
if audio:
    st.audio(audio)
    st.info(f"Файл готов к обработке: **{audio.name}** · {audio.size / 1024 / 1024:.1f} МБ")

if audio and st.button("Создать протокол", type="primary"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio.name).suffix) as temp:
        temp.write(audio.getbuffer())
        audio_path = temp.name
    with st.spinner("Распознаю аудио локально. Первый запуск скачает модель..."):
        segments = transcribe_audio(audio_path, model_size)
    transcript = segments_to_text(segments)
    tasks = extract_tasks(transcript)
    summary = make_summary(transcript, tasks)
    st.session_state["result"] = {"transcript": transcript, "tasks": tasks, "summary": summary}

result = st.session_state.get("result")
if result:
    st.success("Протокол сформирован. Проверьте поручения перед отправкой коллегам.")
    tasks = result["tasks"]
    summary = result["summary"]
    transcript = result["transcript"]
    left, middle, right = st.columns(3)
    left.metric("Фрагментов речи", len([x for x in transcript.splitlines() if x.strip()]))
    middle.metric("Поручений найдено", len(tasks))
    right.metric("Формат экспорта", "DOCX")
    speakers = sorted(set(re.findall(r"\]\s*([^:]+):", transcript)))
    if len(speakers) <= 1:
        st.warning("Голоса пока не удалось надёжно разделить. Метки SPEAKER будут проверены вручную перед финальной отправкой.")
    edited_tasks = tasks
    result_tab, transcript_tab, export_tab = st.tabs(["📌 Результат", "🎙️ Транскрипт", "📤 Экспорт"])
    with result_tab:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("Саммари")
        st.write(summary)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("Поручения")
        if tasks:
            display_tasks = [dict(item, status="Не проверено") for item in tasks]
            edited_tasks = st.data_editor(display_tasks, use_container_width=True, hide_index=True, disabled=["task", "responsible", "deadline"], column_config={
                "task": st.column_config.TextColumn("Поручение", width="large"),
                "responsible": st.column_config.TextColumn("Ответственный"),
                "deadline": st.column_config.TextColumn("Срок"),
                "status": st.column_config.SelectboxColumn("Статус", options=["Не проверено", "В работе", "Выполнено", "Просрочено"]),
            })
            st.caption("Статус можно изменить перед экспортом протокола.")
        else:
            st.warning("Поручения не найдены. Проверьте транскрипт вручную.")
        st.markdown('</div>', unsafe_allow_html=True)

    with transcript_tab:
        st.subheader("Полный транскрипт")
        st.caption("Временные метки и локальные метки говорящих помогают быстро проверить результат.")
        st.text_area("Текст совещания", transcript, height=480, label_visibility="collapsed")

    structured = {"summary": summary, "tasks": edited_tasks, "transcript": transcript, "speakers": speakers}
    with export_tab:
        st.subheader("Скачать результат")
        st.write("DOCX удобно отправить коллегам, JSON — передать в СЭД, CRM или будущий дашборд.")
        export_col, pdf_col, json_col = st.columns(3)
        with export_col:
            st.download_button("⬇️ Скачать протокол DOCX", data=make_docx(transcript, summary, edited_tasks), file_name="protocol.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", type="primary")
        with pdf_col:
            st.download_button("📄 Скачать PDF", data=make_pdf(transcript, summary, edited_tasks), file_name="protocol.pdf", mime="application/pdf")
        with json_col:
            st.download_button("↗️ Скачать JSON для интеграции", data=json.dumps(structured, ensure_ascii=False, indent=2), file_name="protocol.json", mime="application/json")
