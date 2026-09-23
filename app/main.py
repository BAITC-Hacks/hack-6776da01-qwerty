import tempfile
from pathlib import Path

import streamlit as st

from analyzer import extract_tasks, make_summary
from exporter import make_docx
from transcription import segments_to_text, transcribe_audio


st.set_page_config(page_title="Автопротоколирование", page_icon="📝", layout="wide")
st.title("Система автопротоколирования совещаний")
st.caption("Локальный прототип: аудио обрабатывается на вашем компьютере.")

audio = st.file_uploader("Загрузите запись совещания", type=["mp3", "wav", "m4a", "mp4"])
model_size = st.selectbox("Размер локальной модели", ["small", "medium"], index=0)

if audio and st.button("Создать протокол", type="primary"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio.name).suffix) as temp:
        temp.write(audio.getbuffer())
        audio_path = temp.name
    with st.spinner("Распознаю аудио локально. Первый запуск скачает модель..."):
        segments = transcribe_audio(audio_path, model_size)
    transcript = segments_to_text(segments)
    tasks = extract_tasks(transcript)
    summary = make_summary(transcript, tasks)
    st.subheader("Саммари")
    st.write(summary)
    st.subheader("Поручения")
    st.dataframe(tasks, use_container_width=True)
    with st.expander("Транскрипт"):
        st.text(transcript)
    st.download_button("Скачать протокол DOCX", data=make_docx(transcript, summary, tasks), file_name="protocol.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
