import json
import re
import tempfile
from datetime import date
from pathlib import Path

import streamlit as st

from analyzer import extract_tasks, make_summary
from exporter import make_docx
from local_llm import analyze_locally
from pdf_exporter import make_pdf
from quality import validate_protocol
from transcription import segments_to_text, transcribe_audio

st.set_page_config(page_title="MeetingMind", page_icon="📝", layout="wide", initial_sidebar_state="expanded")
theme_label = st.sidebar.selectbox("Тема интерфейса", ["Тёмная", "Светлая"], index=0, key="interface_theme")
st.markdown("""
<style>
.block-container {max-width: 1120px; padding-top: 1.8rem; padding-bottom: 4rem;}
body {background: #070d18;}
.topbar {display:flex; align-items:center; justify-content:space-between; margin-bottom:1.4rem; color:#94a3b8; font-size:.78rem; letter-spacing:.04em; text-transform:uppercase;}
.brand {display:flex; align-items:center; gap:.7rem; color:#f8fafc; font-weight:800; letter-spacing:.12em;}
.brand-mark {display:grid; place-items:center; width:34px; height:34px; border-radius:11px; background:linear-gradient(135deg,#38bdf8,#6366f1); color:white; font-size:.88rem; box-shadow:0 8px 22px rgba(56,189,248,.25);}
.status-pill {display:inline-flex; align-items:center; gap:.45rem; padding:.42rem .7rem; border:1px solid rgba(148,163,184,.2); border-radius:999px; background:rgba(15,23,42,.72); color:#a7f3d0; text-transform:none; letter-spacing:0;}
.status-dot {width:7px; height:7px; border-radius:50%; background:#34d399; box-shadow:0 0 0 4px rgba(52,211,153,.12);}
.hero {display:grid; grid-template-columns:1.45fr .75fr; min-height:280px; overflow:hidden; background:linear-gradient(135deg,#101d3a 0%,#172554 52%,#263e9e 100%); border:1px solid rgba(129,140,248,.24); border-radius:26px; color:white; margin-bottom:1.25rem; box-shadow:0 22px 55px rgba(2,6,23,.38);}
.hero-copy {padding:2.65rem 2.8rem; display:flex; flex-direction:column; justify-content:center;}
.hero-quote {display:none;}
.hero h1 {margin:0; max-width:650px; font-size:3rem; line-height:1.04; letter-spacing:-.055em; font-weight:800; color:#fff;}
.hero p {max-width:590px; margin:1.1rem 0 0; color:#c7d2fe; font-size:1.05rem; line-height:1.6;}
.hero-visual {position:relative; display:grid; place-items:center; overflow:hidden; background:radial-gradient(circle at 50% 45%,rgba(96,165,250,.45),transparent 25%),linear-gradient(145deg,rgba(30,64,175,.55),rgba(15,23,42,.15));}
.hero-visual:before {content:""; position:absolute; width:210px; height:210px; border:1px solid rgba(165,180,252,.35); border-radius:50%; box-shadow:0 0 0 30px rgba(129,140,248,.08),0 0 0 62px rgba(56,189,248,.06);}
.hero-visual:after {content:"AI"; position:relative; display:grid; place-items:center; width:108px; height:108px; border-radius:32px; transform:rotate(8deg); background:linear-gradient(145deg,#38bdf8,#6366f1); color:#fff; font-size:2.4rem; font-weight:800; letter-spacing:.08em; box-shadow:0 18px 40px rgba(56,189,248,.28);}
.hero-meta {display:flex; gap:.65rem; margin-top:1.5rem; flex-wrap:wrap;}
.hero-meta span {padding:.5rem .75rem; border:1px solid rgba(191,219,254,.22); border-radius:9px; background:rgba(15,23,42,.25); color:#dbeafe; font-size:.76rem;}
.section-heading {display:flex; align-items:end; justify-content:space-between; margin:1.55rem 0 .7rem;}
.section-heading h2 {margin:0; color:#f8fafc; font-size:1.25rem; letter-spacing:-.02em;}
.section-heading span {color:#64748b; font-size:.78rem;}
.section {margin-top: 1.2rem; padding: .25rem 0; border: 0; background: transparent;}
.eyebrow {font-size: .76rem; letter-spacing: .16em; text-transform: uppercase; color: #d9efff; font-weight: 700; margin-bottom: .65rem;}
.privacy {display: inline-block; margin-top: 1.15rem; padding: .42rem .7rem; border: 1px solid rgba(255,255,255,.34); border-radius: 3px; color: #eff6ff; font-size: .82rem; background: rgba(3,35,73,.22);}
.upload-card {padding:1.2rem 1.35rem .7rem; margin:0 0 1.4rem; border:1px solid rgba(148,163,184,.2); border-radius:18px; background:rgba(15,23,42,.72); box-shadow:0 12px 28px rgba(2,6,23,.18);}
.empty-title {font-size: 1.45rem; font-weight: 700; margin: .35rem 0 .25rem;}
.empty-subtitle {color: #94a3b8; margin-bottom: 1.15rem;}
.step-card {padding: 1rem; min-height: 105px; border: 1px solid rgba(148,163,184,.22); border-radius: 14px; background: rgba(30,41,59,.32);}
.step-number {color: #65b7e8; font-size: .78rem; font-weight: 800; letter-spacing: .08em;}
.step-card strong {display: block; margin: .35rem 0; color: #f8fafc;}
.step-card span {color: #94a3b8; font-size: .88rem;}
[data-testid="stMetric"] {padding: 1rem 1.1rem; border: 1px solid rgba(148,163,184,.2); border-radius: 16px; background: rgba(30,41,59,.32);}
[data-testid="stMetricLabel"] {color: #94a3b8;}
[data-testid="stMetricValue"] {color: #f8fafc;}
.stButton > button, .stDownloadButton > button {border-radius: 10px; font-weight: 650; min-height: 2.6rem;}
</style>
<div class="topbar"><div class="brand"><div class="brand-mark">M</div><span>MEETINGMIND</span></div><div class="status-pill"><i class="status-dot"></i> Локальный контур</div></div>
<div class="hero"><div class="hero-copy"><div class="eyebrow">AI MEETING INTELLIGENCE</div><h1>Порядок после каждого разговора</h1><p>Превращайте запись совещания в понятный протокол с решениями, ответственными и сроками — без ручной расшифровки.</p><div class="hero-meta"><span>◉ Русский · Қазақша</span><span>▣ DOCX · PDF · JSON</span></div></div><div class="hero-visual"></div></div>
""", unsafe_allow_html=True)
st.caption("🔒 Аудио и текст обрабатываются локально и не отправляются во внешние облачные API.")
if theme_label == "Светлая":
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: #f6f7f9; }
    [data-testid="stAppViewContainer"] { color: #1f2937; }
    [data-testid="stSidebar"] { background: #eef2f7; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #1f2937; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #64748b; }
    [data-testid="stSidebar"] hr { border-color: #cbd5e1; }
    .topbar { color: #64748b; }
    .brand { color: #172033; }
    .status-pill { background: #ffffff; border-color: #dbe3ee; color: #047857; }
    .section-heading h2 { color: #172033; }
    .upload-card, .step-card, [data-testid="stMetric"] { background: #ffffff; border-color: #dbe3ee; }
    .empty-title, .step-card strong { color: #172033; }
    .empty-subtitle, .step-card span, [data-testid="stMetricLabel"] { color: #64748b; }
    [data-testid="stMetricValue"] { color: #172033; }
    </style>
    """, unsafe_allow_html=True)
st.info("Внимание: ведётся запись и ИИ-транскрибация совещания. Участники должны быть уведомлены. Обработка данных происходит строго локально (On-Premise) без передачи во внешние облачные API.")


def _speakers(transcript: str) -> list[str]:
    return sorted(set(re.findall(r"\]\s*([^:]+):", transcript)))


def _rename_speakers(transcript: str, mapping: dict[str, str]) -> str:
    for source, target in mapping.items():
        if target.strip():
            transcript = re.sub(rf"(?<=\])\s*{re.escape(source)}(?=:)", f" {target.strip()}", transcript)
    return transcript


def _demo_split(transcript: str, speaker_count: int = 2) -> str:
    lines = [line for line in transcript.splitlines() if line.strip()]
    return "\n".join(re.sub(r"(\]\s*)[^:]+(?=:)", rf"\1SPEAKER_{index % speaker_count:02d}", line) for index, line in enumerate(lines))


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
    st.markdown("## MeetingMind")
    st.caption("Автопротоколирование совещаний")
    st.divider()
    st.subheader("Настройки")
    model_size = st.selectbox("Размер локальной модели", ["small", "medium", "large-v3-turbo"], index=0)
    language_label = st.selectbox("Язык записи", ["Автоопределение", "Русский", "Казахский"], index=0, help="Для чисто русской записи выбор языка повышает качество распознавания.")
    language = {"Автоопределение": None, "Русский": "ru", "Казахский": "kk"}[language_label]
    smart_mode = st.checkbox("Умный локальный анализ", help="Использует Ollama на этом компьютере. Если Ollama недоступен, включится обычный анализатор.")
    diarization_mode = st.selectbox("Разделение спикеров", ["Авто: pyannote + fallback", "Демо-разметка по очереди"], help="Авто-режим сам определяет число спикеров, если локальная pyannote-модель настроена.")
    demo_mode = diarization_mode == "Демо-разметка по очереди"
    demo_speaker_count = st.slider("Количество спикеров для демо", min_value=2, max_value=6, value=3, disabled=not demo_mode)
    if model_size == "medium":
        st.caption("Medium точнее только при достаточных ресурсах; на CPU может быть медленнее small.")
    elif model_size == "large-v3-turbo":
        st.caption("Рекомендуется для GPU. На CPU эта модель будет очень медленной.")
    st.divider()
    st.markdown("**Как это работает**")
    st.markdown("1. Уведомьте участников и загрузите запись\n2. Создайте протокол\n3. Назначьте имена и проверьте поручения\n4. Сформируйте уведомления и скачайте протокол")

st.markdown('<div class="section-heading"><h2>Новое совещание</h2><span>Шаг 01 · Загрузите аудиозапись</span></div>', unsafe_allow_html=True)
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
st.markdown('<div class="empty-title">Загрузите запись совещания</div><div class="empty-subtitle">MP3, WAV, M4A или MP4 · обработка выполняется локально</div>', unsafe_allow_html=True)
audio = st.file_uploader("Выберите аудиофайл", type=["mp3", "wav", "m4a", "mp4"], help="Поддерживаются MP3, WAV, M4A и MP4", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)
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
            segments = transcribe_audio(audio_path, model_size, language=language, diarization=True)
        raw_transcript = segments_to_text(segments)
        if demo_mode and len(_speakers(raw_transcript)) <= 1:
            raw_transcript = _demo_split(raw_transcript, demo_speaker_count)
        ai_result = analyze_locally(raw_transcript) if smart_mode else None
        st.session_state["result"] = {"raw_transcript": raw_transcript, "ai_result": ai_result}
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
    ai_result = result.get("ai_result")
    if ai_result:
        tasks = ai_result.get("tasks", [])
        summary = ai_result.get("summary", "")
        st.success("Протокол дополнительно проанализирован локальной языковой моделью.")
    else:
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
        st.warning("Автоматическая диаризация не нашла несколько голосов. Настройте локальную pyannote-модель или включите демо-разметку.")
    if demo_mode and len(raw_speakers) > 1:
        st.warning("Включён демо-режим разделения по очереди сегментов; результат нельзя считать акустической диаризацией.")

    result_tab, transcript_tab, export_tab = st.tabs(["Результат", "Транскрипт", "Экспорт"])
    with result_tab:
        st.subheader("Саммари")
        st.markdown(summary.replace("\n", "  \n"))
        st.subheader("Поручения и контроль сроков")
        if tasks:
            display_tasks = []
            for item in tasks:
                row = dict(item)
                row["status"] = "Просрочено" if _overdue(item.get("deadline", "")) else "Не проверено"
                display_tasks.append(row)
            edited_tasks = st.data_editor(display_tasks, use_container_width=True, hide_index=True, column_config={
                "task": st.column_config.TextColumn("Поручение", width="large"),
                "responsible": st.column_config.TextColumn("Ответственный"),
                "deadline": st.column_config.TextColumn("Срок"),
                "status": st.column_config.SelectboxColumn("Статус", options=["Не проверено", "В работе", "Выполнено", "Просрочено"]),
            })
        else:
            edited_tasks = []
            st.warning("Поручения не найдены. Проверьте транскрипт вручную.")
        st.subheader("Контроль качества")
        st.metric("Оценка протокола", f"{quality['quality_score']}/100")
        for warning in quality["warnings"]:
            st.warning(warning)
        if tasks and st.button("Сформировать уведомления ответственным"):
            st.subheader("Готовые напоминания")
            for item in edited_tasks:
                st.info(f"Уважаемый(ая) {item.get('responsible', 'коллега')}, напоминаем о поручении «{item.get('task', '')}» со сроком {item.get('deadline', 'не указан')}.")

    with transcript_tab:
        st.subheader("Полный транскрипт с именами")
        edited_transcript = st.text_area("Текст совещания", transcript, height=480, label_visibility="collapsed")
        if st.button("Пересчитать поручения из исправленного текста"):
            st.session_state["result"] = {"raw_transcript": edited_transcript, "ai_result": analyze_locally(edited_transcript) if smart_mode else None}
            st.rerun()

    structured = {"summary": summary, "tasks": edited_tasks, "transcript": transcript, "speakers": speakers, "quality": quality}
    with export_tab:
        st.subheader("Скачать результат")
        export_col, pdf_col, json_col = st.columns(3)
        with export_col:
            st.download_button("Скачать DOCX", data=make_docx(transcript, summary, edited_tasks), file_name="protocol.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", type="primary")
        with pdf_col:
            st.download_button("Скачать PDF", data=make_pdf(transcript, summary, edited_tasks), file_name="protocol.pdf", mime="application/pdf")
        with json_col:
            st.download_button("Скачать JSON", data=json.dumps(structured, ensure_ascii=False, indent=2), file_name="protocol.json", mime="application/json")

