from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont, TTFError
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _register_font() -> str:
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for path in candidates:
        try:
            pdfmetrics.registerFont(TTFont("MeetingFont", path))
            return "MeetingFont"
        except (OSError, TTFError):
            continue
    return "Helvetica"


def make_pdf(transcript: str, summary: str, tasks: list[dict[str, str]]) -> bytes:
    font = _register_font()
    output = BytesIO()
    document = SimpleDocTemplate(output, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("MeetingTitle", parent=styles["Title"], fontName=font, fontSize=20, leading=24, textColor=colors.HexColor("#172554"))
    heading = ParagraphStyle("MeetingHeading", parent=styles["Heading2"], fontName=font, fontSize=13, leading=16, textColor=colors.HexColor("#1e3a8a"), spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("MeetingBody", parent=styles["BodyText"], fontName=font, fontSize=9.5, leading=13)
    story = [Paragraph("Протокол совещания", title), Spacer(1, 6), Paragraph("Саммари", heading), Paragraph(summary, body), Paragraph("Поручения", heading)]
    rows = [[Paragraph("Поручение", body), Paragraph("Ответственный", body), Paragraph("Срок", body)]]
    for item in tasks:
        rows.append([Paragraph(item.get("task", ""), body), Paragraph(item.get("responsible", ""), body), Paragraph(item.get("deadline", ""), body)])
    table = Table(rows, colWidths=[105 * mm, 42 * mm, 28 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([table, Paragraph("Транскрипт", heading), Paragraph(transcript.replace("\n", "<br/>"), body)])
    document.build(story)
    return output.getvalue()
