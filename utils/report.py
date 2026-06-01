"""PDF report generation."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_prediction_pdf(result: dict[str, Any], summary: str, recommendation: str) -> BytesIO:
    """Build a compact PDF report for the latest prediction."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=42)
    styles = getSampleStyleSheet()
    values = result.get("input", {})
    probability = result.get("rain_probability", 0) * 100
    confidence = result.get("confidence", 0) * 100
    label = "Rainfall Expected" if result.get("prediction") == 1 else "No Rainfall Expected"

    story = [
        Paragraph("Rainfall Detection using Machine Learning", styles["Title"]),
        Spacer(1, 10),
        Paragraph("Prediction Report", styles["Heading2"]),
        Paragraph(f"<b>Result:</b> {label}", styles["BodyText"]),
        Paragraph(f"<b>Rain Probability:</b> {probability:.1f}%", styles["BodyText"]),
        Paragraph(f"<b>Confidence:</b> {confidence:.1f}%", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph(f"<b>Summary:</b> {summary}", styles["BodyText"]),
        Paragraph(f"<b>Recommendation:</b> {recommendation}", styles["BodyText"]),
        Spacer(1, 14),
    ]

    rows = [["Parameter", "Value"]]
    for key in [
        "City",
        "State",
        "Temperature_Avg (°C)",
        "Humidity (%)",
        "Wind_Speed (km/h)",
        "AQI",
        "AQI_Category",
        "Pressure (hPa)",
        "Cloud_Cover (%)",
        "Year",
        "Month",
        "Day",
    ]:
        rows.append([key, str(values.get(key, "-"))])

    table = Table(rows, colWidths=[190, 260])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#13213f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#94a3b8")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf2ff")]),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(table)
    doc.build(story)
    buffer.seek(0)
    return buffer

