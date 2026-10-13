import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report(pdf_path: str, assessment: dict, metadata: dict, annotated_img_path: str):
    # Establish document boundaries with optimized margins for bottom layout spacing
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    
    styles = getSampleStyleSheet()
    navy_blue = colors.HexColor("#1A365D")
    charcoal = colors.HexColor("#2D3748")
    amber_red = colors.HexColor("#C53030")
    warning_yellow = colors.HexColor("#744210")
    
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold',
        fontSize=20, textColor=navy_blue, spaceAfter=12
    )
    body_style = ParagraphStyle(
        'DocBody', parent=styles['Normal'], fontName='Helvetica',
        fontSize=10, textColor=charcoal, leading=14, spaceAfter=8
    )
    alert_style = ParagraphStyle(
        'DocAlert', parent=styles['Normal'], fontName='Helvetica-Bold',
        fontSize=11, textColor=amber_red, leading=15, spaceAfter=10
    )
    warning_style = ParagraphStyle(
        'DocWarning', parent=styles['Normal'], fontName='Helvetica',
        fontSize=9, textColor=warning_yellow, leading=13
    )

    # Report Header Panel
    story.append(Paragraph("CLINICAL MULTI-AGENT INFERENCE REVIEWS", title_style))
    story.append(Spacer(1, 10))
    
    # Structural Table Metadata Mapping
    meta_data = [
        [Paragraph(f"<b>Patient ID:</b> Unknown Anonymized", body_style),
         Paragraph(f"<b>Scan Modality:</b> {metadata.get('scan_type', 'N/A')}", body_style)],
        [Paragraph(f"<b>Demographics:</b> Age {metadata.get('patient_age', 'N/A')}", body_style),
         Paragraph(f"<b>System Signature:</b> OLLAMA-VLM-ASYNC-FIFO", body_style)]
    ]
    t = Table(meta_data, colWidths=[260, 260])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))
    
    # Diagnosis Block Section
    story.append(Paragraph("<b>CLINICAL IMPRESSION & DIAGNOSIS ASSESSMENT:</b>", body_style))
    diag_clean = assessment.get('diagnosis', 'Inconclusive').replace("Diagnosis:", "").strip()
    story.append(Paragraph(f"<u>👉 {diag_clean}</u>", alert_style))
    story.append(Spacer(1, 15))
    
    # Embedded Visual Specimen Scan Section
    story.append(Paragraph("<b>INGESTED VISUAL SPECIMEN SCAN REFERENCE:</b>", body_style))
    story.append(Spacer(1, 5))
    if os.path.exists(annotated_img_path):
        img = Image(annotated_img_path, width=220, height=220)
        img.hAlign = 'CENTER'
        story.append(img)
    else:
        story.append(Paragraph("<font color='red'><i>Reference matrix view omitted from filesystem registry.</i></font>", body_style))
    story.append(Spacer(1, 15))
    
    # Extracted Physiological Findings Section
    story.append(Paragraph("<b>EXTRACTED PHYSIOLOGICAL FINDINGS:</b>", body_style))
    story.append(Paragraph(assessment.get('findings', 'N/A'), body_style))
    story.append(Spacer(1, 10))
    
    # Suggested Remedies Section
    story.append(Paragraph("<b>SUGGESTED TREATMENT PATHWAYS:</b>", body_style))
    story.append(Paragraph(assessment.get('remedies', 'N/A'), body_style))
    story.append(Spacer(1, 15))
    
    # FIXED: Comprehensive Programmatic Statutory Warning Embedded Callout Box Layout
    warning_text = (
        "<b>⚠️ STATUTORY WARNING:</b> This automated evaluation artifact was compiled by an AI multi-agent "
        "orchestration layer running open-source CPU inference models. It is structured for technical screening "
        "validation workflows and research development only. Do not rely on these observations for clinical "
        "care decisions. All findings must be critically cross-referenced, reviewed, and signed off "
        "by a certified Subject Matter Expert (SME) physician before acting on patient diagnostic pathways."
    )
    warning_table = Table([[Paragraph(warning_text, warning_style)]], colWidths=[520])
    warning_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FEFCBF")), # Matches front-end light yellow background
        ('PADDING', (0,0), (-1,-1), 10),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#D69E2E")),     # Amber border trace rules
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(warning_table)
    story.append(Spacer(1, 25))
    
    # FIXED: Added structured side-by-side lines for Signature and Date Time values
    sign_data = [
        [
            Paragraph("_______________________________________<br/><b>SME Authorized Specialist Signature</b>", body_style),
            Paragraph("___________________________<br/><b>Review Date / Time</b>", body_style)
        ]
    ]
    sign_table = Table(sign_data, colWidths=[320, 200])
    sign_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (0,0), (0,0), 'LEFT'),
        ('ALIGN', (1,0), (1,0), 'RIGHT'),
        ('TOPPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(sign_table)
    
    doc.build(story)