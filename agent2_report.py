"""
AGENT 2 -- Report Generation + Fully Automated Appointment Booking
===================================================================
Filename: agent2_report.py

WHAT'S CHANGED vs the old version:
  ✅ DOCTOR_SCHEDULES added here — doctor busy/free times are defined in one place
  ✅ find_next_free_slot() — checks schedule and picks the next genuinely free slot
  ✅ run_agent2_auto() — new fully automated function:
       Agent1 verdict → route doctor → check schedule → pick slot → PDF → email → calendar
       NO manual slot selection needed. Everything fires automatically.
  ✅ Old run_agent2() and confirm_appointment() kept for backward compatibility

EMAIL FLOW (new, simplified):
  run_agent2_auto()  ->  does everything in one call, no follow-up needed
  Old flow still works: run_agent2() -> confirm_appointment() (if you want manual slot picking)

GOOGLE CALENDAR SETUP (3 steps):
  1. console.cloud.google.com -> Enable "Google Calendar API"
     -> Credentials -> OAuth 2.0 Client ID (Desktop App)
     -> Download JSON -> rename to "google_calendar_creds.json"
     -> Place next to this file
  2. pip install google-auth google-auth-oauthlib google-api-python-client
  3. First run opens browser once for OAuth -> token.pickle saved -> automatic after

GMAIL APP PASSWORD:
  myaccount.google.com -> Security -> App Passwords -> generate 16-char code
  pip install reportlab
"""

from __future__ import annotations

# ================================================================
# CONFIGURATION -- edit only this section
# ================================================================
SENDER_EMAIL    = "shruthidhandapani2021@gmail.com"
SENDER_PASSWORD = "scml kxjs ushj fzjy"        # 16-char Google App Password

DOCTOR_ROUTING: dict = {
    "5": {
        "name":  "Dr. P. Sundarajan",
        "email": "shruthidhandapani.official@gmail.com",
        "dept":  "Oncology Department",
        "role":  "Senior Oncologist",
    },
    "4C": {
        "name":  "Dr. P. Sundarajan",
        "email": "shruthi.perso@gmail.com",
        "dept":  "Oncology Department",
        "role":  "Senior Oncologist",
    },
    "4B": {
        "name":  "Dr. R. Meenakshi",
        "email": "sri.radha.23.8@gmail.com",
        "dept":  "Radiology Department",
        "role":  "Radiologist Specialist",
    },
    "4A": {
        "name":  "Dr. R. Meenakshi",
        "email": "srikala202001@gmail.com",
        "dept":  "Radiology Department",
        "role":  "Radiologist Specialist",
    },
    "3": {
        "name":  "Dr. S. Lakshmi",
        "email": "kartk0171@gmail.com",
        "dept":  "Radiology Department",
        "role":  "General Radiologist",
    },
    "1-2": {
        "name":  "Dr. A. Priya",
        "email": "infraccoun@gmail.com",
        "dept":  "General Medicine",
        "role":  "General Physician",
    },
    "default": {
        "name":  "Dr. R. Meenakshi",
        "email": "infaccoun@gmail.com",
        "dept":  "Radiology Department",
        "role":  "Radiologist",
    },
}

# ================================================================
# DOCTOR SCHEDULES
# New in this version — defines each doctor's working hours and
# which time blocks are already busy. The auto-scheduler checks
# this before booking any slot.
#
# busy: dict of day -> list of "HH:MM-HH:MM" strings (24h format)
#       Any slot that falls inside a busy block will be skipped.
# ================================================================
DOCTOR_SCHEDULES: dict = {
    "Dr. P. Sundarajan": {
        "working_days":  ["Mon", "Tue", "Wed", "Thu", "Fri"],
        "working_hours": (8, 17),   # start hour, end hour (24h)
        "busy": {
            "Mon": ["09:00-10:00", "14:00-15:00"],
            "Tue": ["11:00-12:00", "15:00-16:00"],
            "Wed": ["09:00-11:00"],
            "Thu": ["10:00-11:00", "14:00-15:00"],
            "Fri": ["13:00-14:00"],
        },
    },
    "Dr. R. Meenakshi": {
        "working_days":  ["Mon", "Tue", "Wed", "Thu", "Fri"],
        "working_hours": (9, 18),
        "busy": {
            "Mon": ["10:00-11:00"],
            "Tue": ["09:00-10:00", "14:00-16:00"],
            "Wed": ["11:00-12:00"],
            "Thu": ["09:00-10:00", "15:00-16:00"],
            "Fri": ["10:00-11:00", "13:00-14:00"],
        },
    },
    "Dr. S. Lakshmi": {
        "working_days":  ["Mon", "Tue", "Thu", "Fri"],
        "working_hours": (9, 16),
        "busy": {
            "Mon": ["09:30-10:30"],
            "Tue": ["14:00-15:00"],
            "Thu": ["11:00-12:00"],
            "Fri": ["09:00-10:00"],
        },
    },
    "Dr. A. Priya": {
        "working_days":  ["Mon", "Wed", "Thu", "Fri"],
        "working_hours": (8, 15),
        "busy": {
            "Mon": ["08:00-09:00"],
            "Wed": ["11:00-12:00", "13:00-14:00"],
            "Thu": ["09:00-10:00"],
            "Fri": ["08:00-09:00", "14:00-15:00"],
        },
    },
}

# Urgency -> how many days ahead to start looking for a slot
URGENCY_WINDOW: dict = {
    "emergency": (0, 1),    # today or tomorrow
    "urgent":    (1, 3),    # within 3 days
    "priority":  (3, 7),    # within a week
    "routine":   (7, 14),   # within 2 weeks
}

# Preferred appointment times (tried in order)
PREFERRED_TIMES = ["09:30", "10:30", "11:30", "14:00", "15:00", "16:00"]

HOSPITAL_NAME     = "Apollo Hospitals"
HOSPITAL_ADDRESS  = "Bannerghatta Rd, Bengaluru"
GOOGLE_CREDS_FILE = "google_calendar_creds.json"
# ================================================================

import json
import pickle
import datetime
import smtplib
from pathlib import Path
from typing import Any, Dict, List, Optional
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

SCRIPT_DIR  = Path(__file__).parent
REPORTS_DIR = SCRIPT_DIR / "outputs" / "agent_reports"
RESULTS_DIR = SCRIPT_DIR / "outputs" / "results"
DOC_REPORTS = SCRIPT_DIR / "outputs" / "doctor_reports"
APPT_FILE   = SCRIPT_DIR / "appointments.json"

DOC_REPORTS.mkdir(parents=True, exist_ok=True)


# ====================================================================
# SCHEDULE HELPERS  (new)
# ====================================================================

def _slot_is_busy(doctor_name: str, day_short: str, t_str: str) -> bool:
    """
    Check if a given time slot is in the doctor's busy list.
    t_str: "HH:MM" (24h)
    day_short: "Mon", "Tue", etc.
    """
    sched = DOCTOR_SCHEDULES.get(doctor_name, {})
    h, m  = map(int, t_str.split(":"))
    slot_m = h * 60 + m
    for period in sched.get("busy", {}).get(day_short, []):
        s, e   = period.split("-")
        sh, sm = map(int, s.split(":"))
        eh, em = map(int, e.split(":"))
        if sh * 60 + sm <= slot_m < eh * 60 + em:
            return True
    return False


def find_next_free_slot(doctor_name: str, urgency: str) -> Dict:
    """
    Find the next genuinely free slot for a doctor, respecting their
    working days, working hours, and busy blocks.

    Returns a slot dict ready for use in bookings and emails.
    """
    min_d, max_d = URGENCY_WINDOW.get(urgency.lower(), (7, 14))
    sched        = DOCTOR_SCHEDULES.get(doctor_name, {})
    working_days = sched.get("working_days", ["Mon", "Tue", "Wed", "Thu", "Fri"])
    ws, we       = sched.get("working_hours", (9, 17))
    now          = datetime.datetime.now()

    # Try each day in the urgency window (+5 buffer days in case all are busy)
    for offset in range(min_d, max_d + 5):
        day = now + datetime.timedelta(days=offset)
        day_short = day.strftime("%a")   # "Mon", "Tue", etc.

        if day_short not in working_days:
            continue

        for t in PREFERRED_TIMES:
            h = int(t.split(":")[0])
            # Skip outside working hours
            if h < ws or h >= we:
                continue
            # Skip if busy
            if _slot_is_busy(doctor_name, day_short, t):
                continue
            # Found a free slot
            minute = int(t.split(":")[1])
            slot_dt = day.replace(hour=h, minute=minute, second=0, microsecond=0)
            return {
                "slot_id":  "SLOT-AUTO",
                "date":     day.strftime("%A, %d %B %Y"),
                "time":     slot_dt.strftime("%I:%M %p"),
                "time_24":  t,
                "day":      day_short,
                "doctor":   doctor_name,
                "hospital": HOSPITAL_NAME,
                "address":  HOSPITAL_ADDRESS,
                "label":    f"{day.strftime('%A, %d %B %Y')} at {slot_dt.strftime('%I:%M %p')}",
            }

    # Absolute fallback: use the last day in window at 09:30
    fb = now + datetime.timedelta(days=max_d)
    return {
        "slot_id":  "SLOT-AUTO",
        "date":     fb.strftime("%A, %d %B %Y"),
        "time":     "09:30 AM",
        "time_24":  "09:30",
        "day":      fb.strftime("%a"),
        "doctor":   doctor_name,
        "hospital": HOSPITAL_NAME,
        "address":  HOSPITAL_ADDRESS,
        "label":    f"{fb.strftime('%A, %d %B %Y')} at 09:30 AM",
    }


# ====================================================================
# DOCTOR ROUTING
# ====================================================================

def get_doctor(birads: str) -> Dict[str, str]:
    """Return doctor dict based on BI-RADS. Falls back to default."""
    key    = str(birads).strip()
    doctor = DOCTOR_ROUTING.get(key, DOCTOR_ROUTING["default"])
    print(f"  [Agent2] Routed: BI-RADS {key} -> {doctor['role']} ({doctor['name']})")
    return doctor


# ====================================================================
# SECTION 1 -- LOAD AGENT 1 VERDICT
# ====================================================================

def load_verdict(patient_id: str, scan_number: str) -> Optional[Dict]:
    path = REPORTS_DIR / f"{patient_id}_{scan_number}_verdict.json"
    if not path.exists():
        print(f"  [Agent2] Verdict not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


# ====================================================================
# SECTION 2 -- PDF REPORT GENERATION
# ====================================================================

def generate_pdf_report(verdict: Dict, scan_image_path: Optional[str]) -> Path:
    """
    Build a professional PDF doctor report.
    Requires: pip install reportlab
    Falls back to .txt if reportlab not installed.
    """
    patient_id  = str(verdict.get("patient_id",  "UNKNOWN"))
    scan_number = str(verdict.get("scan_number", "UNKNOWN"))
    scan_date   = str(verdict.get("scan_date",   str(datetime.datetime.now())[:16]))
    birads      = verdict.get("birads_estimate", "default")
    doctor      = get_doctor(str(birads))
    out_pdf     = DOC_REPORTS / f"{patient_id}_{scan_number}_doctor_report.pdf"

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer,
            Image as RLImage, Table, TableStyle, HRFlowable,
        )

        doc   = SimpleDocTemplate(
            str(out_pdf), pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
        )
        story: List[Any] = []

        v_col_map: Dict[str, Any] = {
            "APPROVED": colors.HexColor("#1a7a1a"),
            "FLAGGED":  colors.HexColor("#cc7700"),
            "REJECTED": colors.HexColor("#cc0000"),
        }
        b_col_map: Dict[str, Any] = {
            "1-2": colors.HexColor("#1a7a1a"), "3":  colors.HexColor("#4488cc"),
            "4A":  colors.HexColor("#cc9900"), "4B": colors.HexColor("#cc6600"),
            "4C":  colors.HexColor("#cc2200"), "5":  colors.HexColor("#880000"),
        }
        v_color: Any = v_col_map.get(str(verdict.get("verdict", "")), colors.grey)
        b_color: Any = b_col_map.get(str(verdict.get("birads_estimate", "")), colors.grey)

        def ps(name: str, **kw: Any) -> ParagraphStyle:
            return ParagraphStyle(name, **kw)

        title_s = ps("T",  fontSize=18, fontName="Helvetica-Bold",
                     textColor=colors.HexColor("#0a0a2e"), spaceAfter=4)
        sub_s   = ps("ST", fontSize=10, fontName="Helvetica",
                     textColor=colors.HexColor("#555555"), spaceAfter=12)
        sec_s   = ps("SC", fontSize=9,  fontName="Helvetica-Bold",
                     textColor=colors.HexColor("#333333"),
                     spaceBefore=12, spaceAfter=4, leading=14)
        body_s  = ps("B",  fontSize=9,  fontName="Helvetica",
                     textColor=colors.HexColor("#222222"), leading=14, spaceAfter=6)
        sm_s    = ps("SM", fontSize=7.5, fontName="Helvetica",
                     textColor=colors.HexColor("#666666"), leading=11)
        hosp_s  = ps("H",  fontSize=9,  fontName="Helvetica",
                     textColor=colors.HexColor("#888888"), spaceAfter=2)
        flag_s  = ps("FL", fontSize=9,  fontName="Helvetica",
                     textColor=colors.HexColor("#cc6600"),
                     leftIndent=12, spaceAfter=3)
        foot_s  = ps("F2", fontSize=7,  fontName="Helvetica",
                     textColor=colors.HexColor("#aaaaaa"), leading=10)
        vt_s    = ps("VT", fontSize=16, fontName="Helvetica-Bold",
                     textColor=colors.white, leading=20)
        vs_s    = ps("VS", fontSize=9,  fontName="Helvetica",
                     textColor=colors.HexColor("#dddddd"), leading=12)

        def hr() -> HRFlowable:
            return HRFlowable(width="100%", thickness=0.5,
                              color=colors.HexColor("#cccccc"), spaceAfter=8)

        # Header
        story += [
            Paragraph("BREAST IMAGING CENTER", hosp_s),
            Paragraph("AI-Assisted Breast Ultrasound Report", title_s),
            Paragraph(
                f"Generated by Agent 1 (Groq Llama-4-Scout) + Agent 2  |  "
                f"Assigned to: {doctor['name']} ({doctor['role']})  |  For physician use only",
                sub_s),
            HRFlowable(width="100%", thickness=1,
                       color=colors.HexColor("#dddddd"), spaceAfter=12),
        ]

        # Patient info table
        info_t = Table([
            ["Patient ID",  patient_id,           "Scan Number",  scan_number],
            ["Scan Date",   scan_date,             "Report Date",  str(datetime.date.today())],
            ["AI Backend",  "Groq Llama-4-Scout",  "Assigned To",  doctor["name"]],
            ["Department",  doctor["dept"],         "Doctor Role",  doctor["role"]],
        ], colWidths=[3.5*cm, 5.5*cm, 3.5*cm, 5*cm])
        info_t.setStyle(TableStyle([
            ("FONTNAME",       (0,0),(-1,-1), "Helvetica"),
            ("FONTNAME",       (0,0),(0,-1),  "Helvetica-Bold"),
            ("FONTNAME",       (2,0),(2,-1),  "Helvetica-Bold"),
            ("FONTSIZE",       (0,0),(-1,-1), 8.5),
            ("TEXTCOLOR",      (0,0),(0,-1),  colors.HexColor("#555555")),
            ("TEXTCOLOR",      (2,0),(2,-1),  colors.HexColor("#555555")),
            ("ROWBACKGROUNDS", (0,0),(-1,-1),
             [colors.HexColor("#f8f8f8"), colors.HexColor("#f0f0f0")]),
            ("GRID",           (0,0),(-1,-1), 0.5, colors.HexColor("#dddddd")),
            ("TOPPADDING",     (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
            ("LEFTPADDING",    (0,0),(-1,-1), 8),
        ]))
        story += [info_t, Spacer(1, 14)]

        # Verdict banner
        vl = str(verdict.get("verdict", "UNKNOWN"))
        vt = Table([[
            Paragraph(f"<b>{vl}</b>", vt_s),
            Paragraph(
                f"Exit code {verdict.get('exit_code','?')}  |  "
                f"Urgency: {str(verdict.get('urgency','')).upper()}  |  "
                f"BI-RADS: {verdict.get('birads_estimate','?')}",
                vs_s),
        ]], colWidths=[5*cm, 12.5*cm])
        vt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), v_color),
            ("LEFTPADDING",   (0,0),(-1,-1), 14),
            ("TOPPADDING",    (0,0),(-1,-1), 10),
            ("BOTTOMPADDING", (0,0),(-1,-1), 10),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ]))
        story += [vt, Spacer(1, 14)]

        # Metrics table
        conf    = verdict.get("extracted_confidence", "?")
        pred    = str(verdict.get("predicted_class",  "?")).upper()
        urg     = str(verdict.get("urgency",          "?")).upper()
        vis_con = str(verdict.get("visual_consistency","?")).upper()

        mt = Table([
            ["METRIC",             "VALUE",       "INTERPRETATION"],
            ["AI Prediction",      pred,          "Model classification output"],
            ["Confidence",         f"{conf}%",    ">=85% APPROVED | 70-85% FLAGGED | <70% REJECTED"],
            ["BI-RADS Estimate",   str(birads),   "ACR 5th Ed category"],
            ["Urgency Level",      urg,           "Clinical routing priority"],
            ["Visual Consistency", vis_con,       "Heatmap / mask agreement with prediction"],
            ["Assigned Doctor",    doctor["name"],doctor["role"]],
        ], colWidths=[4.5*cm, 3.5*cm, 9.5*cm])
        mt.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),(-1,0),  colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",      (0,0),(-1,0),  colors.white),
            ("FONTNAME",       (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTNAME",       (0,1),(-1,-1), "Helvetica"),
            ("FONTSIZE",       (0,0),(-1,-1), 8.5),
            ("ROWBACKGROUNDS", (0,1),(-1,-1),
             [colors.HexColor("#ffffff"), colors.HexColor("#f4f4f4")]),
            ("GRID",           (0,0),(-1,-1), 0.5, colors.HexColor("#cccccc")),
            ("TOPPADDING",     (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
            ("LEFTPADDING",    (0,0),(-1,-1), 8),
            ("TEXTCOLOR",      (1,3),(1,3),   b_color),
        ]))
        story += [mt, Spacer(1, 14)]

        # Scan image
        story.append(Paragraph("SCAN IMAGE (6-Panel AI Output)", sec_s))
        story.append(hr())
        sp = Path(scan_image_path) if scan_image_path else None
        if sp and sp.exists():
            story.append(RLImage(str(sp), width=17*cm, height=8.5*cm))
            story.append(Paragraph(f"Figure 1: {sp.name}", sm_s))
        else:
            story.append(Paragraph("[Scan image not available]", sm_s))
        story.append(Spacer(1, 14))

        # Clinical reasoning
        story.append(Paragraph("CLINICAL REASONING", sec_s))
        story.append(hr())
        story.append(Paragraph(
            str(verdict.get("clinical_reasoning", "No reasoning provided.")), body_s))
        story.append(Spacer(1, 8))

        # Flag reasons
        flags: List[str] = verdict.get("flag_reasons", []) or []
        if flags:
            story.append(Paragraph("FLAG REASONS", sec_s))
            story.append(hr())
            for flag in flags:
                story.append(Paragraph(f"* {flag}", flag_s))
            story.append(Spacer(1, 8))

        # Recommended action
        story.append(Paragraph("RECOMMENDED ACTION", sec_s))
        story.append(hr())
        story.append(Paragraph(
            f"-> {verdict.get('recommended_action', 'No action specified.')}", body_s))
        story.append(Spacer(1, 20))

        # Footer
        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor("#cccccc"), spaceAfter=8))
        story.append(Paragraph(
            "DISCLAIMER: AI-generated. Supports, does not replace, clinical judgment. "
            "Final diagnosis must be made by a qualified radiologist. "
            "BI-RADS per ACR 5th Ed. WHO 2022.", sm_s))
        story.append(Paragraph(
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | "
            f"Groq Llama-4-Scout | Confidential — Physician use only", foot_s))

        doc.build(story)
        print(f"  [Agent2] PDF saved: {out_pdf}")
        return out_pdf

    except ImportError:
        print("  [Agent2] reportlab not installed -> .txt fallback")
        txt = out_pdf.with_suffix(".txt")
        flags_txt: List[str] = verdict.get("flag_reasons", []) or []
        lines = [
            "BREAST IMAGING CENTER — AI-Assisted Breast Ultrasound Report",
            "=" * 60,
            f"Patient ID   : {patient_id}",
            f"Scan Number  : {scan_number}",
            f"Date         : {scan_date}",
            f"Assigned To  : {doctor['name']} ({doctor['role']})",
            "=" * 60,
            f"VERDICT      : {verdict.get('verdict')}  (exit {verdict.get('exit_code')})",
            f"Prediction   : {str(verdict.get('predicted_class','?')).upper()}",
            f"Confidence   : {verdict.get('extracted_confidence')}%",
            f"BI-RADS      : {verdict.get('birads_estimate')}",
            f"Urgency      : {str(verdict.get('urgency','')).upper()}",
            f"Consistency  : {verdict.get('visual_consistency')}",
            "-" * 60,
            "CLINICAL REASONING:",
            str(verdict.get("clinical_reasoning", "")),
            "-" * 60,
            "RECOMMENDED ACTION:",
            str(verdict.get("recommended_action", "")),
            "-" * 60,
            "FLAG REASONS:",
        ] + [f"  - {f}" for f in flags_txt] + [
            "=" * 60,
            "DISCLAIMER: AI-generated. For physician use only.",
        ]
        with open(txt, "w") as f:
            f.write("\n".join(lines))
        return txt


# ====================================================================
# SECTION 3 -- APPOINTMENT SLOTS (kept for backward compatibility)
# ====================================================================

def propose_appointment_slots(urgency: str, doctor: Dict) -> List[Dict]:
    """
    Backward-compatible: returns 3 slots ignoring busy times.
    For new code, use find_next_free_slot() instead.
    """
    now = datetime.datetime.now()
    offsets: Dict[str, List[int]] = {
        "emergency": [0, 0, 1],
        "urgent":    [1, 2, 2],
        "priority":  [3, 4, 5],
        "routine":   [7, 8, 10],
    }
    days  = offsets.get(str(urgency).lower(), [7, 8, 10])
    times = ["09:30 AM", "11:00 AM", "02:30 PM"]

    slots: List[Dict] = []
    for i, d in enumerate(days):
        date_str = (now + datetime.timedelta(days=d)).strftime("%A, %d %B %Y")
        slots.append({
            "slot_id":  f"SLOT-{i+1}",
            "date":     date_str,
            "time":     times[i],
            "doctor":   doctor["name"],
            "role":     doctor["role"],
            "dept":     doctor["dept"],
            "hospital": HOSPITAL_NAME,
            "address":  HOSPITAL_ADDRESS,
            "label":    f"Option {i+1}: {date_str} at {times[i]}",
        })
    return slots


# ====================================================================
# SECTION 4 -- EMAIL
# ====================================================================

def _attach_pdf(msg: MIMEMultipart, pdf_path: Path) -> None:
    if not pdf_path.exists():
        return
    with open(pdf_path, "rb") as f:
        data = f.read()
    part = MIMEBase("application", "octet-stream")
    part.set_payload(data)
    encoders.encode_base64(part)
    part.add_header("Content-Disposition",
                    f'attachment; filename="{pdf_path.name}"')
    msg.attach(part)


def _send_smtp(msg: MIMEMultipart, to_addr: str) -> bool:
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_addr, msg.as_string())
        print(f"  [Agent2] Email sent -> {to_addr}")
        return True
    except smtplib.SMTPAuthenticationError:
        print("  [Agent2] Gmail auth failed. Check App Password.")
        return False
    except Exception as ex:
        print(f"  [Agent2] Email error: {ex}")
        return False


def _build_patient_email(
    chosen_slot: Dict, patient_name: str, patient_email: str,
    verdict: Dict, pdf_path: Path, doctor: Dict,
) -> MIMEMultipart:
    pid        = str(verdict.get("patient_id",  "?"))
    scan       = str(verdict.get("scan_number", "?"))
    slot_date  = str(chosen_slot.get("date",    "?"))
    slot_time  = str(chosen_slot.get("time",    "?"))
    hosp       = str(chosen_slot.get("hospital","?"))
    addr       = str(chosen_slot.get("address", "?"))
    doc_name   = str(chosen_slot.get("doctor",  doctor.get("name","?")))
    doc_role   = str(chosen_slot.get("role",    doctor.get("role","?")))
    doc_dept   = str(chosen_slot.get("dept",    doctor.get("dept","?")))

    html = f"""
<html><body style='font-family:Arial,sans-serif;color:#222;max-width:620px;margin:auto;'>
<div style='background:#0a0a2e;padding:20px 24px;border-radius:8px 8px 0 0;'>
  <h2 style='color:white;margin:0;'>Breast Imaging Center</h2>
  <p style='color:#aaa;margin:4px 0 0;'>Appointment Confirmed &mdash; {HOSPITAL_NAME}</p>
</div>
<div style='padding:24px;border:1px solid #eee;border-top:none;border-radius:0 0 8px 8px;'>
  <p>Dear <b>{patient_name}</b>,</p>
  <p>Your appointment has been confirmed. Details below.</p>
  <div style='background:#f0fff0;border-left:4px solid #1a7a1a;
              padding:14px 18px;border-radius:0 8px 8px 0;margin:16px 0;'>
    <b style='color:#1a7a1a;font-size:15px;'>&#10003; Appointment Confirmed</b><br><br>
    <table style='font-size:13px;width:100%;border-collapse:collapse;'>
      <tr><td style='color:#555;width:130px;padding:4px 0;'>Date</td>
          <td><b>{slot_date}</b></td></tr>
      <tr><td style='color:#555;padding:4px 0;'>Time</td>
          <td><b>{slot_time}</b></td></tr>
      <tr><td style='color:#555;padding:4px 0;'>Hospital</td>
          <td>{hosp}</td></tr>
      <tr><td style='color:#555;padding:4px 0;'>Address</td>
          <td>{addr}</td></tr>
      <tr><td style='color:#555;padding:4px 0;'>Doctor</td>
          <td><b>{doc_name}</b> &mdash; {doc_role}</td></tr>
      <tr><td style='color:#555;padding:4px 0;'>Department</td>
          <td>{doc_dept}</td></tr>
      <tr><td style='color:#555;padding:4px 0;'>Patient ID</td>
          <td>{pid} / {scan}</td></tr>
    </table>
  </div>
  <p style='color:#555;font-size:13px;'>
    Your doctor report is attached as a PDF. Please bring it to your appointment.</p>
  <p style='color:#555;font-size:13px;'>
    &#128276; You will receive a Google Calendar invite with a reminder before your appointment.</p>
  <hr style='border:none;border-top:1px solid #eee;margin:20px 0;'>
  <p style='color:#888;font-size:11px;'>
    Automated notification &mdash; Breast Imaging AI System &nbsp;|&nbsp; {pid} / {scan}</p>
</div></body></html>"""

    msg = MIMEMultipart("mixed")
    msg["Subject"] = f"[CONFIRMED] Your Appointment — {slot_date} at {slot_time} | {HOSPITAL_NAME}"
    msg["From"] = SENDER_EMAIL
    msg["To"]   = patient_email
    msg.attach(MIMEText(html, "html"))
    _attach_pdf(msg, pdf_path)
    return msg


def _build_doctor_email(
    chosen_slot: Dict, patient_name: str, patient_email: str,
    verdict: Dict, pdf_path: Path, doctor: Dict,
) -> MIMEMultipart:
    pid       = str(verdict.get("patient_id",         "?"))
    scan      = str(verdict.get("scan_number",        "?"))
    pred      = str(verdict.get("predicted_class",    "?")).upper()
    conf      = verdict.get("extracted_confidence",   "?")
    birads    = str(verdict.get("birads_estimate",    "?"))
    urgency   = str(verdict.get("urgency",            "routine")).upper()
    reasoning = str(verdict.get("clinical_reasoning", ""))[:600]
    action    = str(verdict.get("recommended_action", ""))
    vl        = str(verdict.get("verdict",            "?"))
    v_hex     = {"APPROVED":"#1a7a1a","FLAGGED":"#cc7700","REJECTED":"#cc0000"}.get(vl,"#555")
    slot_date  = str(chosen_slot.get("date",    "?"))
    slot_time  = str(chosen_slot.get("time",    "?"))
    hosp       = str(chosen_slot.get("hospital","?"))
    addr       = str(chosen_slot.get("address", "?"))

    html = f"""
<html><body style='font-family:Arial,sans-serif;color:#222;max-width:660px;margin:auto;'>
<div style='background:#0a0a2e;padding:20px 24px;border-radius:8px 8px 0 0;'>
  <h2 style='color:white;margin:0;'>Breast Imaging Center</h2>
  <p style='color:#aaa;margin:4px 0 0;'>New Patient Appointment &mdash; AI Report Attached</p>
</div>
<div style='padding:24px;border:1px solid #eee;border-top:none;border-radius:0 0 8px 8px;'>
  <p>Dear <b>{doctor['name']}</b> ({doctor['role']}),</p>
  <p>A new patient appointment has been auto-booked. Full AI report is attached.</p>
  <div style='background:#f4f4f4;border-radius:8px;padding:16px 20px;margin:16px 0;'>
    <b style='font-size:13px;color:#333;'>Patient &amp; Appointment Details</b>
    <table style='width:100%;font-size:13px;border-collapse:collapse;margin-top:8px;'>
      <tr><td style='color:#555;width:150px;padding:3px 0;'>Patient Name</td>
          <td><b>{patient_name}</b></td></tr>
      <tr><td style='color:#555;padding:3px 0;'>Patient Email</td>
          <td>{patient_email}</td></tr>
      <tr><td style='color:#555;padding:3px 0;'>Patient ID</td>
          <td>{pid} / {scan}</td></tr>
      <tr><td style='color:#555;padding:3px 0;'>Appointment Date</td>
          <td><b style='color:#0a0a2e;'>{slot_date}</b></td></tr>
      <tr><td style='color:#555;padding:3px 0;'>Appointment Time</td>
          <td><b style='color:#0a0a2e;'>{slot_time}</b></td></tr>
      <tr><td style='color:#555;padding:3px 0;'>Location</td>
          <td>{hosp} &middot; {addr}</td></tr>
    </table>
  </div>
  <div style='border-left:4px solid {v_hex};background:#fafafa;
              padding:14px 18px;border-radius:0 6px 6px 0;margin:16px 0;'>
    <b style='font-size:15px;color:{v_hex};'>AI Verdict: {vl}</b>
    <table style='font-size:12px;margin-top:10px;width:100%;border-collapse:collapse;'>
      <tr><td style='color:#555;width:150px;padding:3px 0;'>AI Prediction</td>
          <td><b>{pred}</b></td></tr>
      <tr><td style='color:#555;padding:3px 0;'>Confidence</td>
          <td>{conf}%</td></tr>
      <tr><td style='color:#555;padding:3px 0;'>BI-RADS Estimate</td>
          <td><b>{birads}</b></td></tr>
      <tr><td style='color:#555;padding:3px 0;'>Urgency</td>
          <td><b>{urgency}</b></td></tr>
    </table>
  </div>
  <p style='font-size:13px;'><b>Clinical Reasoning:</b></p>
  <p style='font-size:13px;color:#444;background:#f9f9f9;padding:12px;
            border-radius:6px;line-height:1.6;'>{reasoning}</p>
  <p style='font-size:13px;'><b>Recommended Action:</b></p>
  <p style='font-size:13px;color:#444;'>&rarr; {action}</p>
  <hr style='border:none;border-top:1px solid #eee;margin:20px 0;'>
  <p style='color:#888;font-size:11px;'>
    AI-generated &mdash; Groq Llama-4-Scout &nbsp;|&nbsp; BI-RADS 5th Ed &nbsp;|&nbsp; WHO 2022<br>
    <b>DISCLAIMER:</b> Supports, does not replace, clinical judgment.
  </p>
</div></body></html>"""

    msg = MIMEMultipart("mixed")
    msg["Subject"] = f"[NEW PATIENT] {patient_name} | {pid} | {slot_date} {slot_time}"
    msg["From"] = SENDER_EMAIL
    msg["To"]   = doctor["email"]
    msg.attach(MIMEText(html, "html"))
    _attach_pdf(msg, pdf_path)
    return msg


def send_both_emails(
    chosen_slot: Dict, patient_name: str, patient_email: str,
    verdict: Dict, pdf_path: Path, doctor: Dict,
) -> Dict[str, bool]:
    print(f"  [Agent2] Sending confirmation email to patient: {patient_email}")
    patient_msg = _build_patient_email(
        chosen_slot, patient_name, patient_email, verdict, pdf_path, doctor)
    patient_ok = _send_smtp(patient_msg, patient_email)

    print(f"  [Agent2] Sending report email to doctor: {doctor['email']}")
    doctor_msg = _build_doctor_email(
        chosen_slot, patient_name, patient_email, verdict, pdf_path, doctor)
    doctor_ok = _send_smtp(doctor_msg, doctor["email"])

    return {"patient": patient_ok, "doctor": doctor_ok}


# ====================================================================
# SECTION 5 -- APPOINTMENT STATE
# ====================================================================

def save_appointment(
    patient_id: str, scan_number: str, patient_email: str, patient_name: str,
    slots: List[Dict], pdf_path: Path, doctor: Dict, status: str = "pending",
) -> Dict:
    data: Dict = {}
    if APPT_FILE.exists():
        with open(APPT_FILE) as f:
            data = json.load(f)
    key = f"{patient_id}_{scan_number}"
    data[key] = {
        "patient_id":        patient_id,
        "scan_number":       scan_number,
        "patient_name":      patient_name,
        "patient_email":     patient_email,
        "assigned_doctor":   doctor,
        "status":            status,
        "proposed_slots":    slots,
        "chosen_slot":       None,
        "pdf_report":        str(pdf_path),
        "calendar_event_id": None,
        "emails_sent":       False,
        "created_at":        str(datetime.datetime.now())[:16],
        "updated_at":        str(datetime.datetime.now())[:16],
    }
    with open(APPT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return data[key]


def load_appointment(patient_id: str, scan_number: str) -> Optional[Dict]:
    if not APPT_FILE.exists():
        return None
    with open(APPT_FILE) as f:
        data: Dict = json.load(f)
    return data.get(f"{patient_id}_{scan_number}")


def update_appointment(patient_id: str, scan_number: str, updates: Dict) -> None:
    data: Dict = {}
    if APPT_FILE.exists():
        with open(APPT_FILE) as f:
            data = json.load(f)
    key = f"{patient_id}_{scan_number}"
    if key in data:
        data[key].update(updates)
        data[key]["updated_at"] = str(datetime.datetime.now())[:16]
    with open(APPT_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ====================================================================
# SECTION 6 -- GOOGLE CALENDAR
# ====================================================================

def _get_calendar_credentials() -> Any:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request as GRequest

    SCOPES     = ["https://www.googleapis.com/auth/calendar"]
    token_path = SCRIPT_DIR / "token.pickle"
    creds: Any = None

    if token_path.exists():
        with open(token_path, "rb") as tf:
            creds = pickle.load(tf)

    if creds is not None and getattr(creds, "valid", False):
        return creds

    if (creds is not None
            and getattr(creds, "expired", False)
            and getattr(creds, "refresh_token", None)):
        creds.refresh(GRequest())
        with open(token_path, "wb") as tf:
            pickle.dump(creds, tf)
        print("  [Agent2] Calendar token refreshed automatically.")
        return creds

    creds_file = SCRIPT_DIR / GOOGLE_CREDS_FILE
    if not creds_file.exists():
        raise FileNotFoundError(
            f"Google Calendar credentials file not found: {creds_file}\n"
            "  1. Go to https://console.cloud.google.com\n"
            "  2. Enable Google Calendar API\n"
            "  3. Create OAuth 2.0 Client ID (Desktop App)\n"
            f"  4. Download JSON -> rename to '{GOOGLE_CREDS_FILE}'\n"
            "  5. Place it next to agent2_report.py"
        )

    flow  = InstalledAppFlow.from_client_secrets_file(str(creds_file), SCOPES)
    creds = flow.run_local_server(port=0)
    with open(token_path, "wb") as tf:
        pickle.dump(creds, tf)
    print("  [Agent2] Calendar credentials saved to token.pickle.")
    return creds


def book_google_calendar(
    patient_name: str, patient_email: str,
    slot: Dict, verdict: Dict, pdf_path: Path, doctor: Dict,
) -> str:
    try:
        from googleapiclient.discovery import build

        creds   = _get_calendar_credentials()
        service = build("calendar", "v3", credentials=creds)

        slot_dt: datetime.datetime = datetime.datetime.now() + datetime.timedelta(days=7)
        for fmt in ("%A, %d %B %Y %I:%M %p", "%A, %d %B %Y %H:%M"):
            try:
                slot_dt = datetime.datetime.strptime(
                    f"{slot['date']} {slot['time']}", fmt)
                break
            except ValueError:
                pass
        end_dt  = slot_dt + datetime.timedelta(hours=1)
        urgency = str(verdict.get("urgency", "routine")).upper()
        color_map = {"EMERGENCY":"11","URGENT":"6","PRIORITY":"5","ROUTINE":"2"}

        event: Dict[str, Any] = {
            "summary": (
                f"Breast Ultrasound Follow-up — "
                f"{verdict.get('patient_id','?')} [{urgency}]"
            ),
            "location": slot.get("address", HOSPITAL_ADDRESS),
            "description": (
                f"Patient: {patient_name} ({patient_email})\n"
                f"Patient ID: {verdict.get('patient_id','?')}\n"
                f"Doctor: {doctor['name']} ({doctor['role']})\n"
                f"Department: {doctor['dept']}\n\n"
                f"AI Result: {str(verdict.get('predicted_class','?')).upper()} "
                f"({verdict.get('extracted_confidence','?')}%) | "
                f"BI-RADS {verdict.get('birads_estimate','?')} | {urgency}\n\n"
                f"Auto-booked by Breast Imaging AI system."
            ),
            "start": {"dateTime": slot_dt.isoformat(), "timeZone": "Asia/Kolkata"},
            "end":   {"dateTime": end_dt.isoformat(),  "timeZone": "Asia/Kolkata"},
            "attendees": [
                {"email": patient_email,  "displayName": patient_name},
                {"email": doctor["email"],"displayName": doctor["name"]},
            ],
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "email",  "minutes": 24 * 60},
                    {"method": "popup",  "minutes": 60},
                ],
            },
            "colorId": color_map.get(urgency, "2"),
        }

        created = service.events().insert(
            calendarId="primary", body=event, sendUpdates="all",
        ).execute()
        eid = str(created.get("id", ""))
        print(f"  [Agent2] Calendar event created: {eid}")
        return eid

    except ImportError:
        msg = "GCAL_NOT_INSTALLED: pip install google-auth google-auth-oauthlib google-api-python-client"
        print(f"  [Agent2] {msg}")
        return msg
    except FileNotFoundError as ex:
        print(f"  [Agent2] Calendar credentials missing: {ex}")
        return f"GCAL_NO_CREDS: {ex}"
    except Exception as ex:
        print(f"  [Agent2] Calendar error: {ex}")
        return f"GCAL_ERROR: {ex}"


# ====================================================================
# SECTION 7 -- FULLY AUTOMATED RUNNER  (new)
# ====================================================================

def run_agent2_auto(
    patient_id: str,
    scan_number: str,
    patient_name: str,
    patient_email: str,
    scan_image_path: Optional[str] = None,
    verdict: Optional[Dict] = None,
) -> Dict:
    """
    NEW: Fully automated Agent 2.
    Called from the dashboard after Agent 1 completes.
    No manual slot selection needed.

    Steps:
      1. Load or use the provided Agent 1 verdict
      2. Route doctor by BI-RADS
      3. Find next free slot (checks DOCTOR_SCHEDULES + urgency window)
      4. Generate PDF report
      5. Send emails to both patient and doctor
      6. Book Google Calendar event
      7. Save confirmed appointment to appointments.json

    Returns a full result dict.
    """
    print(f"\n{'='*62}")
    print(f"  AGENT 2 AUTO -- Full Pipeline")
    print(f"  Patient : {patient_id}  |  Scan: {scan_number}")
    print(f"{'='*62}")

    # 1 ── Load verdict
    if verdict is None:
        print("[1/6] Loading Agent 1 verdict from disk...")
        verdict = load_verdict(patient_id, scan_number)
    else:
        print("[1/6] Using provided Agent 1 verdict...")

    if verdict is None:
        return {"error": f"No verdict found for {patient_id}/{scan_number}. Run Agent 1 first."}
    if verdict.get("verdict") == "REJECTED":
        return {
            "error": (
                "Agent 1 REJECTED this scan (confidence too low). "
                "Manual review required. No appointment scheduled."
            ),
            "verdict": verdict,
        }

    # 2 ── Route doctor
    birads  = str(verdict.get("birads_estimate", "default"))
    urgency = str(verdict.get("urgency", "routine")).lower()
    doctor  = get_doctor(birads)
    print(f"[2/6] Routed to: {doctor['name']} ({doctor['role']})")

    # 3 ── Find next free slot (schedule-aware)
    print(f"[3/6] Finding next free slot for urgency={urgency}...")
    chosen_slot = find_next_free_slot(doctor["name"], urgency)
    chosen_slot.update({
        "role":  doctor["role"],
        "dept":  doctor["dept"],
        "label": f"{chosen_slot['date']} at {chosen_slot['time']}",
    })
    print(f"       Slot: {chosen_slot['label']}")

    # 4 ── Auto-detect scan image if not provided
    if not scan_image_path:
        num        = patient_id.replace("PT-", "").zfill(3)
        candidates = list(RESULTS_DIR.glob(f"prediction_patient_{num}_*.png"))
        scan_image_path = str(candidates[0]) if candidates else None

    # 5 ── Generate PDF
    print("[4/6] Generating PDF report...")
    try:
        pdf_path = generate_pdf_report(verdict, scan_image_path)
    except Exception as exc:
        pdf_path = DOC_REPORTS / f"{patient_id}_{scan_number}_doctor_report.txt"
        pdf_path.write_text(f"PDF generation failed: {exc}\n\n{json.dumps(verdict,indent=2)}")

    email_ok = {"patient": False, "doctor": False}
    event_id = "NOT_SENT"

    # 6 ── Send emails + book calendar
    print("[5/6] Sending emails and booking calendar...")
    try:
        email_ok = send_both_emails(
            chosen_slot, patient_name, patient_email,
            verdict, Path(str(pdf_path)), doctor)
    except Exception as exc:
        print(f"  [Agent2] Email error: {exc}")

    try:
        event_id = book_google_calendar(
            patient_name, patient_email,
            chosen_slot, verdict, Path(str(pdf_path)), doctor)
    except Exception as exc:
        event_id = f"GCAL_ERROR: {exc}"

    # 7 ── Save confirmed appointment
    print("[6/6] Saving confirmed appointment...")
    data: Dict = {}
    if APPT_FILE.exists():
        with open(APPT_FILE) as f:
            data = json.load(f)
    key = f"{patient_id}_{scan_number}"
    data[key] = {
        "patient_id":        patient_id,
        "scan_number":       scan_number,
        "patient_name":      patient_name,
        "patient_email":     patient_email,
        "assigned_doctor":   doctor,
        "status":            "confirmed",
        "proposed_slots":    [chosen_slot],
        "chosen_slot":       chosen_slot,
        "pdf_report":        str(pdf_path),
        "calendar_event_id": event_id,
        "emails_sent":       True,
        "patient_email_ok":  email_ok["patient"],
        "doctor_email_ok":   email_ok["doctor"],
        "created_at":        str(datetime.datetime.now())[:16],
        "updated_at":        str(datetime.datetime.now())[:16],
    }
    with open(APPT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n  ✓ Pipeline complete.")
    print(f"  Patient email : {email_ok['patient']}")
    print(f"  Doctor email  : {email_ok['doctor']}")
    print(f"  Calendar ID   : {event_id}")

    return {
        "verdict":          verdict,
        "doctor":           doctor,
        "chosen_slot":      chosen_slot,
        "pdf_path":         str(pdf_path),
        "email_ok":         email_ok,
        "event_id":         event_id,
        "patient_id":       patient_id,
        "scan_number":      scan_number,
        "appointment":      data[key],
    }


# ====================================================================
# SECTION 8 -- ORIGINAL RUNNERS (kept for backward compatibility)
# ====================================================================

def run_agent2(
    patient_id: str, scan_number: str,
    patient_name: str, patient_email: str,
    scan_image_path: Optional[str] = None,
) -> Dict:
    """
    Original Agent 2 Step 1 — PDF + proposed slots, no email.
    Kept for backward compatibility. Use run_agent2_auto() for new code.
    """
    print(f"\n[Agent2] run_agent2 (manual flow) for {patient_id}/{scan_number}")
    verdict = load_verdict(patient_id, scan_number)
    if verdict is None:
        return {"error": f"No Agent 1 verdict found for {patient_id}/{scan_number}."}
    if verdict.get("verdict") == "REJECTED":
        return {"error": "Agent 1 REJECTED scan. Human review required.", "verdict": verdict}

    birads  = str(verdict.get("birads_estimate", "default"))
    doctor  = get_doctor(birads)
    urgency = str(verdict.get("urgency", "routine"))

    if not scan_image_path:
        num        = patient_id.replace("PT-", "")
        candidates = list(RESULTS_DIR.glob(f"prediction_patient_{num}_*.png"))
        scan_image_path = str(candidates[0]) if candidates else None

    pdf_path = generate_pdf_report(verdict, scan_image_path)
    slots    = propose_appointment_slots(urgency, doctor)
    appointment = save_appointment(
        patient_id, scan_number, patient_email, patient_name,
        slots, pdf_path, doctor, status="pending",
    )

    return {
        "verdict":        verdict,
        "pdf_path":       str(pdf_path),
        "slots":          slots,
        "email_sent":     False,
        "appointment":    appointment,
        "agent1_verdict": verdict.get("verdict"),
        "doctor":         doctor,
    }


def confirm_appointment(
    patient_id: str, scan_number: str, chosen_slot_id: str,
    patient_name: str, patient_email: str,
) -> Dict:
    """
    Original Agent 2 Step 2 — sends emails after patient picks a slot.
    Kept for backward compatibility.
    """
    print(f"\n[Confirm] {patient_id}/{scan_number} -> slot {chosen_slot_id}")
    appt = load_appointment(patient_id, scan_number)
    if appt is None:
        return {"error": "No pending appointment found. Run Agent 2 first."}
    if appt.get("status") == "confirmed" and appt.get("emails_sent"):
        return {"error": "Already confirmed. No duplicate emails sent."}

    chosen: Optional[Dict] = next(
        (s for s in appt.get("proposed_slots", []) if s.get("slot_id") == chosen_slot_id),
        None,
    )
    if chosen is None:
        return {"error": f"Slot {chosen_slot_id} not found."}

    verdict  = load_verdict(patient_id, scan_number)
    if verdict is None:
        return {"error": "Cannot load Agent 1 verdict."}

    pdf_path: Path = Path(str(appt.get("pdf_report", "")))
    doctor: Dict   = appt.get("assigned_doctor") or get_doctor(
        str(verdict.get("birads_estimate", "default")))

    email_results = send_both_emails(
        chosen, patient_name, patient_email, verdict, pdf_path, doctor)
    event_id = book_google_calendar(
        patient_name, patient_email, chosen, verdict, pdf_path, doctor)
    update_appointment(patient_id, scan_number, {
        "status":            "confirmed",
        "chosen_slot":       chosen,
        "calendar_event_id": event_id,
        "emails_sent":       True,
        "patient_email_ok":  email_results["patient"],
        "doctor_email_ok":   email_results["doctor"],
    })

    return {
        "confirmed":        True,
        "chosen_slot":      chosen,
        "event_id":         event_id,
        "calendar_ok":      event_id and not event_id.startswith(("GCAL_", "ERROR")),
        "pdf_path":         str(pdf_path),
        "doctor":           doctor,
        "patient_email_ok": email_results["patient"],
        "doctor_email_ok":  email_results["doctor"],
    }