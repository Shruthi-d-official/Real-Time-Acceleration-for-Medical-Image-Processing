"""
STREAMLIT DASHBOARD -- Agent 1 + Agent 2
Breast Ultrasound AI Clinical Validation Dashboard

HOW TO RUN:
  pip install streamlit reportlab
  cd E:\\Agentic
  streamlit run streamlit_app.py
"""

import streamlit as st
import json
import sys
import time
from pathlib import Path
from PIL import Image
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent))
from agent1_validation import run_agent1, RESULTS_DIR, REPORTS_DIR, GROQ_MODEL
from agent2_report import run_agent2, confirm_appointment, load_appointment

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Breast Ultrasound AI -- Validation Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# CSS
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.verdict-approved {
    background: #0a1f0a; border: 1px solid #1a7a1a;
    border-radius: 8px; padding: 1.2rem 1.5rem; color: #4dff4d;
}
.verdict-flagged {
    background: #1f1500; border: 1px solid #cc7700;
    border-radius: 8px; padding: 1.2rem 1.5rem; color: #ffaa00;
}
.verdict-rejected {
    background: #1f0000; border: 1px solid #cc0000;
    border-radius: 8px; padding: 1.2rem 1.5rem; color: #ff4444;
}
.verdict-quota {
    background: #1a1a00; border: 1px solid #888800;
    border-radius: 8px; padding: 1.2rem 1.5rem; color: #ffff44;
}
.metric-box {
    background: #1a1a1a; border: 1px solid #2a2a2a;
    border-radius: 6px; padding: 0.8rem 1rem; text-align: center;
}
.metric-label {
    font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.1em; color: #666; margin-bottom: 4px;
}
.metric-value {
    font-size: 1.3rem; font-weight: 600; color: #eee;
    font-family: 'IBM Plex Mono', monospace;
}
.reasoning-box {
    background: #141414; border-left: 3px solid #444;
    border-radius: 0 6px 6px 0; padding: 1rem 1.2rem;
    font-size: 0.88rem; color: #bbb; line-height: 1.7;
}
.flag-item {
    background: #1f1200; border: 1px solid #cc6600;
    border-radius: 4px; padding: 0.4rem 0.8rem;
    font-size: 0.82rem; color: #ffaa44; margin: 4px 0;
}
.section-header {
    font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.12em; color: #555;
    margin: 1.2rem 0 0.5rem;
    border-bottom: 1px solid #222; padding-bottom: 4px;
}
.history-item {
    background: #161616; border: 1px solid #2a2a2a;
    border-radius: 4px; padding: 0.5rem 0.8rem; margin: 4px 0;
    font-size: 0.82rem; color: #aaa;
    font-family: 'IBM Plex Mono', monospace;
}
.slot-preview {
    background: #111827; border: 1px solid #1e3a5f;
    border-radius: 8px; padding: 1rem 1.2rem; margin: 10px 0;
    font-size: 0.87rem; color: #c9d1d9; line-height: 1.8;
}
.confirmed-box {
    background: #0a1f0a; border: 2px solid #1a7a1a;
    border-radius: 10px; padding: 1.4rem 1.6rem; margin-top: 4px;
    text-align: center;
}
.doctor-badge {
    background: #111827; border: 1px solid #2a3f5f;
    border-radius: 6px; padding: 0.6rem 1rem; margin: 8px 0;
    font-size: 0.83rem; color: #88aadd;
}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# HELPERS
# =====================================================================

def verdict_icon(v: str) -> str:
    return {"APPROVED": "✓", "FLAGGED": "⚠", "REJECTED": "✕",
            "QUOTA_EXCEEDED": "⏳"}.get(v, "?")

def verdict_css(v: str) -> str:
    return {"APPROVED": "verdict-approved", "FLAGGED": "verdict-flagged",
            "REJECTED": "verdict-rejected", "QUOTA_EXCEEDED": "verdict-quota",
            }.get(v, "verdict-rejected")

def conf_color(c: object) -> str:
    if not isinstance(c, (int, float)): return "#888"
    if c >= 85: return "#4dff4d"
    if c >= 70: return "#ffaa00"
    return "#ff4444"

def birads_color(b: str) -> str:
    return {"1-2": "#4dff4d", "3": "#88ccff", "4A": "#ffcc44",
            "4B": "#ff8800", "4C": "#ff4400", "5": "#ff0000"}.get(str(b), "#888")

def urgency_color(u: str) -> str:
    return {"EMERGENCY": "#ff0000", "URGENT": "#ff4400",
            "PRIORITY": "#ffaa00", "ROUTINE": "#4dff4d"}.get(str(u).upper(), "#888")

def load_patient_history(patient_id: str) -> list:
    hf = Path(__file__).parent / "patient_history.json"
    if not hf.exists():
        return []
    with open(hf) as f:
        data = json.load(f)
    return data.get(patient_id, {}).get("scans", [])

def is_valid_email(email: str) -> bool:
    email = email.strip()
    if "@" not in email:
        return False
    parts = email.split("@")
    if len(parts) != 2:
        return False
    domain = parts[1]
    if "." not in domain:
        return False
    if len(parts[0]) == 0 or len(domain) < 3:
        return False
    return True

def safe_str(val: object, default: str = "") -> str:
    """
    Safely convert session_state values to str.
    Fixes Pylance 'Any | Unknown | None is not assignable to str' errors.
    """
    if val is None:
        return default
    return str(val)

def get_next_patient_id() -> str:
    """Generate next patient ID"""
    existing = sorted(RESULTS_DIR.glob("prediction_patient_*.png"))
    if not existing:
        return "PT-001"
    nums = []
    for img in existing:
        try:
            num_str = img.name.split("_")[2]
            nums.append(int(num_str))
        except:
            pass
    return f"PT-{max(nums, default=0) + 1:03d}"


# =====================================================================
# SESSION STATE INIT
# =====================================================================

def _init_state() -> None:
    defaults: dict = {
        "current_verdict":  None,
        "current_image":    None,
        "agent2_result":    None,
        "booking_done":     False,
        "booking_result":   None,
        "selected_slot":    None,
        "last_patient_id":  "",
        "last_scan":        "",
        "last_pname":       "",
        "last_pemail":      "",
        "mode":             "existing",
        "new_patient_file": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


# =====================================================================
# PAGE HEADER
# =====================================================================

st.markdown(
    "<h1 style='font-size:1.4rem;font-weight:500;color:#ddd;margin-bottom:0;'>"
    "Breast Ultrasound AI &mdash; Clinical Validation Dashboard</h1>"
    "<p style='color:#555;font-size:0.8rem;margin-top:2px;'>"
    "Agent 1: Groq Llama-4-Scout &nbsp;|&nbsp; "
    "Agent 2: PDF Report + Email + Calendar &nbsp;|&nbsp; "
    "BI-RADS 5th Ed &nbsp;|&nbsp; WHO 2022</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# =====================================================================
# SIDEBAR — MODE SELECTOR
# =====================================================================

with st.sidebar:
    st.markdown("### 🔬 Validation Dashboard")
    st.markdown(
        f"<div style='font-size:0.72rem;color:#555;margin-bottom:8px;'>"
        f"Groq · {GROQ_MODEL}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    mode: str = st.radio(
        "Select Mode",
        ["Existing Patient", "New Patient"],
        horizontal=False,
    )
    st.session_state.mode = mode
    st.markdown("---")


# =====================================================================
# MODE 1: EXISTING PATIENT
# =====================================================================

if st.session_state.mode == "Existing Patient":
    with st.sidebar:
        all_images = sorted(RESULTS_DIR.glob("prediction_patient_*.png"))
        if not all_images:
            st.error(
                f"No scan images found.\n\n"
                f"Place PNG files in:\n`{RESULTS_DIR}`\n\n"
                f"Filename format:\n`prediction_patient_001_S1.png`"
            )
            st.stop()

        selected_img = st.selectbox(
            "Select patient scan",
            [img.name for img in all_images],
            format_func=lambda x: x.replace("prediction_", "").replace(".png", ""),
        )
        if selected_img is None:
            st.error("No images available. Please try again.")
            st.stop()
        selected_img: str = str(selected_img)

        st.markdown("---")
        risk_factors: str = st.text_input(
            "Risk factors (optional)",
            placeholder="e.g. BRCA1, family history",
        )
        run_agent1_btn: bool = st.button(
            "▶  Run Agent 1 Validation", use_container_width=True, type="primary"
        )

        st.markdown("---")
        st.markdown("**Agent 2 — Report + Booking**")
        patient_name: str  = st.text_input("Patient name",  placeholder="e.g. Shruthi Dhandapani", key="pname")
        patient_email: str = st.text_input("Patient email", placeholder="e.g. shruthi@gmail.com",  key="pemail")

        email_ok: bool = is_valid_email(patient_email) if patient_email.strip() else False
        name_ok:  bool = len(patient_name.strip()) > 0

        if patient_email.strip() and not email_ok:
            st.markdown(
                "<div style='color:#ff6666;font-size:0.78rem;margin-top:-8px;'>"
                "⚠ Enter a valid email address</div>",
                unsafe_allow_html=True,
            )

        run_agent2_btn: bool = st.button("▶  Run Agent 2", use_container_width=True)

        st.markdown("---")
        run_batch_btn: bool = st.button("Run ALL images (Agent 1)", use_container_width=True)

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.68rem;color:#333;'>"
            "Agent 1 · Groq Llama-4-Scout (Free)<br>"
            "Agent 2 · PDF + Email + Calendar<br>"
            "BI-RADS 5th Ed · WHO 2022"
            "</div>",
            unsafe_allow_html=True,
        )

    image_path = RESULTS_DIR / selected_img

    # ─────────────────────────────────────────────────────────────────
    # BATCH MODE
    # ─────────────────────────────────────────────────────────────────

    if run_batch_btn:
        images = sorted(RESULTS_DIR.glob("prediction_patient_*.png"))
        st.markdown(f"### Batch validation — {len(images)} image(s)")
        progress = st.progress(0)
        status   = st.empty()
        results  = []
        for i, img in enumerate(images):
            status.text(f"Processing {img.name} ...")
            v = run_agent1(img)
            if v:
                results.append(v)
            progress.progress((i + 1) / len(images))
            time.sleep(1)
        status.text("Batch complete!")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total",      len(results))
        c2.metric("Approved",   sum(1 for v in results if v.get("verdict") == "APPROVED"))
        c3.metric("Flagged",    sum(1 for v in results if v.get("verdict") == "FLAGGED"))
        c4.metric("Rejected",   sum(1 for v in results if v.get("verdict") == "REJECTED"))
        c5.metric("Rate Limit", sum(1 for v in results if v.get("quota_exceeded")))
        st.dataframe(
            [{
                "Patient":    v.get("patient_id", "?"),
                "Scan":       v.get("scan_number", "?"),
                "Prediction": str(v.get("predicted_class", "?")).upper(),
                "Confidence": f"{v.get('extracted_confidence', '?')}%",
                "BI-RADS":    v.get("birads_estimate", "?"),
                "Verdict":    v.get("verdict", "?"),
                "Urgency":    str(v.get("urgency", "?")).upper(),
            } for v in results],
            use_container_width=True,
        )
        st.stop()

    # ─────────────────────────────────────────────────────────────────
    # CLEAR STATE WHEN SCAN CHANGES
    # ─────────────────────────────────────────────────────────────────

    if safe_str(st.session_state.current_image) != selected_img:
        st.session_state.current_image   = selected_img
        st.session_state.current_verdict = None
        st.session_state.agent2_result   = None
        st.session_state.booking_done    = False
        st.session_state.booking_result  = None
        st.session_state.selected_slot   = None

    # ─────────────────────────────────────────────────────────────────
    # RUN AGENT 1
    # ─────────────────────────────────────────────────────────────────

    if run_agent1_btn:
        st.session_state.agent2_result  = None
        st.session_state.booking_done   = False
        st.session_state.booking_result = None
        st.session_state.selected_slot  = None

        with st.spinner("Calling Groq Vision API (Llama-4-Scout)..."):
            agent1_result = run_agent1(image_path, risk_factors=risk_factors or "none")

        if agent1_result is None:
            st.error("Agent 1 returned no result. Check terminal for errors.")
        elif agent1_result.get("quota_exceeded"):
            st.warning("⏳ Groq rate limit hit. Wait 60 seconds and try again.")
            st.session_state.current_verdict = agent1_result
        else:
            st.success("✅ Agent 1 validation complete.")
            st.session_state.current_verdict = agent1_result

    verdict: dict = st.session_state.current_verdict or {}

    # ─────────────────────────────────────────────────────────────────
    # RUN AGENT 2
    # ─────────────────────────────────────────────────────────────────

    if run_agent2_btn:
        if not name_ok:
            st.sidebar.error("Please enter the patient name.")
        elif not email_ok:
            st.sidebar.error("Please enter a valid email (e.g. name@gmail.com).")
        elif not verdict:
            st.sidebar.error("Run Agent 1 successfully first.")
        elif verdict.get("quota_exceeded"):
            st.sidebar.error("Agent 1 rate limited. Retry Agent 1 first.")
        elif verdict.get("verdict") == "REJECTED":
            st.sidebar.error("Scan REJECTED by Agent 1. Human review required.")
        else:
            st.session_state.booking_done   = False
            st.session_state.booking_result = None
            st.session_state.selected_slot  = None

            pid:  str = str(verdict.get("patient_id",  "PT-000"))
            scan: str = str(verdict.get("scan_number", "S1"))

            st.session_state.last_patient_id = pid
            st.session_state.last_scan       = scan
            st.session_state.last_pname      = patient_name.strip()
            st.session_state.last_pemail     = patient_email.strip()

            with st.spinner("Agent 2: generating PDF and preparing appointment slots..."):
                a2_result = run_agent2(
                    patient_id      = pid,
                    scan_number     = scan,
                    patient_name    = patient_name.strip(),
                    patient_email   = patient_email.strip(),
                    scan_image_path = str(image_path),
                )

            if "error" in a2_result:
                st.error(f"Agent 2 error: {a2_result['error']}")
            else:
                st.session_state.agent2_result = a2_result
                st.success(
                    "✅ PDF generated. Select a slot below and click "
                    "**Confirm** to book and send emails."
                )

    # ─────────────────────────────────────────────────────────────────
    # MAIN LAYOUT — Image + Verdict
    # ─────────────────────────────────────────────────────────────────

    col_img, col_right = st.columns([1.1, 0.9], gap="large")

    with col_img:
        st.markdown(
            "<div class='section-header'>Model output image (6-panel)</div>",
            unsafe_allow_html=True,
        )
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
            st.markdown(
                f"<div style='font-size:0.72rem;color:#444;font-family:monospace;'>"
                f"{image_path.name}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning(f"Image not found: {image_path}")

    with col_right:

        if not verdict:
            st.markdown(
                "<div style='border:1px dashed #2a2a2a;border-radius:10px;"
                "padding:3rem 2rem;text-align:center;margin-top:1rem;'>"
                "<div style='font-size:2.5rem;margin-bottom:12px;'>🔬</div>"
                "<div style='color:#555;font-size:0.95rem;font-weight:500;'>"
                "No scan analysed yet</div>"
                "<div style='color:#333;font-size:0.82rem;margin-top:8px;'>"
                "Click <b style='color:#666;'>▶ Run Agent 1 Validation</b> "
                "in the sidebar to begin."
                "</div></div>",
                unsafe_allow_html=True,
            )

        elif verdict.get("quota_exceeded"):
            st.markdown(
                "<div class='verdict-quota'>"
                "<div style='font-size:0.7rem;letter-spacing:0.12em;opacity:0.7;'>STATUS</div>"
                "<div style='font-size:1.5rem;font-weight:600;font-family:IBM Plex Mono,monospace;'>"
                "⏳ RATE LIMITED</div>"
                "<div style='font-size:0.8rem;opacity:0.7;margin-top:4px;'>"
                "Wait 60s and retry</div></div>",
                unsafe_allow_html=True,
            )

        else:
            verd    = str(verdict.get("verdict",    "UNKNOWN"))
            conf    = verdict.get("extracted_confidence")
            birads  = str(verdict.get("birads_estimate", "?"))
            urgency = str(verdict.get("urgency",    "?")).upper()
            pred    = str(verdict.get("predicted_class", "?")).upper()
            pid_lbl = str(verdict.get("patient_id", "?"))
            scan_lbl= str(verdict.get("scan_number","?"))

            st.markdown(
                f"<div class='{verdict_css(verd)}'>"
                f"<div style='font-size:0.7rem;letter-spacing:0.12em;opacity:0.7;'>VERDICT</div>"
                f"<div style='font-size:1.8rem;font-weight:600;font-family:IBM Plex Mono,monospace;'>"
                f"{verdict_icon(verd)} {verd}</div>"
                f"<div style='font-size:0.8rem;opacity:0.7;margin-top:4px;'>"
                f"Patient {pid_lbl} &nbsp;·&nbsp; Scan {scan_lbl}</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                conf_disp = "N/A" if conf is None else f"{conf}%"
                st.markdown(
                    f"<div class='metric-box'><div class='metric-label'>Confidence</div>"
                    f"<div class='metric-value' style='color:{conf_color(conf)};'>"
                    f"{conf_disp}</div></div>",
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"<div class='metric-box'><div class='metric-label'>BI-RADS</div>"
                    f"<div class='metric-value' style='color:{birads_color(birads)};'>"
                    f"{birads}</div></div>",
                    unsafe_allow_html=True,
                )
            with m3:
                pc = "#ff4444" if pred == "MALIGNANT" else "#4dff4d"
                st.markdown(
                    f"<div class='metric-box'><div class='metric-label'>Prediction</div>"
                    f"<div class='metric-value' style='color:{pc};font-size:0.95rem;'>"
                    f"{pred}</div></div>",
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f"<div class='metric-box'><div class='metric-label'>Urgency</div>"
                    f"<div class='metric-value' "
                    f"style='color:{urgency_color(urgency)};font-size:0.85rem;'>"
                    f"{urgency}</div></div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<div class='section-header'>Visual consistency</div>",
                unsafe_allow_html=True,
            )
            vc     = str(verdict.get("visual_consistency", "?"))
            vc_col = {"consistent": "#4dff4d", "inconsistent": "#ff4444",
                      "ambiguous": "#ffaa00"}.get(vc, "#888")
            st.markdown(
                f"<span style='color:{vc_col};font-family:monospace;'>● {vc.upper()}</span>",
                unsafe_allow_html=True,
            )

            if isinstance(conf, (int, float)):
                pct = min(int(conf), 100)
                st.markdown(
                    f"<div style='margin:8px 0 4px;'>"
                    f"<div style='height:6px;background:#222;border-radius:3px;overflow:hidden;'>"
                    f"<div style='width:{pct}%;height:100%;"
                    f"background:{conf_color(conf)};border-radius:3px;'></div>"
                    f"</div>"
                    f"<div style='font-size:0.68rem;color:#444;margin-top:2px;'>"
                    f"&ge;85%=APPROVED &nbsp;70-84%=FLAGGED &nbsp;&lt;70%=REJECTED</div></div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<div class='section-header'>Clinical reasoning</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='reasoning-box'>"
                f"{verdict.get('clinical_reasoning', 'No reasoning provided.')}</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div class='section-header'>Recommended action</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='background:#111;border:1px solid #2a2a2a;border-radius:6px;"
                f"padding:0.7rem 1rem;font-size:0.87rem;color:#ccc;'>"
                f"&rarr; {verdict.get('recommended_action', 'No action specified.')}</div>",
                unsafe_allow_html=True,
            )

            flags = verdict.get("flag_reasons", []) or []
            if flags:
                st.markdown(
                    "<div class='section-header'>Flag reasons</div>",
                    unsafe_allow_html=True,
                )
                for flag in flags:
                    st.markdown(
                        f"<div class='flag-item'>&#9873; {flag}</div>",
                        unsafe_allow_html=True,
                    )

            history = load_patient_history(pid_lbl)
            if len(history) > 1:
                st.markdown(
                    "<div class='section-header'>Patient history</div>",
                    unsafe_allow_html=True,
                )
                for s in history[-5:]:
                    st.markdown(
                        f"<div class='history-item'>"
                        f"Scan {s.get('scan_number','?')} [{s.get('date','?')}] "
                        f"{str(s.get('predicted_class','?')).upper()} "
                        f"BI-RADS {s.get('birads_estimate','?')} "
                        f"&rarr; {s.get('verdict','?')}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)
            report_file = REPORTS_DIR / f"{pid_lbl}_{scan_lbl}_verdict.json"
            if report_file.exists():
                with open(report_file) as f:
                    st.download_button(
                        "⬇  Download JSON Report",
                        data=f.read(),
                        file_name=report_file.name,
                        mime="application/json",
                        use_container_width=True,
                    )

    # ─────────────────────────────────────────────────────────────────
    # AGENT 2 — BOOKING SECTION
    # ─────────────────────────────────────────────────────────────────

    a2: dict = st.session_state.agent2_result or {}

    if a2 and "error" not in a2:
        st.markdown("---")

        doc_col, info_col = st.columns([1, 1], gap="large")

        with doc_col:
            st.markdown(
                "<div class='section-header'>Agent 2 — Doctor Report</div>",
                unsafe_allow_html=True,
            )

            pdf_path_str: str = safe_str(a2.get("pdf_path", ""))
            if pdf_path_str and Path(pdf_path_str).exists():
                with open(pdf_path_str, "rb") as f:
                    st.download_button(
                        "⬇  Download Doctor PDF Report",
                        data=f,
                        file_name=Path(pdf_path_str).name,
                        mime="application/pdf",
                        use_container_width=True,
                    )

            doctor: dict = a2.get("doctor") or {}
            if doctor:
                st.markdown(
                    f"<div class='doctor-badge'>"
                    f"🩺 <b>Assigned Doctor:</b> {doctor.get('name','?')}<br>"
                    f"&nbsp;&nbsp;&nbsp;&nbsp;<b>Role:</b> {doctor.get('role','?')}<br>"
                    f"&nbsp;&nbsp;&nbsp;&nbsp;<b>Dept:</b> {doctor.get('dept','?')}<br>"
                    f"&nbsp;&nbsp;&nbsp;&nbsp;<b>Email:</b> {doctor.get('email','?')}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            if st.session_state.booking_done and st.session_state.booking_result:
                br: dict   = st.session_state.booking_result or {}
                pat_ok: bool = bool(br.get("patient_email_ok", False))
                doc_ok: bool = bool(br.get("doctor_email_ok",  False))
                pemail_disp: str = safe_str(st.session_state.last_pemail)
                demail_disp: str = safe_str(doctor.get("email", "?"))

                st.markdown(
                    "<div class='section-header'>Emails sent</div>",
                    unsafe_allow_html=True,
                )
                pat_icon = "✓" if pat_ok else "✕"
                doc_icon = "✓" if doc_ok else "✕"
                pat_col  = "#4dff4d" if pat_ok else "#ff6644"
                doc_col2 = "#4dff4d" if doc_ok else "#ff6644"
                st.markdown(
                    f"<div style='font-size:0.84rem;line-height:2.2;'>"
                    f"<span style='color:{pat_col};'>"
                    f"{pat_icon} Patient: {pemail_disp}</span><br>"
                    f"<span style='color:{doc_col2};'>"
                    f"{doc_icon} Doctor: {demail_disp}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        with info_col:
            st.markdown(
                "<div class='section-header'>Agent 2 — Appointment Booking</div>",
                unsafe_allow_html=True,
            )

            if st.session_state.booking_done and st.session_state.booking_result:
                br2: dict    = st.session_state.booking_result or {}
                chosen: dict = br2.get("chosen_slot") or {}
                eid: str     = safe_str(br2.get("event_id", ""))

                st.markdown(
                    f"<div class='confirmed-box'>"
                    f"<div style='font-size:2.2rem;'>✅</div>"
                    f"<div style='font-size:1.25rem;font-weight:600;"
                    f"color:#4dff4d;margin:10px 0 6px;'>Appointment Confirmed!</div>"
                    f"<div style='font-size:0.92rem;color:#ccc;line-height:2;'>"
                    f"<b>{chosen.get('date','?')}</b> at <b>{chosen.get('time','?')}</b><br>"
                    f"🩺 {chosen.get('doctor','?')}<br>"
                    f"🏥 {chosen.get('hospital','?')}<br>"
                    f"📍 {chosen.get('address','?')}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)

                if eid and not eid.startswith(("GCAL", "ERROR")):
                    st.markdown(
                        f"<div style='font-size:0.83rem;color:#4dff4d;'>"
                        f"📅 Google Calendar invite sent to patient + doctor<br>"
                        f"<span style='font-family:monospace;font-size:0.72rem;color:#555;'>"
                        f"Event ID: {eid}</span></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div style='font-size:0.83rem;color:#888;'>"
                        "📅 Calendar not configured — confirmation sent via email.</div>",
                        unsafe_allow_html=True,
                    )

            else:
                slots: list = a2.get("slots") or []
                if slots:
                    st.markdown(
                        "<div style='font-size:0.84rem;color:#888;margin-bottom:10px;'>"
                        "Select a slot and click <b>Confirm</b>.<br>"
                        "Emails will be sent to <b>both patient and doctor</b> "
                        "only after confirmation."
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    slot_labels: list = [str(s.get("label", "")) for s in slots]
                    slot_map:    dict = {str(s.get("label", "")): s for s in slots}

                    chosen_label = st.selectbox(
                        "📅 Select appointment slot",
                        options=slot_labels,
                        index=0,
                        key="slot_dropdown",
                    )
                    if chosen_label is None:
                        chosen_label = slot_labels[0] if slot_labels else ""

                    chosen_preview: dict = slot_map.get(str(chosen_label), {})

                    st.markdown(
                        f"<div class='slot-preview'>"
                        f"📆 <b>{chosen_preview.get('date','?')}</b> at "
                        f"<b>{chosen_preview.get('time','?')}</b><br>"
                        f"🏥 {chosen_preview.get('hospital','?')} &nbsp;·&nbsp; "
                        f"{chosen_preview.get('address','?')}<br>"
                        f"🩺 <b>{chosen_preview.get('doctor','?')}</b> — "
                        f"{chosen_preview.get('role','')}<br>"
                        f"🏬 {chosen_preview.get('dept','')}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        "<div style='font-size:0.78rem;color:#444;margin:6px 0 14px;'>"
                        "📧 On confirmation: patient receives slot details + PDF. "
                        "Doctor receives full AI report + PDF."
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    if st.button(
                        "✓  Confirm This Appointment",
                        type="primary",
                        use_container_width=True,
                        key="confirm_booking_btn",
                    ):
                        pid2:   str = safe_str(st.session_state.last_patient_id, "PT-000")
                        scan2:  str = safe_str(st.session_state.last_scan,       "S1")
                        pname2: str = safe_str(st.session_state.last_pname,      "")
                        peml2:  str = safe_str(st.session_state.last_pemail,     "")
                        sid:    str = safe_str(chosen_preview.get("slot_id", "SLOT-1"))

                        if not pid2 or not scan2 or not pname2 or not peml2:
                            st.error(
                                "Missing patient details. "
                                "Please fill in name and email in the sidebar and run Agent 2 again."
                            )
                        else:
                            with st.spinner(
                                "Confirming appointment and sending emails "
                                "to patient + doctor..."
                            ):
                                confirm_result: dict = confirm_appointment(
                                    patient_id     = pid2,
                                    scan_number    = scan2,
                                    chosen_slot_id = sid,
                                    patient_name   = pname2,
                                    patient_email  = peml2,
                                )

                            if confirm_result.get("confirmed"):
                                st.session_state.booking_done   = True
                                st.session_state.booking_result = confirm_result
                                st.rerun()
                            else:
                                st.error(
                                    confirm_result.get("error", "Booking failed. Check terminal.")
                                )


# =====================================================================
# MODE 2: NEW PATIENT
# =====================================================================

elif st.session_state.mode == "New Patient":
    with st.sidebar:
        st.markdown("**Patient Information**")
        patient_name: str = st.text_input("Full Name *", placeholder="e.g. Ananya Krishnan", key="new_name")
        patient_email: str = st.text_input("Email Address *", placeholder="ananya@email.com", key="new_email")
        patient_age: int = st.number_input("Age", min_value=18, max_value=100, value=45, key="new_age")

        patient_id: str = st.text_input(
            "Patient ID (auto-generated if blank)",
            placeholder="PT-042",
            value="",
            key="new_id",
        )

        risk_factors_new: str = st.text_input(
            "Risk Factors (optional)",
            placeholder="e.g. BRCA1, family history",
            key="new_rf",
        )

        st.markdown("**Scan File**")
        uploaded_file = st.file_uploader(
            "Upload scan image (PNG, JPG)",
            type=["png", "jpg", "jpeg"],
        )

        st.markdown("---")

        run_new_patient_btn: bool = st.button(
            "🚀 Analyse New Patient",
            use_container_width=True,
            type="primary",
        )

    # Handle new patient submission
    if run_new_patient_btn:
        errors = []

        if not patient_name.strip():
            errors.append("Patient full name is required.")
        if not patient_email.strip() or not is_valid_email(patient_email):
            errors.append("A valid email address is required.")
        if uploaded_file is None:
            errors.append("Please upload a scan image.")

        if errors:
            st.markdown("### ❌ Validation Errors")
            for err in errors:
                st.error(err)
        else:
            # Auto-generate patient ID if not provided
            final_pid = patient_id.strip() if patient_id.strip() else get_next_patient_id()
            num_part = final_pid.replace("PT-", "").zfill(3)
            out_png_path = RESULTS_DIR / f"prediction_patient_{num_part}_S1.png"

            RESULTS_DIR.mkdir(parents=True, exist_ok=True)

            # Save uploaded image
            if uploaded_file is not None:
                out_png_path.write_bytes(uploaded_file.getbuffer())
            else:
                st.error("❌ File upload error. Please try again.")
                st.stop()

            st.markdown(f"### 🆕 New Patient: {final_pid}")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # ─────────────────────────────────────────────────────────
            # RUN AGENT 1
            # ─────────────────────────────────────────────────────────

            status_text.text("Running Agent 1 AI Analysis...")
            progress_bar.progress(30)

            a1_result = run_agent1(out_png_path, risk_factors=risk_factors_new or "none")

            if a1_result is None:
                progress_bar.empty()
                status_text.empty()
                st.error("❌ Agent 1 returned no result. Check terminal for errors.")
            elif a1_result.get("quota_exceeded"):
                progress_bar.empty()
                status_text.empty()
                st.warning("⏳ Groq rate limit hit. Please wait 60 seconds and try again.")
            else:
                progress_bar.progress(60)
                status_text.text("Agent 1 complete. Running Agent 2...")

                # Store result
                st.session_state.current_verdict = a1_result
                st.session_state.last_patient_id = final_pid
                st.session_state.last_scan = "S1"
                st.session_state.last_pname = patient_name.strip()
                st.session_state.last_pemail = patient_email.strip()

                # ─────────────────────────────────────────────────────
                # RUN AGENT 2
                # ─────────────────────────────────────────────────────

                a2_result = run_agent2(
                    patient_id=final_pid,
                    scan_number="S1",
                    patient_name=patient_name.strip(),
                    patient_email=patient_email.strip(),
                    scan_image_path=str(out_png_path),
                )

                progress_bar.progress(90)

                if "error" in a2_result:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Agent 2 error: {a2_result['error']}")
                else:
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()

                    st.session_state.agent2_result = a2_result

                    # ─────────────────────────────────────────────────
                    # DISPLAY RESULTS
                    # ─────────────────────────────────────────────────

                    st.success("✅ New patient registered and analysis complete!")

                    col_img, col_right = st.columns([1.1, 0.9], gap="large")

                    with col_img:
                        st.markdown(
                            "<div class='section-header'>6-Panel Model Output</div>",
                            unsafe_allow_html=True,
                        )
                        st.image(str(out_png_path), use_container_width=True)

                    with col_right:
                        verd = str(a1_result.get("verdict", "UNKNOWN"))
                        conf = a1_result.get("extracted_confidence")
                        birads = str(a1_result.get("birads_estimate", "?"))
                        urgency = str(a1_result.get("urgency", "?")).upper()
                        pred = str(a1_result.get("predicted_class", "?")).upper()

                        st.markdown(
                            f"<div class='{verdict_css(verd)}'>"
                            f"<div style='font-size:0.7rem;letter-spacing:0.12em;opacity:0.7;'>VERDICT</div>"
                            f"<div style='font-size:1.8rem;font-weight:600;font-family:IBM Plex Mono,monospace;'>"
                            f"{verdict_icon(verd)} {verd}</div>"
                            f"<div style='font-size:0.8rem;opacity:0.7;margin-top:4px;'>"
                            f"Patient {final_pid} &nbsp;·&nbsp; Scan S1</div></div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("<br>", unsafe_allow_html=True)

                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            conf_disp = "N/A" if conf is None else f"{conf}%"
                            st.markdown(
                                f"<div class='metric-box'><div class='metric-label'>Confidence</div>"
                                f"<div class='metric-value' style='color:{conf_color(conf)};'>"
                                f"{conf_disp}</div></div>",
                                unsafe_allow_html=True,
                            )
                        with m2:
                            st.markdown(
                                f"<div class='metric-box'><div class='metric-label'>BI-RADS</div>"
                                f"<div class='metric-value' style='color:{birads_color(birads)};'>"
                                f"{birads}</div></div>",
                                unsafe_allow_html=True,
                            )
                        with m3:
                            pc = "#ff4444" if pred == "MALIGNANT" else "#4dff4d"
                            st.markdown(
                                f"<div class='metric-box'><div class='metric-label'>Prediction</div>"
                                f"<div class='metric-value' style='color:{pc};font-size:0.95rem;'>"
                                f"{pred}</div></div>",
                                unsafe_allow_html=True,
                            )
                        with m4:
                            st.markdown(
                                f"<div class='metric-box'><div class='metric-label'>Urgency</div>"
                                f"<div class='metric-value' "
                                f"style='color:{urgency_color(urgency)};font-size:0.85rem;'>"
                                f"{urgency}</div></div>",
                                unsafe_allow_html=True,
                            )

                        st.markdown(
                            f"<div class='section-header'>Clinical Reasoning</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div class='reasoning-box'>"
                            f"{a1_result.get('clinical_reasoning', 'No reasoning provided.')}</div>",
                            unsafe_allow_html=True,
                        )

                    # Booking section
                    st.markdown("---")
                    st.markdown("### 📅 Appointment Booking")

                    slots: list = a2_result.get("slots") or []
                    if slots:
                        slot_labels: list = [str(s.get("label", "")) for s in slots]
                        slot_map: dict = {str(s.get("label", "")): s for s in slots}

                        chosen_label = st.selectbox(
                            "Select appointment slot",
                            options=slot_labels,
                            key="new_slot_dropdown",
                        )
                        if chosen_label is None:
                            chosen_label = slot_labels[0] if slot_labels else ""

                        chosen_preview: dict = slot_map.get(str(chosen_label), {})

                        st.markdown(
                            f"<div class='slot-preview'>"
                            f"📆 <b>{chosen_preview.get('date','?')}</b> at "
                            f"<b>{chosen_preview.get('time','?')}</b><br>"
                            f"🏥 {chosen_preview.get('hospital','?')}<br>"
                            f"🩺 {chosen_preview.get('doctor','?')}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        if st.button(
                            "✓  Confirm Appointment",
                            type="primary",
                            use_container_width=True,
                            key="new_confirm_btn",
                        ):
                            with st.spinner("Confirming and sending emails..."):
                                confirm_result: dict = confirm_appointment(
                                    patient_id=final_pid,
                                    scan_number="S1",
                                    chosen_slot_id=safe_str(
                                        chosen_preview.get("slot_id", "SLOT-1")
                                    ),
                                    patient_name=patient_name.strip(),
                                    patient_email=patient_email.strip(),
                                )

                            if confirm_result.get("confirmed"):
                                st.success("✅ Appointment confirmed! Emails sent.")
                                st.balloons()
                            else:
                                st.error(
                                    confirm_result.get(
                                        "error", "Booking failed. Check terminal."
                                    )
                                )
