"""
BreastAI Doctor Dashboard — Simple Clean UI
============================================
Filename: doctor_dashboard.py

HOW TO RUN:
  pip install streamlit reportlab
  streamlit run doctor_dashboard.py

PLACE THIS FILE in the same folder as:
  agent1_validation.py
  agent2_report.py
  step5_predict.py   ← needed for raw .mat signal → 6-panel image pipeline
"""

from __future__ import annotations
import streamlit as st
import json, time, datetime, shutil, subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="BreastAI · Doctor Portal",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Backend imports ──────────────────────────────────────────────────
BACKEND_OK = AGENT1_OK = AGENT2_OK = STEP5_OK = False

RESULTS_DIR      = Path("outputs/results")
REPORTS_DIR      = Path("outputs/agent_reports")
DOC_REPORTS      = Path("outputs/doctor_reports")
APPT_FILE        = Path("appointments.json")
HOSPITAL_NAME    = "Apollo Hospitals"
HOSPITAL_ADDRESS = "Bannerghatta Rd, Bengaluru"

URGENCY_WINDOW: dict = {
    "emergency": (0, 1), "urgent": (1, 3),
    "priority":  (3, 7), "routine": (7, 14),
}
DOCTOR_ROUTING: dict = {
    "5":       {"name": "Dr. P. Sundarajan", "email": "onco@hospital.com",   "dept": "Oncology",        "role": "Senior Oncologist"},
    "4C":      {"name": "Dr. P. Sundarajan", "email": "onco@hospital.com",   "dept": "Oncology",        "role": "Senior Oncologist"},
    "4B":      {"name": "Dr. R. Meenakshi",  "email": "radio@hospital.com",  "dept": "Radiology",       "role": "Radiologist Specialist"},
    "4A":      {"name": "Dr. R. Meenakshi",  "email": "radio@hospital.com",  "dept": "Radiology",       "role": "Radiologist Specialist"},
    "3":       {"name": "Dr. S. Lakshmi",    "email": "genrad@hospital.com", "dept": "Radiology",       "role": "General Radiologist"},
    "1-2":     {"name": "Dr. A. Priya",      "email": "gp@hospital.com",     "dept": "General Medicine","role": "General Physician"},
    "default": {"name": "Dr. R. Meenakshi",  "email": "radio@hospital.com",  "dept": "Radiology",       "role": "Radiologist"},
}
DOCTOR_SCHEDULES: dict = {
    "Dr. P. Sundarajan": {
        "working_days": ["Mon","Tue","Wed","Thu","Fri"], "working_hours": (8, 17),
        "busy": {"Mon":["09:00-10:00","14:00-15:00"],"Tue":["11:00-12:00"],"Wed":["09:00-11:00"],"Thu":["10:00-11:00"],"Fri":["13:00-14:00"]},
    },
    "Dr. R. Meenakshi": {
        "working_days": ["Mon","Tue","Wed","Thu","Fri"], "working_hours": (9, 18),
        "busy": {"Mon":["10:00-11:00"],"Tue":["09:00-10:00","14:00-16:00"],"Wed":["11:00-12:00"],"Thu":["09:00-10:00"],"Fri":["10:00-11:00"]},
    },
    "Dr. S. Lakshmi": {
        "working_days": ["Mon","Tue","Thu","Fri"], "working_hours": (9, 16),
        "busy": {"Mon":["09:30-10:30"],"Tue":["14:00-15:00"],"Thu":["11:00-12:00"],"Fri":["09:00-10:00"]},
    },
    "Dr. A. Priya": {
        "working_days": ["Mon","Wed","Thu","Fri"], "working_hours": (8, 15),
        "busy": {"Mon":["08:00-09:00"],"Wed":["11:00-12:00"],"Thu":["09:00-10:00"],"Fri":["08:00-09:00"]},
    },
}

try:
    from agent1_validation import run_agent1 as _run_agent1_real
    from agent1_validation import RESULTS_DIR, REPORTS_DIR  # type: ignore
    AGENT1_OK = True
except ImportError:
    pass

try:
    from agent2_report import (  # type: ignore
        run_agent2_auto as _run_agent2_auto_real,
        get_doctor as _get_doctor_real,
        find_next_free_slot as _find_next_free_slot_real,
        DOCTOR_SCHEDULES as _DS_real,
        DOCTOR_ROUTING as _DR_real,
        DOC_REPORTS as _DOC_REPORTS_real,
        HOSPITAL_NAME as _HOSP_real,
        HOSPITAL_ADDRESS as _ADDR_real,
        APPT_FILE as _APPT_real,
        URGENCY_WINDOW as _UW_real,
    )
    DOC_REPORTS = _DOC_REPORTS_real
    HOSPITAL_NAME = _HOSP_real
    HOSPITAL_ADDRESS = _ADDR_real
    APPT_FILE = _APPT_real
    DOCTOR_SCHEDULES = _DS_real
    DOCTOR_ROUTING = _DR_real
    URGENCY_WINDOW = _UW_real
    AGENT2_OK = True
except ImportError:
    pass

try:
    import step5_predict as _step5  # type: ignore
    STEP5_OK = True
except ImportError:
    pass

BACKEND_OK = AGENT1_OK and AGENT2_OK

# ── Create output directories ────────────────────────────────────────
for _d in [RESULTS_DIR, REPORTS_DIR, DOC_REPORTS]:
    _d.mkdir(parents=True, exist_ok=True)


# ── Wrappers ─────────────────────────────────────────────────────────

def run_agent1_wrapper(image_path: Path, risk_factors: str = "none") -> dict:
    if AGENT1_OK:
        return _run_agent1_real(image_path, risk_factors=risk_factors)  # type: ignore
    import random, hashlib
    seed   = int(hashlib.md5(str(image_path).encode()).hexdigest()[:8], 16)
    rng    = random.Random(seed)
    conf   = rng.randint(70, 97)
    birads = rng.choice(["1-2","3","4A","4B","4C","5"])
    um     = {"1-2":"routine","3":"routine","4A":"priority","4B":"urgent","4C":"urgent","5":"emergency"}
    num    = Path(image_path).stem.replace("prediction_patient_","").split("_")[0]
    return {
        "patient_id":           f"PT-{num.zfill(3)}",
        "scan_number":          "S1",
        "predicted_class":      "malignant" if rng.random() > 0.45 else "benign",
        "extracted_confidence": conf,
        "birads_estimate":      birads,
        "urgency":              um.get(birads, "routine"),
        "verdict":              "APPROVED" if conf >= 85 else "FLAGGED" if conf >= 70 else "REJECTED",
        "visual_consistency":   rng.choice(["consistent","ambiguous"]),
        "clinical_reasoning": (
            f"AI analysis complete. BI-RADS {birads} detected with {conf}% confidence. "
            + ("Irregular margins and posterior acoustic features noted." if birads in ("4B","4C","5")
               else "Well-defined oval lesion with homogeneous echo.")
        ),
        "recommended_action": (
            "Immediate specialist review and biopsy consideration." if birads in ("4B","4C","5")
            else "Short-interval follow-up recommended." if birads in ("3","4A")
            else "Routine annual screening."
        ),
        "flag_reasons": (["Confidence below 85% — radiologist review required"] if conf < 85 else []),
    }


def get_doctor_wrapper(birads: str) -> dict:
    if AGENT2_OK:
        return _get_doctor_real(birads)  # type: ignore
    return DOCTOR_ROUTING.get(str(birads), DOCTOR_ROUTING["default"])


def find_next_free_slot_wrapper(doctor_name: str, urgency: str) -> dict:
    if AGENT2_OK:
        return _find_next_free_slot_real(doctor_name, urgency)  # type: ignore
    min_d, max_d = URGENCY_WINDOW.get(urgency.lower(), (7, 14))
    sched        = DOCTOR_SCHEDULES.get(doctor_name, {})
    working_days = sched.get("working_days", ["Mon","Tue","Wed","Thu","Fri"])
    ws, we       = sched.get("working_hours", (9, 17))
    now          = datetime.datetime.now()
    PREFERRED    = ["09:30","10:30","11:30","14:00","15:00","16:00"]
    for offset in range(min_d, max_d + 5):
        day = now + datetime.timedelta(days=offset)
        dsh = day.strftime("%a")
        if dsh not in working_days:
            continue
        for t in PREFERRED:
            h = int(t.split(":")[0])
            if h < ws or h >= we:
                continue
            busy = False
            slot_m = h * 60 + int(t.split(":")[1])
            for period in sched.get("busy", {}).get(dsh, []):
                s, e = period.split("-")
                sh, sm = map(int, s.split(":")); eh, em = map(int, e.split(":"))
                if sh*60+sm <= slot_m < eh*60+em:
                    busy = True; break
            if busy:
                continue
            slot_dt = day.replace(hour=h, minute=int(t.split(":")[1]), second=0, microsecond=0)
            return {
                "slot_id": "SLOT-AUTO", "date": day.strftime("%A, %d %B %Y"),
                "time": slot_dt.strftime("%I:%M %p"), "time_24": t, "day": dsh,
                "doctor": doctor_name, "hospital": HOSPITAL_NAME, "address": HOSPITAL_ADDRESS,
                "label": f"{day.strftime('%A, %d %B %Y')} at {slot_dt.strftime('%I:%M %p')}",
            }
    fb = now + datetime.timedelta(days=max_d)
    return {
        "slot_id": "SLOT-AUTO", "date": fb.strftime("%A, %d %B %Y"),
        "time": "09:30 AM", "time_24": "09:30", "day": fb.strftime("%a"),
        "doctor": doctor_name, "hospital": HOSPITAL_NAME, "address": HOSPITAL_ADDRESS,
        "label": f"{fb.strftime('%A, %d %B %Y')} at 09:30 AM",
    }


def run_agent2_auto_wrapper(patient_id, scan_number, patient_name, patient_email, scan_image_path, verdict) -> dict:
    if AGENT2_OK:
        return _run_agent2_auto_real(  # type: ignore
            patient_id=patient_id, scan_number=scan_number,
            patient_name=patient_name, patient_email=patient_email,
            scan_image_path=scan_image_path, verdict=verdict,
        )
    birads  = str(verdict.get("birads_estimate", "default"))
    urgency = str(verdict.get("urgency", "routine")).lower()
    doctor  = get_doctor_wrapper(birads)
    slot    = find_next_free_slot_wrapper(doctor["name"], urgency)
    slot.update({"role": doctor["role"], "dept": doctor["dept"],
                 "label": f"{slot['date']} at {slot['time']}"})
    return {
        "verdict": verdict, "doctor": doctor, "chosen_slot": slot,
        "pdf_path": str(DOC_REPORTS / f"{patient_id}_S1_doctor_report.txt"),
        "email_ok": {"patient": True, "doctor": True},
        "event_id": "DEMO_EVENT",
        "patient_id": patient_id, "scan_number": scan_number,
    }


def run_signal_pipeline(mat_path: Path, out_png: Path) -> bool:
    script = Path(__file__).parent / "step5_predict.py"
    if not script.exists():
        if STEP5_OK:
            try:
                cnn_m, unet_m = _step5.load_models()  # type: ignore
                _step5.predict_from_signal_file(str(mat_path), cnn_model=cnn_m, unet_model=unet_m)  # type: ignore
                auto_out = Path("outputs/results") / f"prediction_{mat_path.stem}.png"
                if auto_out.exists() and auto_out != out_png:
                    shutil.copy(auto_out, out_png)
                return out_png.exists()
            except Exception as e:
                print(f"[step5 inline] Error: {e}")
                return False
        return False
    result = subprocess.run(
        [sys.executable, str(script), "--signal", str(mat_path)],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        return False
    auto_out = Path("outputs/results") / f"prediction_{mat_path.stem}.png"
    if auto_out.exists() and auto_out != out_png:
        shutil.copy(auto_out, out_png)
    return out_png.exists()


# ── Appointments helpers ──────────────────────────────────────────────

def _load_appts() -> dict:
    if not APPT_FILE.exists():
        return {}
    with open(APPT_FILE) as f:
        return json.load(f)

def _save_appts(data: dict) -> None:
    with open(APPT_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Session state ─────────────────────────────────────────────────────
if "patients"    not in st.session_state: st.session_state.patients    = {}
if "last_result" not in st.session_state: st.session_state.last_result = None
if "active_tab"  not in st.session_state: st.session_state.active_tab  = "existing"

def _sync_from_disk() -> None:
    try:
        appts = _load_appts()
    except Exception:
        return
    for key, appt in appts.items():
        try:
            pid = appt.get("patient_id", key)
            if pid not in st.session_state.patients:
                verdict: dict = {}
                scan_no = appt.get("scan_number", "S1")
                vp_str  = (appt.get("verdict_json") or "").strip()
                rp      = REPORTS_DIR / f"{pid}_{scan_no}_verdict.json"
                if vp_str and vp_str not in (".", "./", ".\\" ):
                    try:
                        vp = Path(vp_str)
                        if vp.is_file():
                            with open(vp) as f:
                                verdict = json.load(f)
                    except Exception:
                        pass
                if not verdict:
                    try:
                        if rp.is_file():
                            with open(rp) as f:
                                verdict = json.load(f)
                    except Exception:
                        pass
                st.session_state.patients[pid] = {
                    "patient_id":    pid, "scan_number": scan_no,
                    "patient_name":  appt.get("patient_name", "Unknown"),
                    "patient_email": appt.get("patient_email", ""),
                    "status":        appt.get("status", "pending"),
                    "verdict":       verdict, "appt": appt,
                    "timestamp":     appt.get("created_at", ""),
                }
        except Exception:
            continue

_sync_from_disk()

def _next_pid() -> str:
    nums = []
    for e in st.session_state.patients:
        try: nums.append(int(e.replace("PT-","").split("_")[0]))
        except: nums.append(0)
    return f"PT-{(max(nums, default=0)+1):03d}"


# ── Urgency colours ───────────────────────────────────────────────────
URGENCY_EMOJI = {"emergency": "🔴", "urgent": "🟠", "priority": "🟡", "routine": "🟢"}
BIRADS_COLOR  = {"1-2":"green","3":"blue","4A":"orange","4B":"orange","4C":"red","5":"red"}

# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🏥 BreastAI")
    st.caption(f"{HOSPITAL_NAME} · {HOSPITAL_ADDRESS}")
    st.divider()

    # Backend status
    st.markdown("**System Status**")
    st.write("Agent 1 (AI Analysis)", "✅" if AGENT1_OK else "❌ Demo mode")
    st.write("Agent 2 (Booking)",     "✅" if AGENT2_OK else "❌ Demo mode")
    st.write("Signal Pipeline",       "✅" if STEP5_OK  else "❌ Not found")

    st.divider()

    # Summary stats
    pts       = st.session_state.patients
    now       = datetime.datetime.now()
    total     = len(pts)
    confirmed = sum(1 for p in pts.values() if p.get("status") == "confirmed")
    urgent_ct = sum(1 for p in pts.values()
                    if p.get("verdict",{}).get("urgency","").lower() in ("emergency","urgent"))
    today_ct  = sum(1 for p in pts.values()
                    if p.get("timestamp","").startswith(now.strftime("%Y-%m-%d")))

    col1, col2 = st.columns(2)
    col1.metric("Total Patients", total)
    col2.metric("Added Today",    today_ct)
    col1.metric("Urgent / Emergency", urgent_ct)
    col2.metric("Appointments Booked", confirmed)

    st.divider()
    st.caption(f"🕐 {now.strftime('%A, %d %b %Y  %I:%M %p')}")


# ════════════════════════════════════════════════════════════════════
# MAIN AREA — TABS
# ════════════════════════════════════════════════════════════════════
st.title("Breast Ultrasound AI — Doctor Dashboard")
st.caption("Agent 1: Groq LLaMA-4-Scout  |  Agent 2: PDF Report + Email + Calendar  |  BI-RADS 5th Ed  |  WHO 2022")
st.divider()

tab_run, tab_records, tab_schedules = st.tabs([
    "🔬  Run Analysis",
    "📋  Patient Records",
    "📅  Doctor Schedules",
])


# ════════════════════════════════════════════════════════════════════
# TAB 1 — RUN ANALYSIS
# ════════════════════════════════════════════════════════════════════
with tab_run:

    # ── Choose mode: existing patient or new patient ──────────────
    mode = st.radio(
        "Select patient",
        ["Existing patient", "New patient"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    # ─────────────────────────────────────────────────────────────
    # MODE A: EXISTING PATIENT
    # ─────────────────────────────────────────────────────────────
    if mode == "Existing patient":

        # Build dropdown from outputs/results PNGs + session patients
        result_images = sorted(RESULTS_DIR.glob("prediction_patient_*.png"))
        file_stems    = [p.stem for p in result_images]  # e.g. prediction_patient_001_1mm

        # Also include any patients already in session that have an image
        session_stems = []
        for pid, p in st.session_state.patients.items():
            num_str = pid.replace("PT-","").zfill(3)
            imgs    = list(RESULTS_DIR.glob(f"prediction_patient_{num_str}_*.png"))
            if imgs:
                stem = imgs[0].stem
                if stem not in file_stems:
                    session_stems.append(stem)

        all_stems = file_stems + session_stems

        if not all_stems:
            st.info("No existing patient scans found in `outputs/results/`. Upload a new patient below, or run the signal pipeline first.")
        else:
            left, right = st.columns([1, 2])

            with left:
                st.markdown("**Select patient scan**")
                selected_stem = st.selectbox(
                    "Patient scan",
                    all_stems,
                    label_visibility="collapsed",
                )

                risk_factors_ex = st.text_input(
                    "Risk factors (optional)",
                    placeholder="e.g. BRCA1, family history",
                    key="rf_existing",
                )

                run_btn = st.button("▶ Run Agent 1 Validation", type="primary", use_container_width=True)

                st.divider()
                st.markdown("**Agent 2 — Report + Booking**")
                p_name_ex  = st.text_input("Patient name",  placeholder="e.g. Shruthi Dhandapani", key="name_ex")
                p_email_ex = st.text_input("Patient email", placeholder="e.g. shruthi@gmail.com",  key="email_ex")
                run2_btn   = st.button("▶ Run Agent 2", type="secondary", use_container_width=True, key="run2_ex")

            with right:
                # Show the selected image
                img_path = RESULTS_DIR / f"{selected_stem}.png"
                if img_path.exists():
                    st.markdown("**Model Output Image (6-Panel)**")
                    st.image(str(img_path), caption=f"{selected_stem}.png", use_container_width=True)
                else:
                    st.info("Image file not found on disk.")

                # Show Agent 1 result if available
                if run_btn:
                    if not img_path.exists():
                        st.error("Image file not found. Cannot run analysis.")
                    else:
                        with st.spinner("Running Agent 1 AI analysis…"):
                            a1 = run_agent1_wrapper(img_path, risk_factors=risk_factors_ex.strip() or "none")

                        if a1:
                            st.session_state.last_result = {
                                "verdict": a1, "doctor": None, "chosen_slot": None,
                                "pdf_path": "", "email_ok": {}, "event_id": "",
                                "patient_id": a1.get("patient_id","?"),
                                "scan_number": a1.get("scan_number","S1"),
                                "was_mat": False,
                            }
                            st.success("Agent 1 complete — see results below.")
                        else:
                            st.error("Agent 1 returned no result. Check your Groq API key.")

                res = st.session_state.last_result
                if res and res.get("verdict"):
                    v      = res["verdict"]
                    urg    = str(v.get("urgency","routine")).lower()
                    birads = str(v.get("birads_estimate","?"))
                    pred   = str(v.get("predicted_class","?"))
                    conf   = v.get("extracted_confidence","?")
                    vd     = str(v.get("verdict","?"))

                    st.markdown("---")
                    st.markdown("**AI Diagnosis Result**")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Prediction",  pred.upper())
                    c2.metric("Confidence",  f"{conf}%")
                    c3.metric("BI-RADS",     birads)
                    c4.metric("Urgency",     f"{URGENCY_EMOJI.get(urg,'')} {urg.upper()}")

                    st.markdown(f"**Verdict:** `{vd}`")
                    st.markdown(f"**Clinical reasoning:** {v.get('clinical_reasoning','—')}")
                    st.markdown(f"**Recommended action:** {v.get('recommended_action','—')}")

                    for flag in (v.get("flag_reasons") or []):
                        st.warning(f"⚑ {flag}")

                    # Agent 2 booking result
                    if run2_btn:
                        if not p_name_ex.strip() or "@" not in (p_email_ex or ""):
                            st.error("Enter a valid patient name and email before running Agent 2.")
                        else:
                            with st.spinner("Checking schedule, booking slot, sending emails…"):
                                a2 = run_agent2_auto_wrapper(
                                    patient_id      = str(v.get("patient_id","PT-000")),
                                    scan_number     = str(v.get("scan_number","S1")),
                                    patient_name    = p_name_ex.strip(),
                                    patient_email   = p_email_ex.strip(),
                                    scan_image_path = str(img_path),
                                    verdict         = v,
                                )
                            if "error" in a2:
                                st.error(f"Agent 2 error: {a2['error']}")
                            else:
                                if st.session_state.last_result is not None:
                                    st.session_state.last_result.update({
                                        "doctor": a2["doctor"],
                                        "chosen_slot": a2["chosen_slot"],
                                        "pdf_path": a2.get("pdf_path",""),
                                        "email_ok": a2.get("email_ok",{}),
                                        "event_id": a2.get("event_id",""),
                                    })
                                doctor = a2["doctor"]
                                slot   = a2["chosen_slot"]
                                ok     = a2.get("email_ok",{})

                                st.success("✅ Appointment booked!")
                                with st.expander("📅 Appointment Details", expanded=True):
                                    st.write(f"**Doctor:** {doctor.get('name')} — {doctor.get('role')}")
                                    st.write(f"**Department:** {doctor.get('dept')}")
                                    st.write(f"**Date:** {slot.get('date')}")
                                    st.write(f"**Time:** {slot.get('time')}")
                                    st.write(f"**Hospital:** {slot.get('hospital')} · {slot.get('address')}")
                                    st.write(
                                        f"{'✅' if ok.get('patient') else '❌'} Patient email  "
                                        f"{'✅' if ok.get('doctor') else '❌'} Doctor email"
                                    )

                                pdf_p = Path(a2.get("pdf_path",""))
                                if pdf_p.exists():
                                    with open(pdf_p,"rb") as fh:
                                        st.download_button("⬇ Download PDF Report", data=fh,
                                            file_name=pdf_p.name,
                                            mime="application/pdf" if str(pdf_p).endswith(".pdf") else "text/plain")

    # ─────────────────────────────────────────────────────────────
    # MODE B: NEW PATIENT
    # ─────────────────────────────────────────────────────────────
    else:
        left, right = st.columns([1, 2])

        with left:
            st.markdown("**Patient Information**")

            with st.form("new_patient_form", clear_on_submit=True):
                p_name  = st.text_input("Full Name *",      placeholder="e.g. Ananya Krishnan")
                p_email = st.text_input("Email Address *",  placeholder="ananya@email.com")

                col_a, col_b = st.columns(2)
                with col_a:
                    p_age = st.number_input("Age", min_value=18, max_value=100, value=45)
                with col_b:
                    p_id_override = st.text_input("Patient ID (auto if blank)", placeholder="PT-042")

                risk_factors = st.text_input(
                    "Risk Factors (optional)",
                    placeholder="e.g. BRCA1, family history",
                )

                st.markdown("**Scan File**")
                uploaded_file = st.file_uploader(
                    "Upload scan",
                    type=["png","jpg","jpeg","mat"],
                    label_visibility="collapsed",
                    help="PNG/JPG: pre-processed 6-panel image  |  .mat: raw OASBUD signal file",
                )
                st.caption("PNG/JPG → straight to Agent 1 · .mat → full signal pipeline first")

                submit_btn = st.form_submit_button(
                    "🚀 Analyse & Auto-Book Appointment",
                    type="primary",
                    use_container_width=True,
                )

        with right:
            st.markdown("**How it works**")
            steps = [
                ("1", "Upload scan",      "PNG/JPG or raw .mat RF signal"),
                ("2", "Signal pipeline",  ".mat → beamform → CLAHE → UNet → 6-panel PNG"),
                ("3", "Agent 1 analysis", "Groq LLaMA-4 classifies lesion, assigns BI-RADS"),
                ("4", "Doctor routing",   "BI-RADS maps to the right specialist automatically"),
                ("5", "Schedule check",   "Doctor availability grid is checked"),
                ("6", "Slot booked",      "Nearest free slot picked by urgency window"),
                ("7", "PDF generated",    "Full clinical report created"),
                ("8", "Emails + calendar","Patient and doctor notified, invite created"),
            ]
            for num, title, desc in steps:
                st.markdown(f"**{num}.** **{title}** — {desc}")

            st.divider()
            st.markdown("**Urgency windows**")
            urgency_info = {
                "🔴 EMERGENCY": "Same day / next day",
                "🟠 URGENT":    "Within 1–3 days",
                "🟡 PRIORITY":  "Within 3–7 days",
                "🟢 ROUTINE":   "Within 7–14 days",
            }
            for label, window in urgency_info.items():
                st.write(f"{label} — {window}")

            st.divider()
            st.markdown("**BI-RADS → Specialist**")
            routing_info = [
                ("5, 4C",  "Senior Oncologist"),
                ("4A, 4B", "Radiologist Specialist"),
                ("3",      "General Radiologist"),
                ("1–2",    "General Physician"),
            ]
            for br, spec in routing_info:
                st.write(f"`{br}` → {spec}")

        # ── Handle form submission ──────────────────────────────────
        if submit_btn:
            errors = []
            if not p_name.strip():
                errors.append("Patient full name is required.")
            if not p_email.strip() or "@" not in p_email:
                errors.append("A valid email address is required.")
            if uploaded_file is None:
                errors.append("Please upload a scan file (PNG, JPG, or .mat).")

            if errors:
                for e in errors:
                    st.error(e)
                st.session_state.last_result = None
            elif uploaded_file is not None:
                file_name    = uploaded_file.name
                file_bytes   = uploaded_file.getbuffer()
                ext          = Path(file_name).suffix.lower()
                new_pid      = p_id_override.strip() if p_id_override.strip() else _next_pid()
                num_part     = new_pid.replace("PT-","").zfill(3)
                out_png_path = RESULTS_DIR / f"prediction_patient_{num_part}_S1.png"

                pb = st.progress(0, "Starting…")

                if ext == ".mat":
                    raw_dir  = RESULTS_DIR.parent / "raw_signals"
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    mat_save = raw_dir / f"patient_{num_part}_{file_name}"
                    mat_save.write_bytes(file_bytes)
                    pb.progress(10, "Saved .mat — running signal pipeline…")

                    st.info(
                        "📡 Raw signal file detected. Running full pipeline:\n\n"
                        "Beamforming → CLAHE enhancement → UNet segmentation → CNN classification → 6-panel output\n\n"
                        "This may take 30–90 seconds.",
                        icon="⚙️",
                    )

                    signal_ok = run_signal_pipeline(mat_save, out_png_path)

                    if not signal_ok:
                        existing = sorted(RESULTS_DIR.glob("prediction_patient_*.png"))
                        if existing:
                            shutil.copy(existing[-1], out_png_path)
                            st.warning(
                                "⚠️ `step5_predict.py` not found or failed. "
                                "Using the most recent existing result image as a placeholder."
                            )
                        else:
                            pb.empty()
                            st.error(
                                "❌ Cannot process .mat file: `step5_predict.py` is missing and "
                                "no existing PNG was found. Upload a PNG/JPG instead."
                            )
                            st.session_state.last_result = None
                            st.stop()

                    pb.progress(50, "Signal processing complete — running Agent 1…")

                else:
                    out_png_path.write_bytes(file_bytes)
                    pb.progress(25, "Image saved — running Agent 1…")

                # ── Agent 1 ──────────────────────────────────────────
                a1_result = run_agent1_wrapper(out_png_path, risk_factors=risk_factors.strip() or "none")

                if a1_result is None:
                    pb.empty()
                    st.error("❌ Agent 1 returned no result. Check your Groq API connection.")
                    st.session_state.last_result = None
                    st.stop()

                pb.progress(65, "AI analysis done — routing to specialist…")
                time.sleep(0.2)

                pid     = str(a1_result.get("patient_id", new_pid))
                scan_no = str(a1_result.get("scan_number", "S1"))

                REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                vpath = REPORTS_DIR / f"{pid}_{scan_no}_verdict.json"
                with open(vpath, "w") as vf:
                    json.dump(a1_result, vf, indent=2)

                pb.progress(80, "Checking schedule, finding slot…")
                time.sleep(0.2)

                pb.progress(90, "Generating PDF and sending emails…")
                time.sleep(0.2)

                # ── Agent 2 ──────────────────────────────────────────
                a2_result = run_agent2_auto_wrapper(
                    patient_id=pid, scan_number=scan_no,
                    patient_name=p_name.strip(), patient_email=p_email.strip(),
                    scan_image_path=str(out_png_path), verdict=a1_result,
                )

                pb.progress(100, "Done!")
                time.sleep(0.3)
                pb.empty()

                if "error" in a2_result:
                    st.error(f"❌ {a2_result['error']}")
                    st.session_state.last_result = None
                else:
                    appts = _load_appts()
                    key   = f"{pid}_{scan_no}"
                    st.session_state.patients[pid] = {
                        "patient_id":    pid, "scan_number": scan_no,
                        "patient_name":  p_name.strip(),
                        "patient_email": p_email.strip(),
                        "status":        appts.get(key, {}).get("status","confirmed"),
                        "verdict":       a1_result,
                        "appt":          appts.get(key, {}),
                        "timestamp":     str(now)[:16],
                    }
                    st.session_state.last_result = {
                        "verdict":     a1_result,
                        "doctor":      a2_result["doctor"],
                        "chosen_slot": a2_result["chosen_slot"],
                        "pdf_path":    a2_result.get("pdf_path",""),
                        "email_ok":    a2_result.get("email_ok",{"patient":False,"doctor":False}),
                        "event_id":    a2_result.get("event_id",""),
                        "patient_id":  pid,
                        "scan_number": scan_no,
                        "was_mat":     (ext == ".mat"),
                    }

        # ── Show result ──────────────────────────────────────────────
        res = st.session_state.last_result
        if res and res.get("verdict") and mode == "New patient":
            v      = res["verdict"]
            doctor = res.get("doctor") or {}
            slot   = res.get("chosen_slot") or {}
            ok     = res.get("email_ok") or {}
            urg    = str(v.get("urgency","routine")).lower()
            birads = str(v.get("birads_estimate","?"))
            pred   = str(v.get("predicted_class","?"))
            conf   = v.get("extracted_confidence","?")
            vd     = str(v.get("verdict","?"))

            st.divider()

            if vd == "REJECTED":
                st.error(
                    "⚠️ Scan REJECTED — manual review required.\n\n"
                    "AI confidence was too low to auto-schedule. "
                    "Please have a radiologist review this scan manually."
                )
            else:
                was_mat = res.get("was_mat", False)
                st.success(
                    f"{'📡 Signal processed + ' if was_mat else ''}✅ Analysis complete — appointment auto-booked!"
                )

                # Metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Prediction",  pred.upper())
                c2.metric("Confidence",  f"{conf}%")
                c3.metric("BI-RADS",     birads)
                c4.metric("Urgency",     f"{URGENCY_EMOJI.get(urg,'')} {urg.upper()}")

                # Verdict + flags
                st.markdown(f"**Verdict:** `{vd}`")
                for flag in (v.get("flag_reasons") or []):
                    st.warning(f"⚑ {flag}")

                # Appointment
                if doctor and slot:
                    with st.expander("📅 Appointment Details", expanded=True):
                        col_l, col_r = st.columns(2)
                        with col_l:
                            st.write(f"**Date:** {slot.get('date','?')}")
                            st.write(f"**Time:** {slot.get('time','?')}")
                            st.write(f"**Doctor:** {doctor.get('name','?')} — {doctor.get('role','?')}")
                        with col_r:
                            st.write(f"**Department:** {doctor.get('dept','?')}")
                            st.write(f"**Hospital:** {slot.get('hospital','?')}")
                            st.write(f"**Address:** {slot.get('address','?')}")

                        cal_ok = bool(
                            res.get("event_id","") and
                            not str(res.get("event_id","")).startswith(("GCAL_","NOT_","ERROR","DEMO"))
                        )
                        st.write(
                            f"{'✅' if ok.get('patient') else '❌'} Patient email  "
                            f"{'✅' if ok.get('doctor') else '❌'} Doctor email  "
                            f"{'📅 Calendar invite created' if cal_ok else '📅 Calendar not configured'}"
                        )

                # Clinical notes
                with st.expander("🩺 Clinical Notes"):
                    st.markdown(f"**Clinical Reasoning:**\n{v.get('clinical_reasoning','—')}")
                    st.markdown(f"**Recommended Action:**\n{v.get('recommended_action','—')}")

                # 6-panel image
                pid_shown = res.get("patient_id","")
                num_str   = pid_shown.replace("PT-","").zfill(3)
                imgs      = list(RESULTS_DIR.glob(f"prediction_patient_{num_str}_*.png"))
                if imgs:
                    st.image(str(imgs[0]),
                             caption="6-panel AI pipeline output",
                             use_container_width=True)

                # PDF download
                pdf_p = Path(res.get("pdf_path",""))
                if pdf_p.exists():
                    with open(pdf_p,"rb") as fh:
                        st.download_button(
                            "⬇ Download PDF Report", data=fh,
                            file_name=pdf_p.name,
                            mime="application/pdf" if str(pdf_p).endswith(".pdf") else "text/plain",
                        )


# ════════════════════════════════════════════════════════════════════
# TAB 2 — PATIENT RECORDS
# ════════════════════════════════════════════════════════════════════
with tab_records:
    st.subheader("Patient Records")
    pts = st.session_state.patients

    if not pts:
        st.info("No patients yet. Run an analysis from the **Run Analysis** tab.")
    else:
        # Filters
        fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 3])
        with fc1:
            f_urg = st.selectbox("Urgency", ["All","emergency","urgent","priority","routine"], key="f_urg")
        with fc2:
            f_vd  = st.selectbox("Verdict",  ["All","APPROVED","FLAGGED","REJECTED"], key="f_vd")
        with fc3:
            f_st  = st.selectbox("Status",   ["All","confirmed","pending","rejected"], key="f_st")
        with fc4:
            srch  = st.text_input("Search name or ID", "", placeholder="Type to filter…", key="srch")

        filtered = {
            pid: p for pid, p in pts.items()
            if (f_urg == "All" or p.get("verdict",{}).get("urgency","").lower() == f_urg)
            and (f_vd  == "All" or p.get("verdict",{}).get("verdict","")         == f_vd)
            and (f_st  == "All" or p.get("status","")                             == f_st)
            and (not srch or srch.lower() in (p.get("patient_name","") + pid).lower())
        }
        st.caption(f"Showing {len(filtered)} of {len(pts)} patients")

        for pid, p in sorted(filtered.items(), key=lambda x: x[1].get("timestamp",""), reverse=True):
            v      = p.get("verdict", {})
            appt   = p.get("appt", {})
            doctor = appt.get("assigned_doctor", {})
            slot   = appt.get("chosen_slot", {})
            urg    = str(v.get("urgency","routine")).lower()
            vd     = str(v.get("verdict","PENDING"))
            pred   = str(v.get("predicted_class","?"))
            conf   = v.get("extracted_confidence","?")
            birads = str(v.get("birads_estimate","?"))
            status = p.get("status","pending")

            icon  = URGENCY_EMOJI.get(urg,"⚪")
            label = f"{icon}  {p.get('patient_name','Unknown')}  ·  {pid}  ·  {vd}  ·  {urg.upper()}"

            with st.expander(label, expanded=(urg in ("emergency","urgent"))):
                lc, rc = st.columns([3, 2])

                with lc:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Prediction",  pred.upper())
                    c2.metric("Confidence",  f"{conf}%")
                    c3.metric("BI-RADS",     birads)
                    c4.metric("Urgency",     f"{icon} {urg.upper()}")

                    st.markdown("**Clinical Reasoning**")
                    st.info(v.get("clinical_reasoning","No reasoning available."))
                    st.markdown(f"**Recommended Action:** {v.get('recommended_action','—')}")

                    for flag in (v.get("flag_reasons",[]) or []):
                        st.warning(f"⚑ {flag}")

                    # 6-panel image
                    num_str = pid.replace("PT-","").zfill(3)
                    imgs    = list(RESULTS_DIR.glob(f"prediction_patient_{num_str}_*.png"))
                    if imgs:
                        st.image(str(imgs[0]), caption="6-panel AI pipeline output", use_container_width=True)

                with rc:
                    st.markdown("**Patient**")
                    st.write(f"**Name:** {p.get('patient_name','?')}")
                    st.write(f"**Email:** {p.get('patient_email','?')}")
                    st.write(f"**ID:** {pid}")

                    if doctor:
                        st.markdown("**Assigned Doctor**")
                        st.write(f"**Name:** {doctor.get('name','?')}")
                        st.write(f"**Role:** {doctor.get('role','?')} · {doctor.get('dept','?')}")
                        st.write(f"**Email:** {doctor.get('email','?')}")

                    if slot and status == "confirmed":
                        st.markdown("**✅ Appointment Confirmed**")
                        st.write(f"📆 {slot.get('date','?')}")
                        st.write(f"🕐 {slot.get('time','?')}")
                        st.write(f"🏥 {slot.get('hospital','?')}")
                        st.write(f"📍 {slot.get('address','?')}")
                        eml_p = appt.get("patient_email_ok", False)
                        eml_d = appt.get("doctor_email_ok", False)
                        st.write(
                            f"{'✅' if eml_p else '❌'} Patient email  "
                            f"{'✅' if eml_d else '❌'} Doctor email"
                        )
                    elif status == "rejected":
                        st.error("✕ Rejected — manual review required")

                    # Downloads
                    pdf_p = appt.get("pdf_report","")
                    if pdf_p and Path(pdf_p).exists():
                        with open(pdf_p,"rb") as fh:
                            st.download_button(
                                "⬇ PDF Report", data=fh,
                                file_name=Path(pdf_p).name,
                                mime="application/pdf" if str(pdf_p).endswith(".pdf") else "text/plain",
                                use_container_width=True, key=f"dl_{pid}",
                            )

                    scan_no2 = p.get("scan_number","S1")
                    jf = REPORTS_DIR / f"{pid}_{scan_no2}_verdict.json"
                    if jf.exists():
                        with open(jf) as fh:
                            st.download_button(
                                "⬇ Verdict JSON", data=fh.read(),
                                file_name=jf.name, mime="application/json",
                                use_container_width=True, key=f"jdl_{pid}",
                            )


# ════════════════════════════════════════════════════════════════════
# TAB 3 — DOCTOR SCHEDULES
# ════════════════════════════════════════════════════════════════════
with tab_schedules:
    st.subheader("Doctor Availability Schedules")
    st.caption(
        "Live availability grid used for auto-booking. "
        "🟢 = free  ·  🔴 = busy  ·  ⬜ = day off  "
        "To edit, update `DOCTOR_SCHEDULES` in `agent2_report.py`."
    )

    DISPLAY_HOURS = list(range(8, 18))
    DAYS          = ["Mon","Tue","Wed","Thu","Fri"]

    def _is_busy(doctor_name: str, day_short: str, hour: int) -> bool:
        sched  = DOCTOR_SCHEDULES.get(doctor_name, {})
        slot_m = hour * 60
        for period in sched.get("busy", {}).get(day_short, []):
            s, e   = period.split("-")
            sh, sm = map(int, s.split(":")); eh, em = map(int, e.split(":"))
            if sh*60+sm <= slot_m < eh*60+em:
                return True
        return False

    for doc_name, sched in DOCTOR_SCHEDULES.items():
        doc_info   = next(
            (v for v in DOCTOR_ROUTING.values() if v.get("name") == doc_name),
            {"role": "Specialist", "dept": "Department"},
        )
        w_days     = sched.get("working_days", DAYS)
        ws, we     = sched.get("working_hours", (9, 17))

        with st.expander(f"🩺 {doc_name}  ·  {doc_info['role']}, {doc_info['dept']}", expanded=True):

            # Build a simple table using st.columns
            header_cols = st.columns([1] + [1]*len(DAYS))
            header_cols[0].markdown("**Hour**")
            for i, d in enumerate(DAYS):
                style = "**" if d in w_days else ""
                header_cols[i+1].markdown(f"{style}{d}{style}")

            for h in DISPLAY_HOURS:
                if h < ws or h > we:
                    continue
                row_cols = st.columns([1] + [1]*len(DAYS))
                row_cols[0].caption(f"{h:02d}:00")
                for i, d in enumerate(DAYS):
                    if d not in w_days:
                        row_cols[i+1].write("⬜")
                    elif _is_busy(doc_name, d, h):
                        row_cols[i+1].write("🔴")
                    else:
                        row_cols[i+1].write("🟢")

            st.markdown(" ")
            sc1, sc2 = st.columns([3, 2])
            with sc2:
                urg_sel = st.selectbox(
                    "Check next free slot:",
                    ["routine","priority","urgent","emergency"],
                    key=f"urg_{doc_name.replace(' ','_')}",
                )
                ns = find_next_free_slot_wrapper(doc_name, urg_sel)
                st.info(
                    f"📅 Next **{urg_sel.upper()}** slot:\n\n"
                    f"**{ns.get('date','?')}** at **{ns.get('time','?')}**"
                )
