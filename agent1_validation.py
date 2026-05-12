"""
AGENT 1 - CLINICAL VALIDATION AGENT
Vision backend: GROQ (FREE)

SETUP:
  1. Go to https://console.groq.com
  2. Sign up free (no credit card)
  3. Click API Keys -> Create API Key
  4. Paste key below

FOLDER STRUCTURE REQUIRED:
  Agentic/
  |- agent1_validation.py
  |- agent2_report.py
  |- streamlit_app.py
  |- patient_history.json       (auto-created)
  |- appointments.json          (auto-created)
  +- outputs/
     |- results/
     |  +- prediction_patient_001_S1.png   <- your scan images
     +- agent_reports/                     (auto-created)

RUN STANDALONE (single image):
  python agent1_validation.py outputs/results/prediction_patient_001_S1.png

RUN BATCH (all images):
  python agent1_validation.py
"""

# ================================================================
# CONFIGURATION  -- only edit this section
# ================================================================
GROQ_API_KEY = ""   # get free key at console.groq.com
GROQ_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"
# ================================================================

import json
import base64
import datetime
import sys
import re
from pathlib import Path
import urllib.request
import urllib.error

# -- Directory paths --------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
RESULTS_DIR  = SCRIPT_DIR / "outputs" / "results"
REPORTS_DIR  = SCRIPT_DIR / "outputs" / "agent_reports"
HISTORY_FILE = SCRIPT_DIR / "patient_history.json"
MAX_RETRIES  = 3

# Auto-create folders on first run
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class QuotaExceededError(Exception):
    """Raised on HTTP 429 -- Groq rate limit. Wait ~60s and retry."""
    pass


# -- System prompt sent to Groq vision model --------------------------
SYSTEM_PROMPT = """You are a senior radiology AI validation system for breast ultrasound.
Review the 6-panel AI output image and validate whether the prediction is clinically trustworthy.

PANEL LAYOUT:
  Top-left      = Panel 1: Raw RF signal
  Top-center    = Panel 2: Beamformed ultrasound image
  Top-right     = Panel 3: Enhanced image
  Bottom-left   = Panel 4: Tumor probability heatmap (bright = high probability)
  Bottom-center = Panel 5: Binary tumor mask (white = tumor, % area in title)
  Bottom-right  = Panel 6: DIAGNOSIS - predicted class + confidence % + bar chart

BI-RADS BENIGN (2-3): oval/round, parallel orientation, circumscribed margin, homogeneous echo
BI-RADS MALIGNANT (4-5): irregular shape, non-parallel, spiculated margin, hypoechoic, shadowing

VERDICT THRESHOLDS:
  confidence >= 85% AND consistent visuals         = APPROVED  (exit 0)
  confidence 70-84% OR minor visual inconsistency  = FLAGGED   (exit 1)
  confidence < 70%  OR major inconsistency         = REJECTED  (exit 2)
  two classes within 15% of each other             = always FLAGGED

STEPS TO FOLLOW:
  1. Read Panel 6: predicted class and exact confidence %
  2. Read Panel 4: heatmap brightness and coverage area
  3. Read Panel 5: mask area % and shape regularity
  4. Compare visual features against expected BI-RADS morphology
  5. Issue verdict

IMPORTANT: Return ONLY the JSON object below. No markdown fences, no explanation text.

{
  "patient_id": "string",
  "scan_number": "string",
  "predicted_class": "benign",
  "extracted_confidence": 91.5,
  "birads_estimate": "3",
  "verdict": "APPROVED",
  "exit_code": 0,
  "urgency": "routine",
  "visual_consistency": "consistent",
  "flag_reasons": [],
  "close_competition_detected": false,
  "history_escalation": false,
  "clinical_reasoning": "Detailed paragraph explaining findings and reasoning.",
  "recommended_action": "What should happen next clinically."
}

FIELD RULES:
  predicted_class      : exactly "benign" or "malignant"
  extracted_confidence : number 0-100 (no % sign)
  birads_estimate      : one of "1-2", "3", "4A", "4B", "4C", "5"
  verdict              : one of "APPROVED", "FLAGGED", "REJECTED"
  exit_code            : 0=APPROVED, 1=FLAGGED, 2=REJECTED
  urgency              : one of "routine", "priority", "urgent", "emergency"
  visual_consistency   : one of "consistent", "inconsistent", "ambiguous"
  flag_reasons         : list of strings, empty [] if none
"""


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def encode_image(image_path):
    """Read image from disk and return base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_history(patient_id):
    """Load prior scan history for a patient from patient_history.json."""
    if not HISTORY_FILE.exists():
        return {}
    with open(HISTORY_FILE, "r") as f:
        return json.load(f).get(patient_id, {})


def save_history(patient_id, scan_number, verdict):
    """Append this scan's verdict to the patient history file."""
    data = {}
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)

    data.setdefault(patient_id, {"scans": []})
    data[patient_id]["scans"].append({
        "scan_number":     scan_number,
        "date":            str(datetime.datetime.now())[:10],
        "predicted_class": verdict.get("predicted_class"),
        "birads_estimate": verdict.get("birads_estimate"),
        "verdict":         verdict.get("verdict"),
        "urgency":         verdict.get("urgency"),
    })

    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


def build_history_text(patient_id):
    """Format patient history as a plain-text block for the prompt."""
    h = load_history(patient_id)
    if not h or not h.get("scans"):
        return "No prior history. This is the first recorded scan."

    lines = [f"PRIOR SCANS for {patient_id}:"]
    for s in h["scans"]:
        lines.append(
            f"  Scan {s['scan_number']} ({s['date']}): "
            f"{s['predicted_class']} | BI-RADS {s['birads_estimate']} | {s['verdict']}"
        )
    return "\n".join(lines)


def parse_filename(filename):
    """
    Extract patient ID and scan number from image filename.
    Expected format: prediction_patient_001_S1.png
    Returns: ("PT-001", "S1")
    """
    stem  = Path(filename).stem
    parts = stem.split("_")
    num   = parts[2] if len(parts) > 2 else "000"
    scan  = parts[3] if len(parts) > 3 else "S1"
    return f"PT-{num}", scan


# =====================================================================
# GROQ VISION API
# =====================================================================

def call_groq(image_path, user_prompt):
    import requests

    image_b64 = encode_image(image_path)
    image_url = f"data:image/png;base64,{image_b64}"

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
    }

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90
        )

        print(f"  [Groq] Status: {resp.status_code}")

        if resp.status_code == 200:
            text = resp.json()["choices"][0]["message"]["content"].strip()
            print(f"  [Groq] ✅ OK — {len(text)} chars")
            print(f"  [Groq] Preview: {text[:300]}")
            return text

        elif resp.status_code == 401:
            raise Exception("AUTH FAILED (401) — wrong API key. Go to console.groq.com and create a new key.")
        elif resp.status_code == 429:
            raise Exception("RATE LIMIT (429) — wait 1 minute and retry.")
        elif resp.status_code == 403:
            raise Exception(f"BLOCKED (403) — Cloudflare blocking. Detail: {resp.text[:200]}")
        else:
            raise Exception(f"Groq API error HTTP {resp.status_code}: {resp.text[:300]}")

    except requests.exceptions.ConnectionError:
        raise Exception("NETWORK ERROR — cannot reach api.groq.com. Check internet connection.")
    except requests.exceptions.Timeout:
        raise Exception("TIMEOUT — Groq took too long. Image may be too large.")

# =====================================================================
# JSON PARSING
# =====================================================================

def parse_json_safe(raw, image_path, prompt):
    """
    Parse the model's text response as JSON.

    Handles common model quirks:
      - Markdown code fences (```json ... ```)
      - Extra explanation text before/after the JSON block
      - Incomplete JSON (retries with a stricter instruction)

    Returns a valid dict on success, or a REJECTED fallback dict on failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Strip markdown fences like ```json ... ```
            cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

            if cleaned.startswith("{"):
                return json.loads(cleaned)

            # Find JSON object anywhere in the response (greedy)
            match = re.search(r"\{[\s\S]+\}", cleaned)
            if match:
                return json.loads(match.group())

        except json.JSONDecodeError as je:
            print(f"  [JSON attempt {attempt}/{MAX_RETRIES}] Parse error: {je}")

        print(f"  [JSON attempt {attempt}] Raw (first 200): {repr(raw[:200])}")

        if attempt < MAX_RETRIES:
            try:
                raw = call_groq(
                    image_path,
                    prompt + (
                        "\n\nIMPORTANT: Your last response could not be parsed as JSON.\n"
                        "Return ONLY a valid JSON object.\n"
                        "Start with { and end with }.\n"
                        "No markdown, no code fences, no explanation."
                    )
                )
            except QuotaExceededError:
                raise   # Propagate immediately -- no point retrying
            except Exception as ex:
                print(f"  [Retry API call failed] {ex}")

    # All retries failed -- return safe fallback
    print("  [Safety fallback] All JSON parse attempts failed. Returning REJECTED verdict.")
    return {
        "patient_id":               "UNKNOWN",
        "scan_number":              "UNKNOWN",
        "predicted_class":          "unknown",
        "extracted_confidence":     0,
        "birads_estimate":          "unknown",
        "verdict":                  "REJECTED",
        "exit_code":                2,
        "urgency":                  "urgent",
        "visual_consistency":       "ambiguous",
        "flag_reasons":             ["JSON parse failure -- see terminal for details"],
        "close_competition_detected": False,
        "history_escalation":       False,
        "clinical_reasoning":       (
            "Safety fallback: the Groq vision model did not return "
            "parseable JSON after 3 attempts. This scan requires mandatory "
            "human radiologist review."
        ),
        "recommended_action":       "Mandatory human radiologist review required. Do not use AI output for clinical decision.",
    }


# =====================================================================
# THRESHOLD LOGIC
# =====================================================================

def apply_thresholds(v):
    """
    Override verdict and urgency based on confidence + visual consistency.

    This always runs after the model's response to enforce consistent rules,
    regardless of what the model itself decided.
    """
    conf = float(v.get("extracted_confidence", 0) or 0)

    # -- Verdict --
    if (
        conf >= 85
        and v.get("visual_consistency") != "inconsistent"
        and not v.get("close_competition_detected")
    ):
        v["verdict"]   = "APPROVED"
        v["exit_code"] = 0
    elif conf >= 70:
        v["verdict"]   = "FLAGGED"
        v["exit_code"] = 1
    else:
        v["verdict"]   = "REJECTED"
        v["exit_code"] = 2

    # -- Urgency --
    birads = str(v.get("birads_estimate", "1-2"))
    if birads == "5" or v.get("history_escalation"):
        v["urgency"] = "emergency"
    elif birads in ["4B", "4C"] or v.get("visual_consistency") == "inconsistent":
        v["urgency"] = "urgent"
    elif v["verdict"] == "FLAGGED":
        v["urgency"] = "priority"
    else:
        v["urgency"] = "routine"

    return v


# =====================================================================
# MAIN RUNNER
# =====================================================================

def run_agent1(image_path, patient_id=None, scan_number=None, risk_factors="none"):
    """
    Main Agent 1 pipeline. Called by streamlit_app.py and from CLI.

    Steps:
      1. Resolve and validate image path
      2. Parse patient ID + scan number from filename (if not provided)
      3. Build prompt with patient history context
      4. Call Groq vision API
      5. Parse JSON response (with retries)
      6. Apply threshold rules (confidence + BI-RADS)
      7. Save scan to patient_history.json
      8. Save full verdict JSON to outputs/agent_reports/

    Returns:
      dict  -- verdict on success
      None  -- if image file not found
    """
    image_path = Path(image_path)

    # Resolve image path
    if not image_path.exists():
        alt = RESULTS_DIR / image_path.name
        if alt.exists():
            image_path = alt
        else:
            print(f"\n  ERROR: Image not found.")
            print(f"  Tried: {image_path}")
            print(f"  Tried: {alt}")
            print(f"  Place scan images in: {RESULTS_DIR}")
            print(f"  Filename format: prediction_patient_001_S1.png")
            return None

    # Parse patient metadata from filename
    if patient_id is None or scan_number is None:
        auto_id, auto_scan = parse_filename(image_path.name)
        patient_id  = patient_id  or auto_id
        scan_number = scan_number or auto_scan

    print(f"\n{'='*62}")
    print(f"  AGENT 1 -- Clinical Validation")
    print(f"  Image   : {image_path.name}")
    print(f"  Patient : {patient_id}  |  Scan: {scan_number}")
    print(f"  Backend : GROQ | {GROQ_MODEL}")
    print(f"{'='*62}")

    history_text = build_history_text(patient_id)
    now          = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    user_prompt = (
        f"Patient ID   : {patient_id}\n"
        f"Scan Number  : {scan_number}\n"
        f"Date         : {now}\n"
        f"Risk Factors : {risk_factors}\n\n"
        f"{history_text}\n\n"
        f"Validate the 6-panel breast ultrasound AI output image shown above.\n"
        f"Return ONLY the JSON verdict object. No other text."
    )

    # Step 1: Call Groq
    print(f"\n[1/4] Calling Groq Vision API...")
    try:
        raw = call_groq(str(image_path), user_prompt)
        print(f"  Response length: {len(raw)} chars")

    except QuotaExceededError as qe:
        print(f"\n  RATE LIMIT HIT: {qe}")
        return {
            "quota_exceeded":       True,
            "quota_message":        str(qe),
            "patient_id":           patient_id,
            "scan_number":          scan_number,
            "scan_date":            now,
            "verdict":              "QUOTA_EXCEEDED",
            "extracted_confidence": None,
            "birads_estimate":      None,
            "urgency":              None,
            "visual_consistency":   None,
            "predicted_class":      None,
            "clinical_reasoning":   "Groq rate limit reached. Wait ~60 seconds and retry.",
            "recommended_action":   "Wait 60 seconds, then click Run Validation again.",
            "flag_reasons":         ["Groq rate limit -- HTTP 429"],
            "backend":              "groq",
            "model":                GROQ_MODEL,
        }

    except Exception as ex:
        print(f"  API call failed: {ex}")
        raw = "{}"

    # Step 2: Parse JSON
    print(f"[2/4] Parsing JSON response...")
    verdict = parse_json_safe(raw, str(image_path), user_prompt)

    # Step 3: Apply thresholds
    print(f"[3/4] Applying thresholds...")
    verdict = apply_thresholds(verdict)

    # Step 4: Enrich and save
    print(f"[4/4] Saving verdict...")
    verdict["patient_id"]  = patient_id
    verdict["scan_number"] = scan_number
    verdict["scan_date"]   = now
    verdict["image_path"]  = str(image_path)
    verdict["backend"]     = "groq"
    verdict["model"]       = GROQ_MODEL

    save_history(patient_id, scan_number, verdict)

    report_path = REPORTS_DIR / f"{patient_id}_{scan_number}_verdict.json"
    with open(report_path, "w") as f:
        json.dump(verdict, f, indent=2)

    print(f"\n  VERDICT    : {verdict.get('verdict')}  (exit {verdict.get('exit_code')})")
    print(f"  Prediction : {str(verdict.get('predicted_class','')).upper()}")
    print(f"  Confidence : {verdict.get('extracted_confidence')}%")
    print(f"  BI-RADS    : {verdict.get('birads_estimate')}")
    print(f"  Urgency    : {str(verdict.get('urgency','')).upper()}")
    print(f"  Report     : {report_path.name}")

    return verdict


def run_all():
    """
    Batch mode: process all prediction_patient_*.png files in outputs/results/.
    Run as: python agent1_validation.py
    """
    images = sorted(RESULTS_DIR.glob("prediction_patient_*.png"))

    if not images:
        print(f"\n  No images found in: {RESULTS_DIR}")
        print(f"  Expected filename format: prediction_patient_001_S1.png")
        return

    print(f"\n  Found {len(images)} image(s). Running batch with Groq...")

    results   = []
    approved  = flagged = rejected = quota_err = 0

    for i, img in enumerate(images, 1):
        print(f"\n  [{i}/{len(images)}] Processing: {img.name}")
        v = run_agent1(img)
        if v:
            results.append(v)
            if v.get("quota_exceeded"):          quota_err += 1
            elif v.get("verdict") == "APPROVED": approved  += 1
            elif v.get("verdict") == "FLAGGED":  flagged   += 1
            elif v.get("verdict") == "REJECTED": rejected  += 1

    print(f"\n{'='*50}")
    print(f"  BATCH COMPLETE")
    print(f"  Total processed : {len(results)}")
    print(f"  APPROVED        : {approved}")
    print(f"  FLAGGED         : {flagged}")
    print(f"  REJECTED        : {rejected}")
    print(f"  QUOTA ERRORS    : {quota_err}")
    print(f"{'='*50}")


# -- Entry point -------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = run_agent1(sys.argv[1])
        if result:
            sys.exit(result.get("exit_code", 2))
    else:
        run_all()