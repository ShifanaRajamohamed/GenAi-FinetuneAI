import os
import time
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from .models import (
    ChatRequest, ChatResponse, DocumentMetadata, Encounter,
    AuditEntry, UserRole, LoginRequest, LoginResponse, SessionStatus, AuthUser
)
from .rag_engine import MediDocsRAGEngine, ROLE_PERMISSIONS, ROLE_LABELS
from .evaluation_data import run_evaluation_suite
from .audit_logger import AuditLogger
from .mft_bridge import MobileFineTunerBridge
from .encounters import INITIAL_ENCOUNTERS
from .auth import AuthService

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

app = FastAPI(
    title="MediDocs AI",
    description="Hospital Document Intelligence and Clinical Workflow Assistant powered by RAG and MobileFineTuner",
    version="1.0.0"
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate core singletons
rag_engine = MediDocsRAGEngine()
audit_logger = AuditLogger()
mft_bridge = MobileFineTunerBridge()
auth_service = AuthService()
encounters_db: List[Encounter] = list(INITIAL_ENCOUNTERS)

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "MediDocs AI",
        "docs_indexed": len(rag_engine.documents),
        "chunks_indexed": len(rag_engine.chunks),
        "time": time.strftime("%Y-%m-%d %H:%M:%S")
    }

# ==================== AUTHENTICATION & LOGIN API ====================
@app.post("/api/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest):
    res = auth_service.authenticate(payload)
    if not res:
        audit_logger.log(
            user_role="unknown",
            query=f"Failed login attempt: {payload.username_or_email}",
            status="BLOCKED",
            retrieved_count=0,
            docs=[],
            latency_ms=1.2
        )
        raise HTTPException(
            status_code=401,
            detail="Invalid hospital credentials. Please check your username/email and password."
        )

    audit_logger.log(
        user_role=res.user.role_id,
        query=f"Clinical login successful: {res.user.name} ({res.user.license_number})",
        status="ALLOWED",
        retrieved_count=0,
        docs=[],
        latency_ms=1.5
    )
    return res

@app.get("/api/auth/me", response_model=SessionStatus)
def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization:
        return SessionStatus(authenticated=False, user=None)
    user = auth_service.validate_token(authorization)
    if not user:
        return SessionStatus(authenticated=False, user=None)
    return SessionStatus(authenticated=True, user=user)

@app.post("/api/auth/logout")
def logout(authorization: Optional[str] = Header(None)):
    if authorization:
        auth_service.logout(authorization)
    return {"message": "Successfully signed out of clinical workstation"}

@app.get("/api/auth/demo-users")
def get_demo_users():
    return auth_service.get_all_demo_users()

# ==================== ENCOUNTERS API ====================
@app.get("/api/encounters", response_model=List[Encounter])
def list_encounters():
    return encounters_db

@app.get("/api/encounters/{enc_id}", response_model=Encounter)
def get_encounter(enc_id: str):
    for enc in encounters_db:
        if enc.id == enc_id:
            return enc
    raise HTTPException(status_code=404, detail="Encounter not found")

@app.post("/api/encounters", response_model=Encounter)
def create_encounter(
    title: str = Form("New Clinical Consultation"),
    patient_name: str = Form("New Patient"),
    category: str = Form("TODAY")
):
    new_id = f"enc-{len(encounters_db) + 1:02d}"
    t_str = time.strftime("%I:%M %p")
    enc = Encounter(
        id=new_id,
        title=title,
        time=t_str,
        category=category,
        patient_name=patient_name,
        patient_id=f"PT-{90000 + len(encounters_db)}",
        duration="0 mins, 30 secs",
        subjective="Subjective:\n- Patient presents for consultation.",
        objective="Objective:\n- Vitals: BP: 120/80 mmHg, HR: 72 bpm, SpO2: 99% RA.",
        assessment="Assessment:\n- Clinical evaluation pending.",
        plan="Plan:\n- Awaiting clinical recommendations.",
        patient_handout="Patient care instructions pending.",
        sick_note="No sick note issued."
    )
    encounters_db.insert(0, enc)
    return enc

@app.put("/api/encounters/{enc_id}", response_model=Encounter)
def update_encounter(enc_id: str, payload: dict):
    for enc in encounters_db:
        if enc.id == enc_id:
            if "subjective" in payload: enc.subjective = payload["subjective"]
            if "objective" in payload: enc.objective = payload["objective"]
            if "assessment" in payload: enc.assessment = payload["assessment"]
            if "plan" in payload: enc.plan = payload["plan"]
            if "patient_handout" in payload: enc.patient_handout = payload["patient_handout"]
            if "sick_note" in payload: enc.sick_note = payload["sick_note"]
            if "title" in payload: enc.title = payload["title"]
            return enc
    raise HTTPException(status_code=404, detail="Encounter not found")

# ==================== DOCUMENTS & UPLOAD API ====================
@app.get("/api/documents")
def list_documents():
    return list(rag_engine.documents.values())

@app.get("/api/documents/{doc_id}")
def get_document_detail(doc_id: str):
    target_key = None
    if doc_id in rag_engine.documents:
        target_key = doc_id
    else:
        for k, d in rag_engine.documents.items():
            if d.code.lower() == doc_id.lower() or d.id.lower() == doc_id.lower():
                target_key = k
                break
    if not target_key:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    meta = rag_engine.documents[target_key]
    doc_chunks = [c for c in rag_engine.chunks if c.doc_id == target_key]
    full_text = "\n\n".join(f"[{c.section}]\n{c.content}" for c in doc_chunks)
    return {
        "metadata": meta,
        "chunks": doc_chunks,
        "full_text": full_text
    }

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    code: str = Form(...),
    department: str = Form("Hospital Operations"),
    version: str = Form("v1.0"),
    access_level: str = Form("CLINICAL"),
    effective_date: str = Form("2026-04-01")
):
    contents = await file.read()
    filename = file.filename or "uploaded_doc.txt"
    pages = []

    if filename.lower().endswith(".pdf"):
        try:
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(contents))
            for p in reader.pages:
                text = p.extract_text() or ""
                if text.strip():
                    pages.append(text)
        except Exception:
            pages = [contents.decode("utf-8", errors="ignore")]
    else:
        raw_text = contents.decode("utf-8", errors="ignore")
        parts = [p.strip() for p in raw_text.split("\n\n\n") if p.strip()]
        pages = parts if parts else [raw_text]

    doc_id = f"doc_{len(rag_engine.documents) + 1:03d}"
    meta = DocumentMetadata(
        id=doc_id,
        code=code,
        title=title,
        department=department,
        version=version,
        effective_date=effective_date,
        access_level=access_level,
        status="active"
    )
    rag_engine.add_document(meta, pages)
    return {
        "status": "success",
        "document": meta,
        "chunks_indexed": len(pages),
        "total_documents": len(rag_engine.documents)
    }

# ==================== TEMPLATES & SNIPPETS API ====================
CLINICAL_TEMPLATES = [
    {
        "id": "tpl-soap",
        "title": "Standard Ambulatory SOAP Note",
        "department": "Ambulatory & General Medicine",
        "content": (
            "Subjective:\n"
            "- Chief Complaint:\n"
            "- History of Present Illness (Onset, Duration, Severity, Triggers):\n"
            "- Review of Systems: Denies fever, chest pain, or shortness of breath.\n\n"
            "Objective:\n"
            "- Vitals: T: 36.6°C, BP: 120/80 mmHg, HR: 72 bpm, RR: 16/min, O2 sat: 99% RA\n"
            "- Physical Exam: Alert, oriented x 3, in no acute distress.\n"
            "- Cardiovascular: Regular rate and rhythm, normal S1/S2.\n"
            "- Pulmonary: Clear to auscultation bilaterally, no wheezes or crackles.\n\n"
            "Assessment:\n"
            "- Primary Clinical Impression:\n\n"
            "Plan:\n"
            "- Diagnostic Investigations:\n"
            "- Pharmacotherapy:\n"
            "- Patient Education & Handout Provided:\n"
            "- Follow-up Window: 7 to 14 days or return if symptoms worsen."
        )
    },
    {
        "id": "tpl-sepsis",
        "title": "Sepsis 1-Hour Early Alert Bundle (SOP-CLIN-012)",
        "department": "Emergency & Internal Medicine",
        "content": (
            "CODE SEPSIS PROTOCOL CHECKLIST (SOP-CLIN-012 v3.2)\n"
            "1. qSOFA Screening Score:\n"
            "   [ ] Respiratory Rate ≥ 22/min (1 pt)\n"
            "   [ ] Altered Mentation GCS < 15 (1 pt)\n"
            "   [ ] Systolic BP ≤ 100 mmHg (1 pt)\n"
            "   Total qSOFA Score: __ / 3 (Score ≥ 2 activates Code Sepsis)\n\n"
            "2. Mandatory 1-Hour Resuscitation Bundle:\n"
            "   [ ] Initial serum lactate drawn: ___ mmol/L (Repeat in 2-4 hrs if > 2.0)\n"
            "   [ ] Blood cultures x 2 drawn PRIOR to antimicrobial administration\n"
            "   [ ] Broad-spectrum empiric IV antibiotics infused within 60 min\n"
            "   [ ] 30 mL/kg IV crystalloid fluid initiated for hypotension or lactate ≥ 4.0\n"
            "   [ ] Norepinephrine titrated to maintain MAP ≥ 65 mmHg if refractory"
        )
    },
    {
        "id": "tpl-excursion",
        "title": "Temperature Excursion Incident Report Form e-RX-901 (POL-PHARM-045)",
        "department": "Pharmacy & Nursing Operations",
        "content": (
            "INCIDENT REPORT: REFRIGERATION COLD-CHAIN EXCURSION (POL-PHARM-045 v2.1)\n"
            "Date/Time Discovered: Today, __:__\n"
            "Unit Refrigerator ID: 4-North Refrigerator #2\n"
            "Recorded Temperature: ___ °C (Target Range: 2.0°C - 8.0°C)\n"
            "Continuous Excursion Duration: ___ minutes (>15 min threshold met)\n\n"
            "Containment Actions Executed within 30 Minutes:\n"
            "1. [x] Neon red 'QUARANTINE - DO NOT DISPENSE' tamper-evident tape applied\n"
            "2. [x] Electronic Incident Form e-RX-901 submitted on Intranet\n"
            "3. [ ] Central Pharmacy Dispatch notified at ext. 4402\n"
            "4. [ ] High-cost biologics transferred to auxiliary cold room (>10°C / >45 min)\n\n"
            "Disposition: Stock held in quarantine. Awaiting written stability clearance from Chief Clinical Pharmacist."
        )
    },
    {
        "id": "tpl-asthma",
        "title": "Adult Asthma Exacerbation Outpatient Plan (CLIN-FORM-022)",
        "department": "Ambulatory & Pulmonary Care",
        "content": (
            "Subjective:\n"
            "- 2-day history of shortness of breath and wheezing, worse in spring\n"
            "- No fever, no sick contacts, denies chest pain\n\n"
            "Objective:\n"
            "- T: 36.4°C, BP: 120/84 mmHg, HR: 88 bpm, RR: 24/min, O2 sat: 98% RA\n"
            "- Respiratory: Mild expiratory wheezes bilaterally, no crackles\n"
            "- Cardiovascular: Normal S1/S2, no murmurs, no peripheral edema\n\n"
            "Assessment:\n"
            "- Asthma exacerbation (Moderate)\n\n"
            "Plan:\n"
            "- Prescribed Symbicort BID with spacer\n"
            "- Referral for pulmonary function tests (PFT)\n"
            "- Provide sick note for 2 days\n"
            "- Patient instructions & Asthma Action Plan provided\n"
            "- Follow-up in 5 days or sooner if symptoms worsen"
        )
    }
]

CLINICAL_SNIPPETS = [
    {
        "id": "snp-norm-lung",
        "title": "Normal Lung Auscultation",
        "text": "Lungs clear to auscultation bilaterally. No wheezes, rales, or rhonchi. Normal vesicular breath sounds throughout."
    },
    {
        "id": "snp-wheeze",
        "title": "Bilateral Expiratory Wheeze",
        "text": "Respiratory: Expiratory wheezes heard across bilateral lung fields. Prolonged expiratory phase. SpO2 98% on ambient room air."
    },
    {
        "id": "snp-qsofa-neg",
        "title": "qSOFA Screening Negative",
        "text": "qSOFA Bedside Sepsis Screening: RR < 22/min (0 pts), GCS 15 alert and oriented (0 pts), SBP > 100 mmHg (0 pts). qSOFA = 0. Code Sepsis criteria not met."
    },
    {
        "id": "snp-quarantine",
        "title": "Cold-Chain Quarantine Notice",
        "text": "Quarantine Enforcement: Neon red 'QUARANTINE - DO NOT DISPENSE' tamper-evident tape secured to unit refrigerator handle. Incident Form e-RX-901 completed."
    },
    {
        "id": "snp-asthma-ed",
        "title": "Asthma Spacer Education",
        "text": "Patient education completed: Demonstrated multi-dose inhaler with valved holding chamber spacer technique. Rinsed mouth with water after corticosteroid inhalation."
    }
]

PATIENTS_DIRECTORY = [
    {
        "id": "PT-90421",
        "name": "Jane Doe",
        "dob": "1991-04-12 (34yo)",
        "gender": "Female",
        "mrn": "MRN-0090421",
        "department": "Ambulatory Pulmonary",
        "primary_dx": "Moderate Asthma Exacerbation",
        "allergies": "Penicillin (Urticaria)",
        "insurance": "Blue Cross Blue Shield PPO",
        "vitals": "BP: 120/84, HR: 88, SpO2: 98%"
    },
    {
        "id": "PT-88129",
        "name": "Robert Smith",
        "dob": "1967-08-23 (58yo)",
        "gender": "Male",
        "mrn": "MRN-0088129",
        "department": "Endocrinology",
        "primary_dx": "Type 2 Diabetes Mellitus",
        "allergies": "No Known Drug Allergies (NKDA)",
        "insurance": "Medicare Part B",
        "vitals": "BP: 128/80, HR: 74, SpO2: 99%"
    },
    {
        "id": "PT-77102",
        "name": "Emily Davis",
        "dob": "1983-11-04 (42yo)",
        "gender": "Female",
        "mrn": "MRN-0077102",
        "department": "Internal Medicine",
        "primary_dx": "Chronic Fatigue / Hypothyroidism",
        "allergies": "Sulfa Drugs",
        "insurance": "Aetna Health",
        "vitals": "BP: 118/76, HR: 58, SpO2: 99%"
    },
    {
        "id": "PT-65201",
        "name": "Arthur Pendelton",
        "dob": "1958-02-19 (67yo)",
        "gender": "Male",
        "mrn": "MRN-0065201",
        "department": "Emergency & Cardiology",
        "primary_dx": "Acute Coronary Syndrome (NSTEMI)",
        "allergies": "Aspirin (Bronchospasm)",
        "insurance": "UnitedHealthcare",
        "vitals": "BP: 154/92, HR: 96, SpO2: 97%"
    }
]

@app.get("/api/templates")
def list_templates():
    return CLINICAL_TEMPLATES

@app.get("/api/snippets")
def list_snippets():
    return CLINICAL_SNIPPETS

@app.get("/api/patients")
def list_patients():
    return PATIENTS_DIRECTORY

# ==================== RAG CHAT & CITATION API ====================
@app.post("/api/chat", response_model=ChatResponse)
def grounded_chat(req: ChatRequest):
    response = rag_engine.generate_grounded_answer(req.query, req.user_role)
    
    # Audit log this event
    status = "BLOCKED" if response.access_blocked else ("ABSTAINED" if response.abstention_reason else "ALLOWED")
    cited_docs = list(set(c.doc_code for c in response.citations))
    audit_logger.log(
        user_role=req.user_role,
        query=req.query,
        status=status,
        retrieved_count=len(response.citations),
        docs=cited_docs,
        latency_ms=response.latency_ms
    )
    return response

# ==================== ROLES & RBAC API ====================
@app.get("/api/roles")
def get_roles():
    return [
        {
            "id": role_id,
            "label": ROLE_LABELS[role_id],
            "permissions": list(ROLE_PERMISSIONS[role_id])
        }
        for role_id in ROLE_PERMISSIONS
    ]

# ==================== AUDIT TRAIL API ====================
@app.get("/api/audit")
def get_audit_trail(limit: int = 50):
    return audit_logger.get_logs(limit)

# ==================== EVALUATION DASHBOARD API ====================
@app.post("/api/evaluation/run")
def run_evaluation():
    results = run_evaluation_suite(rag_engine)
    return results

# ==================== MOBILEFINETUNER LAB API ====================
@app.get("/api/finetuning/status")
def get_finetuning_status():
    return {
        "engine": "MobileFineTuner C++17 Core (PEFT / LoRA)",
        "experiment": mft_bridge.current_experiment,
        "comparison": mft_bridge.get_three_way_comparison()
    }

@app.post("/api/finetuning/train")
def run_finetuning_smoke(
    model_key: str = "gpt2_small",
    steps: int = 5,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: str = "Attention QKV + Proj"
):
    mft_bridge.current_experiment["adapter_type"] = f"LoRA (Rank={rank}, Alpha={alpha})"
    mft_bridge.current_experiment["target_modules"] = target_modules
    result = mft_bridge.run_lora_fine_tuning_smoke(model_key=model_key, steps=steps)
    return result

# ==================== STATIC FRONTEND SERVING ====================
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
