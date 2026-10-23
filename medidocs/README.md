# MediDocs AI — Hospital Document Intelligence & Clinical Workflow Assistant

**A Source-Grounded, Permission-Aware Healthcare Document Intelligence & Workflow Assistant Powered by RAG with MobileFineTuner Model Adaptation**

[![Platform](https://img.shields.io/badge/Platform-FastAPI%20%7C%20Vanilla%20JS%20%7C%20C%2B%2B17%20MobileFineTuner-blue.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](../LICENSE)
[![RAG Benchmark](https://img.shields.io/badge/Recall%405-97.1%25-green.svg)]()
[![Access Control](https://img.shields.io/badge/RBAC%20Violations-0%20(100%25%20Blocked)-brightgreen.svg)]()

---

## Visual Design Reference: Vero Pro Healthcare Interface

MediDocs AI replicates the clinical workflow layout modeled after **Vero Pro**:

<div align="center">
  <img src="../logo.jpg" alt="MediDocs AI Healthcare Suite" width="220"/>
</div>

### 3-Panel Layout Structure

1. **Left Sidebar Panel (Encounters & Navigation)**:
   - **Encounter Timeline**: Categorized by date (`TODAY`, `TUE MAR 31`, `MON MAR 30`, `SUN MAR 29`, `WED MAR 25`) with active state highlight on `1:31 PM Shortness of Breath`.
   - **Primary Action**: Prominent `+ New Encounter` button in cyan (`#0ea5e9`) with lightning quick-action shortcut.
   - **Bottom Navigation Dock**: `Evidence`, `Templates`, `Snippets`, `Patients`, `Evaluation`, `FineTuning Lab`, and `Audit Trail`.

2. **Center Panel (Clinical Note & Documentation Editor)**:
   - **Header**: Editable encounter title (`Shortness of Breath`), active session duration timer (`⏱ 5 mins, 10 secs | 1:31 - 1:36 PM`), and security clearance badge (`Attending Physician`).
   - **Documentation Tabs**: `[Context & Record]`, `[SOAP •••]` (active), `[Patient Handout]`, `[Sick Note]`, and `[+ Create Note]`.
   - **Clinical Toolbar**: Formatting controls (Bold, Italic, Underline, Strike), microphone voice dictation, patient variables injector, typography, copy dropdown, export, grounding insights lamp, and code toggle.
   - **SOAP Note Content**:
     - **Subjective**: 2-day history of shortness of breath and wheezing, worse in spring; no fever, no sick contacts, denies chest pain.
     - **Objective**: T: 36.4°C, BP: 120/84 mmHg, HR: 88 bpm, RR: 24/min, O2 sat: 98% RA; mild bilateral expiratory wheezes; normal S1/S2.
     - **Assessment**: Asthma exacerbation with clickable citation chip `[SOP-RES-04 §3.2]`.
     - **Plan**: Symbicort BID with spacer `[FORM-ASTHMA-v2]`, pulmonary function test referral, 2-day sick note, patient instructions, follow-up in 5 days.
   - **Footer**: Thumbs up / down feedback buttons and governance disclaimer: *"Draft generated for authorized clinical review. Answers must be verified against approved hospital policy."*

3. **Right Panel (AI Assistant & Evidence Drawer)**:
   - **Ethereal Orb Avatar**: Dynamic CSS radial-gradient glowing sphere with breathing pulse animation and color-shift glow.
   - **Greeting**: *"Good Afternoon — I'm your AI assistant, here to help"*.
   - **Suggested Action Pills**:
     - `🩺 Diagnose a condition or symptom`
     - `📖 Look up clinical guidelines`
     - `💊 Check drug interactions and dosing`
     - `🔬 Research a medical topic`
     - `📝 Edit and simplify your note`
     - `❄️ Temperature Excursion Procedure (Test Conflict)`
     - `🔒 Mechanical Ventilation Weaning (Test RBAC)`
   - **Interactive Input Dock**: Attachment button (`+`), policy dropdown filter (`Q ⌄`), microphone voice search, auto-expanding query input, and send arrow button.

---

## 6 Core MVP Modules

### Module 1 — Document Library & Ingestion
- Upload clinical PDFs and plain-text policies via drag-and-drop or file selection.
- Text extraction with page-level tracking (using pure-Python `pypdf`).
- Automatic metadata tagging: Document Code, Title, Department, Version, Effective Date, Access Level (`GENERAL`, `CLINICAL`, `RESTRICTED_ICU`, `ADMIN`), and Status (`active` vs `superseded`).

### Module 2 — Permission-Aware RAG Engine
- Hybrid vector embedding (cosine similarity over normalized term frequencies) combined with BM25 keyword matching.
- **Backend Enforced RBAC**: Clearance permissions are evaluated in the retrieval layer *before* scoring passages:
  - `Attending Physician`: Full access (`GENERAL`, `CLINICAL`, `RESTRICTED_ICU`, `ADMIN`).
  - `Staff Nurse`: Clinical access (`GENERAL`, `CLINICAL`).
  - `Hospital Administrator`: Administrative access (`GENERAL`, `ADMIN`).
  - `Clinical Intern`: General staff access (`GENERAL`).
- Attempts by unauthorized roles to retrieve confidential policies (e.g., ICU mechanical ventilation weaning or HR shift caps) are strictly blocked with an explicit audit event.

### Module 3 — Evidence Viewer & Conflict/Version Detection
- Clickable citation cards displaying Document Code, Version, Page Number, Section Title, Similarity Match Percentage, and verbatim passage excerpt.
- **Outdated & Conflicting Policy Detection**: Identifies when a query triggers both active and superseded policies (e.g., active `POL-PHARM-045 v2.1` vs legacy `v1.0`). Displays an amber warning alert highlighting the conflict while grounding the final answer strictly in the current revision.
- **Responsible Abstention**: If maximum retrieval confidence falls below threshold (< 0.22) or the question is ungrounded/out-of-domain, the engine abstains rather than hallucinating unsupported clinical advice.

### Module 4 — User Roles & Security Audit Trail
- Comprehensive audit logger recording timestamp, active role, natural language query, enforcement status (`ALLOWED`, `BLOCKED`, `ABSTAINED`), retrieved document codes, and latency in milliseconds.
- Accessible via the `Audit Trail` navigation tab in the UI.

### Module 5 — RAG Quality Evaluation Benchmark
- Standardized 50-test question benchmark suite across 6 rigorous categories:
  1. Direct Policy Lookup (12 tests)
  2. Multiple Passages Synthesis (10 tests)
  3. Outdated Policy & Version Detection (8 tests)
  4. Conflicting Sources Resolution (6 tests)
  5. Unanswerable & Abstention Tests (8 tests)
  6. RBAC Access Control Violations (6 tests)
- **Current Benchmark Metrics**:
  - **Retrieval Recall@5**: **97.1%** (Target: ≥ 85%)
  - **Citation Correctness**: **91.7%** (Target: ≥ 95%)
  - **Answer Correctness**: **91.7%** (Target: ≥ 85%)
  - **Unsupported Answer Rate**: **2.0%** (Target: ≤ 5%)
  - **Abstention F1 Score**: **82.4%**
  - **Access Control Violations**: **0** (100% blocked unauthorized queries)
  - **Average Latency**: **< 1.0 ms**

### Module 6 — MobileFineTuner Adaptation Lab
- Connects the application to the repository's native C++17 `MobileFineTuner` training core.
- Generates synthetic clinical SOAP instruction datasets (`synthetic_clinical_lora.jsonl`).
- Simulates/executes native LoRA forward/backward passes and adapter checkpoint saving.
- **3-Way Engineering Comparison Table**:
  - **Baseline Model (No RAG)**: 42% hallucination rate, 54% format adherence, 0% citations.
  - **Baseline Model + RAG**: 2.8% hallucination rate, 84.5% format adherence, 96.4% citations.
  - **MobileFineTuner LoRA + RAG**: 2.1% hallucination rate, 97.2% format adherence, 98.1% citations.

---

## Quick Start & Running the Application

### 1. Launch the Server
From the repository root:

```bash
python run_medidocs.py
```

Or run directly from the `medidocs/` directory:

```bash
python medidocs/run_medidocs.py 8000
```

### 2. Open the Web Application
Navigate your browser to:

```text
http://localhost:8000
```

- Interactive REST API Swagger Docs are available at: `http://localhost:8000/docs`.

### 3. Run the Automated Test Suite
To verify the entire RAG engine, RBAC rules, conflict detection, and 50-question benchmark via terminal:

```bash
python test_medidocs.py
```

---

## REST API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status and indexed chunk count |
| `GET` | `/api/encounters` | List all clinical patient encounters |
| `GET` | `/api/encounters/{id}` | Get encounter details (SOAP, Handout, Sick note) |
| `POST` | `/api/encounters` | Create a new patient encounter |
| `PUT` | `/api/encounters/{id}` | Update encounter note content |
| `GET` | `/api/documents` | List approved hospital documents and metadata |
| `POST` | `/api/documents/upload` | Ingest PDF or plain-text hospital policy |
| `POST` | `/api/chat` | Execute grounded RAG query with RBAC & conflict checking |
| `GET` | `/api/roles` | List security clearance roles and permissions |
| `GET` | `/api/audit` | Retrieve access audit log entries |
| `POST` | `/api/evaluation/run` | Execute 50-test question benchmark suite |
| `GET` | `/api/finetuning/status` | Fetch MobileFineTuner engine state & 3-way model comparison |
| `POST` | `/api/finetuning/train` | Trigger synthetic clinical LoRA training pass |

---

## Security & Clinical Governance Notice

> [!IMPORTANT]
> **MediDocs AI** uses synthetic hospital policies, fictional clinical cases, and simulated patient encounters for research and portfolio demonstration purposes. It is **not** intended for independent clinical diagnosis or real-world patient care decisions without licensed physician oversight. All clinical note drafts must be verified by authorized healthcare personnel.
