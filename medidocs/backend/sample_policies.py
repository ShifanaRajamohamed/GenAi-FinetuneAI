from typing import List, Dict
from .models import DocumentMetadata, DocumentChunk

SAMPLE_POLICIES: List[Dict] = [
    {
        "meta": DocumentMetadata(
            id="doc_pharm_045_v2",
            code="POL-PHARM-045",
            title="Medication Storage and Cold-Chain Temperature Excursion Reporting",
            department="Pharmacy & Nursing Operations",
            version="v2.1",
            effective_date="2025-11-01",
            status="active",
            access_level="CLINICAL",
            file_type="pdf",
            pages_count=3,
            description="Mandatory reporting workflow, immediate quarantine protocol, and pharmacist assessment criteria for refrigerated vaccine and biologic storage excursions."
        ),
        "chunks": [
            DocumentChunk(
                chunk_id="chk_pharm_045_2_1",
                doc_id="doc_pharm_045_v2",
                doc_code="POL-PHARM-045",
                doc_title="Medication Storage and Cold-Chain Temperature Excursion Reporting",
                version="v2.1",
                effective_date="2025-11-01",
                access_level="CLINICAL",
                page_number=1,
                section="Section 1.0: Target Temperature Ranges & Detection",
                content=(
                    "All standard clinical medication refrigerators must maintain an internal continuous temperature between 2.0°C and 8.0°C (36°F to 46°F). "
                    "A temperature excursion is defined as any continuous or cumulative recorded reading outside 2.0°C to 8.0°C exceeding 15 minutes. "
                    "Automated wireless NIST-calibrated sensors will trigger Tier-1 audible and SMS alerts to the charge nurse and hospital pharmacist on duty."
                )
            ),
            DocumentChunk(
                chunk_id="chk_pharm_045_2_2",
                doc_id="doc_pharm_045_v2",
                doc_code="POL-PHARM-045",
                doc_title="Medication Storage and Cold-Chain Temperature Excursion Reporting",
                version="v2.1",
                effective_date="2025-11-01",
                access_level="CLINICAL",
                page_number=2,
                section="Section 2.1: Immediate Reporting and Containment Workflow",
                content=(
                    "Upon noticing or receiving an excursion alert, the nursing supervisor must execute three immediate steps within 30 minutes:\n"
                    "1. Physical Quarantine: Apply neon red 'QUARANTINE - DO NOT DISPENSE' tamper-evident tape across refrigerator handles.\n"
                    "2. Form Submission: Complete electronic Event Incident Form e-RX-901 in the Hospital Intranet portal within 30 minutes of discovery.\n"
                    "3. Pharmacy Transfer: Contact central pharmacy dispatch at ext. 4402 to transport high-cost biologics (insulin, erythropoietin, vaccines) to the auxiliary backup cold room if temperature exceeds 10.0°C for >45 minutes."
                )
            ),
            DocumentChunk(
                chunk_id="chk_pharm_045_2_3",
                doc_id="doc_pharm_045_v2",
                doc_code="POL-PHARM-045",
                doc_title="Medication Storage and Cold-Chain Temperature Excursion Reporting",
                version="v2.1",
                effective_date="2025-11-01",
                access_level="CLINICAL",
                page_number=3,
                section="Section 4.3: Stability Determination & Release Criteria",
                content=(
                    "Under no circumstances may unit staff discard or re-administer quarantined medications without written approval from the Chief Clinical Pharmacist. "
                    "Manufacturer stability dossiers must be consulted. If cumulative temperature was between 8.1°C and 15.0°C for under 2 hours, select monoclonal antibodies may retain viability, but requires dual pharmacist sign-off."
                )
            )
        ]
    },
    {
        "meta": DocumentMetadata(
            id="doc_pharm_045_v1",
            code="POL-PHARM-045-OLD",
            title="Medication Storage & Excursions Protocol (SUPERSEDED 2021)",
            department="Pharmacy",
            version="v1.0",
            effective_date="2021-03-01",
            status="superseded",
            access_level="CLINICAL",
            file_type="pdf",
            pages_count=2,
            description="Outdated legacy guideline allowing manual thermometer logging and up to 2 hours reporting window. Superseded by v2.1 in Nov 2025."
        ),
        "chunks": [
            DocumentChunk(
                chunk_id="chk_pharm_045_1_1",
                doc_id="doc_pharm_045_v1",
                doc_code="POL-PHARM-045-OLD",
                doc_title="Medication Storage & Excursions Protocol (SUPERSEDED 2021)",
                version="v1.0",
                effective_date="2021-03-01",
                access_level="CLINICAL",
                page_number=1,
                section="Section 2: Legacy Excursion Notification",
                content=(
                    "[SUPERSEDED POLICY - FOR AUDIT PURPOSES ONLY]: Staff are requested to check the manual clipboard log twice daily. "
                    "If temperature is above 12°C, staff should notify the pharmacy within 2 hours of discovery on the shift paper logbook. "
                    "Medications may remain in the unit refrigerator unless ambient temperature exceeds 20°C."
                )
            )
        ]
    },
    {
        "meta": DocumentMetadata(
            id="doc_clin_012_v3",
            code="SOP-CLIN-012",
            title="Adult Sepsis Screening & Early Alert Bundle Protocol",
            department="Emergency & Internal Medicine",
            version="v3.2",
            effective_date="2026-01-15",
            status="active",
            access_level="CLINICAL",
            file_type="pdf",
            pages_count=4,
            description="Standardized bedside qSOFA scoring, automated EHR alert criteria, and mandatory 1-Hour Sepsis Management Bundle."
        ),
        "chunks": [
            DocumentChunk(
                chunk_id="chk_clin_012_3_1",
                doc_id="doc_clin_012_v3",
                doc_code="SOP-CLIN-012",
                doc_title="Adult Sepsis Screening & Early Alert Bundle Protocol",
                version="v3.2",
                effective_date="2026-01-15",
                access_level="CLINICAL",
                page_number=1,
                section="Section 1.2: Bedside qSOFA Screening Criteria",
                content=(
                    "Bedside screening for sepsis must be performed on all adult admissions presenting with suspected or confirmed infection. "
                    "The quick Sequential Organ Failure Assessment (qSOFA) criteria assigns 1 point each for:\n"
                    "• Respiratory Rate ≥ 22 breaths per minute\n"
                    "• Altered Mentation (Glasgow Coma Scale < 15)\n"
                    "• Systolic Blood Pressure ≤ 100 mmHg\n"
                    "A qSOFA score ≥ 2 indicates high risk of poor outcome and requires immediate Code Sepsis team activation."
                )
            ),
            DocumentChunk(
                chunk_id="chk_clin_012_3_2",
                doc_id="doc_clin_012_v3",
                doc_code="SOP-CLIN-012",
                doc_title="Adult Sepsis Screening & Early Alert Bundle Protocol",
                version="v3.2",
                effective_date="2026-01-15",
                access_level="CLINICAL",
                page_number=2,
                section="Section 2.0: The 1-Hour Bundle Requirements",
                content=(
                    "Upon identification of sepsis, all elements of the 1-Hour Sepsis Bundle must be initiated within 60 minutes:\n"
                    "1. Measure serum lactate level; remeasure within 2 to 4 hours if initial lactate > 2.0 mmol/L.\n"
                    "2. Obtain two sets of blood cultures prior to administration of antibiotics (1 aerobic, 1 anaerobic).\n"
                    "3. Administer broad-spectrum empiric IV antimicrobials (e.g. Cefepime + Vancomycin per antibiogram).\n"
                    "4. Rapidly infuse 30 mL/kg crystalloid fluid for hypotension or lactate ≥ 4.0 mmol/L.\n"
                    "5. Apply vasopressors (Norepinephrine first choice) if patient is hypotensive during or after fluid resuscitation to maintain MAP ≥ 65 mmHg."
                )
            )
        ]
    },
    {
        "meta": DocumentMetadata(
            id="doc_form_022_v2",
            code="CLIN-FORM-022",
            title="Ambulatory Adult Asthma Exacerbation Assessment & Note Drafting",
            department="Ambulatory & Pulmonary Care",
            version="v2.0",
            effective_date="2025-06-10",
            status="active",
            access_level="CLINICAL",
            file_type="pdf",
            pages_count=2,
            description="Standardized SOAP documentation framework, severity stratification, medication dosing for Symbicort/Albuterol, and discharge criteria."
        ),
        "chunks": [
            DocumentChunk(
                chunk_id="chk_form_022_2_1",
                doc_id="doc_form_022_v2",
                doc_code="CLIN-FORM-022",
                doc_title="Ambulatory Adult Asthma Exacerbation Assessment & Note Drafting",
                version="v2.0",
                effective_date="2025-06-10",
                access_level="CLINICAL",
                page_number=1,
                section="Section 2.1: SOAP Documentation Standards",
                content=(
                    "Clinical documentation for adult acute wheezing/shortness of breath must follow the standardized SOAP structure:\n"
                    "• Subjective: Duration, trigger exposure (seasonal/spring allergen), presence of fever, sick contacts, chest pain rule-out.\n"
                    "• Objective: Complete vital signs (Temp, BP, HR, RR, SpO2 on room air), auscultation findings (bilateral expiratory wheezing vs crackles), cardiovascular exam.\n"
                    "• Assessment: Severity classification (Mild, Moderate, Severe exacerbation).\n"
                    "• Plan: Bronchodilator therapy, anti-inflammatory prescription, pulmonary function test (PFT) referral, patient educational handout, and specified follow-up window (typically 3 to 5 days)."
                )
            ),
            DocumentChunk(
                chunk_id="chk_form_022_2_2",
                doc_id="doc_form_022_v2",
                doc_code="CLIN-FORM-022",
                doc_title="Ambulatory Adult Asthma Exacerbation Assessment & Note Drafting",
                version="v2.0",
                effective_date="2025-06-10",
                access_level="CLINICAL",
                page_number=2,
                section="Section 3.4: Recommended First-Line Outpatient Pharmacotherapy",
                content=(
                    "For moderate exacerbations responsive to initial therapy with SpO2 > 95%:\n"
                    "1. Budesonide/formoterol (Symbicort) 160/4.5 mcg: 2 inhalations twice daily (BID) using spacer technique.\n"
                    "2. Sick note entitlement: Clinical staff may issue a work/school sick note for up to 2 calendar days.\n"
                    "3. Action Plan Handout: Provide written Asthma Action Plan detailing warning signs (inability to complete sentences, persistent RR > 28) requiring immediate Emergency Department return."
                )
            )
        ]
    },
    {
        "meta": DocumentMetadata(
            id="doc_icu_088_v4",
            code="SOP-ICU-088",
            title="Mechanical Ventilation Weaning and Extubation Protocol",
            department="Intensive Care Unit (ICU)",
            version="v4.0",
            effective_date="2026-02-01",
            status="active",
            access_level="RESTRICTED_ICU",
            file_type="pdf",
            pages_count=3,
            description="Restricted clinical protocol for Spontaneous Breathing Trials (SBT), Rapid Shallow Breathing Index (RSBI) calculation, and extubation readiness in ICU."
        ),
        "chunks": [
            DocumentChunk(
                chunk_id="chk_icu_088_4_1",
                doc_id="doc_icu_088_v4",
                doc_code="SOP-ICU-088",
                doc_title="Mechanical Ventilation Weaning and Extubation Protocol",
                version="v4.0",
                effective_date="2026-02-01",
                access_level="RESTRICTED_ICU",
                page_number=1,
                section="Section 1.1: Readiness Screening Criteria [RESTRICTED ACCESS]",
                content=(
                    "[CONFIDENTIAL ICU CLINICAL PROTOCOL - ATTENDING PHYSICIANS & ICU FELLOWS ONLY]\n"
                    "Daily weaning eligibility requires meeting all four baseline physiological thresholds:\n"
                    "1. Resolution or improvement of primary cause of acute respiratory failure.\n"
                    "2. PaO2/FiO2 ratio ≥ 200 with PEEP ≤ 8 cmH2O and FiO2 ≤ 0.40.\n"
                    "3. Hemodynamic stability: Minimal or zero inotrope/vasopressor support (Norepinephrine ≤ 0.05 mcg/kg/min).\n"
                    "4. Intact spontaneous respiratory drive and ability to initiate spontaneous breaths."
                )
            ),
            DocumentChunk(
                chunk_id="chk_icu_088_4_2",
                doc_id="doc_icu_088_v4",
                doc_code="SOP-ICU-088",
                doc_title="Mechanical Ventilation Weaning and Extubation Protocol",
                version="v4.0",
                effective_date="2026-02-01",
                access_level="RESTRICTED_ICU",
                page_number=2,
                section="Section 2.3: Spontaneous Breathing Trial (SBT) & RSBI Threshold",
                content=(
                    "Conduct a 30-minute T-piece or low-pressure support (PS 5 cmH2O, PEEP 5 cmH2O) trial. "
                    "At 30 minutes, measure the Rapid Shallow Breathing Index (RSBI = f / Vt in L). "
                    "Extubation readiness requires RSBI < 105 breaths/min/L, cuff leak test positive, and airway suction frequency less than every 2 hours."
                )
            )
        ]
    },
    {
        "meta": DocumentMetadata(
            id="doc_adm_101_v1",
            code="ADM-HR-101",
            title="Emergency Department Staff Fatigue and Shift Duration Policy",
            department="Hospital Administration & Human Resources",
            version="v1.5",
            effective_date="2024-08-01",
            status="active",
            access_level="ADMIN",
            file_type="pdf",
            pages_count=2,
            description="Administrative safety constraints governing clinical shift lengths, mandatory rest intervals, and overtime caps to prevent provider burnout."
        ),
        "chunks": [
            DocumentChunk(
                chunk_id="chk_adm_101_1_1",
                doc_id="doc_adm_101_v1",
                doc_code="ADM-HR-101",
                doc_title="Emergency Department Staff Fatigue and Shift Duration Policy",
                version="v1.5",
                effective_date="2024-08-01",
                access_level="ADMIN",
                page_number=1,
                section="Section 2.0: Shift Caps and Mandatory Rest Periods",
                content=(
                    "To ensure patient safety and minimize diagnostic fatigue errors, no medical doctor or registered nurse may work greater than 14 consecutive scheduled hours. "
                    "A mandatory minimum rest interval of 10 continuous hours is required between consecutive shifts. "
                    "Exceptions during declared Mass Casualty Incidents require electronic authorization by the Chief Medical Officer."
                )
            )
        ]
    },
    {
        "meta": DocumentMetadata(
            id="doc_infect_009_v1",
            code="POL-INFECT-009",
            title="Airborne Isolation Precautions and N95 Respirator Fit Protocol",
            department="Infection Prevention & Control",
            version="v1.8",
            effective_date="2025-02-14",
            status="active",
            access_level="GENERAL",
            file_type="pdf",
            pages_count=2,
            description="Facility-wide infection control guidelines for airborne pathogens (TB, Varicella, Measles), negative pressure rooms, and fit-tested PPE."
        ),
        "chunks": [
            DocumentChunk(
                chunk_id="chk_infect_009_1_1",
                doc_id="doc_infect_009_v1",
                doc_code="POL-INFECT-009",
                doc_title="Airborne Isolation Precautions and N95 Respirator Fit Protocol",
                version="v1.8",
                effective_date="2025-02-14",
                access_level="GENERAL",
                page_number=1,
                section="Section 1.1: Mandatory Airborne PPE Requirements",
                content=(
                    "Staff entering designated Airborne Infection Isolation Rooms (AIIR / Negative Pressure Room with ≥ 12 air changes/hour) must wear: "
                    "1. NIOSH-approved, fit-tested N95 respirator or powered air-purifying respirator (PAPR). "
                    "2. Full protective eye shield or goggles (regular prescription glasses do not qualify). "
                    "3. Clean non-sterile examination gloves and fluid-resistant isolation gown. "
                    "Respirators must be donned prior to entering the anteroom and doffed after room exit."
                )
            )
        ]
    },
    {
        "meta": DocumentMetadata(
            id="doc_code_004_v2",
            code="SOP-CODE-BLUE-004",
            title="In-Hospital Cardiac Arrest Resuscitation Documentation",
            department="Resuscitation Committee & Critical Care",
            version="v2.5",
            effective_date="2026-03-01",
            status="active",
            access_level="CLINICAL",
            file_type="pdf",
            pages_count=2,
            description="Documentation timelines, defibrillator log integration, and attending physician co-signature mandates following Code Blue events."
        ),
        "chunks": [
            DocumentChunk(
                chunk_id="chk_code_004_2_1",
                doc_id="doc_code_004_v2",
                doc_code="SOP-CODE-BLUE-004",
                doc_title="In-Hospital Cardiac Arrest Resuscitation Documentation",
                version="v2.5",
                effective_date="2026-03-01",
                access_level="CLINICAL",
                page_number=1,
                section="Section 3.1: Attending Physician Co-Signature and Timeline",
                content=(
                    "The primary resuscitation record (Form CPR-04) must be completed by the designated recorder immediately following cessation of chest compressions. "
                    "The Attending Physician leading the Code Blue must review the defibrillator rhythm strip, verify epinephrine/amiodarone dosages, and apply their verified electronic co-signature within 4 hours of event termination. "
                    "If the patient expired, family notification must be formally logged in the chart within 60 minutes."
                )
            )
        ]
    }
]
