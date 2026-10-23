from typing import List, Dict
from .models import Encounter

INITIAL_ENCOUNTERS: List[Encounter] = [
    Encounter(
        id="enc-01",
        title="Shortness of Breath",
        time="1:31 PM",
        category="TODAY",
        patient_name="Jane Doe",
        patient_id="PT-90421",
        duration="5 mins, 10 secs | 1:31 - 1:36 PM",
        subjective=(
            "Subjective:\n"
            "- 2-day history of shortness of breath and wheezing, worse in spring\n"
            "- No fever, no sick contacts, denies chest pain"
        ),
        objective=(
            "Objective:\n"
            "- T: 36.4°C, BP: 120/84 mmHg, HR: 88 bpm, RR: 24/min, O2 sat: 98% RA\n"
            "- Respiratory: Mild expiratory wheezes bilaterally, no crackles\n"
            "- Cardiovascular: Normal S1/S2, no murmurs, no peripheral edema"
        ),
        assessment=(
            "Assessment:\n"
            "- Asthma exacerbation"
        ),
        plan=(
            "Plan:\n"
            "- Prescribed Symbicort BID with spacer\n"
            "- Referral for pulmonary function tests\n"
            "- Provide sick note for 2 days\n"
            "- Patient instructions provided\n"
            "- Follow-up in 5 days or sooner if symptoms worsen"
        ),
        patient_handout=(
            "ASTHMA ACTION PLAN & INSTRUCTIONS FOR JANE DOE\n"
            "Diagnosis: Mild-to-Moderate Asthma Flare (Triggered by Spring Seasonal Pollen)\n\n"
            "Medication Directions:\n"
            "• Symbicort (Budesonide/Formoterol 160/4.5 mcg): Take 2 inhalations twice daily using your spacer.\n"
            "• Rinse your mouth with water after each use.\n\n"
            "Warning Signs requiring IMMEDIATE Emergency Department Care:\n"
            "1. Severe difficulty breathing or inability to speak in full sentences.\n"
            "2. Lips or fingernails turning blue or pale.\n"
            "3. No relief within 15 minutes of rescue inhaler use.\n\n"
            "Scheduled Follow-up: 5 days at Ambulatory Pulmonary Clinic (Ext. 3120)."
        ),
        sick_note=(
            "MEDICAL EXCUSE CERTIFICATE / SICK NOTE\n"
            "Date: Today\n"
            "Patient: Jane Doe (DOB: 1991-04-12)\n\n"
            "To Whom It May Concern,\n"
            "Jane Doe was examined at St. Jude Memorial Hospital Ambulatory Clinic today. "
            "Due to an acute respiratory exacerbation, she is medically unfit for work duties for two (2) calendar days.\n\n"
            "Permitted Return-to-Work Date: In 48 hours.\n"
            "Authorized Clinician: Attending Physician, Pulmonary & Ambulatory Medicine"
        )
    ),
    Encounter(
        id="enc-02",
        title="Lab Review",
        time="1:20 PM",
        category="TODAY",
        patient_name="Robert Smith",
        patient_id="PT-88129",
        duration="4 mins, 15 secs | 1:20 - 1:24 PM",
        subjective="Subjective:\n- Routine follow-up on metabolic panel and HbA1c\n- Reports compliant medication adherence, no hypoglycemic episodes",
        objective="Objective:\n- BP: 128/80 mmHg, HR: 74 bpm\n- Fasting glucose: 112 mg/dL, HbA1c: 6.4% (improved from 7.1%)",
        assessment="Assessment:\n- Type 2 Diabetes Mellitus - well controlled on Metformin",
        plan="Plan:\n- Continue Metformin 500mg BID\n- Re-check HbA1c in 6 months\n- Ophthalmology retinal exam referral ordered",
        patient_handout="Patient Handout: Continue low glycemic index diet and daily exercise routine.",
        sick_note="No sick note required."
    ),
    Encounter(
        id="enc-03",
        title="Fatigue",
        time="1:05 PM",
        category="TODAY",
        patient_name="Emily Davis",
        patient_id="PT-77102",
        duration="8 mins, 20 secs | 1:05 - 1:13 PM",
        subjective="Subjective:\n- 3-week worsening daytime somnolence and cold intolerance\n- Unexplained 4kg weight gain over past 2 months",
        objective="Objective:\n- T: 36.1°C, HR: 58 bpm (bradycardia), BP: 118/76 mmHg\n- Thyroid palpation: Mild symmetrical fullness without nodules",
        assessment="Assessment:\n- Suspected Hypothyroidism",
        plan="Plan:\n- Order comprehensive TSH, Free T4, Anti-TPO antibodies\n- Complete blood count for anemia\n- Review results at 1-week follow up",
        patient_handout="Patient Handout: Fasting blood draw scheduled tomorrow at Central Diagnostic Laboratory.",
        sick_note="Sick note: Excused for morning diagnostic laboratory tests."
    ),
    Encounter(
        id="enc-04",
        title="Chest Pain",
        time="10:46 PM",
        category="TUE MAR 31",
        patient_name="Arthur Pendelton",
        patient_id="PT-65201",
        duration="12 mins, 40 secs | 10:46 - 10:59 PM",
        subjective="Subjective:\n- Sudden retrosternal heaviness radiating to left shoulder while climbing stairs",
        objective="Objective:\n- BP: 154/92 mmHg, HR: 96 bpm, SpO2: 97%\n- ECG: 1mm ST-depression in leads V4-V6\n- Initial Troponin-I: 0.08 ng/mL (borderline)",
        assessment="Assessment:\n- Acute Coronary Syndrome (NSTEMI vs Unstable Angina)",
        plan="Plan:\n- Chewable Aspirin 325mg given, Sublingual Nitroglycerin x 1\n- STAT Cardiology consult, Cardiology telemetry bed transfer\n- Repeat Troponin series at 2 hours and 6 hours",
        patient_handout="Inpatient admission protocol initiated.",
        sick_note="Patient admitted to hospital cardiac monitoring service."
    ),
    Encounter(
        id="enc-05",
        title="Bloodwork Review",
        time="12:25 PM",
        category="MON MAR 30",
        patient_name="Grace Chen",
        patient_id="PT-51203",
        duration="6 mins, 10 secs | 12:25 - 12:31 PM",
        subjective="Subjective:\n- Follow-up lipid panel results following 3 months on Atorvastatin",
        objective="Objective:\n- Total cholesterol: 168 mg/dL, LDL: 88 mg/dL (target < 100), ALT/AST normal",
        assessment="Assessment:\n- Hyperlipidemia - responsive to statin therapy",
        plan="Plan:\n- Maintain Atorvastatin 20mg daily\n- Repeat hepatic enzymes annually",
        patient_handout="Handout: Heart-healthy dietary guidance.",
        sick_note="No sick note required."
    ),
    Encounter(
        id="enc-06",
        title="New Patient Intake",
        time="11:59 AM",
        category="MON MAR 30",
        patient_name="Marcus Vance",
        patient_id="PT-49210",
        duration="15 mins, 00 secs | 11:59 - 12:14 PM",
        subjective="Subjective:\n- New resident establishing primary care. No active complaints",
        objective="Objective:\n- Physical exam unremarkable. Vitals within normal limits",
        assessment="Assessment:\n- Routine preventative health maintenance",
        plan="Plan:\n- Age-appropriate cancer screening scheduled (Colonoscopy referral)\n- Tdap booster administered today",
        patient_handout="Handout: Hospital patient portal activation code provided.",
        sick_note="No sick note required."
    ),
    Encounter(
        id="enc-07",
        title="Physiotherapy",
        time="9:40 PM",
        category="SUN MAR 29",
        patient_name="David Miller",
        patient_id="PT-38411",
        duration="10 mins, 30 secs | 9:40 - 9:50 PM",
        subjective="Subjective:\n- Post-ACL reconstruction rehabilitation week 6",
        objective="Objective:\n- Knee flexion: 110 degrees, extension: 0 degrees. Minimal joint effusion",
        assessment="Assessment:\n- Right knee post-ACL reconstruction - expected recovery trajectory",
        plan="Plan:\n- Advance closed kinetic chain quad strengthening\n- Ice 20 min post-exercise",
        patient_handout="Handout: Home exercise guide stage 2.",
        sick_note="No sick note required."
    ),
    Encounter(
        id="enc-08",
        title="Allergic Rhinitis",
        time="9:24 PM",
        category="SUN MAR 29",
        patient_name="Sarah Jenkins",
        patient_id="PT-32104",
        duration="5 mins, 40 secs | 9:24 - 9:30 PM",
        subjective="Subjective:\n- Nasal congestion, clear rhinorrhea, itchy conjunctiva",
        objective="Objective:\n- Boggy pale nasal turbinates, no sinus tenderness",
        assessment="Assessment:\n- Seasonal allergic rhinitis",
        plan="Plan:\n- Fluticasone propionate nasal spray 1 spray each nostril daily\n- Loratadine 10mg daily as needed",
        patient_handout="Handout: Allergen avoidance strategies.",
        sick_note="No sick note required."
    ),
    Encounter(
        id="enc-09",
        title="Chronic Eczema",
        time="9:22 PM",
        category="SUN MAR 29",
        patient_name="Liam Wilson",
        patient_id="PT-29188",
        duration="7 mins, 10 secs | 9:22 - 9:29 PM",
        subjective="Subjective:\n- Pruritic erythematous plaques on antecubital and popliteal fossae",
        objective="Objective:\n- Lichenification and xerosis over flexural surfaces, no impetiginization",
        assessment="Assessment:\n- Atopic dermatitis flare",
        plan="Plan:\n- Triamcinolone 0.1% ointment BID for 7 days\n- Ceramide-based emollient immediately after bathing",
        patient_handout="Handout: Skin barrier care and bathing guidelines.",
        sick_note="No sick note required."
    ),
    Encounter(
        id="enc-10",
        title="Dizziness",
        time="10:57 AM",
        category="WED MAR 25",
        patient_name="Eleanor Rigby",
        patient_id="PT-19402",
        duration="11 mins, 20 secs | 10:57 - 11:08 AM",
        subjective="Subjective:\n- Episodic spinning vertigo when turning head in bed, lasts ~30 seconds",
        objective="Objective:\n- Dix-Hallpike test positive right side with torsional upbeating nystagmus",
        assessment="Assessment:\n- Benign Paroxysmal Positional Vertigo (BPPV) right posterior canal",
        plan="Plan:\n- Epley maneuver performed in clinic with resolution of nystagmus\n- Brandt-Daroff home exercises if recurrence",
        patient_handout="Handout: Post-Epley instructions and fall prevention.",
        sick_note="Sick note: 24-hour driving avoidance."
    ),
    Encounter(
        id="enc-11",
        title="Respiratory Infection",
        time="10:56 AM",
        category="WED MAR 25",
        patient_name="Oscar Wilde",
        patient_id="PT-18299",
        duration="9 mins, 15 secs | 10:56 - 11:05 AM",
        subjective="Subjective:\n- 4-day productive cough with yellowish phlegm, low-grade fever 37.8°C",
        objective="Objective:\n- Lung auscultation: Scattered rhonchi clearing with cough, no focal consolidation",
        assessment="Assessment:\n- Acute Bronchitis (viral etiology suspected)",
        plan="Plan:\n- Supportive therapy: Hydration, honey for cough, Dextromethorphan PRN\n- Antibiotics not indicated at this time\n- Return if fever > 38.5°C or dyspnea develops",
        patient_handout="Handout: Viral cough recovery timelines.",
        sick_note="Sick note: 2 days rest."
    )
]
