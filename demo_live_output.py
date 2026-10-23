import sys
import os
import json
import urllib.request
import urllib.error

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_URL = "http://127.0.0.1:8000"

def get(path):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def print_separator(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def main():
    print_separator("1. MEDIDOCS AI SERVER HEALTH & LIVE STATUS")
    health = get("/api/health")
    print(f"Status:          {health['status']}")
    print(f"Service:         {health['service']}")
    print(f"Docs Indexed:    {health['docs_indexed']} approved hospital documents")
    print(f"Chunks Indexed:  {health['chunks_indexed']} permission-tagged passage chunks")
    print(f"Timestamp:       {health['time']}")

    print_separator("2. APPROVED HOSPITAL POLICIES & CLEARANCE LEVELS")
    docs = get("/api/documents")
    for d in docs:
        status_tag = f"[{d['status'].upper()}]"
        print(f"• {d['code']} ({d['version']}, {d['effective_date']}) - {status_tag} {d['title']}")
        print(f"  Department: {d['department']} | Clearance Required: {d['access_level']}")

    print_separator("3. CLINICAL ENCOUNTER: SHORTNESS OF BREATH (VERO PRO REFERENCE)")
    encounter = get("/api/encounters/enc-01")
    print(f"Patient:    {encounter['patient_name']} ({encounter['patient_id']})")
    print(f"Encounter:  {encounter['title']} [{encounter['time']}] - Duration: {encounter['duration']}")
    print("\n[SOAP Clinical Note Content]:")
    print(encounter['subjective'])
    print(encounter['objective'])
    print(encounter['assessment'])
    print(encounter['plan'])

    print_separator("4. RAG QUERY WITH CONFLICT & VERSION DETECTION")
    query_conflict = "What is the procedure for reporting a medication storage temperature excursion?"
    print(f"User Query: \"{query_conflict}\"")
    print("User Role:  Attending Physician")
    resp_conflict = post("/api/chat", {"query": query_conflict, "user_role": "attending_physician"})
    
    if resp_conflict.get("conflict_warning"):
        print(f"\n[ALERT BOX]:\n{resp_conflict['conflict_warning']}")
    
    print(f"\n[GROUNDED ANSWER]:\n{resp_conflict['answer']}")
    print("\n[VERIFIED CITATIONS]:")
    for c in resp_conflict['citations']:
        print(f"  • {c['doc_code']} ({c['version']}, p.{c['page_number']}) [{c['section']}] - Match: {int(c['confidence']*100)}%")
        print(f"    Excerpt: \"{c['passage_excerpt'][:120]}...\"")
    print(f"Latency: {resp_conflict['latency_ms']} ms | Mode: {resp_conflict['model_mode']}")

    print_separator("5. BACKEND-ENFORCED RBAC SECURITY ACCESS CONTROL")
    query_rbac = "Show me the confidential mechanical ventilation weaning protocol SOP-ICU-088"
    print(f"User Query: \"{query_rbac}\"")
    print("User Role:  Clinical Intern (General Clearance Only)")
    resp_rbac = post("/api/chat", {"query": query_rbac, "user_role": "clinical_intern"})
    print(f"Access Blocked:    {resp_rbac['access_blocked']}")
    print(f"Enforcement Mode:  {resp_rbac['model_mode']}")
    print(f"Security Response: {resp_rbac['answer']}")

    print("\n--- Now Running Same Query as Attending Physician (ICU Clearance) ---")
    resp_attending = post("/api/chat", {"query": query_rbac, "user_role": "attending_physician"})
    print(f"Access Blocked:    {resp_attending['access_blocked']}")
    print(f"Citations Found:   {[c['doc_code'] for c in resp_attending['citations']]}")
    print(f"Grounded Response: {resp_attending['answer'][:200]}...")

    print_separator("6. SAFE ABSTENTION ENGINE (UNANSWERABLE QUERY)")
    query_abstain = "What is the Wi-Fi password for the patient visitor cafeteria lounge?"
    print(f"User Query: \"{query_abstain}\"")
    resp_abstain = post("/api/chat", {"query": query_abstain, "user_role": "staff_nurse"})
    print(f"Model Mode:        {resp_abstain['model_mode']}")
    print(f"Abstention Reason: {resp_abstain['abstention_reason']}")
    print(f"System Response:   {resp_abstain['answer']}")

    print_separator("7. 50-TEST QUESTION EVALUATION BENCHMARK SUITE")
    print("[*] Executing benchmark suite across all 6 categories...")
    eval_suite = post("/api/evaluation/run", {})
    summary = eval_suite['summary']
    print("\n" + "-" * 45)
    print("  BENCHMARK METRICS RESULTS SUMMARY")
    print("-" * 45)
    print(f"Total Test Cases:               {summary['total_tests']}")
    print(f"Tests Passed:                   {summary['passed_tests']} ({summary['pass_rate_percent']}%)")
    print(f"Retrieval Recall@5:             {summary['retrieval_recall_at_5']}%  (Target: ≥ 85%)")
    print(f"Citation Correctness:           {summary['citation_correctness']}%  (Target: ≥ 95%)")
    print(f"Answer Correctness:             {summary['answer_correctness']}%  (Target: ≥ 85%)")
    print(f"Unsupported Answer Rate:        {summary['unsupported_answer_rate']}%   (Target: ≤ 5%)")
    print(f"Abstention F1 Score:            {summary['abstention_f1']}%")
    print(f"Access Control Violations:      {summary['unauthorized_disclosures']}      (Target: 0 disclosures)")
    print(f"Average System Latency:         {summary['avg_latency_ms']} ms")
    print("-" * 45)

    print("\nCategory Breakdown:")
    for cat, stats in eval_suite['category_breakdown'].items():
        print(f"  • {cat:35s}: {stats['passed']}/{stats['total']} passed ({int(stats['passed']/stats['total']*100)}%)")

    print_separator("8. MOBILEFINETUNER C++ ENGINE & 3-WAY COMPARISON")
    mft_status = get("/api/finetuning/status")
    print(f"Adaptation Engine: {mft_status['engine']}")
    print("\n--- Running 5-Step LoRA Adaptation Smoke on C++ Core ---")
    train_res = post("/api/finetuning/train?model_key=gpt2_small&steps=5", {})
    print(f"Training Status:   {train_res['status']}")
    print(f"Model Architecture: {train_res['model_family']}")
    print(f"Steps Executed:    {train_res['steps_completed']}")
    print(f"Loss Trajectory:   {' -> '.join(str(l) for l in train_res['training_loss'])}")
    print(f"Adapter Saved:     {train_res['adapter_saved']}")

    print("\n--- 3-Way Engineering Comparison: Baseline vs RAG vs LoRA ---")
    for comp in mft_status['comparison']['evaluations']:
        print(f"\n[{comp['mode']}]")
        print(f"  Hallucination: {comp['hallucination_rate']} | Format Adherence: {comp['format_adherence']} | Citation Rate: {comp['citation_rate']}")
        print(f"  Response: \"{comp['sample_response'][:110]}...\"")
        print(f"  Engineering Verdict: {comp['pros_cons']}")

    print_separator("9. SECURITY AUDIT LOG VERIFICATION")
    audit_logs = get("/api/audit?limit=5")
    print(f"Recent Audit Entries ({len(audit_logs)} shown):")
    for log in audit_logs:
        status_symbol = "✓" if log['status'] == "ALLOWED" else ("🔒" if log['status'] == "BLOCKED" else "⚠")
        print(f"  {log['timestamp']} | {log['user_role']:20s} | {status_symbol} {log['status']:8s} | Latency: {log['latency_ms']:5.1f}ms | \"{log['query'][:40]}...\"")

    print_separator("ALL DEMONSTRATION TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
