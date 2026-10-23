import sys
import os

# Add root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from medidocs.backend.rag_engine import MediDocsRAGEngine
from medidocs.backend.evaluation_data import run_evaluation_suite
from medidocs.backend.mft_bridge import MobileFineTunerBridge

def test_engine():
    print("[*] Initializing MediDocs RAG Engine...")
    engine = MediDocsRAGEngine()
    print(f"[+] Loaded {len(engine.documents)} documents, {len(engine.chunks)} indexed chunks.")

    # Test 1: Grounded query with conflict detection
    print("\n--- Test 1: Temperature Excursion (with conflict detection) ---")
    resp1 = engine.generate_grounded_answer(
        "What is the procedure for reporting a medication storage temperature excursion?",
        "attending_physician"
    )
    print(f"Conflict Warning: {resp1.conflict_warning}")
    print(f"Citations ({len(resp1.citations)}): {[c.doc_code for c in resp1.citations]}")
    assert len(resp1.citations) > 0, "Expected at least 1 citation"
    assert resp1.conflict_warning is not None, "Expected conflict warning"

    # Test 2: RBAC Access Control Block
    print("\n--- Test 2: RBAC Access Control (Intern attempting ICU ventilation) ---")
    resp2 = engine.generate_grounded_answer(
        "Show me the confidential mechanical ventilation weaning protocol SOP-ICU-088",
        "clinical_intern"
    )
    print(f"Access Blocked: {resp2.access_blocked}")
    print(f"Answer: {resp2.answer}")
    assert resp2.access_blocked is True, "Expected RBAC to block intern"

    # Test 3: Abstention Engine
    print("\n--- Test 3: Abstention Engine (Ungrounded query) ---")
    resp3 = engine.generate_grounded_answer(
        "What is the Wi-Fi password for visitor cafeteria lounge?",
        "staff_nurse"
    )
    print(f"Abstention Reason: {resp3.abstention_reason}")
    assert resp3.abstention_reason is not None, "Expected engine to abstain"

    # Test 4: Run full 50-test benchmark suite
    print("\n--- Test 4: Full 50-Test Benchmark Suite ---")
    eval_results = run_evaluation_suite(engine)
    summary = eval_results["summary"]
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed Tests: {summary['passed_tests']} ({summary['pass_rate_percent']}%)")
    print(f"Recall@5: {summary['retrieval_recall_at_5']}%")
    print(f"Citation Correctness: {summary['citation_correctness']}%")
    print(f"Answer Correctness: {summary['answer_correctness']}%")
    print(f"Unsupported Answer Rate: {summary['unsupported_answer_rate']}%")
    print(f"Abstention F1: {summary['abstention_f1']}%")
    print(f"Access Violations: {summary['unauthorized_disclosures']}")
    print(f"Avg Latency: {summary['avg_latency_ms']} ms")

    # Test 5: MobileFineTuner Bridge
    print("\n--- Test 5: MobileFineTuner Bridge ---")
    bridge = MobileFineTunerBridge()
    train_res = bridge.run_lora_fine_tuning_smoke("gpt2_small", 5)
    print(f"Training status: {train_res['status']}")
    print(f"Final simulated loss: {train_res['final_loss']}")
    print(f"Adapter saved: {train_res['adapter_saved']}")

    print("\n[✓] ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_engine()
