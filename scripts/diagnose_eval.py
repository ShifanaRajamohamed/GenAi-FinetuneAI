import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from medidocs.backend.rag_engine import MediDocsRAGEngine
from medidocs.backend.evaluation_data import run_evaluation_suite

def main():
    engine = MediDocsRAGEngine()
    res = run_evaluation_suite(engine)
    failed = [t for t in res["test_results"] if not t["passed"]]
    print(f"Total Failed: {len(failed)}")
    for f in failed:
        print(f"[{f['id']}] {f['category']}: Query: '{f['query']}'")
        print(f"      Notes: {f['notes']}\n")

if __name__ == "__main__":
    main()
