import os
import sys
import uvicorn

def main():
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    print("=" * 75)
    print("  MediDocs AI — Hospital Document Intelligence & Clinical Assistant")
    print("  Powered by Grounded RAG + MobileFineTuner Adaptation Engine")
    print("=" * 75)
    print(f"[*] Starting FastAPI application server on http://localhost:{port}")
    print(f"[*] Access the Web Application at: http://localhost:{port}")
    print(f"[*] REST API Documentation at:     http://localhost:{port}/docs")
    print("=" * 75)

    uvicorn.run("medidocs.backend.main:app", host="127.0.0.1", port=port, reload=False)

if __name__ == "__main__":
    main()
