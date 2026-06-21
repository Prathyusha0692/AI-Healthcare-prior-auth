"""
Convenience launcher — starts both FastAPI and Streamlit in parallel.

Usage:
    python run.py            # start both servers
    python run.py --api      # API only
    python run.py --ui       # Streamlit only
    python run.py --test     # run a quick smoke test (no real OpenAI key needed)
"""
import argparse
import subprocess
import sys
import threading
import time


def start_api():
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"],
        check=False,
    )


def start_ui():
    time.sleep(2)  # let API spin up first
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py", "--server.port", "8501"],
        check=False,
    )


def run_smoke_test():
    """Quick import + stub-LLM test — no API key required."""
    print("Running smoke test…")
    from app.config import get_settings
    from app.pii.scrubber import scrub_phi
    from app.rag.retriever import chunk_text
    from app.agents.graph import run_authorization

    # Test PII scrubber
    result = scrub_phi("Patient John Doe, SSN: 123-45-6789, email: john@example.com")
    assert result.phi_detected or result.scrubbing_engine == "regex-fallback"
    print(f"  ✅ PII scrubber OK ({result.scrubbing_engine}): {result.summary}")

    # Test chunker
    chunks = chunk_text("This is a test document. " * 50, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    print(f"  ✅ Text chunker OK: {len(chunks)} chunks")

    # Test full pipeline (stub LLM, empty ChromaDB)
    state = run_authorization(
        clinical_note="Patient John Smith (SSN: 111-22-3333) has knee pain. Requesting TKA.",
        procedure_code="27447",
        diagnosis_code="M17.11",
        insurance_plan="Test Plan",
    )
    assert state.get("authorization_decision") in ("approved", "denied", "pending_info")
    assert state.get("request_id")
    print(f"  ✅ Full pipeline OK: decision={state['authorization_decision']} "
          f"confidence={state.get('confidence_score', 0):.0%}")

    print("\n✅ Smoke test passed! Project is ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        run_smoke_test()
    elif args.api:
        start_api()
    elif args.ui:
        start_ui()
    else:
        print("Starting FastAPI (port 8000) + Streamlit (port 8501)…")
        print("  API docs: http://localhost:8000/docs")
        print("  UI:       http://localhost:8501")
        t1 = threading.Thread(target=start_api, daemon=True)
        t2 = threading.Thread(target=start_ui, daemon=True)
        t1.start()
        t2.start()
        try:
            t1.join()
        except KeyboardInterrupt:
            print("\nShutting down.")
