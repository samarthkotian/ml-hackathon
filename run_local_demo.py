"""
One-command single-laptop launcher for PathSnap (Team SI No 107).
Starts BOTH:
1. The FastAPI Model & Evaluation Server on http://127.0.0.1:8000
2. The Streamlit Evaluator UI Dashboard on http://localhost:8501

Usage:
    python run_local_demo.py
"""
import os
import subprocess
import sys
import time


def main() -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    print("=" * 80)
    print(" STARTING PATHSNAP FULL STACK ON ONE LAPTOP (API + UI)")
    print("=" * 80)
    print(" -> [1/2] Starting FastAPI Server at http://127.0.0.1:8000 ...")

    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=project_root,
    )

    time.sleep(2.5)
    print(" -> [2/2] Starting Streamlit Evaluator UI at http://localhost:8501 ...")
    print("          (Press Ctrl+C to stop both servers)\n")

    try:
        ui_proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "ui/app.py", "--server.port", "8501"],
            cwd=project_root,
        )
        ui_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping PathSnap servers...")
    finally:
        api_proc.terminate()


if __name__ == "__main__":
    main()
