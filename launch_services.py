"""
launch_services.py
Starts both Surya and Qwen OCR services in their respective venvs,
waits until both are healthy, then hands control back to the caller.

Usage:
    python launch_services.py                  # just start services
    python launch_services.py --run-batch "E:\\Lumina\\ocr\\docs\\"
"""

import subprocess
import sys
import time
import argparse
import signal
import os
import requests
from pathlib import Path

# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #
BASE_DIR  = Path(__file__).parent.resolve()

SURYA_VENV    = BASE_DIR / "venv-surya"
QWEN_VENV     = BASE_DIR / "venv-qwen"

SURYA_SERVICE = "services.surya_service:app"
QWEN_SERVICE  = "services.qwen_service:app"

SURYA_PORT = 8001
QWEN_PORT  = 8002

HEALTH_TIMEOUT  = 120   # seconds to wait for each service to come up
HEALTH_INTERVAL = 2     # seconds between health checks

# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def venv_python(venv: Path) -> str:
    """Return path to the venv's Python executable."""
    win = venv / "Scripts" / "python.exe"
    unix = venv / "bin" / "python"
    if win.exists():
        return str(win)
    if unix.exists():
        return str(unix)
    raise FileNotFoundError(f"Cannot find Python in venv: {venv}")


def venv_uvicorn(venv: Path) -> str:
    """Return path to the venv's uvicorn executable."""
    win = venv / "Scripts" / "uvicorn.exe"
    unix = venv / "bin" / "uvicorn"
    if win.exists():
        return str(win)
    if unix.exists():
        return str(unix)
    # Fall back: run as python -m uvicorn
    return None


def start_service(
    name: str,
    venv: Path,
    app: str,
    port: int,
) -> subprocess.Popen:
    uvicorn_bin = venv_uvicorn(venv)

    if uvicorn_bin:
        cmd = [uvicorn_bin, app, "--port", str(port), "--app-dir", str(BASE_DIR)]
    else:
        cmd = [venv_python(venv), "-m", "uvicorn", app,
               "--port", str(port), "--app-dir", str(BASE_DIR)]

    log_file = open(BASE_DIR / f"{name.lower()}_service.log", "w")

    print(f"[Launcher] Starting {name} service on port {port}...")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        cwd=str(BASE_DIR),
    )
    return proc


def wait_healthy(name: str, port: int, timeout: int) -> bool:
    url = f"http://localhost:{port}/docs"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                print(f"[Launcher] {name} is healthy ✓")
                return True
        except Exception:
            pass
        print(f"[Launcher] Waiting for {name}...", end="\r")
        time.sleep(HEALTH_INTERVAL)
    print(f"\n[Launcher] ERROR: {name} did not start within {timeout}s.")
    print(f"           Check {name.lower()}_service.log for details.")
    return False


def shutdown(procs: list):
    print("\n[Launcher] Shutting down services...")
    for name, proc in procs:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"[Launcher] {name} stopped.")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Lumina OCR Service Launcher")
    parser.add_argument(
        "--run-batch", metavar="INPUT", default=None,
        help="After services start, run the batch pipeline on this input path"
    )
    parser.add_argument("--output",   default="output")
    parser.add_argument("--dynamic",  action="store_true", default=True)
    parser.add_argument("--no-dynamic", dest="dynamic", action="store_false")
    parser.add_argument("--static-threshold", type=float, default=0.85)
    parser.add_argument("--bg-opacity", type=float, default=0.15)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--font", default=None)
    args = parser.parse_args()

    procs = []

    def handle_exit(sig, frame):
        shutdown(procs)
        sys.exit(0)

    signal.signal(signal.SIGINT,  handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Start services
    surya_proc = start_service("Surya", SURYA_VENV, SURYA_SERVICE, SURYA_PORT)
    qwen_proc  = start_service("Qwen",  QWEN_VENV,  QWEN_SERVICE,  QWEN_PORT)
    procs = [("Surya", surya_proc), ("Qwen", qwen_proc)]

    # Wait for both to be healthy
    surya_ok = wait_healthy("Surya", SURYA_PORT, HEALTH_TIMEOUT)
    qwen_ok  = wait_healthy("Qwen",  QWEN_PORT,  HEALTH_TIMEOUT)

    if not (surya_ok and qwen_ok):
        shutdown(procs)
        sys.exit(1)

    print("\n[Launcher] Both services are running.")
    print(f"           Surya → http://localhost:{SURYA_PORT}/docs")
    print(f"           Qwen  → http://localhost:{QWEN_PORT}/docs")

    # Optionally run the batch pipeline
    if args.run_batch:
        print(f"\n[Launcher] Running batch pipeline on: {args.run_batch}\n")
        from main_batch import run as run_batch
        run_batch(
            input_path=args.run_batch,
            output_dir=args.output,
            use_dynamic_threshold=args.dynamic,
            static_threshold=args.static_threshold,
            bg_opacity=args.bg_opacity,
            dpi=args.dpi,
            font_path=args.font,
        )
        shutdown(procs)
    else:
        print("\n[Launcher] Press Ctrl+C to stop both services.\n")
        # Keep alive until user interrupts
        try:
            while True:
                # Restart crashed services
                for i, (name, proc) in enumerate(procs):
                    if proc.poll() is not None:
                        print(f"[Launcher] {name} crashed (exit {proc.returncode}), restarting...")
                        venv  = SURYA_VENV if name == "Surya" else QWEN_VENV
                        app   = SURYA_SERVICE if name == "Surya" else QWEN_SERVICE
                        port  = SURYA_PORT if name == "Surya" else QWEN_PORT
                        new_proc = start_service(name, venv, app, port)
                        procs[i] = (name, new_proc)
                time.sleep(5)
        except KeyboardInterrupt:
            shutdown(procs)


if __name__ == "__main__":
    main()