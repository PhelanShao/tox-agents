#!/usr/bin/env python3
"""Automated launcher for the molecular toxicity prediction platform."""

import importlib
import os
import pty
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# Pre-emptively clear ports that may have stale processes attached.
os.system("fuser -k 50001/tcp 2>/dev/null")
os.system("fuser -k 50002/tcp 2>/dev/null")
os.system("fuser -k 50003/tcp 2>/dev/null")
os.system("fuser -k 8000/tcp 2>/dev/null")


class SystemLauncher:
    """Launch and monitor the backend (FastAPI) and frontend (Next.js) services."""

    def __init__(self) -> None:
        self.backend_process: Optional[subprocess.Popen] = None
        self.frontend_process: Optional[subprocess.Popen] = None
        self.frontend_port: Optional[int] = None
        self.frontend_pty_fd: Optional[int] = None
        self.base_dir = Path(__file__).parent
        self.frontend_dir = self.base_dir / "frontend"
        self.backend_dir = self.base_dir / "frontend" / "backend"

    # ------------------------------------------------------------------
    # Environment and dependency checks
    # ------------------------------------------------------------------
    def check_conda_environment(self) -> bool:
        """Report the active environment and warn if unimol_tools is missing."""
        print("🐍 Checking Python runtime...")

        current_env = os.environ.get("CONDA_DEFAULT_ENV")
        if current_env:
            print(f"🧪 Active Conda environment: {current_env}")
        else:
            print("ℹ️  No Conda environment detected; using the current Python interpreter.")

        try:
            importlib.import_module("unimol_tools")
            print("✅ unimol_tools detected (UniMol transfer inference is available).")
        except ImportError:
            print("⚠️ unimol_tools is not installed; UniMol transfer inference may be unavailable.")
            print("   Install with: pip install unimol_tools")

        return True

    def check_dependencies(self) -> bool:
        """Ensure critical directories and executables are available."""
        print("🔍 Checking system dependencies...")
        print(f"✅ Python: {sys.version.split()[0]}")

        if not self.frontend_dir.exists():
            print(f"❌ Frontend directory not found: {self.frontend_dir}")
            return False
        print(f"✅ Frontend directory: {self.frontend_dir}")

        backend_file = self.backend_dir / "main_fixed.py"
        if not backend_file.exists():
            print(f"❌ Backend entrypoint not found: {backend_file}")
            return False
        print(f"✅ Backend entrypoint: {backend_file}")

        try:
            result = subprocess.run(["node", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Node.js: {result.stdout.strip()}")
            else:
                print("⚠️ Node.js not available; the frontend will be skipped.")
        except FileNotFoundError:
            print("⚠️ Node.js not available; the frontend will be skipped.")

        try:
            result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ npm: {result.stdout.strip()}")
            else:
                print("⚠️ npm not available; the frontend will be skipped.")
        except FileNotFoundError:
            print("⚠️ npm not available; the frontend will be skipped.")

        return True

    def install_frontend_deps(self) -> bool:
        """Install frontend dependencies if node_modules is absent."""
        print("\n📦 Checking frontend dependencies...")

        node_modules = self.frontend_dir / "node_modules"
        package_json = self.frontend_dir / "package.json"

        if not package_json.exists():
            print(f"⚠️ package.json not found: {package_json}")
            return False

        if not node_modules.exists():
            print("📥 Installing frontend dependencies...")
            try:
                result = subprocess.run(
                    ["npm", "install"],
                    cwd=self.frontend_dir,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )
                if result.returncode == 0:
                    print("✅ Frontend dependencies installed.")
                else:
                    print("⚠️ npm install failed; frontend launch will be skipped.")
                    print(result.stderr)
                    return False
            except subprocess.TimeoutExpired:
                print("⚠️ npm install timed out; frontend launch will be skipped.")
                return False
            except Exception as exc:  # pylint: disable=broad-except
                print(f"⚠️ npm install error: {exc}; frontend launch will be skipped.")
                return False
        else:
            print("✅ Frontend dependencies already present.")

        return True

    # ------------------------------------------------------------------
    # Service launch helpers
    # ------------------------------------------------------------------
    def start_backend(self) -> bool:
        """Launch the FastAPI backend."""
        print("\n🚀 Starting backend service...")

        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.base_dir)

            python_executable = sys.executable
            self.backend_process = subprocess.Popen(
                [python_executable, "main_fixed.py"],
                cwd=self.backend_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            backend_thread = threading.Thread(
                target=self._monitor_backend_logs,
                args=(self.backend_process,),
                daemon=True,
            )
            backend_thread.start()

            time.sleep(3)

            if self.backend_process.poll() is None:
                print("✅ Backend service is running (port 8000).")
                return True

            print("❌ Backend process exited during startup.")
            return False

        except Exception as exc:  # pylint: disable=broad-except
            print(f"❌ Backend launch error: {exc}")
            return False

    @staticmethod
    def check_port_available(port: int) -> bool:
        """Return True if the given port can be bound."""
        import socket

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("localhost", port))
            return True
        except OSError:
            return False

    def start_frontend(self) -> bool:
        """Attempt to launch the Next.js frontend, trying multiple ports if needed."""
        print("\n🚀 Starting frontend service...")

        candidate_ports = [50001, 50002, 50003, 3000, 3001, 3002]
        self.frontend_process = None
        self.frontend_port = None

        for port in candidate_ports:
            if not self.check_port_available(port):
                print(f"⚠️ Port {port} is busy; trying the next candidate.")
                continue

            print(f"🔄 Attempting to launch frontend on port {port}...")
            try:
                env = os.environ.copy()
                env.setdefault("NEXT_DISABLE_SWC_NATIVE", "1")
                env["CI"] = "false"
                next_bin = self.frontend_dir / "node_modules" / ".bin" / "next"
                if next_bin.exists():
                    cmd = [str(next_bin), "dev", "-H", "0.0.0.0", "-p", str(port)]
                else:
                    cmd = ["npx", "next", "dev", "-H", "0.0.0.0", "-p", str(port)]

                master_fd, slave_fd = pty.openpty()
                try:
                    process = subprocess.Popen(
                        cmd,
                        cwd=self.frontend_dir,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=slave_fd,
                        stderr=slave_fd,
                        close_fds=True,
                    )
                finally:
                    os.close(slave_fd)

                self.frontend_pty_fd = master_fd

                monitor_thread = threading.Thread(
                    target=self._monitor_frontend_logs,
                    args=(process, master_fd),
                    daemon=True,
                )
                monitor_thread.start()

                # Wait until the server binds to the port (ready) or the process exits
                ready = False
                for _ in range(20):  # ~20 seconds max
                    if process.poll() is not None:
                        break
                    if not self.check_port_available(port):
                        ready = True
                        break
                    time.sleep(1)

                if ready and process.poll() is None:
                    self.frontend_process = process
                    self.frontend_port = port
                    print(f"✅ Frontend service is running (port {port}).")
                    print(f"🌐 Frontend URL: http://localhost:{port}")
                    return True

                exit_code = process.returncode
                print(f"⚠️ Frontend did not stay running on port {port} (code {exit_code}).")
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except Exception:  # pylint: disable=broad-except
                    process.kill()
                finally:
                    if self.frontend_pty_fd is not None:
                        os.close(self.frontend_pty_fd)
                        self.frontend_pty_fd = None

            except Exception as exc:  # pylint: disable=broad-except
                print(f"⚠️ Frontend launch error on port {port}: {exc}")
                if self.frontend_pty_fd is not None:
                    os.close(self.frontend_pty_fd)
                    self.frontend_pty_fd = None

        print("❌ Failed to launch the frontend on any candidate port.")
        return False

    # ------------------------------------------------------------------
    def start_frontend_prod(self) -> bool:
        """Fallback: run Next.js in production mode (build + start)."""
        print("\n🚀 Starting frontend in production mode (build + start)...")

        candidate_ports = [50001, 50002, 50003, 3000, 3001, 3002]
        self.frontend_process = None
        self.frontend_port = None

        # Build once
        env = os.environ.copy()
        env.setdefault("NEXT_DISABLE_SWC_NATIVE", "1")
        env["CI"] = "false"
        try:
            print("🔧 Building Next.js app (this may take a while)...")
            build = subprocess.run(
                ["npm", "run", "build"],
                cwd=self.frontend_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if build.returncode != 0:
                print("❌ next build failed:")
                print(build.stdout)
                print(build.stderr)
                return False
        except subprocess.TimeoutExpired:
            print("❌ next build timed out.")
            return False
        except Exception as exc:  # pylint: disable=broad-except
            print(f"❌ next build error: {exc}")
            return False

        # Start
        for port in candidate_ports:
            if not self.check_port_available(port):
                print(f"⚠️ Port {port} is busy; trying the next candidate.")
                continue

            print(f"🔄 Attempting to launch production server on port {port}...")
            try:
                next_bin = self.frontend_dir / "node_modules" / ".bin" / "next"
                if next_bin.exists():
                    cmd = [str(next_bin), "start", "-H", "0.0.0.0", "-p", str(port)]
                else:
                    cmd = ["npx", "next", "start", "-H", "0.0.0.0", "-p", str(port)]

                master_fd, slave_fd = pty.openpty()
                try:
                    process = subprocess.Popen(
                        cmd,
                        cwd=self.frontend_dir,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=slave_fd,
                        stderr=slave_fd,
                        close_fds=True,
                    )
                finally:
                    os.close(slave_fd)

                self.frontend_pty_fd = master_fd

                monitor_thread = threading.Thread(
                    target=self._monitor_frontend_logs,
                    args=(process, master_fd),
                    daemon=True,
                )
                monitor_thread.start()

                # Wait for bind
                ready = False
                for _ in range(30):
                    if process.poll() is not None:
                        break
                    if not self.check_port_available(port):
                        ready = True
                        break
                    time.sleep(1)

                if ready and process.poll() is None:
                    self.frontend_process = process
                    self.frontend_port = port
                    print(f"✅ Frontend (production) is running (port {port}).")
                    print(f"🌐 Frontend URL: http://localhost:{port}")
                    return True

                exit_code = process.returncode
                print(f"⚠️ Production server did not stay running on port {port} (code {exit_code}).")
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except Exception:
                    process.kill()
                finally:
                    if self.frontend_pty_fd is not None:
                        os.close(self.frontend_pty_fd)
                        self.frontend_pty_fd = None
            except Exception as exc:  # pylint: disable=broad-except
                print(f"⚠️ Frontend production launch error on port {port}: {exc}")
                if self.frontend_pty_fd is not None:
                    os.close(self.frontend_pty_fd)
                    self.frontend_pty_fd = None

        print("❌ Failed to launch the production frontend on any candidate port.")
        return False

    # ------------------------------------------------------------------
    # Log monitors
    # ------------------------------------------------------------------
    @staticmethod
    def _monitor_backend_logs(process: Optional[subprocess.Popen]) -> None:
        if not process:
            return

        for line in iter(process.stdout.readline, ""):
            if line:
                print(f"[Backend] {line.strip()}")

    def _monitor_frontend_logs(self, process: Optional[subprocess.Popen], fd: int) -> None:
        if not process:
            os.close(fd)
            return

        try:
            with os.fdopen(fd, "rb", buffering=0) as stream:
                while True:
                    try:
                        line = stream.readline()
                    except OSError:
                        break
                    if not line:
                        if process.poll() is not None:
                            break
                        continue
                    text = line.decode(errors="ignore").rstrip()
                    lower_line = text.lower()
                    if any(keyword in lower_line for keyword in ("webpack", "compiled", "ready", "error", "warn", "local:")):
                        print(f"[Frontend] {text}")
                    elif "http://localhost" in text or "https://localhost" in text:
                        print(f"[Frontend] {text}")
                    else:
                        print(f"[Frontend] {text}")
        finally:
            self.frontend_pty_fd = None

    # ------------------------------------------------------------------
    # Shutdown helpers
    # ------------------------------------------------------------------
    def stop_services(self) -> None:
        """Terminate both services if they are running."""
        print("\n🛑 Shutting down services...")

        if self.frontend_process:
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=5)
                print("✅ Frontend service stopped.")
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
                print("⚠️ Frontend service was force killed.")
            except Exception as exc:  # pylint: disable=broad-except
                print(f"❌ Error stopping frontend: {exc}")
            finally:
                if self.frontend_pty_fd is not None:
                    os.close(self.frontend_pty_fd)
                    self.frontend_pty_fd = None

        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
                print("✅ Backend service stopped.")
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
                print("⚠️ Backend service was force killed.")
            except Exception as exc:  # pylint: disable=broad-except
                print(f"❌ Error stopping backend: {exc}")

    # ------------------------------------------------------------------
    # Main orchestration
    # ------------------------------------------------------------------
    def run(self) -> bool:
        """Execute the full launch sequence and monitor the processes."""
        print("🧬 Molecular toxicity platform - automated launch")
        print("=" * 50)

        try:
            if not self.check_conda_environment():
                return False

            if not self.check_dependencies():
                print("\n⚠️ Dependency check failed; attempting to start the backend anyway.")

            frontend_ready = self.install_frontend_deps()

            print("\n🚀 Auto-starting backend...")
            if not self.start_backend():
                print("\n❌ Backend failed to start.")
                return False

            if frontend_ready:
                print("\n🚀 Auto-starting frontend...")
                if self.start_frontend() or self.start_frontend_prod():
                    print("\n" + "=" * 50)
                    print("🎉 Frontend and backend are both running.")
                    print(f"📱 Frontend URL: http://localhost:{self.frontend_port}")
                    print("🔧 Backend URL: http://localhost:8000")
                    print("📚 API docs: http://localhost:8000/docs")
                else:
                    print("\n⚠️ Frontend launch failed; backend remains available.")
                    print("🔧 Backend URL: http://localhost:8000")
                    print("📚 API docs: http://localhost:8000/docs")
            else:
                print("\n⚠️ Frontend dependencies are missing; only the backend is running.")
                print("🔧 Backend URL: http://localhost:8000")
                print("📚 API docs: http://localhost:8000/docs")

            print("=" * 50)
            print("✨ Services are running; press Ctrl+C to exit.")
            print("💡 Tip: install Node.js and npm to enable the frontend.")

            try:
                while True:
                    time.sleep(1)
                    if self.backend_process and self.backend_process.poll() is not None:
                        print("\n⚠️ Backend service exited unexpectedly.")
                        break
                    if self.frontend_process and self.frontend_process.poll() is not None:
                        print("\n⚠️ Frontend service exited unexpectedly; backend is still running.")
                        self.frontend_process = None
            except KeyboardInterrupt:
                print("\n\n👋 Received interrupt signal...")

            return True

        except Exception as exc:  # pylint: disable=broad-except
            print(f"\n❌ System launch error: {exc}")
            return False
        finally:
            self.stop_services()


def signal_handler(signum, frame) -> None:  # type: ignore[unused-arg]
    """Handle termination signals cleanly."""
    print("\n\n👋 Received termination signal, shutting down...")
    sys.exit(0)


def main() -> None:
    """Script entrypoint."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    launcher = SystemLauncher()
    success = launcher.run()

    if success:
        print("\n✅ Shutdown complete.")
    else:
        print("\n❌ Startup failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
