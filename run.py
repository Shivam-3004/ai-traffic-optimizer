# """
# run.py — convenience launcher
#
# When executed this script will:
#  - start the Flask backend (backend/api/app.py) as a subprocess
#  - serve the `frontend/` directory using Python's built-in HTTP server
#  - open the dashboard in the default browser
#  - cleanly shut down both services on Ctrl+C
#
# This is intended as a convenience developer workflow only.
# """

import sys
import os
import subprocess
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler
import socketserver
import functools


ROOT = os.path.abspath(os.path.dirname(__file__))
FRONTEND_DIR = os.path.join(ROOT, 'frontend')
FRONTEND_PORT = 8000


def start_frontend_server(directory, port):
    # run a simple static HTTP server that serves files from `directory`
    # use functools.partial to pass directory to handler (avoids changing CWD)
    handler = functools.partial(SimpleHTTPRequestHandler, directory=directory)

    class ReuseTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    httpd = ReuseTCPServer(("", port), handler)

    def serve():
        try:
            httpd.serve_forever()
        except Exception:
            pass

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    return httpd, th


def start_backend(python_exe=sys.executable):
    # start the backend Flask app as a subprocess (python -m backend.api.app)
    cmd = [python_exe, '-m', 'backend.api.app']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT, text=True)
    return proc


def stream_process_output(proc):
    try:
        for line in proc.stdout:
            print(line, end='')
    except Exception:
        pass


def main():
    print('Starting frontend server (serving repository root on port', FRONTEND_PORT, ')')
    if not os.path.isdir(FRONTEND_DIR):
        print('Error: frontend directory not found at', FRONTEND_DIR)
        return

    # serve from repository root so relative paths like ../backend/... resolve
    httpd, th = start_frontend_server(ROOT, FRONTEND_PORT)

    print('Starting backend (Flask)')
    backend_proc = start_backend()
    out_thread = threading.Thread(target=stream_process_output, args=(backend_proc,), daemon=True)
    out_thread.start()

    dashboard_url = f'http://127.0.0.1:{FRONTEND_PORT}/frontend/dashboard.html'
    print('Opening dashboard at', dashboard_url)
    try:
        webbrowser.open(dashboard_url)
    except Exception:
        pass

    try:
        while True:
            time.sleep(0.5)
            if backend_proc.poll() is not None:
                print('Backend process exited with code', backend_proc.returncode)
                break
    except KeyboardInterrupt:
        print('Shutting down (KeyboardInterrupt)')
    finally:
        print('Stopping frontend server...')
        try:
            httpd.shutdown()
        except Exception:
            pass
        print('Stopping backend process...')
        try:
            backend_proc.terminate()
            time.sleep(0.5)
            if backend_proc.poll() is None:
                backend_proc.kill()
        except Exception:
            pass


if __name__ == '__main__':
    main()