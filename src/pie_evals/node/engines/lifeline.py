import signal
import subprocess
import sys
import threading


def main() -> int:
    child = subprocess.Popen(sys.argv[1:])

    def stop(*_):
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()

    signal.signal(signal.SIGTERM, lambda *a: (stop(), sys.exit(0)))
    signal.signal(signal.SIGINT, lambda *a: (stop(), sys.exit(0)))
    threading.Thread(target=lambda: (sys.stdin.read(), stop()), daemon=True).start()
    return child.wait()


if __name__ == "__main__":
    sys.exit(main())
