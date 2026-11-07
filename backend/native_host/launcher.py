#!/usr/bin/env python3
import json
import socket
import struct
import subprocess
import sys
from pathlib import Path


def is_port_in_use(host: str = "127.0.0.1", port: int = 8765) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) == 0


def read_native_message() -> dict | None:
    raw_len = sys.stdin.buffer.read(4)
    if len(raw_len) < 4:
        return None
    msg_len = struct.unpack("@I", raw_len)[0]
    payload = sys.stdin.buffer.read(msg_len).decode("utf-8")
    return json.loads(payload)


def send_native_message(obj: dict) -> None:
    encoded = json.dumps(obj).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("@I", len(encoded)))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    while True:
        msg = read_native_message()
        if msg is None:
            break
        if msg.get("command") == "START_FASTAPI_SERVER":
            port = int(msg.get("port", 8765))
            if not is_port_in_use("127.0.0.1", port):
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "backend.main:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                    ],
                    cwd=str(repo_root),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            send_native_message({"status": "running", "port": port})


if __name__ == "__main__":
    main()
