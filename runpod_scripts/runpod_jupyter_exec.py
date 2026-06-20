from __future__ import annotations

import argparse
import json
import re
import shlex
import ssl
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
import websocket
import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class JupyterTerminal:
    base_url: str
    password: str
    request_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.ws = None
        self.xsrf = ""

    def login(self) -> None:
        login_url = f"{self.base_url}/login"
        response = self.session.get(login_url, verify=False, timeout=self.request_timeout_s)
        response.raise_for_status()
        self.xsrf = self.session.cookies.get("_xsrf", "")
        response = self.session.post(
            f"{login_url}?next=%2Flab",
            headers={"Referer": login_url},
            data={"password": self.password, "_xsrf": self.xsrf},
            verify=False,
            allow_redirects=False,
            timeout=self.request_timeout_s,
        )
        if response.status_code not in (302, 303):
            raise RuntimeError(f"Jupyter login failed: {response.status_code} {response.text[:200]}")
        self.xsrf = self.session.cookies.get("_xsrf", "")

    def open(self) -> None:
        response = self.session.post(
            f"{self.base_url}/api/terminals",
            headers={"Referer": f"{self.base_url}/lab", "X-XSRFToken": self.xsrf},
            verify=False,
            timeout=self.request_timeout_s,
        )
        response.raise_for_status()
        terminal_name = response.json()["name"]
        cookies = "; ".join(f"{key}={value}" for key, value in self.session.cookies.get_dict().items())
        ws_url = self.base_url.replace("https://", "wss://").replace("http://", "ws://")
        self.ws = websocket.create_connection(
            f"{ws_url}/terminals/websocket/{terminal_name}",
            header=[
                f"Cookie: {cookies}",
                f"X-XSRFToken: {self.xsrf}",
                f"Referer: {self.base_url}/lab",
            ],
            sslopt={"cert_reqs": ssl.CERT_NONE},
            timeout=self.request_timeout_s,
        )
        self.ws.settimeout(2)
        self._drain()

    def _drain(self) -> str:
        chunks: list[str] = []
        while True:
            try:
                message = self.ws.recv()
            except websocket.WebSocketTimeoutException:
                break
            event, payload = json.loads(message)
            if event == "stdout":
                chunks.append(payload)
        return "".join(chunks)

    def exec(self, command: str, timeout_s: float, stdin_text: str | None = None) -> tuple[int, str]:
        marker = "__CODEX_CMD_DONE__"
        shell_command = f"{command}; rc=$?; printf '{marker}:%s\\n' \"$rc\""
        wrapped = f"bash -lc {shlex.quote(shell_command)}"
        self.ws.send(json.dumps(["stdin", wrapped + "\r"]))
        if stdin_text is not None:
            time.sleep(0.2)
            self.ws.send(json.dumps(["stdin", stdin_text + "\r"]))
        deadline = time.time() + timeout_s
        chunks: list[str] = []
        status = None
        while time.time() < deadline:
            try:
                message = self.ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            event, payload = json.loads(message)
            if event != "stdout":
                continue
            chunks.append(payload)
            joined = "".join(chunks)
            matches = list(re.finditer(rf"{marker}:(\d+)", joined))
            if not matches:
                continue
            match = matches[-1]
            status = int(match.group(1))
            output = joined[: match.start()]
            return status, output
        raise TimeoutError(f"Command timed out after {timeout_s} seconds")

    def close(self) -> None:
        if self.ws is not None:
            self.ws.close()
            self.ws = None

    def fetch(self, remote_path: str) -> bytes:
        response = self.session.get(
            f"{self.base_url}/files/{remote_path.lstrip('/')}",
            headers={"Referer": f"{self.base_url}/lab", "X-XSRFToken": self.xsrf},
            verify=False,
            timeout=self.request_timeout_s,
        )
        response.raise_for_status()
        return response.content

    def upload_text(self, remote_path: str, content: str) -> None:
        response = self.session.put(
            f"{self.base_url}/api/contents/{remote_path.lstrip('/')}",
            headers={"Referer": f"{self.base_url}/lab", "X-XSRFToken": self.xsrf},
            json={"type": "file", "format": "text", "content": content},
            verify=False,
            timeout=self.request_timeout_s,
        )
        response.raise_for_status()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a shell command through a Runpod Jupyter terminal.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    exec_parser = subparsers.add_parser("exec", help="Execute a shell command via a Jupyter terminal")
    exec_parser.add_argument("--base-url", required=True, help="Proxy URL, for example https://<pod>-8888.proxy.runpod.net")
    exec_parser.add_argument("--password", help="Jupyter password")
    exec_parser.add_argument("--password-file", help="File containing the Jupyter password")
    exec_parser.add_argument("--request-timeout", type=float, default=30.0, help="HTTP request timeout in seconds")
    exec_parser.add_argument("--stdin-file", help="Local file whose contents are sent to the remote command stdin")
    exec_parser.add_argument("--timeout", type=float, default=30.0, help="Command timeout in seconds")
    exec_parser.add_argument("command", help="Shell command to execute via bash -lc")

    fetch_parser = subparsers.add_parser("fetch", help="Download a file through the authenticated Jupyter file endpoint")
    fetch_parser.add_argument("--base-url", required=True, help="Proxy URL, for example https://<pod>-8888.proxy.runpod.net")
    fetch_parser.add_argument("--password", help="Jupyter password")
    fetch_parser.add_argument("--password-file", help="File containing the Jupyter password")
    fetch_parser.add_argument("--request-timeout", type=float, default=120.0, help="HTTP request timeout in seconds")
    fetch_parser.add_argument("--remote-path", required=True, help="Remote file path relative to the Jupyter root")
    fetch_parser.add_argument("--output", required=True, help="Local output path")

    upload_parser = subparsers.add_parser("upload", help="Upload a text file through the authenticated Jupyter API")
    upload_parser.add_argument("--base-url", required=True, help="Proxy URL, for example https://<pod>-8888.proxy.runpod.net")
    upload_parser.add_argument("--password", help="Jupyter password")
    upload_parser.add_argument("--password-file", help="File containing the Jupyter password")
    upload_parser.add_argument("--request-timeout", type=float, default=120.0, help="HTTP request timeout in seconds")
    upload_parser.add_argument("--local-path", required=True, help="Local text file to upload")
    upload_parser.add_argument("--remote-path", required=True, help="Remote path relative to the Jupyter root")
    return parser.parse_args()


def read_password(args: argparse.Namespace) -> str:
    if args.password is not None and args.password_file is not None:
        raise SystemExit("--password and --password-file are mutually exclusive")
    if args.password_file is not None:
        return Path(args.password_file).read_text().strip()
    if args.password is not None:
        return args.password
    raise SystemExit("one of --password or --password-file is required")


def main() -> None:
    args = parse_args()
    terminal = JupyterTerminal(args.base_url.rstrip("/"), read_password(args), request_timeout_s=args.request_timeout)
    terminal.login()
    if args.mode == "exec":
        terminal.open()
        try:
            stdin_text = Path(args.stdin_file).read_text() if args.stdin_file else None
            status, output = terminal.exec(args.command, timeout_s=args.timeout, stdin_text=stdin_text)
        finally:
            terminal.close()
        sys.stdout.write(output)
        sys.exit(status)

    if args.mode == "fetch":
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(terminal.fetch(args.remote_path))
        return

    if args.mode == "upload":
        terminal.upload_text(args.remote_path, Path(args.local_path).read_text())
        return


if __name__ == "__main__":
    main()
