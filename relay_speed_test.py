from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import secrets
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit


CHUNK_SIZE = 1024 * 1024
TOKEN_ENV = "RELAY_TEST_TOKEN"


def _human_speed(byte_count: int, seconds: float) -> str:
    seconds = max(seconds, 0.001)
    mib_s = byte_count / 1024 / 1024 / seconds
    return f"{mib_s:.2f} MiB/s（{mib_s * 8:.2f} Mbps）"


class Progress:
    def __init__(self, label: str, total: int):
        self.label = label
        self.total = max(1, total)
        self.started = time.monotonic()
        self.last_print = 0.0

    def update(self, current: int, *, force: bool = False):
        now = time.monotonic()
        if not force and now - self.last_print < 1.0:
            return
        self.last_print = now
        percent = min(100, int(current * 100 / self.total))
        print(
            f"\r{self.label} {percent:3d}%  "
            f"{_human_speed(current, now - self.started)}",
            end="\n" if force else "",
            flush=True,
        )


def _token(value: str | None) -> str:
    result = value or os.environ.get(TOKEN_ENV, "")
    if not result:
        raise ValueError(f"请通过 --token 或环境变量 {TOKEN_ENV} 提供临时访问令牌")
    if len(result) < 16:
        raise ValueError("访问令牌至少需要 16 个字符")
    return result


def _safe_name(value: str) -> str:
    name = unquote(value).strip()
    if not name or name in {".", ".."} or Path(name).name != name or "\\" in name:
        raise ValueError("文件名不安全")
    return name


def _connection(url: str):
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL 必须以 http:// 或 https:// 开头")
    cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    base = parsed.path.rstrip("/")
    return cls(parsed.hostname, port, timeout=60), base


def _read_response(response: http.client.HTTPResponse) -> dict:
    body = response.read().decode("utf-8", errors="replace")
    if response.status >= 400:
        raise RuntimeError(f"HTTP {response.status}: {body}")
    return json.loads(body) if body else {}


def generate_file(path: Path, size_mb: int) -> dict:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    remaining = size_mb * 1024 * 1024
    digest = hashlib.sha256()
    progress = Progress("生成测试文件", remaining)
    written = 0
    with path.open("wb") as stream:
        while remaining:
            block = os.urandom(min(CHUNK_SIZE, remaining))
            stream.write(block)
            digest.update(block)
            written += len(block)
            remaining -= len(block)
            progress.update(written)
    progress.update(written, force=True)
    result = {"path": str(path), "bytes": written, "sha256": digest.hexdigest()}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def upload_file(base_url: str, path: Path, token: str) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK_SIZE):
            digest.update(block)
    checksum = digest.hexdigest()
    connection, base = _connection(base_url)
    target = f"{base}/files/{quote(path.name)}"
    progress = Progress("上传到 UCloud", size)
    sent = 0
    try:
        connection.request("HEAD", f"{base}/auth", headers={"Authorization": f"Bearer {token}"})
        auth_response = connection.getresponse()
        if auth_response.status >= 400:
            _read_response(auth_response)
        else:
            auth_response.read()
        connection.putrequest("PUT", target)
        connection.putheader("Authorization", f"Bearer {token}")
        connection.putheader("Content-Type", "application/octet-stream")
        connection.putheader("Content-Length", str(size))
        connection.putheader("X-Content-SHA256", checksum)
        connection.endheaders()
        with path.open("rb") as stream:
            while block := stream.read(CHUNK_SIZE):
                connection.send(block)
                sent += len(block)
                progress.update(sent)
        response = connection.getresponse()
        result = _read_response(response)
    finally:
        connection.close()
    progress.update(sent, force=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def download_file(base_url: str, name: str, output: Path, token: str) -> dict:
    name = _safe_name(name)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    connection, base = _connection(base_url)
    target = f"{base}/files/{quote(name)}"
    temporary = output.with_name(output.name + ".part")
    try:
        connection.request("GET", target, headers={"Authorization": f"Bearer {token}"})
        response = connection.getresponse()
        if response.status >= 400:
            _read_response(response)
        total = int(response.getheader("Content-Length", "0"))
        expected = response.getheader("X-Content-SHA256", "")
        progress = Progress("从 UCloud 下载", total)
        digest = hashlib.sha256()
        received = 0
        with temporary.open("wb") as stream:
            while block := response.read(CHUNK_SIZE):
                stream.write(block)
                digest.update(block)
                received += len(block)
                progress.update(received)
        checksum = digest.hexdigest()
        if expected and not secrets.compare_digest(expected, checksum):
            temporary.unlink(missing_ok=True)
            raise RuntimeError("SHA-256 校验失败，下载文件可能损坏")
        os.replace(temporary, output)
    finally:
        connection.close()
    progress.update(received, force=True)
    result = {"path": str(output), "bytes": received, "sha256": checksum}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def delete_remote(base_url: str, name: str, token: str) -> dict:
    name = _safe_name(name)
    connection, base = _connection(base_url)
    try:
        connection.request(
            "DELETE",
            f"{base}/files/{quote(name)}",
            headers={"Authorization": f"Bearer {token}"},
        )
        result = _read_response(connection.getresponse())
    finally:
        connection.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def make_server(host: str, port: int, storage: Path, token: str, max_bytes: int):
    storage = storage.expanduser().resolve()
    storage.mkdir(parents=True, exist_ok=True)

    class RelayHandler(BaseHTTPRequestHandler):
        server_version = "SubtitleRelayTest/1.0"

        def log_message(self, fmt, *args):
            print(f"[{self.log_date_time_string()}] {self.client_address[0]} {fmt % args}")

        def _json(self, status: int, payload: dict):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {token}"
            if secrets.compare_digest(supplied, expected):
                return True
            self._json(401, {"detail": "访问令牌错误"})
            return False

        def _target(self) -> Path:
            prefix = "/files/"
            if not self.path.startswith(prefix):
                raise ValueError("未知路径")
            return storage / _safe_name(self.path[len(prefix) :].split("?", 1)[0])

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"ok": True, "service": "subtitle-relay-speed-test"})
                return
            if not self._authorized():
                return
            try:
                target = self._target()
                if not target.is_file():
                    self._json(404, {"detail": "文件不存在"})
                    return
                checksum_path = target.with_name(target.name + ".sha256")
                checksum = (
                    checksum_path.read_text(encoding="ascii").strip()
                    if checksum_path.is_file()
                    else ""
                )
                size = target.stat().st_size
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(size))
                self.send_header(
                    "Content-Disposition", f"attachment; filename*=UTF-8''{quote(target.name)}"
                )
                if checksum:
                    self.send_header("X-Content-SHA256", checksum)
                self.end_headers()
                with target.open("rb") as stream:
                    while block := stream.read(CHUNK_SIZE):
                        self.wfile.write(block)
            except (ValueError, OSError) as exc:
                self._json(400, {"detail": str(exc)})

        def do_HEAD(self):
            if not self._authorized():
                return
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_PUT(self):
            if not self._authorized():
                return
            temporary = None
            try:
                target = self._target()
                length = int(self.headers.get("Content-Length", "-1"))
                expected = self.headers.get("X-Content-SHA256", "").lower()
                if length < 0:
                    raise ValueError("请求缺少 Content-Length")
                if length > max_bytes:
                    self._json(413, {"detail": "文件超过服务器设置的大小限制"})
                    return
                if not expected or len(expected) != 64:
                    raise ValueError("请求缺少有效的 X-Content-SHA256")
                temporary = storage / f".{target.name}.{uuid.uuid4().hex}.uploading"
                digest = hashlib.sha256()
                received = 0
                with temporary.open("wb") as stream:
                    while received < length:
                        block = self.rfile.read(min(CHUNK_SIZE, length - received))
                        if not block:
                            raise ConnectionError("上传连接提前断开")
                        stream.write(block)
                        digest.update(block)
                        received += len(block)
                checksum = digest.hexdigest()
                if not secrets.compare_digest(expected, checksum):
                    raise ValueError("SHA-256 校验失败")
                os.replace(temporary, target)
                target.with_name(target.name + ".sha256").write_text(checksum, encoding="ascii")
                self._json(
                    201,
                    {"ok": True, "name": target.name, "bytes": received, "sha256": checksum},
                )
            except (ValueError, OSError, ConnectionError) as exc:
                if temporary:
                    temporary.unlink(missing_ok=True)
                self._json(400, {"detail": str(exc)})

        def do_DELETE(self):
            if not self._authorized():
                return
            try:
                target = self._target()
                existed = target.exists()
                target.unlink(missing_ok=True)
                target.with_name(target.name + ".sha256").unlink(missing_ok=True)
                self._json(200, {"ok": True, "deleted": existed, "name": target.name})
            except (ValueError, OSError) as exc:
                self._json(400, {"detail": str(exc)})

    return ThreadingHTTPServer((host, port), RelayHandler)


def parse_args():
    parser = argparse.ArgumentParser(description="字幕云算力中转链路测速工具")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="生成不含私人数据的随机测试文件")
    generate.add_argument("--file", type=Path, default=Path("relay-test.bin"))
    generate.add_argument("--size-mb", type=int, default=100)

    serve = sub.add_parser("serve", help="在 UCloud 上启动临时中转服务")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8766)
    serve.add_argument("--storage", type=Path, default=Path("./relay-test-data"))
    serve.add_argument("--max-size-mb", type=int, default=2048)
    serve.add_argument("--token")

    upload = sub.add_parser("upload", help="从本机上传并测速")
    upload.add_argument("--url", required=True)
    upload.add_argument("--file", type=Path, required=True)
    upload.add_argument("--token")

    download = sub.add_parser("download", help="在 AutoDL 下载并测速")
    download.add_argument("--url", required=True)
    download.add_argument("--name", required=True)
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--token")

    delete = sub.add_parser("delete", help="删除 UCloud 上的测试文件")
    delete.add_argument("--url", required=True)
    delete.add_argument("--name", required=True)
    delete.add_argument("--token")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "generate":
        if args.size_mb < 1:
            raise ValueError("测试文件至少为 1MB")
        generate_file(args.file, args.size_mb)
    elif args.command == "serve":
        token = _token(args.token)
        server = make_server(
            args.host,
            args.port,
            args.storage,
            token,
            args.max_size_mb * 1024 * 1024,
        )
        print(f"中转测速服务已启动：http://{args.host}:{server.server_port}")
        print(f"临时文件目录：{args.storage.expanduser().resolve()}")
        print("仅用于测速；如需传输真实音轨，请配置 HTTPS。按 Ctrl+C 停止。")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n测速服务已停止")
        finally:
            server.server_close()
    elif args.command == "upload":
        upload_file(args.url, args.file, _token(args.token))
    elif args.command == "download":
        download_file(args.url, args.name, args.output, _token(args.token))
    elif args.command == "delete":
        delete_remote(args.url, args.name, _token(args.token))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
