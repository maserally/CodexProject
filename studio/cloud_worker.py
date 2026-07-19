from __future__ import annotations

import posixpath
import re
import shlex
import time
from pathlib import Path
from typing import Callable

from .config import ROOT
from .schemas import CloudWorkerSettings


class CloudWorkerError(RuntimeError):
    pass


def _paramiko():
    try:
        import paramiko
    except ImportError as exc:
        raise CloudWorkerError(
            "缺少本地 SSH 组件 paramiko，请重新运行依赖安装后再连接云节点"
        ) from exc
    return paramiko


def _validated(settings: CloudWorkerSettings) -> CloudWorkerSettings:
    if not settings.host.strip():
        raise CloudWorkerError("请填写云服务器地址")
    if not re.fullmatch(r"[A-Za-z0-9._:-]+", settings.host.strip()):
        raise CloudWorkerError("云服务器地址包含不支持的字符")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", settings.username.strip()):
        raise CloudWorkerError("云服务器用户名包含不支持的字符")
    if not settings.password and not settings.private_key_path:
        raise CloudWorkerError("请填写 SSH 密码或私钥路径")
    if settings.private_key_path and not Path(settings.private_key_path).expanduser().is_file():
        raise CloudWorkerError("SSH 私钥文件不存在")
    remote_dir = settings.remote_dir.strip().rstrip("/")
    if not remote_dir.startswith("/") or not re.fullmatch(r"/[A-Za-z0-9._/-]+", remote_dir):
        raise CloudWorkerError("云端工作目录必须是只含英文、数字、点、横线的绝对路径")
    result = settings.model_copy(deep=True)
    result.host = result.host.strip()
    result.username = result.username.strip()
    result.remote_dir = remote_dir
    return result


class CloudWhisperWorker:
    def __init__(
        self,
        settings: CloudWorkerSettings,
        *,
        logger: Callable[[str], None] | None = None,
        checkpoint: Callable[[], None] | None = None,
    ):
        self.settings = _validated(settings)
        self.logger = logger or (lambda _message: None)
        self.checkpoint = checkpoint or (lambda: None)
        self.client = None
        self.sftp = None
        self.remote_job_dir = ""
        self.active_control_file = ""

    def connect(self):
        paramiko = _paramiko()
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs = {
            "hostname": self.settings.host,
            "port": self.settings.port,
            "username": self.settings.username,
            "password": self.settings.password or None,
            "key_filename": str(Path(self.settings.private_key_path).expanduser())
            if self.settings.private_key_path
            else None,
            "timeout": 15,
            "banner_timeout": 20,
            "auth_timeout": 20,
            "look_for_keys": not bool(self.settings.password or self.settings.private_key_path),
            "allow_agent": not bool(self.settings.password or self.settings.private_key_path),
        }
        client.connect(**kwargs)
        self.client = client
        self.sftp = client.open_sftp()
        return self

    def close(self):
        if self.sftp:
            self.sftp.close()
            self.sftp = None
        if self.client:
            self.client.close()
            self.client = None

    def _exec(
        self,
        command: str,
        timeout: float | None = None,
        *,
        controllable: bool = False,
    ) -> str:
        if not self.client:
            raise CloudWorkerError("云节点尚未连接")
        channel = self.client.get_transport().open_session()
        if controllable:
            self.active_control_file = posixpath.join(self.remote_job_dir, ".active-process")
            control = shlex.quote(self.active_control_file)
            inner = (
                f"echo $$ > {control}; {command}; status=$?; "
                f"rm -f {control}; exit $status"
            )
            command = "setsid bash -lc " + shlex.quote(inner)
        channel.exec_command(command)
        output: list[str] = []
        errors: list[str] = []
        started = time.monotonic()
        try:
            while not channel.exit_status_ready():
                self.checkpoint()
                if channel.recv_ready():
                    text = channel.recv(65536).decode("utf-8", errors="replace")
                    output.append(text)
                    for line in text.splitlines():
                        if line.strip():
                            self.logger(line.strip())
                if channel.recv_stderr_ready():
                    text = channel.recv_stderr(65536).decode("utf-8", errors="replace")
                    errors.append(text)
                    for line in text.splitlines():
                        if line.strip():
                            self.logger(line.strip())
                if timeout and time.monotonic() - started > timeout:
                    raise CloudWorkerError("云节点命令执行超时")
                time.sleep(0.15)
            while channel.recv_ready():
                output.append(channel.recv(65536).decode("utf-8", errors="replace"))
            while channel.recv_stderr_ready():
                errors.append(channel.recv_stderr(65536).decode("utf-8", errors="replace"))
            status = channel.recv_exit_status()
            combined = "".join(output)
            error_text = "".join(errors)
            if status:
                raise CloudWorkerError(
                    f"云节点命令失败（退出码 {status}）：{error_text.strip() or combined.strip()}"
                )
            return combined
        finally:
            channel.close()
            if controllable:
                self.active_control_file = ""

    def _signal_current(self, signal: str):
        if not self.client or not self.active_control_file:
            return
        control = shlex.quote(self.active_control_file)
        command = (
            f"if [ -f {control} ]; then kill -{signal} -- -$(cat {control}) 2>/dev/null || true; fi"
        )
        channel = self.client.get_transport().open_session()
        try:
            channel.exec_command("bash -lc " + shlex.quote(command))
            channel.recv_exit_status()
        finally:
            channel.close()

    def pause_current(self):
        self._signal_current("STOP")

    def resume_current(self):
        self._signal_current("CONT")

    def cancel_current(self):
        self._signal_current("TERM")

    def test_connection(self) -> dict[str, str]:
        self.connect()
        try:
            output = self._exec(
                "printf 'system='; uname -srm; "
                "printf 'gpu='; (nvidia-smi --query-gpu=name,memory.total "
                "--format=csv,noheader 2>/dev/null || printf 'not-found')",
                timeout=20,
            )
            values = {}
            for line in output.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    values[key.strip()] = value.strip()
            return values
        finally:
            self.close()

    def bootstrap(self) -> dict[str, str]:
        if not self.client:
            self.connect()
        remote = shlex.quote(self.settings.remote_dir)
        script = f"""set -e
if ! command -v python3 >/dev/null 2>&1; then
  if [ "$(id -u)" = 0 ]; then apt-get update && apt-get install -y python3 python3-venv; else sudo -n apt-get update && sudo -n apt-get install -y python3 python3-venv; fi
fi
if ! python3 -m venv --help >/dev/null 2>&1; then
  if [ "$(id -u)" = 0 ]; then apt-get update && apt-get install -y python3-venv; else sudo -n apt-get update && sudo -n apt-get install -y python3-venv; fi
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  if [ "$(id -u)" = 0 ]; then apt-get update && apt-get install -y ffmpeg; else sudo -n apt-get update && sudo -n apt-get install -y ffmpeg; fi
fi
mkdir -p {remote}/studio
if [ ! -f {remote}/.worker-ready-v2 ]; then
  if [ ! -x {remote}/.venv/bin/python ]; then python3 -m venv --system-site-packages {remote}/.venv; fi
  {remote}/.venv/bin/python -m pip install --upgrade pip
  {remote}/.venv/bin/python -m pip install numpy openai-whisper transformers
fi
{remote}/.venv/bin/python -c 'import torch, whisper; print("torch=" + torch.__version__); print("cuda=" + str(torch.cuda.is_available())); assert torch.cuda.is_available(), "云节点 PyTorch 未启用 CUDA，请更换带 CUDA/PyTorch 的 GPU 镜像"'
touch {remote}/.worker-ready-v2
"""
        output = self._exec("bash -lc " + shlex.quote(script), timeout=1800)
        return {"output": output.strip(), "remote_dir": self.settings.remote_dir}

    def _mkdir(self, remote_path: str):
        self._exec("mkdir -p " + shlex.quote(remote_path), timeout=30)

    def _upload(self, local_path: Path, remote_path: str, label: str):
        if not self.sftp:
            raise CloudWorkerError("云节点文件通道尚未连接")
        size = max(1, local_path.stat().st_size)
        last_percent = -1

        def callback(sent: int, _total: int):
            nonlocal last_percent
            self.checkpoint()
            percent = int(sent * 100 / size)
            if percent >= last_percent + 10 or percent == 100:
                last_percent = percent
                self.logger(f"{label} {percent}%")

        self.sftp.put(str(local_path), remote_path, callback=callback, confirm=True)

    def _download(self, remote_path: str, local_path: Path):
        if not self.sftp:
            raise CloudWorkerError("云节点文件通道尚未连接")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.sftp.get(remote_path, str(local_path))

    def prepare_job(self, job_id: str, audio_path: Path):
        if not self.client:
            self.connect()
        if self.settings.auto_setup:
            self.logger("检查并安装云节点运算依赖")
            self.bootstrap()
        self.remote_job_dir = posixpath.join(self.settings.remote_dir, "jobs", job_id)
        self._mkdir(posixpath.join(self.remote_job_dir, "studio"))
        for local, remote in (
            (ROOT / "asr_stage.py", posixpath.join(self.remote_job_dir, "asr_stage.py")),
            (ROOT / "large_review.py", posixpath.join(self.remote_job_dir, "large_review.py")),
            (ROOT / "audio_event_gate.py", posixpath.join(self.remote_job_dir, "audio_event_gate.py")),
            (ROOT / "studio" / "__init__.py", posixpath.join(self.remote_job_dir, "studio", "__init__.py")),
            (ROOT / "studio" / "languages.py", posixpath.join(self.remote_job_dir, "studio", "languages.py")),
        ):
            self._upload(local, remote, f"同步 {local.name}")
        self._upload(audio_path, posixpath.join(self.remote_job_dir, "audio.flac"), "上传音轨")

    def run_event_gate(self, vad_path: Path, local_events_path: Path):
        remote_work = posixpath.join(self.remote_job_dir, "event_gate")
        self._mkdir(remote_work)
        remote_vad = posixpath.join(remote_work, "vad.json")
        remote_events = posixpath.join(remote_work, "events.json")
        self._upload(vad_path, remote_vad, "上传 VAD 分段")
        python = posixpath.join(self.settings.remote_dir, ".venv", "bin", "python")
        args = [
            python,
            posixpath.join(self.remote_job_dir, "audio_event_gate.py"),
            posixpath.join(self.remote_job_dir, "audio.flac"),
            "--vad", remote_vad,
            "--output", remote_events,
        ]
        self._exec(" ".join(shlex.quote(value) for value in args), controllable=True)
        self._download(remote_events, local_events_path)

    def run_asr(
        self,
        events_path: Path,
        local_workdir: Path,
        *,
        label: str,
        model: str,
        language: str,
        speech_threshold: float,
        nonlexical_factor: float,
    ):
        remote_work = posixpath.join(self.remote_job_dir, label)
        self._mkdir(remote_work)
        remote_events = posixpath.join(remote_work, "events.json")
        self._upload(events_path, remote_events, "上传识别分段")
        python = posixpath.join(self.settings.remote_dir, ".venv", "bin", "python")
        args = [
            python,
            posixpath.join(self.remote_job_dir, "asr_stage.py"),
            posixpath.join(self.remote_job_dir, "audio.flac"),
            "--events", remote_events,
            "--workdir", remote_work,
            "--model", model,
            "--language", language,
            "--speech-threshold", str(speech_threshold),
            "--nonlexical-factor", str(nonlexical_factor),
        ]
        self._exec(" ".join(shlex.quote(value) for value in args), controllable=True)
        self._download(
            posixpath.join(remote_work, "source_sentences.json"),
            local_workdir / "source_sentences.json",
        )

    def run_review(
        self,
        source_path: Path,
        local_workdir: Path,
        *,
        label: str,
        model: str,
        language: str,
    ):
        remote_work = posixpath.join(self.remote_job_dir, label)
        self._mkdir(remote_work)
        remote_source = posixpath.join(remote_work, "source_sentences.json")
        self._upload(source_path, remote_source, "上传复核文本")
        python = posixpath.join(self.settings.remote_dir, ".venv", "bin", "python")
        args = [
            python,
            posixpath.join(self.remote_job_dir, "large_review.py"),
            posixpath.join(self.remote_job_dir, "audio.flac"),
            "--medium", remote_source,
            "--workdir", remote_work,
            "--model", model,
            "--language", language,
        ]
        self._exec(" ".join(shlex.quote(value) for value in args), controllable=True)
        for name in ("source_final.json", "model_comparison.json"):
            self._download(posixpath.join(remote_work, name), local_workdir / name)

    def cleanup_job(self):
        if self.remote_job_dir:
            self._exec("rm -rf -- " + shlex.quote(self.remote_job_dir), timeout=60)
            self.remote_job_dir = ""
