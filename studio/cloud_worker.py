from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import shlex
import threading
import time
from pathlib import Path
from typing import Callable

from .config import ROOT
from .schemas import CloudWorkerSettings


CLOUD_UPLOAD_CONCURRENCY = max(
    1, min(6, int(os.getenv("SUBTITLE_CLOUD_UPLOAD_CONCURRENCY", "3")))
)
CLOUD_UPLOAD_SLOTS = threading.BoundedSemaphore(CLOUD_UPLOAD_CONCURRENCY)


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
    if settings.private_key_path and not Path(settings.private_key_path).expanduser().is_file():
        raise CloudWorkerError("SSH 私钥文件不存在")
    remote_dir = settings.remote_dir.strip().rstrip("/")
    if not remote_dir.startswith("/") or not re.fullmatch(r"/[A-Za-z0-9._/-]+", remote_dir):
        raise CloudWorkerError("云端工作目录必须是只含英文、数字、点、横线的绝对路径")
    result = settings.model_copy(deep=True)
    result.host = result.host.strip()
    result.username = result.username.strip()
    result.remote_dir = remote_dir
    model_dir = settings.model_dir.strip().rstrip("/")
    if not model_dir.startswith("/") or not re.fullmatch(r"/[A-Za-z0-9._/-]+", model_dir):
        raise CloudWorkerError("云端模型目录必须是只含英文、数字、点、横线的绝对路径")
    result.model_dir = model_dir
    data_dir = settings.data_dir.strip().rstrip("/")
    if not data_dir.startswith("/") or not re.fullmatch(r"/[A-Za-z0-9._/-]+", data_dir):
        raise CloudWorkerError("云端任务数据目录必须是只含英文、数字、点、横线的绝对路径")
    result.data_dir = data_dir
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
        self.legacy_job_dir = ""
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
            try:
                self.sftp.close()
            except (EOFError, OSError):
                pass
            finally:
                self.sftp = None
        if self.client:
            try:
                self.client.close()
            finally:
                self.client = None

    def _exec(
        self,
        command: str,
        timeout: float | None = None,
        *,
        controllable: bool = False,
        stdin_text: str | None = None,
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
            # util-linux setsid may fork when its caller is already a process-group
            # leader. Without --wait the SSH channel can report success while the
            # actual GPU command is still starting, so callers race to download
            # output files that do not exist yet.
            command = "setsid --wait bash -lc " + shlex.quote(inner)
        channel.exec_command(command)
        if stdin_text is not None:
            channel.sendall(stdin_text.encode("utf-8"))
            channel.shutdown_write()
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
export OMP_NUM_THREADS=4
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
if [ ! -f {remote}/.worker-ready-v3 ]; then
  if [ ! -x {remote}/.venv/bin/python ]; then python3 -m venv --system-site-packages {remote}/.venv; fi
  {remote}/.venv/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --upgrade pip \
    || {remote}/.venv/bin/python -m pip install -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com --upgrade pip
  {remote}/.venv/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn numpy soundfile openai-whisper faster-whisper transformers \
    || {remote}/.venv/bin/python -m pip install -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com numpy soundfile openai-whisper faster-whisper transformers
fi
{remote}/.venv/bin/python -c 'import torch, whisper; print("torch=" + torch.__version__); print("cuda=" + str(torch.cuda.is_available())); assert torch.cuda.is_available(), "云节点 PyTorch 未启用 CUDA，请更换带 CUDA/PyTorch 的 GPU 镜像"'
touch {remote}/.worker-ready-v3
"""
        output = self._exec("bash -lc " + shlex.quote(script), timeout=1800)
        return {"output": output.strip(), "remote_dir": self.settings.remote_dir}

    def bootstrap_accuracy(self) -> dict[str, str]:
        if not self.client:
            self.connect()
        remote = shlex.quote(self.settings.remote_dir)
        models = shlex.quote(self.settings.model_dir)
        token_setup = ""
        stdin_text = None
        if self.settings.huggingface_token:
            token_setup = 'IFS= read -r HF_TOKEN; export HF_TOKEN\n'
            stdin_text = self.settings.huggingface_token + "\n"
        script = f"""set -euo pipefail
{token_setup}export HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=4
models_root={models}
mkdir -p "$models_root"
exec 9>"$models_root/.accuracy-install.lock"
if ! flock -w 7200 9; then
  echo "Timed out waiting for the accuracy-model installer lock" >&2
  exit 25
fi
validate_transformer() {{
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not (root / "config.json").is_file():
    raise SystemExit(1)
direct = (root / "model.safetensors", root / "pytorch_model.bin")
if any(path.is_file() and path.stat().st_size > 0 for path in direct):
    raise SystemExit(0)
indexes = (root / "model.safetensors.index.json", root / "pytorch_model.bin.index.json")
for index in indexes:
    if not index.is_file():
        continue
    try:
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {{}})
        shards = {{str(name) for name in weight_map.values()}}
    except (OSError, ValueError, TypeError):
        raise SystemExit(1)
    if shards and all((root / name).is_file() and (root / name).stat().st_size > 0 for name in shards):
        raise SystemExit(0)
raise SystemExit(1)
PY
}}
accuracy_models_complete() {{
  validate_transformer "$models_root/weights/Qwen3-ASR-1.7B" \
    && validate_transformer "$models_root/weights/Qwen3-ForcedAligner-0.6B" \
    && validate_transformer "$models_root/weights/cohere-transcribe-03-2026" \
    && [ -s "$models_root/weights/faster-whisper-large-v3/model.bin" ]
}}
if [ -f "$models_root/.accuracy-ready-v4" ] && accuracy_models_complete; then
  echo "Accuracy ensemble already installed and shard-verified in $models_root"
  exit 0
fi
gpu_available=0
if python3 -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
  gpu_available=1
fi
if [ "$gpu_available" -eq 0 ] && [ -f "$models_root/.accuracy-downloaded-v4" ] && accuracy_models_complete; then
  echo "ACCURACY_MODELS_DOWNLOADED_NO_GPU"
  echo "Accuracy models are fully downloaded; GPU load verification is deferred until GPU startup"
  exit 0
fi
rm -f "$models_root/.accuracy-ready-v3" "$models_root/.accuracy-ready-v4" "$models_root/.accuracy-downloaded-v4"
mkdir -p "$models_root/weights"
available_kb=$(df -Pk {models} | awk 'NR==2 {{print $4}}')
existing_kb=$(du -sk "$models_root/weights" 2>/dev/null | awk '{{print $1}}')
minimum_kb=36700160
if [ "${{existing_kb:-0}}" -ge 1048576 ]; then minimum_kb=8388608; fi
if [ -z "$available_kb" ] || [ "$available_kb" -lt "$minimum_kb" ]; then
  echo "Accuracy models do not have enough repair/download space in $models_root; required_kb=$minimum_kb available_kb=${{available_kb:-unknown}}" >&2
  exit 23
fi
mkdir -p "$models_root/envs" "$models_root/weights" "$models_root/manifests" "$models_root/tmp"
export TMPDIR="$models_root/tmp"
if [ ! -x "$models_root/envs/qwen/bin/python" ]; then
  python3 -m venv --system-site-packages "$models_root/envs/qwen"
fi
if [ ! -x "$models_root/envs/cohere/bin/python" ]; then
  python3 -m venv --system-site-packages "$models_root/envs/cohere"
fi
(
  "$models_root/envs/qwen/bin/python" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn -U pip \
    || "$models_root/envs/qwen/bin/python" -m pip install -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com -U pip
  "$models_root/envs/qwen/bin/python" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn -U qwen-asr modelscope huggingface_hub hf_xet soundfile nagisa soynlp \
    || "$models_root/envs/qwen/bin/python" -m pip install -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com -U qwen-asr modelscope huggingface_hub hf_xet soundfile nagisa soynlp
) & qwen_pip=$!
(
  "$models_root/envs/cohere/bin/python" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn -U pip \
    || "$models_root/envs/cohere/bin/python" -m pip install -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com -U pip
  "$models_root/envs/cohere/bin/python" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn -U 'transformers>=5.4.0' accelerate sentencepiece protobuf soundfile librosa huggingface_hub hf_xet modelscope \
    || "$models_root/envs/cohere/bin/python" -m pip install -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com -U 'transformers>=5.4.0' accelerate sentencepiece protobuf soundfile librosa huggingface_hub hf_xet modelscope
) & cohere_pip=$!
set +e
wait "$qwen_pip"; qwen_pip_status=$?
wait "$cohere_pip"; cohere_pip_status=$?
set -e
if [ "$qwen_pip_status" -ne 0 ] || [ "$cohere_pip_status" -ne 0 ]; then
  echo "Python dependency installation failed: qwen=$qwen_pip_status cohere=$cohere_pip_status" >&2
  exit 27
fi
download_qwen_model() {{
  repo="$1"
  target="$2"
  "$models_root/envs/qwen/bin/modelscope" download --model "$repo" --local_dir "$target" || true
  if ! validate_transformer "$target"; then
    HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=600 \
      "$models_root/envs/qwen/bin/hf" download "$repo" --local-dir "$target"
  fi
  validate_transformer "$target"
}}
download_cohere_model() {{
  target="$models_root/weights/cohere-transcribe-03-2026"
  "$models_root/envs/cohere/bin/modelscope" download --model CohereLabs/cohere-transcribe-03-2026 --local_dir "$target" || true
  if ! validate_transformer "$target"; then
    HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=600 \
      "$models_root/envs/cohere/bin/hf" download CohereLabs/cohere-transcribe-03-2026 --local-dir "$target"
  fi
  validate_transformer "$target"
}}
(
  validate_transformer "$models_root/weights/Qwen3-ASR-1.7B" \
    || download_qwen_model Qwen/Qwen3-ASR-1.7B "$models_root/weights/Qwen3-ASR-1.7B"
) & qwen_model=$!
(
  validate_transformer "$models_root/weights/Qwen3-ForcedAligner-0.6B" \
    || download_qwen_model Qwen/Qwen3-ForcedAligner-0.6B "$models_root/weights/Qwen3-ForcedAligner-0.6B"
) & align_model=$!
(
  validate_transformer "$models_root/weights/cohere-transcribe-03-2026" || download_cohere_model
) & cohere_model=$!
set +e
wait "$qwen_model"; qwen_status=$?
wait "$align_model"; align_status=$?
wait "$cohere_model"; cohere_status=$?
set -e
if [ "$qwen_status" -ne 0 ] || [ "$align_status" -ne 0 ] || [ "$cohere_status" -ne 0 ]; then
  echo "Model repair failed: qwen=$qwen_status aligner=$align_status cohere=$cohere_status. Cohere may require accepted model terms and an authenticated Hugging Face token." >&2
  exit 24
fi
if [ ! -s "$models_root/weights/faster-whisper-large-v3/model.bin" ]; then
  "$models_root/envs/qwen/bin/modelscope" download --model keepitsimple/faster-whisper-large-v3 --local_dir "$models_root/weights/faster-whisper-large-v3"
fi
accuracy_models_complete || {{ echo "Model shard validation failed after repair" >&2; exit 26; }}
for name in Qwen3-ASR-1.7B Qwen3-ForcedAligner-0.6B cohere-transcribe-03-2026; do
  (cd "$models_root/weights/$name" && find . -type f -not -path './.git/*' -print0 | sort -z | xargs -0 sha256sum > "$models_root/manifests/$name.sha256")
done
(cd "$models_root/weights/faster-whisper-large-v3" && find . -type f -not -path './.git/*' -print0 | sort -z | xargs -0 sha256sum > "$models_root/manifests/faster-whisper-large-v3.sha256")
touch "$models_root/.accuracy-downloaded-v4"
if [ "$gpu_available" -eq 0 ]; then
  echo "ACCURACY_MODELS_DOWNLOADED_NO_GPU"
  echo "All model files and shards passed validation; GPU load verification is deferred until GPU startup"
  df -h "$models_root"
  exit 0
fi
"$models_root/envs/qwen/bin/python" - "$models_root/weights/Qwen3-ASR-1.7B" <<'PY'
import sys
import torch
from qwen_asr import Qwen3ASRModel
model = Qwen3ASRModel.from_pretrained(sys.argv[1], dtype=torch.bfloat16, device_map="cuda:0", max_inference_batch_size=1, max_new_tokens=8)
del model
torch.cuda.empty_cache()
PY
"$models_root/envs/cohere/bin/python" - "$models_root/weights/cohere-transcribe-03-2026" <<'PY'
import sys
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
AutoProcessor.from_pretrained(sys.argv[1], local_files_only=True)
model = CohereAsrForConditionalGeneration.from_pretrained(sys.argv[1], local_files_only=True, dtype=torch.bfloat16, device_map="cuda:0")
del model
torch.cuda.empty_cache()
PY
touch "$models_root/.accuracy-ready-v4"
df -h "$models_root"
"""
        output = self._exec(
            "bash -lc " + shlex.quote(script), timeout=7200, stdin_text=stdin_text
        )
        return {
            "output": output.strip(),
            "remote_dir": self.settings.remote_dir,
            "model_dir": self.settings.model_dir,
            "gpu_verified": "ACCURACY_MODELS_DOWNLOADED_NO_GPU" not in output,
        }

    def _mkdir(self, remote_path: str):
        self._exec("mkdir -p " + shlex.quote(remote_path), timeout=30)

    def set_job_dir(self, job_id: str, *, migrate_legacy: bool = True):
        """Put uploaded audio and all per-job intermediates on the data disk."""
        self.remote_job_dir = posixpath.join(self.settings.data_dir, job_id)
        self.legacy_job_dir = posixpath.join(self.settings.remote_dir, "jobs", job_id)
        if not migrate_legacy:
            return
        target_parent = shlex.quote(self.settings.data_dir)
        target = shlex.quote(self.remote_job_dir)
        legacy = shlex.quote(self.legacy_job_dir)
        output = self._exec(
            f"mkdir -p {target_parent}; "
            f"if [ ! -e {target} ] && [ -d {legacy} ]; then "
            f"mv -- {legacy} {target}; echo migrated-legacy-job; "
            "fi; "
            f"mkdir -p {target}; df -Pk {target} | tail -n 1",
            timeout=600,
        )
        if "migrated-legacy-job" in output:
            self.logger("已将旧版系统盘预上传任务迁移到云端数据盘")

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

    def _local_file_info(self, local_path: Path) -> tuple[int, str]:
        size = local_path.stat().st_size
        digest = hashlib.sha256()
        with local_path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                self.checkpoint()
                digest.update(chunk)
        return size, digest.hexdigest()

    def _local_prefix_sha256(self, local_path: Path, length: int) -> str:
        remaining = length
        digest = hashlib.sha256()
        with local_path.open("rb") as stream:
            while remaining:
                self.checkpoint()
                chunk = stream.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise CloudWorkerError("本地音轨长度在上传期间发生变化")
                digest.update(chunk)
                remaining -= len(chunk)
        return digest.hexdigest()

    def _remote_file_info(self, remote_path: str) -> tuple[int, str] | None:
        quoted = shlex.quote(remote_path)
        output = self._exec(
            f"if [ -f {quoted} ]; then stat -c %s {quoted}; sha256sum {quoted} | cut -d' ' -f1; fi",
            timeout=900,
        )
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return None
        if len(lines) != 2 or not lines[0].isdigit() or not re.fullmatch(r"[0-9a-f]{64}", lines[1]):
            raise CloudWorkerError("云端文件校验结果格式异常")
        return int(lines[0]), lines[1]

    def _upload_resumable(self, local_path: Path, remote_path: str, label: str):
        if not self.sftp:
            raise CloudWorkerError("云节点文件通道尚未连接")
        size = local_path.stat().st_size
        try:
            remote_size = self.sftp.stat(remote_path).st_size
        except FileNotFoundError:
            remote_size = 0
        if remote_size > size:
            self._exec("rm -f -- " + shlex.quote(remote_path), timeout=30)
            remote_size = 0
        elif remote_size:
            remote_info = self._remote_file_info(remote_path)
            local_prefix = self._local_prefix_sha256(local_path, remote_size)
            if remote_info != (remote_size, local_prefix):
                self.logger("云端临时分片前缀校验失败，丢弃后从头重传")
                self._exec("rm -f -- " + shlex.quote(remote_path), timeout=30)
                remote_size = 0
            elif remote_size < size:
                self.logger(f"检测到完整临时分片，从 {remote_size / 1024 / 1024:.1f} MB 处断点续传")

        sent = remote_size
        last_percent = int(sent * 100 / max(1, size)) - 10
        with local_path.open("rb") as source:
            source.seek(remote_size)
            with self.sftp.file(remote_path, "ab" if remote_size else "wb") as target:
                target.set_pipelined(True)
                while chunk := source.read(1024 * 1024):
                    self.checkpoint()
                    target.write(chunk)
                    sent += len(chunk)
                    percent = int(sent * 100 / max(1, size))
                    if percent >= last_percent + 5 or percent == 100:
                        last_percent = percent
                        self.logger(f"{label} {percent}%")

    def _reconnect(self):
        self.close()
        time.sleep(1)
        self.connect()

    def _acquire_upload_slot(self):
        self.logger(f"等待上传通道 · 最多并行 {CLOUD_UPLOAD_CONCURRENCY} 个任务")
        while not CLOUD_UPLOAD_SLOTS.acquire(timeout=0.25):
            self.checkpoint()
        self.logger("已取得上传通道")

    def _ensure_verified_audio(
        self, audio_path: Path, *, upload_slot_held: bool = False
    ) -> dict[str, object]:
        if not self.remote_job_dir:
            raise CloudWorkerError("尚未设置云端任务目录")
        remote_audio = posixpath.join(self.remote_job_dir, "audio.flac")
        remote_part = posixpath.join(self.remote_job_dir, "audio.flac.uploading")
        remote_manifest = posixpath.join(self.remote_job_dir, "audio.ready.json")
        local_size, local_sha256 = self._local_file_info(audio_path)
        expected = (local_size, local_sha256)
        if self._remote_file_info(remote_audio) == expected:
            self.logger(f"复用已校验云端音轨 · {local_size / 1024 / 1024:.1f} MB · SHA-256 {local_sha256[:12]}…")
            return {"size": local_size, "sha256": local_sha256, "reused": True}
        if not upload_slot_held:
            self._acquire_upload_slot()
        try:
            for attempt in range(1, 4):
                self.checkpoint()
                try:
                    self._upload_resumable(
                        audio_path,
                        remote_part,
                        f"上传音轨（连接尝试 {attempt}/3）",
                    )
                    remote_info = self._remote_file_info(remote_part)
                except (EOFError, OSError, _paramiko().SSHException) as exc:
                    if attempt == 3:
                        raise CloudWorkerError(
                            f"音轨上传连续 3 次连接中断：{type(exc).__name__}"
                        ) from exc
                    self.logger(
                        f"上传连接中断（{type(exc).__name__}），正在重连并从已校验分片继续"
                    )
                    self._reconnect()
                    continue
                if remote_info == expected:
                    manifest = json.dumps(
                        {"version": 1, "size": local_size, "sha256": local_sha256},
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                    manifest_part = remote_manifest + ".uploading"
                    command = (
                        f"mv -f -- {shlex.quote(remote_part)} {shlex.quote(remote_audio)}; "
                        f"printf '%s\\n' {shlex.quote(manifest)} > {shlex.quote(manifest_part)}; "
                        f"mv -f -- {shlex.quote(manifest_part)} {shlex.quote(remote_manifest)}"
                    )
                    self._exec(command, timeout=30)
                    self.logger(f"音轨校验通过 · {local_size / 1024 / 1024:.1f} MB · SHA-256 {local_sha256[:12]}…")
                    return {"size": local_size, "sha256": local_sha256, "reused": False}
                self.logger(f"音轨校验失败，第 {attempt}/3 次传输不完整，准备校验分片并续传")

            self._exec("rm -f -- " + shlex.quote(remote_part), timeout=30)
            raise CloudWorkerError("音轨连续 3 次未通过文件大小与 SHA-256 校验，已拒绝进入识别阶段")
        finally:
            if not upload_slot_held:
                CLOUD_UPLOAD_SLOTS.release()
                self.logger("已释放上传通道")

    def _download(self, remote_path: str, local_path: Path):
        if not self.sftp:
            raise CloudWorkerError("云节点文件通道尚未连接")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.sftp.get(remote_path, str(local_path))
        except FileNotFoundError as exc:
            raise CloudWorkerError(
                f"云节点没有生成预期结果文件：{remote_path}。"
                "远程运算可能提前退出，请查看该任务在此错误之前的云端日志"
            ) from exc

    def stage_job_audio(self, job_id: str, audio_path: Path) -> dict[str, object]:
        self._acquire_upload_slot()
        try:
            if not self.client:
                self.connect()
            self.set_job_dir(job_id)
            return self._ensure_verified_audio(audio_path, upload_slot_held=True)
        finally:
            CLOUD_UPLOAD_SLOTS.release()
            self.logger("已释放上传通道")

    def prepare_job(self, job_id: str, audio_path: Path, *, accuracy: bool = False):
        if not self.client:
            self.connect()
        self.set_job_dir(job_id)
        if self.settings.auto_setup:
            self.logger("检查并安装云节点运算依赖")
            if accuracy:
                self.bootstrap()
                self.bootstrap_accuracy()
            else:
                self.bootstrap()
        self._mkdir(posixpath.join(self.remote_job_dir, "studio"))
        for local, remote in (
            (ROOT / "asr_stage.py", posixpath.join(self.remote_job_dir, "asr_stage.py")),
            (ROOT / "large_review.py", posixpath.join(self.remote_job_dir, "large_review.py")),
            (ROOT / "audio_event_gate.py", posixpath.join(self.remote_job_dir, "audio_event_gate.py")),
            (ROOT / "studio" / "__init__.py", posixpath.join(self.remote_job_dir, "studio", "__init__.py")),
            (ROOT / "studio" / "languages.py", posixpath.join(self.remote_job_dir, "studio", "languages.py")),
        ):
            self._upload(local, remote, f"同步 {local.name}")
        self._ensure_verified_audio(audio_path)

        if accuracy:
            for name in (
                "ensemble_common.py",
                "prepare_audio_views.py",
                "prepare_ensemble_windows.py",
                "qwen_primary_stage.py",
                "cohere_review_stage.py",
                "merge_ensemble_reviews.py",
                "whisper_conflict_vote.py",
                "qwen_align_stage.py",
            ):
                self._upload(ROOT / name, posixpath.join(self.remote_job_dir, name), f"同步 {name}")

    def run_accuracy_asr(
        self,
        events_path: Path,
        local_workdir: Path,
        *,
        language: str,
        speech_threshold: float,
        nonlexical_factor: float,
    ):
        remote_work = posixpath.join(self.remote_job_dir, "accuracy_ensemble")
        self._mkdir(remote_work)
        remote_events = posixpath.join(remote_work, "events.json")
        audio_views = posixpath.join(remote_work, "audio_views")
        raw_audio = posixpath.join(audio_views, "raw_view.flac")
        enhanced_audio = posixpath.join(audio_views, "conservative_enhanced_view.flac")
        left_audio = posixpath.join(audio_views, "left_view.flac")
        right_audio = posixpath.join(audio_views, "right_view.flac")
        audio_report = posixpath.join(audio_views, "audio_view_report.json")
        windows = posixpath.join(remote_work, "windows.json")
        primary = posixpath.join(remote_work, "qwen_primary.json")
        cohere = posixpath.join(remote_work, "cohere_reviewed.json")
        reviewed = posixpath.join(remote_work, "ensemble_merged.json")
        voted = posixpath.join(remote_work, "ensemble_voted.json")
        audit = posixpath.join(remote_work, "ensemble_audit.json")
        final = posixpath.join(remote_work, "source_sentences.json")
        self._upload(events_path, remote_events, "上传多模型识别分段")
        models = self.settings.model_dir
        qwen_python = posixpath.join(models, "envs", "qwen", "bin", "python")
        cohere_python = posixpath.join(models, "envs", "cohere", "bin", "python")
        whisper_python = posixpath.join(self.settings.remote_dir, ".venv", "bin", "python")
        scripts = self.remote_job_dir
        audio = posixpath.join(self.remote_job_dir, "audio.flac")

        def command(args: list[str]) -> str:
            return "env OMP_NUM_THREADS=4 " + " ".join(shlex.quote(value) for value in args)

        self.logger("最高精度识别阶段 1/5：分析声道并生成原始/保守增强双视图")
        self._exec(command([
            whisper_python, posixpath.join(scripts, "prepare_audio_views.py"), audio,
            "--workdir", audio_views,
        ]), controllable=True)
        self._exec(command([
            qwen_python, posixpath.join(scripts, "prepare_ensemble_windows.py"), raw_audio,
            "--events", remote_events, "--output", windows,
        ]), controllable=True)

        qwen_command = command([
            qwen_python, posixpath.join(scripts, "qwen_primary_stage.py"), raw_audio,
            "--events", remote_events, "--windows", windows, "--output", primary,
            "--model", posixpath.join(models, "weights", "Qwen3-ASR-1.7B"),
            "--language", language, "--speech-threshold", str(speech_threshold),
            "--nonlexical-factor", str(nonlexical_factor), "--batch-size", "2",
        ])
        cohere_command = command([
            cohere_python, posixpath.join(scripts, "cohere_review_stage.py"), enhanced_audio,
            "--input", windows, "--output", cohere,
            "--model", posixpath.join(models, "weights", "cohere-transcribe-03-2026"),
            "--language", language, "--batch-size", "2", "--review-all",
        ])
        self.logger("最高精度识别阶段 2/5：Qwen 原始音频与 Cohere 增强音频并行识别")
        parallel = (
            f"set +e; ({qwen_command}) & qwen_pid=$!; ({cohere_command}) & cohere_pid=$!; "
            "wait $qwen_pid; qwen_status=$?; wait $cohere_pid; cohere_status=$?; "
            "if [ $qwen_status -ne 0 ] || [ $cohere_status -ne 0 ]; then "
            "echo \"parallel ASR failed: qwen=$qwen_status cohere=$cohere_status\" >&2; exit 31; fi"
        )
        self._exec(parallel, controllable=True)

        self.logger("最高精度识别阶段 3/5：合并双模型证据并标记冲突")
        self._exec(command([
            qwen_python, posixpath.join(scripts, "merge_ensemble_reviews.py"),
            "--qwen", primary, "--cohere", cohere, "--output", reviewed,
        ]), controllable=True)

        self.logger("最高精度识别阶段 4/5：large-v3 仅裁决冲突区间")
        self._exec(command([
            whisper_python, posixpath.join(scripts, "whisper_conflict_vote.py"), raw_audio,
            "--input", reviewed, "--output", voted, "--audit", audit,
            "--model", posixpath.join(models, "weights", "faster-whisper-large-v3"),
            "--language", language, "--left", left_audio, "--right", right_audio,
            "--audio-report", audio_report,
        ]), controllable=True)

        self.logger("最高精度识别阶段 5/5：Qwen ForcedAligner 恢复时间轴")
        self._exec(command([
            qwen_python, posixpath.join(scripts, "qwen_align_stage.py"), raw_audio,
            "--input", voted, "--output", final,
            "--model", posixpath.join(models, "weights", "Qwen3-ForcedAligner-0.6B"),
            "--language", language, "--batch-size", "8",
        ]), controllable=True)
        self._download(final, local_workdir / "source_sentences.json")
        self._download(audit, local_workdir / "ensemble_audit.json")
        self._download(
            posixpath.join(audio_views, "audio_view_report.json"),
            local_workdir / "audio_view_report.json",
        )

    def run_event_gate(self, vad_path: Path, local_events_path: Path):
        remote_work = posixpath.join(self.remote_job_dir, "event_gate")
        self._mkdir(remote_work)
        remote_vad = posixpath.join(remote_work, "vad.json")
        remote_events = posixpath.join(remote_work, "events.json")
        self._upload(vad_path, remote_vad, "上传 VAD 分段")
        python = posixpath.join(self.settings.remote_dir, ".venv", "bin", "python")
        args = [
            "env", "OMP_NUM_THREADS=4",
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
            "env", "OMP_NUM_THREADS=4",
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
            "env", "OMP_NUM_THREADS=4",
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
            targets = [self.remote_job_dir]
            if self.legacy_job_dir and self.legacy_job_dir != self.remote_job_dir:
                targets.append(self.legacy_job_dir)
            self._exec(
                "rm -rf -- " + " ".join(shlex.quote(path) for path in targets),
                timeout=60,
            )
            self.remote_job_dir = ""
            self.legacy_job_dir = ""
