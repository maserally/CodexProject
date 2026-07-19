import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from studio.cloud_worker import CloudWorkerError, _validated
from studio.main import app
from studio.runner import JobState
from studio.schemas import CloudWorkerSettings, JobOptions
from studio.settings_store import load_provider_settings, save_provider_settings


class CloudWorkerTests(unittest.TestCase):
    def test_worker_validation_requires_safe_connection_settings(self):
        valid = _validated(
            CloudWorkerSettings(
                enabled=True,
                host="gpu.example.com",
                port=2222,
                username="root",
                password="secret",
                remote_dir="/root/subtitle-worker",
            )
        )
        self.assertEqual(valid.port, 2222)
        with self.assertRaises(CloudWorkerError):
            _validated(
                CloudWorkerSettings(
                    enabled=True,
                    host="gpu.example.com; reboot",
                    username="root",
                    password="secret",
                )
            )

    def test_worker_credentials_never_enter_task_status(self):
        worker = CloudWorkerSettings(
            enabled=True,
            host="private-host.example",
            username="private-user",
            password="private-password",
        )
        public = JobState(
            id="cloud-job",
            options=JobOptions(input_path="movie.mp4"),
            cloud_worker_settings=worker,
        ).public()
        serialized = str(public)
        self.assertNotIn("private-host", serialized)
        self.assertNotIn("private-user", serialized)
        self.assertNotIn("private-password", serialized)

    @unittest.skipUnless(os.name == "nt", "DPAPI is available on Windows only")
    def test_worker_password_is_dpapi_encrypted_in_local_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "provider_settings.json"
            with patch("studio.settings_store.SETTINGS_PATH", path):
                save_provider_settings(
                    {
                        "cloud_worker": {
                            "enabled": True,
                            "host": "gpu.example.com",
                            "port": 22,
                            "username": "root",
                            "password": "worker-secret",
                            "remote_dir": "/root/subtitle-worker",
                        }
                    }
                )
                self.assertNotIn("worker-secret", path.read_text(encoding="utf-8"))
                loaded = load_provider_settings()
                self.assertEqual(loaded["cloud_worker"]["password"], "worker-secret")

    def test_cloud_worker_routes_are_registered(self):
        paths = {route.path for route in app.routes}
        self.assertIn("/api/cloud-worker/test", paths)
        self.assertIn("/api/cloud-worker/bootstrap", paths)


if __name__ == "__main__":
    unittest.main()
