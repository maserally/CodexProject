import contextlib
import io
import tempfile
import threading
import unittest
from pathlib import Path

from relay_speed_test import delete_remote, download_file, make_server, upload_file


class RelaySpeedTestTests(unittest.TestCase):
    def test_upload_download_checksum_and_delete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            storage = root / "storage"
            source = root / "sample.bin"
            output = root / "downloaded.bin"
            source.write_bytes(bytes(range(256)) * 4096)
            token = "temporary-test-token-1234"
            server = make_server("127.0.0.1", 0, storage, token, 2 * 1024 * 1024)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            url = f"http://127.0.0.1:{server.server_port}"
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    uploaded = upload_file(url, source, token)
                    downloaded = download_file(url, source.name, output, token)
                    deleted = delete_remote(url, source.name, token)
                self.assertEqual(uploaded["sha256"], downloaded["sha256"])
                self.assertEqual(output.read_bytes(), source.read_bytes())
                self.assertTrue(deleted["deleted"])
                self.assertFalse((storage / source.name).exists())
            finally:
                server.shutdown()
                server.server_close()
                worker.join(2)

    def test_wrong_token_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "sample.bin"
            source.write_bytes(b"test")
            server = make_server(
                "127.0.0.1", 0, root / "storage", "correct-token-123456", 1024
            )
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(
                    RuntimeError
                ):
                    upload_file(
                        f"http://127.0.0.1:{server.server_port}",
                        source,
                        "wrong-token-1234567",
                    )
            finally:
                server.shutdown()
                server.server_close()
                worker.join(2)


if __name__ == "__main__":
    unittest.main()
