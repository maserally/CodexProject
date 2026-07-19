import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from studio.main import _choose_local_folder, _video_files_in, app, create_folder_jobs
from studio.schemas import FolderBatchRequest, JobOptions


class BatchFolderTests(unittest.TestCase):
    def test_folder_scan_can_read_supported_top_level_videos_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "B.MKV").write_bytes(b"")
            (root / "a.mp4").write_bytes(b"")
            (root / "note.txt").write_text("ignore", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "hidden.mp4").write_bytes(b"")
            resolved, files = _video_files_in(str(root), recursive=False)
            self.assertEqual(resolved, root.resolve())
            self.assertEqual([path.name for path in files], ["a.mp4", "B.MKV"])

    def test_folder_scan_recurses_by_default_and_sorts_relative_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "top.mp4").write_bytes(b"")
            nested = root / "合集" / "作品"
            nested.mkdir(parents=True)
            (nested / "movie.mkv").write_bytes(b"")
            _, files = _video_files_in(str(root))
            self.assertEqual(
                [str(path.relative_to(root)) for path in files],
                [str(Path("top.mp4")), str(Path("合集") / "作品" / "movie.mkv")],
            )

    def test_batch_creates_one_job_per_video_with_shared_output_dir(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            (source / "one.mp4").write_bytes(b"")
            (source / "one.mkv").write_bytes(b"")
            (source / "two.mov").write_bytes(b"")
            captured = []

            def fake_create(options, worker):
                captured.append(options)
                return SimpleNamespace(public=lambda: {"input": options.input_path})

            request = FolderBatchRequest(
                input_dir=str(source),
                output_dir=str(output),
                options=JobOptions(input_path=""),
            )
            with patch("studio.main.manager.create", side_effect=fake_create), patch(
                "studio.main.load_provider_settings", return_value={"cloud_worker": {}}
            ):
                result = create_folder_jobs(request)
            self.assertEqual(result["count"], 3)
            self.assertTrue(output.is_dir())
            self.assertTrue(all(item.output_dir == str(output.resolve()) for item in captured))
            self.assertEqual(
                [Path(item.input_path).name for item in captured],
                ["one.mkv", "one.mp4", "two.mov"],
            )
            self.assertEqual(
                [item.output_name for item in captured],
                ["one", "one_mp4", "two"],
            )

    def test_batch_creates_jobs_for_selected_files_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            nested = source / "nested"
            nested.mkdir(parents=True)
            (source / "skip.mp4").write_bytes(b"")
            (nested / "keep.mp4").write_bytes(b"")
            captured = []

            def fake_create(options, worker):
                captured.append(options)
                return SimpleNamespace(public=lambda: {"input": options.input_path})

            request = FolderBatchRequest(
                input_dir=str(source),
                selected_files=[str(Path("nested") / "keep.mp4")],
                options=JobOptions(input_path=""),
            )
            with patch("studio.main.manager.create", side_effect=fake_create), patch(
                "studio.main.load_provider_settings", return_value={"cloud_worker": {}}
            ):
                result = create_folder_jobs(request)

            self.assertEqual(result["count"], 1)
            self.assertEqual(Path(captured[0].input_path).name, "keep.mp4")

    def test_batch_rejects_empty_or_stale_file_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "one.mp4").write_bytes(b"")
            empty_request = FolderBatchRequest(
                input_dir=str(root),
                selected_files=[],
                options=JobOptions(input_path=""),
            )
            with self.assertRaises(HTTPException) as empty_error:
                create_folder_jobs(empty_request)
            self.assertEqual(empty_error.exception.status_code, 400)

            stale_request = FolderBatchRequest(
                input_dir=str(root),
                selected_files=["missing.mp4"],
                options=JobOptions(input_path=""),
            )
            with self.assertRaises(HTTPException) as stale_error:
                create_folder_jobs(stale_request)
            self.assertEqual(stale_error.exception.status_code, 400)

    def test_batch_rejects_output_equal_to_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "one.mp4").write_bytes(b"")
            request = FolderBatchRequest(
                input_dir=str(root),
                output_dir=str(root),
                options=JobOptions(input_path=""),
            )
            with self.assertRaises(HTTPException) as raised:
                create_folder_jobs(request)
            self.assertEqual(raised.exception.status_code, 400)

    def test_recursive_batch_preserves_subfolders_and_same_names(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            output = root / "output"
            first = source / "第一部"
            second = source / "第二部"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "movie.mp4").write_bytes(b"")
            (second / "movie.mp4").write_bytes(b"")
            captured = []

            def fake_create(options, worker):
                captured.append(options)
                return SimpleNamespace(public=lambda: {"input": options.input_path})

            request = FolderBatchRequest(
                input_dir=str(source),
                output_dir=str(output),
                options=JobOptions(input_path=""),
            )
            with patch("studio.main.manager.create", side_effect=fake_create), patch(
                "studio.main.load_provider_settings", return_value={"cloud_worker": {}}
            ):
                result = create_folder_jobs(request)

            self.assertEqual(result["count"], 2)
            self.assertEqual([item.output_name for item in captured], ["movie", "movie"])
            self.assertEqual(
                [Path(item.output_dir) for item in captured],
                [output.resolve() / "第一部", output.resolve() / "第二部"],
            )

    def test_batch_rejects_output_nested_under_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "one.mp4").write_bytes(b"")
            request = FolderBatchRequest(
                input_dir=str(root),
                output_dir=str(root / "outputs"),
                options=JobOptions(input_path=""),
            )
            with self.assertRaises(HTTPException) as raised:
                create_folder_jobs(request)
            self.assertEqual(raised.exception.status_code, 400)

    def test_batch_routes_are_registered(self):
        paths = {route.path for route in app.routes}
        self.assertIn("/api/media/folder", paths)
        self.assertIn("/api/jobs/batch", paths)
        self.assertIn("/api/local/pick-folder", paths)

    def test_folder_picker_is_blocked_outside_local_mode(self):
        with patch("studio.main.ALLOW_LOCAL_OPEN", False), self.assertRaises(
            HTTPException
        ) as raised:
            _choose_local_folder()
        self.assertEqual(raised.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
