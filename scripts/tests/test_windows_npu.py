"""Standard-library tests of the Handy Windows NPU protocol; no hardware needed."""
import importlib.util
import io
import struct
import unittest
from pathlib import Path


WORKER_FILE = Path(__file__).resolve().parents[2] / "src-tauri" / "src" / "windows_npu_worker.py"
spec = importlib.util.spec_from_file_location("windows_npu_worker", WORKER_FILE)
npu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(npu)


class FakeResult:
    texts = ["recognized text"]


class FakePipe:
    def __init__(self):
        self.audio = None
        self.kwargs = None

    def generate(self, audio, **kwargs):
        self.audio = list(audio)
        self.kwargs = kwargs
        return FakeResult()


class WindowsNpuProtocolTests(unittest.TestCase):
    def test_pcm_and_language(self):
        pipe = FakePipe()
        reader = io.BytesIO(struct.pack("<3f", 0.0, 0.125, -0.25))
        result = npu.process_request(
            pipe, reader, {"samples": 3, "language": "<|en|>", "translate": False}
        )
        self.assertEqual(result, {"ok": True, "text": "recognized text"})
        self.assertEqual(pipe.audio, [0.0, 0.125, -0.25])
        self.assertEqual(pipe.kwargs, {"task": "transcribe", "language": "<|en|>"})

    def test_translation(self):
        pipe = FakePipe()
        reader = io.BytesIO(struct.pack("<f", 0.0))
        npu.process_request(pipe, reader, {"samples": 1, "translate": True})
        self.assertEqual(pipe.kwargs, {"task": "translate"})

    def test_reject_truncated_pcm(self):
        with self.assertRaises(EOFError):
            npu.process_request(FakePipe(), io.BytesIO(b"\x00"), {"samples": 2})

    def test_reject_bad_length_without_reading(self):
        for length in [0, -1, True, "1", npu.MAX_SAMPLES + 1]:
            with self.subTest(length=length), self.assertRaises(ValueError):
                npu.process_request(FakePipe(), io.BytesIO(b""), {"samples": length})

    def test_reject_out_of_range_audio(self):
        with self.assertRaises(ValueError):
            npu.process_request(FakePipe(), io.BytesIO(struct.pack("<f", 9.0)), {"samples": 1})

    def test_reject_invalid_language(self):
        with self.assertRaises(ValueError):
            npu.process_request(
                FakePipe(), io.BytesIO(struct.pack("<f", 0)),
                {"samples": 1, "language": "<|auto|>"}
            )


if __name__ == "__main__":
    unittest.main()
