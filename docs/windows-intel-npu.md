# Windows Intel NPU support (experimental)

This branch adds **real Intel NPU execution**, not a renamed GPU mode.
Handy's original native transcribe-cpp engine remains the fallback on any failure.

## Scope

- Windows 11 **x86_64** with a **supported Intel NPU** (typically Core Ultra with Intel AI Boost), compatible OpenVINO driver and Python.
- Uses OpenVINO GenAI's WhisperPipeline with explicit device `"NPU"`. The sidecar *refuses* to pretend CPU/GPU execution is NPU execution.
- Applies to Handy's **Whisper-family non-streaming** transcription only, and only when its existing custom-word prompts are empty. Other engines and live streaming stay unchanged.
- This is **opt-in** via process environment variables, not a graphical setting yet. This is an engineering preview, **not a validated consumer release**.
- Existing GGUF/ggml files are **not** compatible with OpenVINO. Export or download a separate OpenVINO Whisper model.
- This first integration is **Intel NPU only**. AMD Ryzen AI, Qualcomm Hexagon, NVIDIA RTX GPUs and Apple M4 ANE are NOT supported.
- No cloud transcription: PCM stays on the local system and moves only between Handy and a local Python child process.

## Setup

1. In Windows **Task Manager > Performance** confirm an NPU is present, then install/up-to-date drivers. To check OpenVINO specifically run:
   ```powershell
   py -3 -c "import openvino; print(openvino.Core().available_devices)"
   ```
   The output MUST contain `NPU`. Run this check **after** the OpenVINO install below, using the virtual environment's Python. If your PC has no physical NPU, this cannot create one.

2. Install Python 3.11/3.12 x64, then create an isolated environment:
   ```powershell
   py -3 -m venv "$env:USERPROFILE\handy-npu-venv"
   & "$env:USERPROFILE\handy-npu-venv\Scripts\python.exe" -m pip install --upgrade pip
   & "$env:USERPROFILE\handy-npu-venv\Scripts\python.exe" -m pip install "openvino>=2025.1" "openvino-genai>=2025.1" "optimum-intel[openvino]"
   ```

3. Export a Whisper model into OpenVINO format (one-time; conversion downloads model weights):
   ```powershell
   & "$env:USERPROFILE\handy-npu-venv\Scripts\optimum-cli.exe" export openvino --model openai/whisper-base "$env:USERPROFILE\handy-whisper-base-ov"
   ```
   Conversion and Windows/Python version compatibility may vary. See [OpenVINO's Whisper NPU guide](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html).

4. Start the **branch's** Windows Handy build with the environment variables set:
   ```powershell
   $env:HANDY_OPENVINO_PYTHON = "$env:USERPROFILE\handy-npu-venv\Scripts\python.exe"
   $env:HANDY_OPENVINO_NPU_MODEL_DIR = "$env:USERPROFILE\handy-whisper-base-ov"
   & "C:\Path\To\Handy.exe"
   ```
   Launch from this same terminal so Handy inherits the variables. If launched normally without the variables, Handy works exactly as before.

5. Select a **Whisper** model in Handy. Transcription goes through OpenVINO's NPU where supported. Check Handy's logs for:
   `Transcribed through OpenVINO Whisper on Intel NPU`.
   When disabled/failing, a warning records the fallback. Task Manager's NPU utilization can offer an additional hardware signal.

## Engineering behavior

- OpenVINO is imported lazily only when explicitly enabled. Handy's main installer does NOT require Python or an NPU to run normally.
- A persistent Python process holds the compiled Whisper NPU model across transcriptions. Requests use a newline-delimited JSON header and little-endian float32 PCM audio over stdin; responses are structured JSON over stdout.
- NPU initialization requires `"NPU" in openvino.Core().available_devices`; model execution requests `WhisperPipeline(model_dir, "NPU")`.
- Audio is mono, normalized float32, 16 kHz (Handy's transcription PCM format).
- Timeouts/errors fall back to Handy's already-loaded transcribe-cpp model. After an unrecoverable startup error or timeout the NPU path is disabled for that Handy session. On per-inference error the worker is restarted on next call.
- Inputs above 10 minutes, inputs with NaNs/infinities, and custom-word prompts use the native path.
- If the Python process locks up, the 120-second client timeout will return to native transcription, but the stuck subprocess may remain until Handy exits.
- **Limitations:** first-run compilation can be slow. Language autodetection metadata, streaming overlays, backend status in Settings, and custom-word prompts aren't implemented for this backend. The sidecar loads a potentially different Whisper model from Handy's selected native model. First use may block transcription while model compiles.
- When using a CPU fallback in Handy you must still install/select a normal Handy Whisper model; this is NOT a stand-alone NPU transcription installation.

## Test commands

- `python -m unittest discover -s scripts/tests -p "test_windows_npu.py"` (no NPU required)
- On Windows, `cd src-tauri; cargo test --lib whisper_languages_are_validated`
- With a supported Intel NPU, actually record/transcribe short English and non-English samples, compare outputs with native Handy, force NPU error to verify fallback, and verify NPU utilization. This hardware validation remains necessary before calling a downloadable installer a *tested NPU release*.

## Known upstream/driver concerns

OpenVINO documents NPU support for Whisper Tiny/Base/Small/Large, but exact driver/runtime/model combinations can fail. In particular OpenVINO 2026.2.x had NPU Whisper inference failures on Lunar Lake in [issue #4222](https://github.com/openvinotoolkit/openvino.genai/issues/4222), and OpenVINO 2026.3.0 has [long-form state concerns](https://github.com/openvinotoolkit/openvino/issues/37937). Test your driver and model, especially >30s speech.

References: [OpenVINO NPU Whisper](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html), [WhisperPipeline API](https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.WhisperPipeline.html), [Windows ML EP catalog](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers).
