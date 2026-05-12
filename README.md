# ane-transformer-optimization

Rules, tooling, and a worked example for refactoring transformer-style models toward layouts that map well to Apple’s **Neural Engine (ANE)** and **Core ML** (`mlprogram`).

---

## Skill package (`ane-transformer-optimization/`)

Use this folder as a **agent skill** (or copy it into your skills path) so the model gets ANE-oriented guidance on layout, attention, masks, LayerNorm, export pitfalls, and validation.

| Path | Role |
|------|------|
| [`ane-transformer-optimization/SKILL.md`](ane-transformer-optimization/SKILL.md) | Rules, decision tables, Core ML pitfalls, production validation workflow |
| [`ane-transformer-optimization/references/ane_modules.py`](ane-transformer-optimization/references/ane_modules.py) | Optional copy-paste PyTorch helpers (BC1S attention, FFN, LayerNorm, KV patterns) |
| [`ane-transformer-optimization/tools/`](ane-transformer-optimization/tools/) | CLI utilities (run with `--help`) |

**Tools**

| Script | Purpose |
|--------|---------|
| `check_structure.py` | AST scan: `Linear` in hot paths, hand-rolled LayerNorm, unsafe `eps`, bool masks, SDPA, layout churn |
| `verify_numerical.py` | PyTorch **original vs ANE-refactored** parity (multi-output, PSNR / SNR / cosine, JSON report, threshold exit codes) |
| `export_coreml.py` | TorchScript → `.mlpackage` (fp16/fp32, int32 token inputs) |
| `verify_coreml.py` | **`ml`**: baseline vs ANE `.mlpackage` vs PyTorch reference, latency on `CPU_AND_NE`, optional real-input block, JSON + Markdown |
| `benchmark.py` | Single-model latency for `.mlpackage` or TorchScript |

**Using the skill in Agents**

Install or symlink this repo (or the `ane-transformer-optimization` subtree) where your agent loads skills, then ask the agent to follow the skill when refactoring or exporting transformers for Apple silicon.

---

## Example: Whisper (`examples/whisper-ane/`)

End-to-end port of **OpenAI Whisper** to an ANE-friendly PyTorch layout (`WhisperANE`), separate **encoder** and **decoder** Core ML packages, parity checks against upstream PyTorch (with `disable_sdpa()`), baseline vs ANE benchmark, and optional **`jfk.flac`** transcription compare.

| Path | Role |
|------|------|
| [`examples/whisper-ane/Whisper_ANE_Export.md`](examples/whisper-ane/Whisper_ANE_Export.md) | Export layout, scripts, commands (Whisper-focused, no skill prose) |
| [`examples/whisper-ane/model_ane.py`](examples/whisper-ane/model_ane.py) | `WhisperANE`, `from_whisper`, export wrappers, `trace_with_roundtrip` |
| [`examples/whisper-ane/export_and_verify.py`](examples/whisper-ane/export_and_verify.py) | Load checkpoint, PyTorch parity, optional Core ML export + smoke `predict` |
| [`examples/whisper-ane/benchmark_coreml.py`](examples/whisper-ane/benchmark_coreml.py) | Export **baseline** (original) vs **ANE** enc/dec, accuracy vs PyTorch, latency, optional real-audio `transcribe` |
| [`examples/whisper-ane/coreml_whisper_adapter.py`](examples/whisper-ane/coreml_whisper_adapter.py) | `CoreMLWhisper`: stock `transcribe` with Core ML encoder + flex decoder |
| [`examples/whisper-ane/validate_skill_tools.py`](examples/whisper-ane/validate_skill_tools.py) | Optional: run the skill `tools/` against this tree (defaults aligned with `benchmark_coreml.py` for `small` + fp16) |

Upstream Whisper is expected as a clone at [`examples/whisper-ane/whisper/`](examples/whisper-ane/whisper/) (local package on `PYTHONPATH`).

### Setup

```bash
cd examples/whisper-ane
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch numpy coremltools
# If whisper/ is not present yet:
git clone https://github.com/openai/whisper.git
pip install -r whisper/requirements.txt
```

### Common commands

```bash
cd examples/whisper-ane
.venv/bin/python export_and_verify.py --model tiny -o coreml_out
.venv/bin/python benchmark_coreml.py --model small -o .output/coreml_benchmark_out_small
```

ANE/Core ML timing and placement are meaningful on **Apple silicon** with `coremltools` and on-device `predict`.

---

## License

No license file included. Add a `LICENSE` if you intend to specify reuse terms.
