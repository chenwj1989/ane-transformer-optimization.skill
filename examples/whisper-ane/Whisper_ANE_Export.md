# Whisper ANE refactor and Core ML export

This example ports OpenAI **Whisper** to an Apple Neural Engine (ANE)–friendly PyTorch layout, exports **encoder** and **decoder** as separate Core ML packages, and provides scripts to verify numerics, benchmark latency, and validate **real-audio** transcription against the upstream model.

---

## What changed (ANE-oriented Whisper)

| Area | Original Whisper | This example (`model_ane.py`) |
|------|-------------------|-------------------------------|
| Layout | `(B, S, C)` linear layers | Attention / FFN use **BC1S** `(B, C, 1, S)` with **`nn.Conv2d(..., kernel_size=1)`** |
| Audio stem | `Conv1d` | **`Conv2d`** with `(1, k)` kernels on mel time |
| Attention | SDPA / batched matmul | **Per-head** `einsum`, **`softmax(dim=1)`** on key axis after K layout |
| Causal mask | `-inf` triu buffer | **Additive float** `-1e4`, sliced to sequence length |
| Layer norm | Custom `LayerNorm` | **`F.layer_norm`** (strategy A), **`eps ≥ 1e-7`** |
| Activations | GELU | **GELU kept** for checkpoint compatibility |
| Weights | — | **`WhisperANE.from_whisper(w)`** copies and reshapes tensors explicitly |

The decoder export path uses **no KV cache** (full prefix each step): slower than production PyTorch with hooks, but **TorchScript- and Core ML–stable**.

---

## Repository layout (`examples/whisper-ane/`)

| Path | Role |
|------|------|
| [`model_ane.py`](model_ane.py) | `WhisperANE`, ANE encoder/decoder modules, `from_whisper`, export wrappers, `trace_with_roundtrip` |
| [`export_and_verify.py`](export_and_verify.py) | Load checkpoint, PyTorch parity vs OpenAI (with `disable_sdpa`), optional Core ML export + smoke `predict` |
| [`benchmark_coreml.py`](benchmark_coreml.py) | Export **baseline** (original) vs **ANE** enc/dec, tensor metrics vs PyTorch, latency, optional **`jfk.flac`** transcribe compare |
| [`coreml_whisper_adapter.py`](coreml_whisper_adapter.py) | `CoreMLWhisper` duck model: stock `transcribe` + Core ML enc/dec + flex decoder packages |

Upstream clone is expected at [`whisper/`](whisper/) (OpenAI `whisper` package root on `PYTHONPATH`).

---

## Dependencies

- Python 3.9+ (as used in this example’s venv)
- PyTorch, **openai-whisper** (local tree under `whisper/`)
- **`coremltools`** for conversion and on-device `predict`

Install Core ML tooling, for example:

```bash
pip install coremltools
```

---

## PyTorch: build and check parity

From `examples/whisper-ane/` (ensure `whisper/` is importable, e.g. `export PYTHONPATH=whisper` or run scripts from this directory as documented in each file’s header):

```python
import whisper
from whisper.model import disable_sdpa
from model_ane import WhisperANE

w = whisper.load_model("tiny", device="cpu").eval()
ane = WhisperANE.from_whisper(w).eval()

mel = ...  # (1, 80, 3000) for default tiny timing
toks = ... # (1, S) long indices

with torch.no_grad(), disable_sdpa():
    ref_e = w.encoder(mel)
    ref_l = w.decoder(toks, ref_e)
with torch.no_grad():
    ane_e = ane.encoder(mel)
    ane_l = ane.decoder(toks, ane_e)
```

Compare `ref_e` / `ane_e` and `ref_l` / `ane_l`; with SDPA disabled they should match to ~**1e-4** fp32.

---

## Core ML export (separate encoder and decoder)

### Principles

1. **TorchScript trace** each export wrapper, then **save → load** once (round-trip) before `coremltools.convert` (avoids shared-module issues in some graphs).
2. Convert with **`compute_units=ALL`**; load for ANE benchmarking with **`compute_units=CPU_AND_NE`**.
3. Token inputs: declare **`dtype=np.int32`** and pass **`int32`** at `predict`.
4. **`FLOAT16`** is typical for deployment; use **`FLOAT32`** when debugging numerics.

### `export_and_verify.py` (ANE only, fixed shapes)

Exports **`WhisperEncoderANE.mlpackage`** and **`WhisperDecoderANE.mlpackage`** using the traced shapes from the run (default mel `3000` frames, short token length for the decoder trace).

```bash
cd examples/whisper-ane
.venv/bin/python export_and_verify.py --model tiny -o coreml_out
# Tighter Core ML vs PyTorch:
.venv/bin/python export_and_verify.py --model tiny --fp32-coreml -o coreml_out_fp32
```

Use **`--skip-coreml`** if you only want PyTorch checks.

---

## Benchmark: baseline vs ANE + real audio

[`benchmark_coreml.py`](benchmark_coreml.py) does the following:

1. Exports **four** fixed-shape packages: baseline encoder/decoder, ANE encoder/decoder.
2. Compares Core ML outputs to **original PyTorch Whisper** (encoder hidden states, decoder logits on a random batch).
3. Measures **`predict`** latency with **`CPU_AND_NE`**.
4. If **`whisper/tests/jfk.flac`** exists (same asset as [`whisper/tests/test_transcribe.py`](whisper/tests/test_transcribe.py)), exports **flexible-sequence** decoder MLPrograms (`RangeDim(1, n_text_ctx)`) under **`transcribe_ml/`**, builds **`CoreMLWhisper`**, and runs **`transcribe`** three ways: PyTorch, baseline Core ML, ANE Core ML. Results are stored in **`benchmark_results.json`** under **`real_audio_transcription`**.

```bash
cd examples/whisper-ane
.venv/bin/python benchmark_coreml.py --model tiny -o .output/coreml_benchmark
# Other checkpoints, e.g. small:
.venv/bin/python benchmark_coreml.py --model small -o .output/coreml_benchmark_small
# Skip real-audio block:
.venv/bin/python benchmark_coreml.py --no-real-audio -o .output/coreml_benchmark
# Custom audio (same decode options as the benchmark harness):
.venv/bin/python benchmark_coreml.py --audio /path/to/audio.flac -o .output/coreml_benchmark
```

**Note:** the real-audio harness uses **`word_timestamps=False`** because word-level timing in upstream Whisper depends on PyTorch decoder internals; text comparison is the primary signal there.

---

## Full `transcribe` with Core ML in your own code

Use [`coreml_whisper_adapter.py`](coreml_whisper_adapter.py):

- Load **`MLModel`** for the **fixed** encoder `.mlpackage` and the **flex** decoder `.mlpackage` from `benchmark_coreml.py`’s **`transcribe_ml/`** output (or re-export with the same `RangeDim` pattern).
- Instantiate **`CoreMLWhisper(reference_whisper, ml_encoder, ml_decoder, enc_io_pair(...), dec_io_triplet(...))`**.
- Call **`model.transcribe(...)`** as with upstream `Whisper`.

`CoreMLWhisper` keeps a reference to the original **`Whisper`** only to satisfy **`DecodingTask`** construction (decoder block introspection); inference uses Core ML for **encoder** and **decoder** via the patched decode path.

---

## Outputs you should expect

| Artifact | Description |
|----------|-------------|
| `BaselineWhisperEncoder.mlpackage` / `BaselineWhisperDecoder.mlpackage` | Original architecture, Core ML |
| `ANEWhisperEncoder.mlpackage` / `ANEWhisperDecoder.mlpackage` | ANE refactored graph, fixed trace shapes |
| `transcribe_ml/*_flex.mlpackage` | Decoders with flexible token length (from benchmark) |
| `benchmark_results.json` | Tensor accuracy, latency, optional `real_audio_transcription` |

---

## Further reading

- Apple: [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
