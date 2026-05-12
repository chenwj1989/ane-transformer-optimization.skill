#!/usr/bin/env python3
"""
End-to-end latency benchmark for ANE / TorchScript models.

Usage:
    # PyTorch / TorchScript
    python tools/benchmark.py --model model.pt --dummy '[1,128]' \
        --warmup 10 --iter 100

    # CoreML mlpackage (force ANE-only execution for honest ANE numbers)
    python tools/benchmark.py --model model.mlpackage --dummy '[1,80,3000]' \
        --compute-unit ANE --warmup 10 --iter 100
"""
import argparse
import json
import time

import numpy as np
import torch


def _coreml_compute_unit(name: str):
    import coremltools as ct
    return {
        "ALL": ct.ComputeUnit.ALL,
        "ANE": ct.ComputeUnit.CPU_AND_NE,
        "CPU": ct.ComputeUnit.CPU_ONLY,
        "GPU": ct.ComputeUnit.CPU_AND_GPU,
    }[name]


def _benchmark_coreml(path: str, shape: list, compute_unit: str,
                      warmup: int, iters: int) -> dict:
    import coremltools as ct
    model = ct.models.MLModel(path, compute_units=_coreml_compute_unit(compute_unit))
    input_name = list(model.get_spec().description.input)[0].name
    x = {input_name: np.random.randn(*shape).astype(np.float32)}

    for _ in range(warmup):
        model.predict(x)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(x)
        times.append((time.perf_counter() - t0) * 1000)

    return _summary(times, label=f"CoreML [{compute_unit}]")


def _benchmark_torch(path: str, shape: list, warmup: int, iters: int) -> dict:
    model = torch.jit.load(path).eval()
    x = torch.randn(shape)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return _summary(times, label="TorchScript")


def _summary(times: list, label: str) -> dict:
    arr = np.array(times)
    out = {
        "label": label,
        "avg_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "iters": len(arr),
    }
    print(f"\n{label}")
    print(f"  Avg : {out['avg_ms']:8.3f} ms   ({out['iters']} iters)")
    print(f"  P50 : {out['p50_ms']:8.3f} ms")
    print(f"  P95 : {out['p95_ms']:8.3f} ms")
    print(f"  Min : {out['min_ms']:8.3f} ms")
    return out


def main():
    parser = argparse.ArgumentParser(description="Benchmark a PyTorch / TorchScript / CoreML model")
    parser.add_argument("--model", required=True, help="Model file (.pt / .mlpackage)")
    parser.add_argument("--dummy", required=True, help='Input shape as JSON list, e.g. "[1,128]"')
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iter", type=int, default=100, help="Measured iterations")
    parser.add_argument("--compute-unit", choices=["ALL", "ANE", "CPU", "GPU"], default="ANE",
                        help="CoreML compute unit (only used for .mlpackage). "
                             "Default ANE = CPU_AND_NE for honest ANE measurements.")
    args = parser.parse_args()

    shape = json.loads(args.dummy)

    if args.model.endswith(".mlpackage"):
        _benchmark_coreml(args.model, shape, args.compute_unit, args.warmup, args.iter)
    else:
        _benchmark_torch(args.model, shape, args.warmup, args.iter)


if __name__ == "__main__":
    main()
