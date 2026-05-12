#!/usr/bin/env python3
"""
Convert TorchScript model to CoreML mlprogram and validate.

Usage:
    python tools/export_coreml.py \
        --model traced.pt --inputs '{"input_ids": [1,128]}' \
        -o model.mlpackage --fp16
"""
import argparse
import json

import coremltools as ct
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Export TorchScript → CoreML mlprogram")
    parser.add_argument("--model", required=True, help="TorchScript traced model (.pt)")
    parser.add_argument("--inputs", required=True,
                        help='JSON: {"name": [shape_dims]}; names containing "id"/"token"/"int" '
                             'are exported as np.int32 (CoreML rejects int64).')
    parser.add_argument("-o", "--output", required=True, help="Output .mlpackage path")
    parser.add_argument("--fp16", action="store_true", help="Compute precision FLOAT16 (default FLOAT32)")
    parser.add_argument("--minimum-deployment-target", default="macOS14",
                        help="One of: macOS13, macOS14, macOS15, iOS16, iOS17, iOS18")
    args = parser.parse_args()

    traced = torch.jit.load(args.model)
    input_desc = json.loads(args.inputs)

    def _dtype_for(name: str):
        # Token / index inputs MUST be int32 — coremltools drops int64.
        if any(s in name.lower() for s in ("id", "token", "int", "pos")):
            return np.int32
        return np.float32

    inputs = [
        ct.TensorType(name=n, shape=s, dtype=_dtype_for(n))
        for n, s in input_desc.items()
    ]

    target = getattr(ct.target, args.minimum_deployment_target)

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16 if args.fp16 else ct.precision.FLOAT32,
        minimum_deployment_target=target,
    )
    mlmodel.save(args.output)
    print(f"Saved: {args.output}")

    loaded = ct.models.MLModel(args.output)
    spec = loaded.get_spec()
    print(f"Spec version: {spec.specificationVersion}")

    # mlprogram models populate spec.mlProgram (NOT spec.neuralNetwork)
    if spec.HasField("mlProgram"):
        # Count total ops across all functions / blocks
        n_ops = 0
        for func_name, func in spec.mlProgram.functions.items():
            for block_name, block in func.block_specializations.items():
                n_ops += len(block.operations)
        print(f"mlProgram ops: {n_ops} (across {len(spec.mlProgram.functions)} function(s))")
        print("Note: actual ANE/GPU/CPU op placement is decided at load time by CoreML; "
              "use Xcode's Performance report or `MLComputePlan` (CoreML 8+) to inspect "
              "per-op device assignment.")
    elif len(spec.neuralNetwork.layers) > 0:
        print(f"neuralNetwork layers: {len(spec.neuralNetwork.layers)} (legacy NN format)")
    else:
        print("Empty op set — check the converted model.")


if __name__ == "__main__":
    main()
