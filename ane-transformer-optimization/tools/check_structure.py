#!/usr/bin/env python3
"""
Check model source for ANE compliance.

Validates:
- Linear → Conv2d replacement (scoped to attention/FFN/projection contexts)
- Hand-rolled LayerNorm (mean + (zero_mean ** 2).mean + rsqrt patterns)
- LayerNorm eps that underflows fp16 (eps < 1e-7)
- Boolean mask usage (masked_fill / where with bool) instead of additive masks
- Layout churn (transpose followed by contiguous + view/reshape)
- F.scaled_dot_product_attention usage (must be disabled before tracing)

Usage:
    python tools/check_structure.py --model path/to/model.py
"""
import argparse
import ast
import re
import sys
from pathlib import Path

# Class-name patterns that should NOT contain nn.Linear when targeting ANE
ANE_SENSITIVE_CLASS_RE = re.compile(
    r"(Attention|MultiHead|FFN|MLP|Encoder|Decoder|Transformer|Block|Projection)",
    re.IGNORECASE,
)


def _has_call(tree, name_predicate):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            attr = None
            if isinstance(target, ast.Attribute):
                attr = target.attr
            elif isinstance(target, ast.Name):
                attr = target.id
            if attr and name_predicate(attr):
                yield node


def _enclosing_class(node, parents):
    """Walk up parents to find enclosing ClassDef name (or None)."""
    cur = node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, ast.ClassDef):
            return cur.name
    return None


def _build_parent_map(tree):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def check_linear_in_ane_classes(tree, parents, source_lines):
    """Flag Linear / nn.Linear inside attention / FFN / projection classes."""
    issues = []
    for call in _has_call(tree, lambda n: n == "Linear"):
        cls_name = _enclosing_class(call, parents)
        if not (cls_name and ANE_SENSITIVE_CLASS_RE.search(cls_name)):
            continue
        # Distinguish nn.Linear (definite) from bare Linear (heuristic — could be
        # a project-local subclass like Whisper's, which is exactly the pattern
        # we want to flag).
        is_nn_qualified = (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "nn"
        )
        qualifier = "nn.Linear" if is_nn_qualified else "Linear(...)"
        issues.append((
            "warning",
            f"{qualifier} inside `{cls_name}` — replace with nn.Conv2d(kernel_size=1) for ANE",
            call.lineno,
        ))
    return issues


def check_handrolled_layernorm(source_lines):
    """Heuristic: contiguous lines containing mean(...keepdim) + .rsqrt() + an eps add."""
    issues = []
    text = "\n".join(source_lines)
    # Look for `(... + eps).rsqrt()` or `.pow(2).mean(...).rsqrt()` patterns close together
    rsqrt_lines = [i for i, ln in enumerate(source_lines, 1) if ".rsqrt(" in ln]
    for ln in rsqrt_lines:
        # Look back up to 6 lines for a `.mean(` keepdim pattern
        window = "\n".join(source_lines[max(0, ln - 7):ln])
        if (".mean(" in window and "keepdim" in window
                and ("- " in window or "zero_mean" in window or "** 2" in window or "*" in window)):
            issues.append((
                "warning",
                "possible hand-rolled LayerNorm (mean+rsqrt). "
                "Use F.layer_norm or Apple's validated LayerNormANE — fp16 will be unstable otherwise",
                ln,
            ))
    return issues


def check_layernorm_eps(source_lines):
    issues = []
    eps_re = re.compile(r"eps\s*=\s*([0-9eE.\-+]+)")
    for i, ln in enumerate(source_lines, 1):
        if "LayerNorm" not in ln and "layer_norm" not in ln:
            continue
        m = eps_re.search(ln)
        if not m:
            continue
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        if val < 1e-7:
            issues.append((
                "error",
                f"LayerNorm eps={val:g} underflows in fp16 (use eps>=1e-7, "
                "Apple's distilbert port uses 1e-7)",
                i,
            ))
    return issues


def check_boolean_masking(tree):
    issues = []
    for call in _has_call(tree, lambda n: n in ("masked_fill", "masked_fill_")):
        issues.append((
            "warning",
            f"{getattr(call.func, 'attr', 'masked_fill')}(...) — convert to additive float "
            "mask (-1e4 = mask out, 0 = keep) and add to attention weights",
            call.lineno,
        ))
    return issues


def check_layout_churn(source_lines):
    issues = []
    for i, ln in enumerate(source_lines, 1):
        if ".transpose(" in ln and ".contiguous()" in ln:
            issues.append((
                "warning",
                "transpose+contiguous chain — incurs memory copy, breaks ANE op fusion",
                i,
            ))
        if ".transpose(" in ln and (".view(" in ln or ".reshape(" in ln):
            issues.append((
                "warning",
                "transpose+view/reshape chain — likely layout churn on hot path",
                i,
            ))
    return issues


def check_sdpa(tree):
    issues = []
    for call in _has_call(tree, lambda n: n == "scaled_dot_product_attention"):
        issues.append((
            "warning",
            "F.scaled_dot_product_attention used — disable before tracing "
            "(version-dependent op decomposition breaks coremltools)",
            call.lineno,
        ))
    return issues


def main():
    parser = argparse.ArgumentParser(description="Check model source for ANE structural compliance")
    parser.add_argument("--model", required=True, help="Model source file (.py)")
    args = parser.parse_args()

    path = Path(args.model)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2

    source = path.read_text()
    source_lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"ERROR: failed to parse {path}: {e}", file=sys.stderr)
        return 2
    parents = _build_parent_map(tree)

    issues = []
    issues += check_linear_in_ane_classes(tree, parents, source_lines)
    issues += check_handrolled_layernorm(source_lines)
    issues += check_layernorm_eps(source_lines)
    issues += check_boolean_masking(tree)
    issues += check_layout_churn(source_lines)
    issues += check_sdpa(tree)

    issues.sort(key=lambda x: x[2])

    errors = [i for i in issues if i[0] == "error"]
    warnings = [i for i in issues if i[0] == "warning"]

    print(f"File:     {path}")
    print(f"Errors:   {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    for sev, msg, ln in issues:
        print(f"  [{sev.upper():7s}] L{ln}: {msg}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
