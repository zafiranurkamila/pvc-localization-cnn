"""Inspect the OTVA dataset of github.com/uripont/arrhythmia-origin-predictor (release `dataset`) before any use.

The data come as pickle files, and unpickling a file from an unknown source can run arbitrary code. This script
therefore first scans the pickle opcodes and lists every class or function the file would import; it loads the file
only if all of them belong to an allow-list of data libraries (pandas, numpy, builtins, ...). --trust skips the check.

What it reports, to decide whether the data can serve as an external validation of the PVC models:
  zip contents; top-level type; columns; number of patients; label columns and their values (side and
  sub-location); the structure of the ECG column (keys = regions / positions, array shapes, sampling); and
  whether the ECGs are spontaneous PVCs or pace maps (names of the recording positions).

Usage (on the PC, after downloading dataset_arrhythmias.zip from the release into data/external/otva_upf/):
    python scripts/29_inspect_external_dataset.py data/external/otva_upf/dataset_arrhythmias.zip
    python scripts/29_inspect_external_dataset.py data/external/otva_upf/all_points_may_2024.pkl
Output: printed report only (nothing is written)
"""
import argparse
import collections
import io
import pickle
import pickletools
import sys
import zipfile
from pathlib import Path

SAFE_MODULES = ("pandas", "numpy", "builtins", "collections", "datetime", "copyreg", "_codecs", "decimal")


def imported_globals(data):
    """(module, name) pairs that the pickle would import, from its opcodes."""
    found, strings = set(), []
    for op, arg, _ in pickletools.genops(io.BytesIO(data)):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
            strings.append(arg)
        if op.name == "GLOBAL":
            mod, name = arg.split(" ", 1)
            found.add((mod, name))
        elif op.name == "STACK_GLOBAL" and len(strings) >= 2:
            found.add((strings[-2], strings[-1]))
    return found


def check(data, trust):
    glob = imported_globals(data)
    unsafe = sorted(g for g in glob if not g[0].split(".")[0] in SAFE_MODULES)
    print(f"  pickle imports {len(glob)} names; outside the allow-list: {unsafe or 'none'}")
    if unsafe and not trust:
        sys.exit("  not loaded: the file imports names outside the allow-list (inspect them, then use --trust if safe)")


def describe_value(v, depth=0, max_items=6):
    pad = "    " * (depth + 2)
    if isinstance(v, dict):
        keys = list(v)
        print(f"{pad}dict with {len(keys)} keys, e.g. {keys[:max_items]}")
        if keys and depth < 3:
            describe_value(v[keys[0]], depth + 1)
    elif isinstance(v, (list, tuple)):
        print(f"{pad}{type(v).__name__} of {len(v)}")
        if v and depth < 3:
            describe_value(v[0], depth + 1)
    elif hasattr(v, "shape"):
        print(f"{pad}{type(v).__name__} shape {getattr(v, 'shape', None)}"
              + (f" columns {list(v.columns)[:12]}" if hasattr(v, "columns") else ""))
    else:
        print(f"{pad}{type(v).__name__}: {str(v)[:80]}")


def inspect_object(obj):
    import pandas as pd
    print(f"  top-level type: {type(obj).__name__}")
    if not isinstance(obj, pd.DataFrame):
        describe_value(obj)
        return
    print(f"  rows {len(obj)}, columns {len(obj.columns)}:")
    print("   ", list(obj.columns))
    for c in obj.columns:
        s = obj[c]
        if s.dtype == object and s.map(lambda x: isinstance(x, (dict, list))).any():
            print(f"  column {c!r} holds nested data; first non-empty entry:")
            describe_value(next(x for x in s if isinstance(x, (dict, list)) and len(x)))
            # names of regions / positions across all patients: pace maps are usually named by catheter position
            names = collections.Counter(k for x in s if isinstance(x, dict) for k in x)
            if names:
                print(f"    keys over all rows ({len(names)} distinct): {names.most_common(25)}")
        elif s.nunique() <= 40:
            print(f"  column {c!r}: {s.value_counts(dropna=False).to_dict()}")
        else:
            print(f"  column {c!r}: {s.nunique()} distinct values, e.g. {s.dropna().head(3).tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="the downloaded zip or an extracted .pkl")
    ap.add_argument("--trust", action="store_true", help="load even if the pickle imports names outside the allow-list")
    args = ap.parse_args()
    path = Path(args.path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            print(f"{path.name}: {len(infos)} files")
            for i in infos[:60]:
                print(f"  {i.filename}  {i.file_size / 1e6:.1f} MB")
            pkls = [i for i in infos if i.filename.endswith((".pkl", ".pickle"))]
            for i in pkls:
                print(f"\n{i.filename}:")
                data = z.read(i)
                check(data, args.trust)
                inspect_object(pickle.loads(data))
    else:
        data = path.read_bytes()
        print(f"{path.name}:")
        check(data, args.trust)
        inspect_object(pickle.loads(data))


if __name__ == "__main__":
    main()
