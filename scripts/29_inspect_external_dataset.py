"""Inspect the OTVA dataset of github.com/uripont/arrhythmia-origin-predictor (release `dataset`) before any use.

The data come as a pickle file, and unpickling a file from an unknown source can run arbitrary code. The file is
therefore loaded with a restricted unpickler that resolves only classes and functions of data libraries (numpy,
pandas, builtins and a few standard modules); any other import stops the load and is reported.

The pickle in the zip is about 7 GB; it is extracted next to the zip once (streamed, not read into memory) and loaded
from disk. Loading still needs roughly as much RAM as the file size.

What it reports, to decide whether the data can serve as an external validation of the PVC models:
  zip contents; top-level type; columns; number of patients; label columns and their values (side and
  sub-location); the structure of the ECG column (keys = regions / positions, array shapes); and the names of the
  recording positions, which show whether the ECGs are spontaneous PVCs or pace maps.

Usage (on the PC, with dataset_arrhythmias.zip in data/external/otva_upf/):
    python scripts/29_inspect_external_dataset.py data/external/otva_upf/dataset_arrhythmias.zip
Output: printed report; the extracted all_points_may_2024.pkl and the two label .xlsx in data/external/otva_upf/
"""
import argparse
import collections
import pickle
import shutil
import sys
import zipfile
from pathlib import Path

SAFE_MODULES = ("numpy", "pandas", "builtins", "collections", "datetime", "copyreg", "_codecs", "decimal")


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that may import only from data libraries; everything else raises."""

    def find_class(self, module, name):
        if module.split(".")[0] in SAFE_MODULES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked import {module}.{name}")


def describe_value(v, depth=0, max_items=8):
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
        print(f"{pad}{type(v).__name__} shape {getattr(v, 'shape', None)} dtype {getattr(v, 'dtype', '')}"
              + (f" columns {list(v.columns)[:14]}" if hasattr(v, "columns") else ""))
    else:
        print(f"{pad}{type(v).__name__}: {str(v)[:100]}")


def short(v):
    """Compact description of one value: type, length or shape, and a preview."""
    if hasattr(v, "shape"):
        return f"{type(v).__name__}{tuple(v.shape)}"
    if isinstance(v, dict):
        return f"dict[{len(v)}]"
    if isinstance(v, (list, tuple)):
        inner = short(v[0]) if v else ""
        return f"{type(v).__name__}[{len(v)}] of {inner}" if inner else f"{type(v).__name__}[0]"
    return f"{type(v).__name__}={str(v)[:40]}"


def scalar(v):
    while isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    return v if isinstance(v, (str, int, float, bool, type(None))) else short(v)


def inspect_patients(obj):
    """Top level = {patient id: {attribute: value}}: attributes, label counts and the ECG structure."""
    ids = list(obj)
    first = obj[ids[0]]
    print(f"  {len(ids)} patients; attributes of {ids[0]}:")
    for k, v in first.items():
        print(f"    {k:22s} {short(v)}   e.g. {str(scalar(v))[:60]}")
    for k in first:
        vals = [scalar(p.get(k)) for p in obj.values()]
        if all(isinstance(v, (str, int, float, bool, type(None))) for v in vals):
            c = collections.Counter(vals)
            if len(c) <= 30:
                print(f"  {k}: {dict(c.most_common())}")
            else:
                print(f"  {k}: {len(c)} distinct values")
    # nested ECG structure: walk the dict levels of every patient and collect key names and array shapes
    for k, v in first.items():
        if not isinstance(v, dict):
            continue
        print(f"\n  nested attribute {k!r}:")
        levels = collections.defaultdict(collections.Counter)
        shapes = collections.Counter()
        n_arrays = []

        def walk(x, depth):
            count = 0
            if isinstance(x, dict):
                for kk, vv in x.items():
                    levels[depth][kk] += 1
                    count += walk(vv, depth + 1)
            elif isinstance(x, (list, tuple)) and x and not hasattr(x[0], "shape") and isinstance(x[0], (dict, list)):
                for vv in x:
                    count += walk(vv, depth + 1)
            elif hasattr(x, "shape") or (hasattr(x, "columns")):
                shapes[str(tuple(x.shape)) + " " + type(x).__name__] += 1
                count += 1
            elif isinstance(x, (list, tuple)):
                shapes[f"list[{len(x)}] of {short(x[0]) if x else ''}"] += 1
                count += 1
            return count

        for p in obj.values():
            if isinstance(p.get(k), dict):
                n_arrays.append(walk(p[k], 1))
        for d in sorted(levels):
            print(f"    level {d}: {len(levels[d])} distinct keys, most common {levels[d].most_common(25)}")
        print(f"    leaf shapes: {shapes.most_common(10)}")
        if n_arrays:
            print(f"    leaves (ECGs) per patient: min {min(n_arrays)}, median {sorted(n_arrays)[len(n_arrays) // 2]}, "
                  f"max {max(n_arrays)}, total {sum(n_arrays)}")
        print("    first patient, path to the first leaf:")
        x, path = v, []
        while isinstance(x, dict) and x:
            kk = next(iter(x)); path.append(kk); x = x[kk]
        print(f"      {' / '.join(map(str, path))} -> {short(x)}")
        if hasattr(x, "columns"):
            print(f"      columns {list(x.columns)}")


def inspect_object(obj):
    import pandas as pd
    print(f"  top-level type: {type(obj).__name__}")
    if isinstance(obj, dict) and obj and all(isinstance(v, dict) for v in list(obj.values())[:5]):
        inspect_patients(obj)
        return
    if not isinstance(obj, pd.DataFrame):
        describe_value(obj)
        return
    print(f"  rows {len(obj)}, columns {len(obj.columns)}:")
    print("   ", list(obj.columns))
    for c in obj.columns:
        s = obj[c]
        nested = s.dtype == object and s.map(lambda x: isinstance(x, (dict, list))).any()
        if nested:
            print(f"  column {c!r} holds nested data; first non-empty entry:")
            first = next(x for x in s if isinstance(x, (dict, list)) and len(x))
            describe_value(first)
            # names of regions and of positions over all rows (pace maps are named by catheter position)
            level1 = collections.Counter(k for x in s if isinstance(x, dict) for k in x)
            level2 = collections.Counter(k2 for x in s if isinstance(x, dict) for v in x.values()
                                         if isinstance(v, dict) for k2 in v)
            if level1:
                print(f"    level-1 keys over all rows ({len(level1)} distinct): {level1.most_common(30)}")
            if level2:
                print(f"    level-2 keys over all rows ({len(level2)} distinct): {level2.most_common(30)}")
        else:
            try:
                n = s.nunique()
            except TypeError:
                print(f"  column {c!r}: unhashable values, e.g. {str(s.iloc[0])[:80]}")
                continue
            if n <= 40:
                print(f"  column {c!r}: {s.value_counts(dropna=False).to_dict()}")
            else:
                print(f"  column {c!r}: {n} distinct values, e.g. {s.dropna().head(3).tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="the downloaded zip or an extracted .pkl")
    args = ap.parse_args()
    path = Path(args.path)
    pkl = path
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            infos = [i for i in z.infolist() if not i.filename.startswith("__MACOSX")]
            print(f"{path.name}: {len(infos)} files")
            for i in infos:
                print(f"  {i.filename}  {i.file_size / 1e6:.1f} MB")
            for i in infos:
                if i.filename.endswith((".pkl", ".xlsx")):
                    out = path.parent / Path(i.filename).name
                    if not out.exists():
                        print(f"extracting {i.filename} -> {out}")
                        with z.open(i) as src, open(out, "wb") as dst:
                            shutil.copyfileobj(src, dst, length=16 << 20)
                    if out.suffix == ".pkl":
                        pkl = out
    print(f"\nloading {pkl.name} with the restricted unpickler ...")
    try:
        with open(pkl, "rb") as f:
            obj = RestrictedUnpickler(f).load()
    except pickle.UnpicklingError as e:
        sys.exit(f"  not loaded: {e}")
    inspect_object(obj)
    import pandas as pd
    for x in sorted(pkl.parent.glob("*.xlsx")):
        print(f"\n{x.name}:")
        for sheet, df in pd.read_excel(x, sheet_name=None).items():
            print(f"  sheet {sheet!r}: {df.shape}, columns {list(df.columns)[:12]}")
            print(df.head(8).to_string(max_colwidth=40))


if __name__ == "__main__":
    main()
