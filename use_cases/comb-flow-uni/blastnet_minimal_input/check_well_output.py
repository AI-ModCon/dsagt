"""Compare two WELL HDF5 files and report structural and numerical differences."""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def collect(path):
    """Return {name: info_dict} for every group and dataset in the file."""
    items = {}
    with h5py.File(path) as f:
        items["__root__"] = {"type": "root", "attrs": dict(f.attrs)}
        def visit(name, obj):
            entry = {"attrs": dict(obj.attrs)}
            if isinstance(obj, h5py.Dataset):
                entry["type"]  = "dataset"
                entry["shape"] = obj.shape
                entry["dtype"] = str(obj.dtype)
            else:
                entry["type"] = "group"
            items[name] = entry
        f.visititems(visit)
    return items


def load_ds(path, name):
    with h5py.File(path) as f:
        return f[name][:]


def spot_check_ds(cand_path, ref_path, name, n_points, rng):
    """Load only n_points randomly sampled points from each file."""
    with h5py.File(cand_path) as cf, h5py.File(ref_path) as rf:
        shape = cf[name].shape
        flat_size = int(np.prod(shape))
        indices = rng.choice(flat_size, size=min(n_points, flat_size), replace=False)
        indices.sort()
        coords = [tuple(int(x) for x in np.unravel_index(int(i), shape)) for i in indices]
        cv = np.array([float(cf[name][c]) for c in coords])
        rv = np.array([float(rf[name][c]) for c in coords])
    return cv, rv, indices, shape, coords


def fmt_attr(v):
    if hasattr(v, "tolist"):
        return str(v.tolist())
    return repr(v)


def run(candidate: Path, reference: Path, rtol: float, atol: float,
        spot_check: bool = False, n_points: int = 5, seed: int = 0) -> int:
    """Return 0 if no differences found, 1 otherwise."""
    cand = collect(candidate)
    ref  = collect(reference)

    failures = 0

    # ------------------------------------------------------------------ #
    # 1. Root attributes
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("ROOT ATTRIBUTES")
    print("=" * 60)
    ca, ra = cand["__root__"]["attrs"], ref["__root__"]["attrs"]
    all_keys = sorted(set(ca) | set(ra))
    for k in all_keys:
        if k not in ca:
            print(f"  MISSING  {k!r} (only in reference: {fmt_attr(ra[k])})")
            failures += 1
        elif k not in ra:
            print(f"  EXTRA    {k!r} = {fmt_attr(ca[k])} (not in reference)")
            failures += 1
        else:
            cv, rv = fmt_attr(ca[k]), fmt_attr(ra[k])
            if cv == rv:
                print(f"  ok       {k!r} = {cv}")
            else:
                print(f"  DIFF     {k!r}:  candidate={cv}  reference={rv}")
                failures += 1

    # ------------------------------------------------------------------ #
    # 2. Structure (groups / datasets)
    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("STRUCTURE")
    print("=" * 60)
    cand_keys = set(cand) - {"__root__"}
    ref_keys  = set(ref)  - {"__root__"}
    only_cand = sorted(cand_keys - ref_keys)
    only_ref  = sorted(ref_keys  - cand_keys)
    shared    = sorted(cand_keys & ref_keys)

    if only_cand:
        for k in only_cand:
            print(f"  EXTRA    {k}")
            failures += 1
    if only_ref:
        for k in only_ref:
            print(f"  MISSING  {k}")
            failures += 1

    shape_ok = True
    for k in shared:
        ci, ri = cand[k], ref[k]
        if ci["type"] != ri["type"]:
            print(f"  TYPE     {k}: candidate={ci['type']}  reference={ri['type']}")
            failures += 1
            shape_ok = False
            continue
        if ci["type"] == "dataset":
            if ci["shape"] != ri["shape"] or ci["dtype"] != ri["dtype"]:
                print(f"  SHAPE    {k}: candidate={ci['shape']} {ci['dtype']}  "
                      f"reference={ri['shape']} {ri['dtype']}")
                failures += 1
                shape_ok = False

    if not only_cand and not only_ref and shape_ok:
        print(f"  ok  ({len(shared)} items, shapes and dtypes match)")

    # ------------------------------------------------------------------ #
    # 3. Dataset attributes
    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("DATASET / GROUP ATTRIBUTES")
    print("=" * 60)
    # field_names / spatial_dims are unordered indexes — compare as sets
    SET_ATTRS = {"field_names", "spatial_dims"}

    attr_failures = 0
    for k in shared:
        ca2, ra2 = cand[k]["attrs"], ref[k]["attrs"]
        all_ak = sorted(set(ca2) | set(ra2))
        for ak in all_ak:
            if ak not in ca2:
                print(f"  MISSING  {k}  @{ak!r} (ref={fmt_attr(ra2[ak])})")
                attr_failures += 1
            elif ak not in ra2:
                print(f"  EXTRA    {k}  @{ak!r} = {fmt_attr(ca2[ak])}")
                attr_failures += 1
            else:
                cv_raw, rv_raw = ca2[ak], ra2[ak]
                if ak in SET_ATTRS:
                    cv_set = set(cv_raw.tolist() if hasattr(cv_raw, "tolist") else cv_raw)
                    rv_set = set(rv_raw.tolist() if hasattr(rv_raw, "tolist") else rv_raw)
                    if cv_set != rv_set:
                        print(f"  DIFF     {k}  @{ak!r} (set):  "
                              f"candidate={sorted(cv_set)}  reference={sorted(rv_set)}")
                        attr_failures += 1
                    else:
                        cv_list = sorted(cv_set)
                        print(f"  ok       {k}  @{ak!r} = {cv_list} (order ignored)")
                else:
                    cv, rv = fmt_attr(cv_raw), fmt_attr(rv_raw)
                    if cv != rv:
                        print(f"  DIFF     {k}  @{ak!r}:  candidate={cv}  reference={rv}")
                        attr_failures += 1
    if attr_failures == 0:
        print("  ok  (all attributes match)")
    failures += attr_failures

    # ------------------------------------------------------------------ #
    # 4. Numerical values
    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    if spot_check:
        rng = np.random.default_rng(seed)
        print(f"NUMERICAL VALUES — SPOT CHECK  "
              f"({n_points} points/dataset, seed={seed}, rtol={rtol:.0e}, atol={atol:.0e})")
    else:
        print(f"NUMERICAL VALUES  (rtol={rtol:.0e}, atol={atol:.0e})")
    print("=" * 60)

    datasets = [k for k in shared if cand[k]["type"] == "dataset"
                and cand[k]["shape"] == ref[k]["shape"]]
    num_failures = 0
    for k in datasets:
        if spot_check:
            cv, rv, indices, shape, coords = spot_check_ds(candidate, reference, k, n_points, rng)
            diff      = np.abs(cv - rv)
            max_abs   = diff.max()
            ref_scale = np.abs(rv).mean() + 1e-30
            mean_rel  = (diff / ref_scale).mean()
            exact     = np.array_equal(cv, rv)
            close     = np.allclose(cv, rv, rtol=rtol, atol=atol)
            status    = "exact" if exact else ("ok" if close else "DIFF")
            flag      = "  " if (exact or close) else "!!"
            print(f"  {flag} {status:<5s}  {k}  (sampled {len(indices)}/{int(np.prod(shape))} points)")
            if not exact or not close:
                for idx, coord, c, r in zip(indices, coords, cv, rv):
                    marker = " <-- DIFF" if abs(c - r) > atol + rtol * abs(r) else ""
                    print(f"         [{coord}]  candidate={c:.6g}  reference={r:.6g}{marker}")
            if not close:
                print(f"         max_abs={max_abs:.3e}  mean_rel={mean_rel:.3e}")
                num_failures += 1
        else:
            cv = load_ds(candidate, k).astype(np.float64)
            rv = load_ds(reference,  k).astype(np.float64)
            diff      = np.abs(cv - rv)
            max_abs   = diff.max()
            mean_abs  = diff.mean()
            ref_scale = np.abs(rv).mean() + 1e-30
            mean_rel  = (diff / ref_scale).mean()
            exact     = np.array_equal(cv, rv)
            close     = np.allclose(cv, rv, rtol=rtol, atol=atol)
            status    = "exact" if exact else ("ok" if close else "DIFF")
            flag      = "  " if (exact or close) else "!!"
            print(f"  {flag} {status:<5s}  {k}")
            if not exact:
                print(f"         max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}  "
                      f"mean_rel={mean_rel:.3e}")
            if not close:
                num_failures += 1
    failures += num_failures

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    if failures == 0:
        print("PASS — candidate matches reference exactly.")
    else:
        print(f"FAIL — {failures} difference(s) found.")
    print("=" * 60)

    return 0 if failures == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Compare a candidate WELL HDF5 file against a reference.")
    parser.add_argument("candidate", help="Path to the file to check")
    parser.add_argument("reference", help="Path to the reference file")
    parser.add_argument("--rtol", type=float, default=1e-5,
                        help="Relative tolerance for np.allclose (default 1e-5)")
    parser.add_argument("--atol", type=float, default=1e-8,
                        help="Absolute tolerance for np.allclose (default 1e-8)")
    parser.add_argument("--spot-check", action="store_true",
                        help="Fast mode: check structure fully, sample n points per dataset")
    parser.add_argument("--n-points", type=int, default=5,
                        help="Number of random points per dataset in spot-check mode (default 5)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for spot-check sampling (default 0)")
    args = parser.parse_args()

    sys.exit(run(Path(args.candidate), Path(args.reference),
                 args.rtol, args.atol, args.spot_check, args.n_points, args.seed))


if __name__ == "__main__":
    main()
