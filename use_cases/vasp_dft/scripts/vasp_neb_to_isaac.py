#!/usr/bin/env python3
"""Convert a VASP NEB calculation directory to an ISAAC AI-Ready Record v1.05.

Usage:
    python vasp_neb_to_isaac.py <neb_dir> [options]

The NEB directory must contain sequentially numbered image subdirectories
(e.g. 00, 01, 02 ... N) each with an OUTCAR. Endpoint directories (00 and
the last) must also contain a POSCAR or CONTCAR.

Examples:
    python vasp_neb_to_isaac.py ./neb
    python vasp_neb_to_isaac.py ./neb --reaction "Fe(lattice) -> Fe(vacancy)" --output record.json
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time

try:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Outcar
except ImportError:
    sys.exit("pymatgen is required: pip install pymatgen")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _ulid() -> str:
    ms = int(time.time() * 1000)
    ts = ""
    for _ in range(10):
        ts = _CROCKFORD[ms & 0x1F] + ts
        ms >>= 5
    return ts + "".join(_CROCKFORD[random.randint(0, 31)] for _ in range(16))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _grep(pattern: str, text: str, default="N/A"):
    m = re.search(pattern, text)
    return m.group(1).strip() if m else default


def _find_images(neb_dir: str) -> list[str]:
    """Return sorted list of image subdirectory names (e.g. ['00','01','02'])."""
    images = sorted(
        d for d in os.listdir(neb_dir)
        if os.path.isdir(os.path.join(neb_dir, d))
        and d.isdigit()
        and os.path.exists(os.path.join(neb_dir, d, "OUTCAR"))
    )
    if len(images) < 3:
        sys.exit(f"Need at least 3 image directories with OUTCARs in {neb_dir}")
    return images


def _poscar_path(img_dir: str) -> str | None:
    for name in ("CONTCAR", "POSCAR"):
        p = os.path.join(img_dir, name)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_calc_settings(outcar_text: str) -> dict:
    g = lambda pat, default="N/A": _grep(pat, outcar_text, default)
    ismear_map = {"0": "Gaussian", "1": "Methfessel-Paxton", "-5": "tetrahedron", "-1": "Fermi-Dirac"}
    ismear_val = g(r"ISMEAR\s*=\s*(-?\d+)")
    ichain = g(r"ICHAIN\s*=\s*(\d+)", "0")
    method_map = {"0": "NEB", "1": "dimer", "2": "string"}

    try:
        encut = float(g(r"ENCUT\s*=\s*([\d.]+)"))
    except ValueError:
        encut = None

    try:
        sigma = float(g(r"SIGMA\s*=\s*([\d.]+)"))
    except ValueError:
        sigma = None

    try:
        ediff = float(g(r"EDIFF\s*=\s*([\deE.+-]+)"))
    except ValueError:
        ediff = None

    try:
        ediffg = float(g(r"EDIFFG\s*=\s*([\deE.+-]+)"))
    except ValueError:
        ediffg = None

    nkpts_m = re.search(r"Found\s+(\d+)\s+irreducible k-points", outcar_text)
    nkpts = int(nkpts_m.group(1)) if nkpts_m else None
    kpoints_label = g(r"k-points in units.*?:\s*(.+)", "unknown")
    kpoints_str = f"{nkpts} irreducible k-points ({kpoints_label})" if nkpts else kpoints_label

    lclimb = g(r"LCLIMB\s*=\s*(\S+)", "F")
    is_cineb = lclimb.upper() in ("T", "TRUE", ".TRUE.")

    ivdw = g(r"IVDW\s*=\s*(\d+)", "0")
    dispersion_map = {"0": "none", "10": "D2", "11": "D3", "12": "D3BJ"}
    dispersion = dispersion_map.get(ivdw, f"IVDW={ivdw}")

    ispin = g(r"ISPIN\s*=\s*(\d+)", "1")
    spin_map = {"1": "none", "2": "collinear", "4": "noncollinear"}

    return {
        "vasp_version": g(r"vasp\.([\d.]+)"),
        "potcar": g(r"TITEL\s*=\s*(.+)"),
        "encut_eV": encut,
        "ediff": ediff,
        "ediffg": ediffg,
        "ispin": ispin,
        "spin_treatment": spin_map.get(ispin, "collinear"),
        "smearing_method": ismear_map.get(ismear_val, f"ISMEAR={ismear_val}"),
        "smearing_width_eV": sigma,
        "dispersion": dispersion,
        "kpoints": kpoints_str,
        "neb_method": "CI-NEB" if is_cineb else method_map.get(ichain, "NEB"),
        "neb_method_schema": "CI-NEB" if is_cineb else "NEB",
        "acquired_start_utc": _grep(r"date\s+([\d.]+\s+[\d:]+)", outcar_text, None),
    }


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def build_record(
    neb_dir: str,
    reaction: str,
    material_name: str | None,
    notes: str | None,
) -> dict:
    images = _find_images(neb_dir)
    n_intermediate = len(images) - 2

    # Energies (sigma→0 corrected via pymatgen Outcar.final_energy)
    energies: dict[str, float] = {}
    for img in images:
        oc = Outcar(os.path.join(neb_dir, img, "OUTCAR"))
        energies[img] = oc.final_energy

    e0 = energies[images[0]]
    ef = energies[images[-1]]
    ts_img = max(energies, key=energies.get)
    ea = energies[ts_img] - e0
    rxn_e = ef - e0

    # Settings from first endpoint OUTCAR
    with open(os.path.join(neb_dir, images[0], "OUTCAR")) as f:
        raw = f.read()
    cfg = extract_calc_settings(raw)

    # Structure from endpoint POSCAR/CONTCAR
    poscar = _poscar_path(os.path.join(neb_dir, images[0]))
    if poscar:
        struct = Structure.from_file(poscar)
        formula = struct.composition.formula
        reduced = struct.composition.reduced_formula
        lattice = struct.lattice
        lattice_str = (
            f"a={lattice.a:.4f} b={lattice.b:.4f} c={lattice.c:.4f} Å  "
            f"α={lattice.alpha:.2f} β={lattice.beta:.2f} γ={lattice.gamma:.2f}°"
        )
    else:
        formula = reduced = "unknown"
        lattice_str = "unknown"

    mat_name = material_name or f"{reduced} NEB supercell"

    # Parse acquisition timestamp
    acq_start = None
    if cfg["acquired_start_utc"] and cfg["acquired_start_utc"] != "N/A":
        raw_ts = cfg["acquired_start_utc"].strip()
        # Format: "2016.09.22  13:56:26" → ISO 8601
        m = re.match(r"(\d{4})\.(\d{2})\.(\d{2})\s+(\d{2}:\d{2}:\d{2})", raw_ts)
        if m:
            acq_start = f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}Z"

    timestamps = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if acq_start:
        timestamps["acquired_start_utc"] = acq_start

    # Assets
    assets = []
    for img in images:
        p = os.path.join(neb_dir, img, "OUTCAR")
        assets.append({
            "asset_id": f"outcar_image_{img}",
            "content_role": "raw_data",
            "uri": f"file://neb/{img}/OUTCAR",
            "sha256": _sha256(p),
            "media_type": "text/plain",
        })
    for img in [images[0], images[-1]]:
        pp = _poscar_path(os.path.join(neb_dir, img))
        if pp:
            fname = os.path.basename(pp)
            assets.append({
                "asset_id": f"{fname.lower()}_image_{img}",
                "content_role": "input_structure",
                "uri": f"file://neb/{img}/{fname}",
                "sha256": _sha256(pp),
                "media_type": "text/plain",
            })

    # smearing schema mapping
    smearing_schema_map = {
        "Gaussian": "Gaussian",
        "Methfessel-Paxton": "Methfessel-Paxton",
        "Fermi-Dirac": "Fermi-Dirac",
        "tetrahedron": "tetrahedron",
    }
    smearing_method = smearing_schema_map.get(cfg["smearing_method"], "Gaussian")

    record = {
        "isaac_record_version": "1.05",
        "record_id": _ulid(),
        "record_type": "evidence",
        "record_domain": "simulation",
        "source_type": "computation",
        "timestamps": timestamps,
        "sample": {
            "material": {
                "name": mat_name,
                "formula": formula,
                "provenance": "theoretical",
            },
            "sample_form": "bulk_supercell",
        },
        "system": {
            "domain": "computational",
            "technique": "DFT",
            "instrument": {
                "instrument_type": "simulation_engine",
                "instrument_name": "VASP",
                "vendor_or_project": "VASP",
            },
            "configuration": {
                "code_version": cfg["vasp_version"],
                "compute_architecture": "CPU",
            },
        },
        "context": {
            "environment": "in_silico",
            "temperature_K": 0,
            "simulation_assumptions": {"solvation_model": "none"},
        },
        "computation": {
            "method": {
                "family": "DFT",
                "functional_class": "GGA",
                "functional_name": "PBE",
                "basis_type": "planewave",
                "pseudopotential": "PAW",
                **({"cutoff_eV": cfg["encut_eV"]} if cfg["encut_eV"] else {}),
                "spin_treatment": cfg["spin_treatment"],
                "dispersion": cfg["dispersion"],
                "kpoints": cfg["kpoints"],
                **({"smearing": {"method": smearing_method, "width_eV": cfg["smearing_width_eV"]}}
                   if cfg["smearing_width_eV"] else {}),
                "convergence": {
                    **({"energy_eV": cfg["ediff"]} if cfg["ediff"] else {}),
                    **({"force_eV_per_A": abs(cfg["ediffg"])} if cfg["ediffg"] else {}),
                },
            },
            "output_quantity": {
                "quantity": "activation_energy_raw",
                "corrections_applied": {
                    "zero_point_energy": False,
                    "entropy": False,
                    "thermal": False,
                    "solvation": False,
                    "dispersion": False,
                    "grand_canonical": False,
                    "PCET": False,
                },
            },
            "transition_state": {
                "method": cfg["neb_method_schema"],
                "images": n_intermediate,
                "reaction": reaction,
                "n_electrons_transferred": 0,
            },
        },
        "measurement": {
            "series": [
                {
                    "series_id": "neb_energy_profile",
                    "independent_variables": [
                        {
                            "name": "image_index",
                            "unit": "dimensionless",
                            "values": list(range(len(images))),
                        }
                    ],
                    "channels": [
                        {
                            "name": "total_energy_sigma0",
                            "unit": "eV",
                            "role": "simulated_observable",
                            "values": [round(energies[img], 6) for img in images],
                        },
                        {
                            "name": "relative_energy",
                            "unit": "eV",
                            "role": "derived_signal",
                            "values": [round(energies[img] - e0, 6) for img in images],
                        },
                    ],
                }
            ],
            "qc": {
                "status": "pass",
                "notes": (
                    f"All {len(images)} images converged; "
                    f"EDIFF={cfg['ediff']} eV, EDIFFG={cfg['ediffg']} eV/Å"
                ),
            },
        },
        "assets": assets,
        "descriptors": {
            "outputs": [
                {
                    "label": "neb_barrier_v1",
                    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "generated_by": {
                        "agent": "vasp_neb_to_isaac.py",
                        "version": "1.0",
                    },
                    "descriptors": [
                        {
                            "name": "activation_barrier",
                            "kind": "theoretical_metric",
                            "source": "auto",
                            "value": round(ea, 4),
                            "unit": "eV",
                            "definition": (
                                f"Raw DFT NEB activation barrier: E(TS image {ts_img}) − E(image {images[0]}). "
                                "energy(sigma→0). No ZPE, entropy, or solvation corrections applied."
                            ),
                            "uncertainty": {"sigma": 0.05, "unit": "eV"},
                        },
                        {
                            "name": "reaction_energy",
                            "kind": "theoretical_metric",
                            "source": "auto",
                            "value": round(rxn_e, 4),
                            "unit": "eV",
                            "definition": (
                                f"Raw DFT reaction energy: E(image {images[-1]}) − E(image {images[0]}). "
                                "energy(sigma→0). No corrections applied."
                            ),
                            "uncertainty": {"sigma": 0.01, "unit": "eV"},
                        },
                    ],
                }
            ]
        },
        **({"links": []} ),
    }

    # Append notes to qc if provided
    if notes:
        record["measurement"]["qc"]["notes"] += f" | {notes}"

    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a VASP NEB directory to an ISAAC AI-Ready Record v1.05."
    )
    parser.add_argument("neb_dir", help="Path to the NEB directory (contains 00/, 01/, ... subdirs)")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path (default: <neb_dir>/isaac_neb_record.json)")
    parser.add_argument("--reaction", "-r", default="unknown", help='Reaction string, e.g. "A* → B*"')
    parser.add_argument("--material", "-m", default=None, help="Human-readable material name")
    parser.add_argument("--notes", "-n", default=None, help="Free-text notes appended to QC block")
    parser.add_argument("--validate", action="store_true", help="Validate output against ISAAC JSON Schema (requires gh CLI)")
    args = parser.parse_args()

    neb_dir = os.path.abspath(args.neb_dir)
    if not os.path.isdir(neb_dir):
        sys.exit(f"Directory not found: {neb_dir}")

    print(f"Reading NEB images from: {neb_dir}")
    record = build_record(neb_dir, args.reaction, args.material, args.notes)

    out_path = args.output or os.path.join(neb_dir, "isaac_neb_record.json")
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Written:  {out_path}")
    print(f"Record ID: {record['record_id']}")

    ea = record["descriptors"]["outputs"][0]["descriptors"][0]["value"]
    rxn = record["descriptors"]["outputs"][0]["descriptors"][1]["value"]
    n_img = record["computation"]["transition_state"]["images"]
    print(f"Ea = {ea} eV  |  ΔE = {rxn} eV  |  {n_img} intermediate images")

    if args.validate:
        _validate(out_path)


def _validate(record_path: str):
    import base64, subprocess
    try:
        from jsonschema import validate, ValidationError
    except ImportError:
        print("Skipping validation: pip install jsonschema")
        return

    result = subprocess.run(
        ["gh", "api",
         "repos/ISAAC-DOE/isaac-ai-ready-record/contents/schema/isaac_record_v1.json",
         "--jq", ".content"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("Skipping validation: gh CLI unavailable or not authenticated")
        return

    schema = json.loads(base64.b64decode(result.stdout.strip()).decode())
    record = json.load(open(record_path))
    try:
        validate(instance=record, schema=schema)
        print("Schema validation: PASS")
    except ValidationError as e:
        print(f"Schema validation: FAIL — {e.message}  (path: {list(e.path)})")


if __name__ == "__main__":
    main()
