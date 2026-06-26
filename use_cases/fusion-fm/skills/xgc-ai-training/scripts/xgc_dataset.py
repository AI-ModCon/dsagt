#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC

"""XGC Graph Dataset for AI training.

Reads preprocessed npz files (from preprocess_xgc.py) and provides
PyG graph-structured training samples.

Each sample: (Data, bcs) where Data has:
    x          : [N, n_steps, 7+F]  – input sequence (pos + psi + region_oh + fields)
    y          : [N, F]             – target field values
    pos        : [N, 2]             – (R, Z) node positions
    edge_index : [2, E]
    edge_attr  : [E, 3]             – (dR, dZ, |d|)
    leadtime   : [1, 1] float32
    phi        : int  – toroidal plane index
    step0      : int  – first input step
    target_step: int  – target step

Training samples are indexed by (phi_plane, start_timestep) pairs.
Splits are made by phi-plane group (all steps from a plane stay together).
"""

import json
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from graph_datasets import BaseCFDGraphDataset, SampleId

# XGC region codes → class indices 0-3
_REGION_MAP = {1: 0, 2: 1, 3: 2, 100: 3}
_NUM_REGION_TYPES = 4
_TOPOLOGY_FILE = "topology.pt"


class XGCGraphDataset(BaseCFDGraphDataset):
    """
    PyG Dataset for XGC plasma turbulence simulations.

    Workflow:
        1. Run preprocess_xgc.py to produce npz files.
        2. Instantiate XGCGraphDataset – on first use it builds a cached
           topology.pt (edge_index, edge_attr, static node context) and
           writes an index.json under path/{split}/processed/.
        3. Each __getitem__ call reads the relevant step_NNNNN.npz files,
           slices the requested phi plane, and assembles a PyG Data graph.

    Args:
        path         : root directory with mesh.npz, meta.json, step_*.npz
        field_names  : fields to use as node features; must be keys in npz
                       (default: all fields in meta.json)
        phi_planes   : list of toroidal plane indices to include
                       (default: all planes 0 … nphi-1)
        require_all_fields : if True (default), drop steps missing any field
        **kwargs     : forwarded to BaseCFDGraphDataset
                       (split, n_steps, leadtime_max, train_val_test, …)
    """

    # ── class-level helpers (static) ────────────────────────────────────────

    @staticmethod
    def _specifics():
        """Placeholder; real values come from _specifics(self) instance method."""
        return ["dpot"], "xgc", None, _NUM_REGION_TYPES

    # ── construction ────────────────────────────────────────────────────────

    def __init__(
        self,
        path: str,
        field_names: Optional[List[str]] = None,
        phi_planes: Optional[List[int]] = None,
        require_all_fields: bool = True,
        **kwargs,
    ):
        meta_path = os.path.join(path, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"meta.json not found in {path}. Run preprocess_xgc.py first."
            )
        with open(meta_path) as fp:
            self._meta = json.load(fp)

        self._nphi    = self._meta["nphi"]
        self._n_nodes = self._meta["n_nodes"]

        # Resolve field names
        avail_fields = self._meta["field_names"]
        if field_names is None:
            self._field_names_data = avail_fields
        else:
            missing = [f for f in field_names if f not in avail_fields]
            if missing:
                raise ValueError(f"Requested fields not in meta.json: {missing}")
            self._field_names_data = list(field_names)

        # Resolve step list (drop steps missing any requested field)
        field_avail = self._meta.get("field_availability", {})
        step_list = self._meta["steps"]
        if require_all_fields and field_avail:
            valid = set(step_list)
            for fn in self._field_names_data:
                if fn in field_avail:
                    valid &= set(field_avail[fn])
            step_list = sorted(valid)
        self._step_list = step_list

        # Resolve phi planes
        self._phi_planes = (
            list(phi_planes) if phi_planes is not None
            else list(range(self._nphi))
        )

        # Cache for topology (lazy-loaded in __getitem__)
        self._topo_cache: Optional[dict] = None

        super().__init__(path, **kwargs)

    # ── specifics (instance override) ───────────────────────────────────────

    def _specifics(self):
        return (
            self._field_names_data,
            "xgc",
            len(self._step_list),
            _NUM_REGION_TYPES,
        )

    # ── one-time processing ──────────────────────────────────────────────────

    def _load_mesh_arrays(self):
        d = np.load(os.path.join(self.path, "mesh.npz"))
        rz      = d["rz"]                      # [N, 2]
        conn    = d["conn"]                    # [C, 3]
        psi     = d["psi"]    if "psi"    in d else None
        region  = d["region"] if "region" in d else None
        return rz, conn, psi, region

    def _map_region(self, region: np.ndarray) -> np.ndarray:
        mapped = np.zeros_like(region)
        for code, idx in _REGION_MAP.items():
            mapped[region == code] = idx
        return mapped

    def process(self):
        """Build cached topology.pt and write index.json. Called once."""
        rz, conn, psi, region = self._load_mesh_arrays()
        n_nodes = self._n_nodes

        pos        = torch.as_tensor(rz, dtype=torch.float32)   # [N, 2]
        edge_index = self.cells_to_edge_index(conn, num_nodes=n_nodes, undirected=True)
        edge_attr  = self.mesh_edge_attr(pos, edge_index)        # [E, 3]

        psi_t = (
            torch.as_tensor(psi, dtype=torch.float32).unsqueeze(1)
            if psi is not None
            else torch.zeros(n_nodes, 1)
        )

        if region is not None:
            reg_idx = torch.as_tensor(self._map_region(region), dtype=torch.long)
            region_oh = self.one_hot_node_type(reg_idx, _NUM_REGION_TYPES)  # [N, 4]
        else:
            region_oh = torch.zeros(n_nodes, _NUM_REGION_TYPES)

        # static_ctx: [R, Z, psi, r0, r1, r2, r3]  → [N, 7]
        static_ctx = torch.cat([pos, psi_t, region_oh], dim=-1)

        topo = dict(
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            static_ctx=static_ctx,
        )
        topo_path = os.path.join(self.processed_dir, _TOPOLOGY_FILE)
        torch.save(topo, topo_path)
        print(f"  Topology saved: {topo_path}")
        print(f"  nodes={n_nodes}  edges={edge_index.shape[1]}")

        index_obj = {
            "version":      1,
            "n_nodes":      n_nodes,
            "nphi":         self._nphi,
            "phi_planes":   self._phi_planes,
            "steps":        self._step_list,
            "field_names":  self._field_names_data,
        }
        with open(self.processed_index, "w") as fp:
            json.dump(index_obj, fp, indent=2)

    # ── sample discovery ─────────────────────────────────────────────────────

    def discover_samples(self) -> List[SampleId]:
        """Return all (phi_plane, start_step) pairs valid as input starts."""
        n_steps = len(self._step_list)
        # need n_in input steps + at least 1 future step for target
        max_start = n_steps - self.nsteps_input - 1
        if max_start < 0:
            return []

        samples = []
        for ip in self._phi_planes:
            group = f"phi_{ip:03d}"
            for si in range(max_start + 1):
                step = self._step_list[si]
                samples.append(
                    SampleId(group=group, item=f"step_{step:05d}", path="", t=step)
                )
        return samples

    # ── getitem ──────────────────────────────────────────────────────────────

    def _load_topo(self) -> dict:
        if self._topo_cache is None:
            topo_path = os.path.join(self.processed_dir, _TOPOLOGY_FILE)
            self._topo_cache = torch.load(topo_path, weights_only=False, map_location="cpu")
        return self._topo_cache

    def _load_phi_fields(self, step: int, ip: int) -> torch.Tensor:
        """Read one phi plane from step_NNNNN.npz → [N, F] float32 tensor."""
        npz_path = os.path.join(self.path, f"step_{step:05d}.npz")
        d = np.load(npz_path)
        vecs = []
        for fn in self._field_names_data:
            if fn in d:
                vecs.append(d[fn][ip].astype(np.float32))   # [N]
            else:
                vecs.append(np.zeros(self._n_nodes, dtype=np.float32))
        return torch.as_tensor(np.stack(vecs, axis=-1), dtype=torch.float32)  # [N, F]

    def __getitem__(self, index):
        base_idx = self.active_indices[index]
        meta = self.samples[base_idx]

        step_start = meta.t
        ip = int(meta.group.split("_")[1])   # phi plane index

        si_start = self._step_list.index(step_start)
        n_in = self.nsteps_input
        max_lead = len(self._step_list) - si_start - n_in
        leadtime = torch.randint(1, max(2, min(self.leadtime_max + 1, max_lead + 1)), (1,))
        si_target = si_start + n_in + leadtime.item() - 1

        topo = self._load_topo()
        static_ctx = topo["static_ctx"]   # [N, 7]
        pos        = topo["pos"]
        edge_index = topo["edge_index"]
        edge_attr  = topo["edge_attr"]

        # Build input sequence: [N, n_in, 7+F]
        x_list = []
        for si in range(si_start, si_start + n_in):
            s = self._step_list[si]
            fields_t = self._load_phi_fields(s, ip)           # [N, F]
            x_list.append(torch.cat([static_ctx, fields_t], dim=-1))  # [N, 7+F]
        x_seq = torch.stack(x_list, dim=0).permute(1, 0, 2)  # [N, n_in, 7+F]

        # Target: field values only  [N, F]
        target_step = self._step_list[si_target]
        y = self._load_phi_fields(target_step, ip)

        data = Data(
            x=x_seq,
            y=y,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        data.step0       = step_start
        data.target_step = target_step
        data.phi         = ip
        data.leadtime    = leadtime.reshape(-1, 1).to(torch.float32)

        bcs = self._get_specific_bcs()
        return data, bcs


# ── convenience helpers ──────────────────────────────────────────────────────

def build_datasets(
    preprocessed_dirs: List[str],
    field_names: Optional[List[str]] = None,
    phi_planes: Optional[List[int]] = None,
    n_steps: int = 1,
    leadtime_max: int = 1,
    train_val_test=(0.7, 0.15, 0.15),
    **kwargs,
) -> dict:
    """Build train/val/test datasets from multiple preprocessed case directories.

    Returns {'train': ds, 'val': ds, 'test': ds} using ConcatDataset.
    """
    from torch.utils.data import ConcatDataset

    splits = {}
    for split in ("train", "val", "test"):
        datasets = []
        for pdir in preprocessed_dirs:
            ds = XGCGraphDataset(
                pdir,
                field_names=field_names,
                phi_planes=phi_planes,
                n_steps=n_steps,
                leadtime_max=leadtime_max,
                split=split,
                train_val_test=list(train_val_test),
                **kwargs,
            )
            datasets.append(ds)
        splits[split] = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    return splits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test XGCGraphDataset")
    parser.add_argument("preprocessed_dir",
                        help="Directory produced by preprocess_xgc.py")
    parser.add_argument("--n_steps", type=int, default=1)
    parser.add_argument("--leadtime_max", type=int, default=1)
    args = parser.parse_args()

    ds = XGCGraphDataset(
        args.preprocessed_dir,
        n_steps=args.n_steps,
        leadtime_max=args.leadtime_max,
        split="train",
        train_val_test=[0.7, 0.15, 0.15],
    )
    print(f"Dataset: {len(ds)} training samples")
    sample, bcs = ds[0]
    print(f"  x shape     : {sample.x.shape}")   # [N, n_steps, 7+F]
    print(f"  y shape     : {sample.y.shape}")   # [N, F]
    print(f"  pos shape   : {sample.pos.shape}")
    print(f"  edge_index  : {sample.edge_index.shape}")
    print(f"  edge_attr   : {sample.edge_attr.shape}")
    print(f"  phi={sample.phi}  step0={sample.step0}  target={sample.target_step}  lead={sample.leadtime.item():.0f}")
