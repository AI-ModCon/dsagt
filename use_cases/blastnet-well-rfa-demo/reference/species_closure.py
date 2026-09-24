"""AIDRIN custom module: species-closure check for a BlastNet point table.

BlastNet's lifted-hydrogen-jet trajectories carry eight species mass fractions
(H, H2, O, O2, OH, H2O, HO2, H2O2) and leave N2, the bulk diluent of the air
stream, out of the stored variables.  N2 is therefore the complement of the
stored species, which makes the physical invariant of a valid point
``0 <= sum(Y_i) <= 1`` with the implied ``Y_N2 = 1 - sum(Y_i)`` a valid mass
fraction.  A conversion that rescales a species, transposes a field, or maps a
variable to the wrong name pushes the sum past one, so this is the check that
catches a conversion bug on a trajectory that has no reference file to compare
against.  ``remedy`` drops the violating rows, which is the curation step a
training table needs; a renormalization would hide the bug by construction.

Run it through the registered ``aidrin`` code:

    aidrin run custom <this file> <point table>.csv metric
    aidrin run custom <this file> <point table>.csv remedy
"""

from typing import Any

import pandas as pd

from aidrin.custom_metrics.base_dr import BaseDRAgent

# Float32 fields summed over eight columns, so the closure test needs a
# tolerance above the accumulated rounding error of the sum itself.
TOLERANCE = 1e-4

# What a correct conversion of this case reports: no violating point, and an
# implied diluent running from the nitrogen of the diluted fuel stream to the
# nitrogen of air.  The two ends are the check worth reading, since a rescaled
# or mismapped species moves them where a violation count may not.
EXPECTED_DILUENT_RANGE = (0.350, 0.767)

SPECIES_PREFIX = "mass_fraction_"


def species_columns(frame: pd.DataFrame) -> list[str]:
    """Return the species mass-fraction columns, in name order."""
    columns = sorted(c for c in frame.columns if c.startswith(SPECIES_PREFIX))
    if not columns:
        raise ValueError(
            f"no {SPECIES_PREFIX}* columns in the table; "
            "sample the WELL file into a point table first"
        )
    return columns


def implied_diluent(frame: pd.DataFrame) -> pd.Series:
    """Return the mass fraction the stored species leave to the diluent."""
    return 1.0 - frame[species_columns(frame)].sum(axis=1)


class CustomDR(BaseDRAgent):
    """Species closure over a sampled WELL point table."""

    def __init__(self, dataset: Any, **kwargs):
        super().__init__(dataset, **kwargs)

    def metric(self, **kwargs):
        """Report how many points imply an out-of-range diluent mass fraction."""
        frame: pd.DataFrame = self.dataset
        columns = species_columns(frame)
        diluent = implied_diluent(frame)
        above = diluent < -TOLERANCE
        below = diluent > 1.0 + TOLERANCE
        negative = (frame[columns] < -TOLERANCE).any(axis=1)
        violations = above | below | negative
        rows = len(frame)
        return {
            "species_columns": columns,
            "tolerance": TOLERANCE,
            "rows": rows,
            "closure_violations": int(violations.sum()),
            "closure_violation_rate": round(float(violations.sum()) / rows, 6),
            "sum_exceeds_one": int(above.sum()),
            "sum_below_zero": int(below.sum()),
            "negative_mass_fraction": int(negative.sum()),
            "implied_diluent_min": round(float(diluent.min()), 6),
            "implied_diluent_max": round(float(diluent.max()), 6),
            "implied_diluent_mean": round(float(diluent.mean()), 6),
            "expected_diluent_range": list(EXPECTED_DILUENT_RANGE),
        }

    def remedy(self, **kwargs) -> pd.DataFrame:
        """Return the table with the closure-violating points removed."""
        frame: pd.DataFrame = self.dataset
        columns = species_columns(frame)
        diluent = implied_diluent(frame)
        keep = (
            (diluent >= -TOLERANCE)
            & (diluent <= 1.0 + TOLERANCE)
            & (frame[columns] >= -TOLERANCE).all(axis=1)
        )
        return frame[keep].reset_index(drop=True)
