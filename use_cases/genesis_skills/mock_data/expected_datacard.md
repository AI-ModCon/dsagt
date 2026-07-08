# Expected Datacard Shape (reference target)

This is the *shape* the generated datacard should cover — not a byte-for-byte
answer. The installed `generating-datacards` skill owns the authoritative
template; this file just lists the sections the agent should populate from the
dataset + the KB-ingested domain docs, so you can spot anything missing.

---

## Dataset: CO2 Methanation Catalyst Screen (2026 Q1)

**Summary.** 8 catalyst formulations screened for CO2 methanation activity and
CH4 selectivity in a fixed-bed reactor. One row per sample.

### Provenance
- Campaign: `methanation-screen-2026Q1`
- Curated from `catalyst_screening.csv`; methods per the lab protocol.
- License: **CC-BY-4.0**

### Schema
A row per `sample_id` with: `composition`, `calcination_temp_C` (°C),
`bet_surface_area_m2_g` (m²/g), `co2_conversion_pct` (%), `ch4_selectivity_pct`
(%), `test_date`, `operator`. (Field units + definitions pulled from the data
dictionary.)

### Collection methodology
- Catalysts: incipient-wetness impregnation, calcined 4 h in static air.
- BET: N2 physisorption, 77 K, multipoint (P/P0 0.05–0.30).
- Activity: 250 °C, 1 atm, H2:CO2 = 4:1, GHSV 12,000 mL·g⁻¹·h⁻¹, steady state at 60 min.

### Descriptive statistics (computed from the data)
- Rows: 8; samples unique; no missing values.
- `co2_conversion_pct`: range ~29–53 %.
- `ch4_selectivity_pct`: range ~80–95 %.

### Limitations / caveats
- Single-run (no replicate error bars).
- Relative ranking, not absolute kinetics.
- Selectivity excludes trace C2+ (< 0.5 %).

### Quality checks
- `sample_id` matches `^CAT-\d{3}$`, unique.
- Percentages in `[0, 100]`; `bet_surface_area_m2_g > 0`.
