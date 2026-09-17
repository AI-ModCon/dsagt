# Measurement Protocol — Methanation Screen v2

This protocol describes how the values in `catalyst_screening.csv` were
produced. It is domain context for the curation agent: ingest it into the
knowledge base so the agent can describe methods + provenance accurately in the
datacard without re-deriving them.

## Catalyst preparation

Catalysts were prepared by incipient-wetness impregnation of the support with
aqueous metal-nitrate precursors, dried at 110 °C overnight, and calcined in
static air for 4 h at the temperature recorded in `calcination_temp_C`.
Promoted samples (Ce, K, Mn) were co-impregnated.

## BET surface area

Specific surface area (`bet_surface_area_m2_g`) was measured by N2 physisorption
at 77 K on a Micromeritics ASAP 2020, multipoint BET in the relative-pressure
range 0.05–0.30, after degassing at 200 °C for 6 h.

## Reactor screen

Activity was measured in a fixed-bed quartz reactor, 50 mg catalyst diluted with
SiC, at **250 °C, 1 atm, H2:CO2 = 4:1, GHSV = 12,000 mL·g⁻¹·h⁻¹**. Steady state
was reached after 60 min on stream. Effluent was analyzed by online GC-TCD/FID.

- `co2_conversion_pct` = (CO2_in − CO2_out) / CO2_in × 100
- `ch4_selectivity_pct` = CH4 carbon / (total converted carbon) × 100

## Known limitations

- Single-run measurements (no replicate error bars in this screen).
- Selectivity excludes trace C2+ (< 0.5 %), lumped into the balance.
- Intended for **relative** ranking of formulations, not absolute kinetics.

## Provenance

- Instrument campaign: `methanation-screen-2026Q1`
- Raw GC traces + reactor logs archived in the lab LIMS under the same campaign id.
- License for the curated dataset: **CC-BY-4.0**.
