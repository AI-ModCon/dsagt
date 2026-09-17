# Data Dictionary — CO2 Methanation Catalyst Screening

One row per catalyst sample tested in the fixed-bed reactor screen.

| Column | Type | Units | Definition |
|---|---|---|---|
| `sample_id` | string | — | Unique catalyst identifier (`CAT-NNN`). Primary key. |
| `composition` | string | — | Active metal / promoter / support, e.g. `Ni-Ce/Al2O3` = Ni + Ce promoter on alumina. |
| `calcination_temp_C` | integer | °C | Calcination temperature during catalyst preparation. |
| `bet_surface_area_m2_g` | float | m²/g | BET specific surface area (N2 physisorption, see protocol). |
| `co2_conversion_pct` | float | % | CO2 converted at steady state (250 °C, 1 atm, H2:CO2 = 4:1). |
| `ch4_selectivity_pct` | float | % | Carbon selectivity to CH4 (balance: CO + higher hydrocarbons). |
| `test_date` | date | ISO 8601 | Date the reactor run was performed. |
| `operator` | string | — | Lab notebook operator initials. |

## Controlled vocabularies

- **Supports:** `Al2O3`, `TiO2`, `SiO2`, `ZrO2`.
- **Promoters:** `Ce`, `K`, `Mn` (optional; absent for unpromoted catalysts).

## Quality rules

- `co2_conversion_pct` and `ch4_selectivity_pct` are in `[0, 100]`.
- `bet_surface_area_m2_g > 0`.
- Every `sample_id` is unique and matches `^CAT-\d{3}$`.
