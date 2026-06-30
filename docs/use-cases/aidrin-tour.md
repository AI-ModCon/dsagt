# AIDRIN Full Feature Tour

**Domain:** AI data readiness — quality, fairness, and privacy assessment

**Tools:** [`aidrin`](https://github.com/idtlab/AIDRIN) (AI Data Readiness Inspector CLI)

**Dataset:** UCI Adult census (bundled with AIDRIN)

**Source:** [`use_cases/aidrin_readiness/`](https://github.com/AI-ModCon/dsagt/tree/main/use_cases/aidrin_readiness/)

## Overview

This use case drives **every** AIDRIN metric through DSAgt on a single tabular dataset —
exercising all 15 metrics across data-quality, impact-of-data-on-AI, fairness-and-bias, and
data-governance, each recorded via `dsagt-run`. The bundled UCI Adult extract has an ID,
quasi-identifiers, sensitive attributes, and a prediction target, so the full suite (including the
fairness and privacy metrics) applies.

## Guides

- [Full AIDRIN Feature Tour](https://github.com/AI-ModCon/dsagt/blob/main/use_cases/aidrin_readiness/aidrin_full_tour_demo.md) — full walkthrough (all 15 metrics on UCI Adult).
