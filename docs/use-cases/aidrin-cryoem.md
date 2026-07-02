# AIDRIN Readiness Gate (Cryo-EM)

**Domain:** AI data readiness — quality assessment of a scientific pipeline

**Codes:** [`aidrin`](https://github.com/idtlab/AIDRIN) (AI Data Readiness Inspector CLI)

**Dataset:** EMPIAR-10017 cryo-EM particle tables (via CryoPPP)

**Source:** [`use_cases/aidrin_readiness/`](https://github.com/AI-ModCon/dsagt/tree/main/use_cases/aidrin_readiness/)

## Overview

This use case shows DSAgt using AIDRIN as a **readiness gate** around a cryo-EM curation step. The
agent registers the AIDRIN CLI, then runs the applicable data-quality and class-balance metrics
*before* and *after* particle curation — with full `dsagt-run` provenance — to measure how much the
pipeline improved AI-readiness. Fairness and privacy metrics are deliberately excluded, since
cryo-EM particle data has no sensitive attributes or personal identifiers.

## Guides

- [Cryo-EM Readiness Gate](https://github.com/AI-ModCon/dsagt/blob/main/use_cases/aidrin_readiness/cryoem_readiness_demo.md) — full walkthrough (self-contained; downloads EMPIAR-10017).
