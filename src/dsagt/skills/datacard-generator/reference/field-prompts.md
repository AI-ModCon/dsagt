# Per-Level Field Prompts

Ask 3–5 fields at a time. Only prompt for fields not already auto-filled.

## Level 1 — Discoverable

- Dataset name and title
- Project name
- Dataset identifiers (DOI, URL)
- Dataset description
- Keywords
- Release status

## Level 2 — Interoperable & Reusable (adds to Level 1)

- Contact point (name, email, affiliation, ORCID)
- Access policy (sensitivity tier, access level, authorization) — see lookup-tables.md for sensitivity tiers
- License (SPDX ID, name, URL)
- Science domain
- Originating research organization
- Funding sources
- Dataset authors and contributors
- Provenance — how was the data generated or collected?
- Related resources (datasets, publications, software)
- Maintenance and stewardship plans
- Key dates (collection start/end, issued, modified)

## Level 3 — Understandable & Trustworthy (adds to Level 2)

- Semantic/schema info — ontologies, controlled vocabularies (must be user-provided)
- Data quality — completeness, known issues, validation methods, noise, uncertainty
- Integrity — checksums, fixity policy, versioning strategy
- AI/ML usage — AI-ready status, training/inference permissions, bias risks, safety notes
- Variable-level documentation — name, description, unit, value labels (one row per variable)
- Missing data codes
- Data processing steps

**If the user selects Level 3 but provides minimal information:** generate the card with N/A for unprovided fields, list them in the review summary, and note which ones most improve trustworthiness.