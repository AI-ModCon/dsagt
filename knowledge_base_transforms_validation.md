# DSAGT Knowledge Base & Transforms Validation

## Summary

This document describes how to set up and run validation tests for DSAGT's agentic pipeline building workflow. These tests verify that **session-specific registries**, **knowledge base exploration**, and **comprehensive elicitation** work correctly for domain-specific data curation.

---

## Key Design Decisions

### Session Registry as Primary Deliverable

Each DSAGT session produces a `runtime/registry.yaml` containing:
- Tools copied from the base registry
- All generated filters, transforms, and assessments
- The complete pipeline registered as a tool

This makes pipelines fully reproducible and portable.

### Three-Part Elicitation (Before Any Data Processing)

| Category | What to Gather |
|----------|----------------|
| **Domain Knowledge** | Papers, standards, schemas, protocols |
| **Data Details** | Schema, provenance, known issues, semantics, temporal aspects |
| **Pipeline Context** | Current state, existing scripts, downstream ML requirements, success criteria |

### The Workflow

**Initial Setup Phase:**
1. Gather context (domain, data, pipeline requirements)
2. Extend KB with user's domain documents via `kb_ingest`
3. Register user's existing custom tools
4. Review available tools and collections

**Iterative Cycle (per data manipulation step):**
```
UNDERSTAND → EXPLORE KB → APPLY HIERARCHY → DESIGN → GENERATE → REGISTER → EXECUTE → EVALUATE → (next step)
```

The pipeline emerges incrementally through conversation with the domain scientist.

---

## Critical Architecture

### Tool-Mediated Data Access
Agent NEVER directly accesses data. All operations through registered tools.

### Tool Preference Hierarchy
```
1. REGISTERED TOOL       — Use existing tool from session registry
        ↓
2. KB PACKAGE TOOL       — Leverage package documented in KB (NeMo Curator, AIDRIN, etc.)
        ↓
3. CUSTOM IMPLEMENTATION — Custom code using general-purpose libraries
```

### Session Isolation
Everything in `runtime/`:
- `registry.yaml` — Session tool registry (copied from base, modified during session)
- `provenance.log` — Audit trail of all tool executions
- Generated artifacts: `checks/`, `filters/`, `transforms/`, `data/`

The **registry builder server** writes directly to `runtime/registry.yaml`, so newly registered tools are immediately available to the pipeline server.

### KB Exploration
Agent explores KB as a library—listing collections to discover domain docs, package references (NeMo Curator, AIDRIN), and implementation patterns. Packages in KB are Level 2 in the hierarchy.

### Per-Operation Checks
Every filter/transform has check tool that runs before AND after.

---

## Current Package State

### What's Ready ✅

| Component | Status |
|-----------|--------|
| Knowledge Base | ✅ FAISS indexing, semantic search, reranking, batched embeddings |
| Knowledge Server | ✅ `kb_list_collections`, `kb_search`, `kb_ingest` |
| Pipeline Server | ✅ Executes registered tools, copies base registry to session |
| Registry Builder | ✅ Writes to session registry (`runtime/registry.yaml` by default) |
| Core Tools | ✅ `load_csv`, `profile_data`, `validate_schema`, `split_data` |
| Setup Script | ✅ `scripts/setup_core_kb.py` indexes NeMo Curator and AIDRIN |

### Core Knowledge Base Collections

Run `python scripts/setup_core_kb.py` to index:

| Collection | Contents |
|------------|----------|
| `nemo_curator` | NVIDIA NeMo Curator docs, code, tutorials — text filtering, deduplication, quality scoring |
| `aidrin` | AIDRIN framework code and papers — data quality metrics, fairness, FAIR compliance |

---

## Validation Use Cases

### 1. Building Energy Time Series

**Domain context:** HVAC sensor data, building metadata, energy consumption patterns

**Preparation:**
- Sample data: LBNL Building 59 subset or synthetic building data
- Optional additional KB collections: ASHRAE guidelines, Brick schema docs

**Validation flow:**
1. Load and profile building sensor CSV
2. Search KB for NeMo Curator quality filters applicable to time series
3. Search KB for AIDRIN completeness/outlier metrics
4. Generate domain-specific filters (e.g., physical range checks for temperatures)
5. Register and execute pipeline
6. Evaluate before/after data quality

### 2. Cryo-Electron Microscopy Data

**Domain context:** Particle image stacks, metadata, preprocessing for reconstruction

**Preparation:**
- Sample data: CryoPPP sample or synthetic micrograph metadata
- Optional additional KB collections: cryoEDU tutorials, CryoCRAB documentation

**Validation flow:**
1. Load and profile particle metadata
2. Search KB for relevant quality assessment approaches
3. Generate filters for CTF parameters, defocus ranges
4. Register and execute pipeline
5. Evaluate data readiness for downstream ML

---

## Running Validation Tests

### Environment Setup

```bash
# Install package
pip install -e .

# Set embedding API key
export LLM_API_KEY="your-api-key"

# Index core knowledge base (NeMo Curator + AIDRIN)
python scripts/setup_core_kb.py

# Verify collections
python -c "from dsagt.knowledge import KnowledgeBase; kb = KnowledgeBase('./kb_index'); print(kb.list_collections())"
```

### Before Validation Checklist

- [ ] Package installed: `pip install -e .`
- [ ] API key set: `export LLM_API_KEY=...`
- [ ] Core KB indexed: `python scripts/setup_core_kb.py`
- [ ] Sample dataset ready
- [ ] Test Goose session with all three MCP servers

### Running the Validation Test

```bash
# Start Goose with all DSAGT servers
goose session \
  --with-extension 'dsagt-pipeline-server --registry registry.yaml' \
  --with-extension 'dsagt-registry-builder' \
  --with-extension 'dsagt-knowledge-server'
```

### Expected Agent Behavior

1. Ask about domain docs, data details, and pipeline context
2. List KB collections to discover available resources
3. Search KB for relevant filtering/assessment approaches
4. Profile data with domain-aware interpretation
5. Present options, elicit custom rules from user
6. Generate code and register in session registry
7. Run pre-assessment → pipeline → post-assessment
8. Present before/after comparison
9. Session registry serves as reproducible deliverable

---

## Files in Package

```
DSAGT/
├── .goosehints              ← Agentic workflow guidance
├── registry.yaml            ← Base tool registry
├── goose.yaml               ← Goose MCP server configuration
├── scripts/
│   └── setup_core_kb.py     ← Knowledge base setup script
├── src/dsagt/
│   ├── knowledge/           ← KB with FAISS indexing
│   ├── servers/             ← MCP servers (pipeline, registry builder, knowledge)
│   └── core/                ← Registry management
├── tools/                   ← Reference tool scripts
├── demo/                    ← Example data
├── kb_index/                ← Knowledge base index (generated)
├── runtime/                 ← Session directory (generated)
│   ├── registry.yaml        ← Session tool registry
│   └── provenance.log       ← Execution audit trail
└── docs/
    └── knowledge_base_transforms_validation.md  ← This file
```

---

## Troubleshooting

### Knowledge Base Setup Fails

**API timeout:** The embedding API may be slow. The setup script uses 300s timeout and batching. If still failing, check API availability.

**Git clone fails for AIDRIN:** Ensure you're using the fixed setup script with correct URL (`kaveenh/AIDRIN`) and branch (`develop`).

### Tools Not Appearing in Session

Verify both servers point to the same runtime directory:
- Pipeline server: `--runtime-dir ./runtime`
- Registry builder: `--registry ./runtime/registry.yaml`

### KB Search Returns No Results

1. Verify collections exist: `kb_list_collections`
2. Check collection was indexed successfully (look for chunk count in setup output)
3. Try broader search terms
