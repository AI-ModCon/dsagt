# HDMF Schema Builder Skill

A Goose skill for designing HDMF (Hierarchical Data Modeling Framework) schemas for organizing scientific data in HDF5 format.

## What is HDMF?

HDMF is a framework for creating structured, self-documenting HDF5 files. It's particularly useful for:

- **AI/ML training data**: Organize datasets for efficient loading during training
- **Data sharing**: Create self-describing files that others can easily understand
- **Reproducibility**: Embed metadata and provenance information
- **Standards compliance**: Build on community standards for scientific data

## When to Use This Skill

Use this skill when you need to design a data schema for:

- Scientific datasets (time-series, imaging, simulation results, sensor data, etc.)
- Multi-modal experimental or simulation data
- Hierarchical or structured data
- Data that will be shared or archived
- Training datasets for machine learning models

## How to Use

### 1. Invoke the Skill

From a Goose session:

```
/hdmf-schema
```

Or simply ask:

```
Help me design an HDMF schema for my experimental data.
```

### 2. Interactive Interview

The agent will ask detailed questions about your data:

- **Domain**: What kind of data is this?
- **Structure**: What are the main entities and relationships?
- **Data types**: What measurements and metadata do you have?
- **Access patterns**: How will you use this data?

Be ready to describe:
- The experimental or simulation setup
- Data entities (samples, runs, frames, measurements, etc.)
- Relationships between entities
- Temporal structure (continuous, discrete timesteps, events)
- Metadata to capture

### 3. Schema Proposal

After gathering requirements, the agent will:

1. Recommend an HDMF pattern (timeseries, tabular, multimodal, etc.)
2. Show a complete schema with all entities and fields
3. Explain the design rationale
4. Offer to generate implementation code

### 4. Implementation

Once the schema is approved:

1. Save the schema YAML file
2. Use DSAGT pipeline builder to create data conversion pipelines
3. Validate data against the schema
4. Generate HDF5 files conforming to the schema

## Example Session

```
User: I need to organize cryoEM tomography data with particle picking
      coordinates and classification results for training a particle
      detection model.

Agent: I'll help you design an HDMF schema. Let me understand your data structure:

      1. What are the main components?
         - 3D tomogram volumes?
         - 2D projections at different tilt angles?
         - Particle coordinates in 3D?
         - Classification or pose information?

User: Yes. We have:
      - Tomograms reconstructed from tilt series (512x512x200 voxels)
      - Particle coordinates with orientation quaternions
      - Per-particle class labels and confidence scores
      - CTF parameters for each tilt series

Agent: Got it. A few more questions:

      2. How are the tilt series structured?
         - How many angles per series?
         - What metadata per tilt image?

      3. How will you access this data?
         - Load entire tomograms at once?
         - Extract sub-volumes around particles?
         - Batch process particles from multiple tomograms?

[... interview continues ...]

Agent: Based on your requirements, I recommend a hierarchical schema:

      ## HDMF Schema: CryoEM Tomography Dataset

      **Pattern**: N-dimensional arrays with tabular metadata

      ### Structure
      Dataset
      ├── tomograms/
      │   ├── tomogram_data [num_tomo, depth, height, width]
      │   └── tomogram_metadata (table)
      │       ├── tomo_id
      │       ├── pixel_size_angstrom
      │       └── dose_total
      ├── particles (table)
      │   ├── particle_id
      │   ├── tomo_id (reference)
      │   ├── coordinates [x, y, z]
      │   ├── orientation [quaternion]
      │   ├── class_label
      │   └── confidence
      └── tilt_series/
          ├── images [series, tilts, height, width]
          ├── angles [series, tilts]
          └── ctf_params (table)

      [... full schema YAML ...]

      Does this meet your needs?

User: Yes, looks perfect!

Agent: Great! I'll:
      1. Save the type definitions to cryoem_types.yaml and a
         namespace.yaml that references it — namespace.yaml is
         the file you share with collaborators and tools
      2. Generate a Python writer class
      3. You can then use DSAGT to build a pipeline that converts
         your existing MRC/STAR files into this format.
```

## Templates

The `templates/` directory contains example schemas for common patterns:

### `tabular_example.yaml`
- **Use case**: Structured tabular data with relationships
- **Features**: DynamicTable with VectorData columns, foreign key references
- **Good for**: Sample metadata, experimental conditions, measurement catalogs

### `ndimensional_example.yaml`
- **Use case**: Multi-dimensional tensor/array data
- **Features**: 2D matrices, 3D image stacks, 4D video/volumetric data
- **Good for**: Image data, simulation outputs, feature embeddings, tensor datasets

### `ragged_array_example.yaml`
- **Use case**: Variable-length sequences and arrays
- **Features**: VectorData + VectorIndex pattern for ragged data
- **Good for**: Time series with different lengths, tokenized text, event sequences

### `complex_columns_example.yaml`
- **Use case**: Tables with array-valued columns
- **Features**: DynamicTable with n-dimensional and ragged array columns
- **Good for**: Datasets where each row has associated images, sequences, or feature vectors

## Namespaces and Schema Sharing

Every HDMF schema intended for sharing is distributed as a **namespace**. The namespace file (`namespace.yaml`) is the single entry point that declares the schema's identity, version, authors, and the list of YAML files containing the type definitions.

### Why a Namespace?

- **Discoverability**: The namespace name (e.g., `xrd-schema`) uniquely identifies your schema and is used by HDMF to load types and resolve `data_type_inc` references.
- **Versioning**: Namespaces carry a semantic version, so consumers can pin to a specific version.
- **Reuse**: Other schemas can declare a dependency on your namespace and extend your types.
- **Single distribution unit**: Share the `namespace.yaml` and its referenced source files together — users point HDMF at the namespace file to load everything.

### Structure

A schema package typically looks like:

```
my-schema/
├── namespace.yaml     ← the file users reference
└── types.yaml         ← data_type_def entries (can be split across multiple files)
```

The `namespace.yaml` declares the dependency on `hdmf-common` (since your types will inherit from `DynamicTable`, `VectorData`, etc.) and lists all source files:

```yaml
namespaces:
  - name: my-schema
    doc: Schema for my scientific dataset
    version: 1.0.0
    author:
      - Jane Smith
    contact:
      - jsmith@example.org
    full_name: My Scientific Dataset Schema
    schema:
      - namespace: hdmf-common   # required if using DynamicTable, VectorData, etc.
      - source: types.yaml
        doc: Type definitions for my dataset
        title: My Dataset Types
```

### Loading in Python

```python
from hdmf.common import get_manager
from hdmf.spec import NamespaceCatalog

catalog = NamespaceCatalog()
catalog.load_namespaces('path/to/namespace.yaml')
```

When the agent proposes a schema, it will always output both the type definition YAML(s) **and** the `namespace.yaml` file. Start by pointing collaborators and tools at `namespace.yaml`.

## Integration with DSAGT

After creating your schema:

### 1. Build Conversion Pipeline

Use DSAGT's pipeline builder to convert existing data:

```
I have the HDMF schema in schema.yaml.
Help me build a pipeline to convert my existing CSV and binary files
into this HDF5 format.
```

The agent will:
- Analyze your current data format
- Propose tools for reading, transforming, and writing
- Create a pipeline that populates the HDMF file

### 2. Validate Data

Create validation tools to ensure data conforms:

```python
# tools/validate_hdmf.py
def validate_schema(hdf5_file, schema_file):
    # Check all required fields exist
    # Validate data types and shapes
    # Verify relationships and references
    pass
```

### 3. Quality Checking

Use profiling tools to inspect HDMF files:

```
Profile my HDMF file dataset.h5 to check:
- Are all required fields populated?
- Are array dimensions consistent?
- Do indices match expected ranges?
- Are timestamps or frame numbers monotonic?
```

## Best Practices

### Start Simple
- Begin with essential fields, add complexity later
- Don't over-engineer the initial schema
- Iterate based on actual usage

### Document Everything
- Every field needs a clear description
- Specify units for all measurements
- Document relationships and references

### Think About Access Patterns
- How will you load data during training?
- What queries will be common?
- Design for your actual use case

### Use Community Standards
- Check if standards exist in your domain
- Build on existing schemas when possible
- Contribute your schema back to the community

### Plan for Evolution
- Leave room for future additions
- Use versioning for schema changes
- Document migration paths

## Tips for Different Data Types

### Timeseries Data
- Choose appropriate sampling rate representation
- Consider chunking strategy for large arrays
- Use timestamps vs. uniform sampling based on your needs

### Event Data
- Use VectorData for variable-length event lists
- Include event metadata (duration, type, etc.)
- Consider indexing for fast queries

### Hierarchical Data
- Use DynamicTables with references
- Keep hierarchy shallow when possible
- Document the relationship graph

### Multi-modal Data
- Use consistent time reference across modalities
- Provide alignment information
- Allow independent access to each modality

## Common Questions

**Q: Should I store raw or processed data?**
A: Often both. Raw in `/acquisition`, processed in `/processing`. Document the processing pipeline.

**Q: How do I handle missing values?**
A: Use NaN for numeric data, empty strings for text, or create a separate "valid" flag column.

**Q: What about very large datasets?**
A: Use chunking and compression. Design for out-of-core access. Store indices for common queries.

**Q: Can I modify the schema later?**
A: Yes, but version it carefully. You may need migration scripts to update existing files.

**Q: Should I store metadata in the HDF5 file or separately?**
A: Store essential metadata in the file (self-describing). Keep extensive documentation in separate files but reference them.

## Next Steps

After using this skill to design your schema:

1. **Save `namespace.yaml` and the type definition YAML(s)** — these files together are your shareable schema package
2. **Review with collaborators** to ensure it meets all needs
3. **Distribute by sharing `namespace.yaml`** (and the accompanying type files) — this is the file users and tools reference
4. **Use DSAGT pipeline builder** to create data conversion tools, pointing it at `namespace.yaml`
5. **Validate** a few test files to ensure the schema works
6. **Version control** the schema package along with your data

## Resources

- [HDMF Documentation](https://hdmf.readthedocs.io/) - Official HDMF documentation
- [HDMF Common Schema](https://hdmf-common-schema.readthedocs.io/) - Common data types and patterns
- [HDF5 Documentation](https://www.hdfgroup.org/solutions/hdf5/) - Underlying file format

## Feedback

This skill is designed to be consultative and iterative. If the agent:
- Asks too many questions, say "let's focus on the essentials"
- Jumps to solutions too quickly, ask it to gather more requirements
- Proposes an overly complex schema, ask for a simpler version

The goal is to design a schema that works for *your* use case, not a theoretical perfect schema.
