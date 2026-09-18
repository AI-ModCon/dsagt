# Microbial Isolate Genomes

Whole genome shotgun sequencing data from 11 bacterial isolates for developing a whole genome gene-content foundation model.

## Description

This dataset comprises paired-end 150bp Illumina reads from 11 non-pathogenic bacterial isolates. The sequencing captures comprehensive genomic information across each organism's genome, providing a snapshot of genetic diversity and gene content across different bacterial species or strains.

This is a representative subset of the larger training data for a whole genome gene-content foundation model. The model will learn to recognize patterns in bacterial gene content, genomic architecture, and relationships between genetic elements across diverse bacterial genomes.

## Files

11 interleaved paired-end FASTQ files (gzipped), ~3.5 GB total:

| File | Size |
|------|------|
| `53162.2.609630.AAAGGCTAGA-GATTCAGTTA.filter-ISO.fastq.gz` | 249 MB |
| `53162.2.609630.AAGGCATATT-TGCAATACTT.filter-ISO.fastq.gz` | 373 MB |
| `53162.2.609630.AGAAACGCAC-GAAATGTTAG.filter-ISO.fastq.gz` | 297 MB |
| `53162.2.609630.CCTCTATAAC-GTGTTAAGCG.filter-ISO.fastq.gz` | 714 MB |
| `53162.2.609630.CTACCAGGGC-TGTATGCGAG.filter-ISO.fastq.gz` | 336 MB |
| `53162.2.609630.GAATTATATC-ATTCTGTACT.filter-ISO.fastq.gz` | 282 MB |
| `53162.2.609630.GATAATCCTT-CAGGGATCAG.filter-ISO.fastq.gz` | 256 MB |
| `53162.2.609630.TCGGTCGCTT-TTCCCGAAGG.filter-ISO.fastq.gz` | 318 MB |
| `53163.3.610169.AAAGGTCTGG-GTAAACAGGC.filter-ISO.fastq.gz` | 348 MB |
| `53163.3.610169.CGATCCGCAT-CATGAGCTAG.filter-ISO.fastq.gz` | 382 MB |
| `53163.3.610169.TGCGAGGATA-GTGGGAGGGT.filter-ISO.fastq.gz` | 305 MB |

### Naming convention

`<project>.<run>.<flowcell>.<index1>-<index2>.filter-ISO.fastq.gz`

- Two sequencing projects: `53162` (8 samples) and `53163` (3 samples)
- Dual-index barcodes identify individual isolates
- `filter-ISO` indicates isolate-filtered reads

## Preprocessing Already Applied

These reads have been demultiplexed and filtered upstream (`filter-ISO`). Sequencing adapters and barcode sequences have been removed. The files are ready for quality-based trimming and assembly.

## Format

- **Type**: Interleaved paired-end FASTQ (gzipped)
- **Platform**: Illumina
- **Read length**: 150 bp
- **Encoding**: Phred+33 quality scores

## Source

Original data located on NERSC: `/global/cfs/projectdirs/amsc002/base_data/example_famous_data`

## Intended Processing Pipeline

1. **Preprocessing & QC**: Quality trimming, low-quality read filtering, read length filtering (fastp)
2. **Genome Assembly**: de novo assembly into contigs (MEGAHIT). The parameters for a laptop are in `docs/fastp_megahit_best_practices.md` beside this data.

3. **Gene Prediction & Annotation**: Gene calling, functional annotation
4. **Feature Extraction**: Gene-content profiles, orthologous group clustering, gene embeddings
5. **Dataset Formatting**: Structured format for model training (e.g., HDF5)
