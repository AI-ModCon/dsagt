**fastp & MEGAHIT Best Practices**

*Microbial Isolate Assembly on Laptop-Class Hardware*

Overview

This document outlines best practices for preprocessing and assembling
microbial isolate short-read sequencing data using fastp for quality
control and MEGAHIT for de novo assembly. The guidance is specifically
tuned for running on a laptop with constrained RAM and CPU resources.

MEGAHIT is the preferred assembler for this context. It uses
significantly less memory than SPAdes while delivering comparable
assembly quality for microbial isolates, and it handles resource limits
gracefully through explicit memory and thread controls.

Step 1: Quality Control with fastp

fastp performs adapter trimming, quality filtering, and QC report
generation in a single pass over the input data. Its single-scan design
minimizes I/O overhead, which is especially beneficial when working from
a laptop hard drive.

Recommended Command

> fastp \\
>
> \--in1 sample_R1.fastq.gz \\
>
> \--in2 sample_R2.fastq.gz \\
>
> \--out1 sample_R1.trimmed.fastq.gz \\
>
> \--out2 sample_R2.trimmed.fastq.gz \\
>
> \--detect_adapter_for_pe \\
>
> -q 20 \\
>
> -l 50 \\
>
> \--thread 4 \\
>
> -h sample_fastp.html \\
>
> -j sample_fastp.json

Key Parameters

  -------------------------- -------------------------------------------------
  **Parameter**              **Description**

  \--detect_adapter_for_pe   Enables automatic adapter detection for
                             paired-end data. Without this flag,
                             auto-detection is only active for single-end
                             input.

  -q 20                      Quality threshold per base. Bases below Phred Q20
                             are considered low quality. A value of 20 is a
                             well-validated standard for microbial isolate
                             assembly pipelines.

  -u 40                      Default: up to 40% of bases in a read may fall
                             below the quality threshold before the read is
                             discarded. Adjust lower (e.g. -u 20) for stricter
                             filtering.

  -l 50                      Discard reads shorter than 50 bp after trimming.
                             Prevents very short remnant reads from degrading
                             assembly graph construction.

  \--thread 4                Limits CPU usage to 4 threads. fastp defaults to
                             using available cores; capping this leaves
                             headroom for the OS on a laptop.

  -h / -j                    Outputs an HTML and JSON QC report showing read
                             quality before and after filtering. Always review
                             the HTML report before proceeding to assembly.
  -------------------------- -------------------------------------------------

Platform-Specific Notes

-   NextSeq / NovaSeq data: fastp automatically detects and trims polyG
    tails, which are a known artifact of the two-color signal system
    used by these instruments. No extra flag needed.

-   TruSeq libraries: If adapter auto-detection is unreliable, supply
    adapters explicitly with \--adapter_sequence
    AGATCGGAAGAGCACACGTCTGAACTCCAGTCA and \--adapter_sequence_r2
    AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT.

-   Trimming stringency: Research on Gram-negative bacterial genomes
    indicates that fastp with Q20 trimming is the highest-performing
    strategy, though overall the effect of trimming on assembly
    completeness is modest. Sensible defaults are sufficient; aggressive
    trimming is not necessary.

Step 2: Assembly with MEGAHIT

MEGAHIT uses a succinct de Bruijn graph approach that allows it to
assemble genomes with substantially less memory than SPAdes. Its default
parameters are tuned for metagenomic complexity, so a few adjustments
are recommended for single microbial isolate data.

Recommended Command

> megahit \\
>
> -1 sample_R1.trimmed.fastq.gz \\
>
> -2 sample_R2.trimmed.fastq.gz \\
>
> -t 4 \\
>
> -m 3000000000 \\
>
> \--min-count 3 \\
>
> \--min-contig-len 500 \\
>
> -o megahit_output/

Key Parameters

  --------------------- -------------------------------------------------
  **Parameter**         **Description**

  -t 4                  Limits threads to 4. MEGAHIT defaults to
                        auto-detecting and using all CPU threads, which
                        will saturate a laptop. Set this explicitly.

  -m 3000000000         Hard memory cap in bytes (3 GB). Using the byte
                        value is more reliable than the fractional flag
                        (-m 0.5), which can misread available RAM and
                        cause segfaults. See memory guidance below.

  \--min-count 3        Filters k-mers appearing fewer than 3 times. The
                        default is 2 (tuned for metagenomes). For isolate
                        data with coverage above \~40x, setting this to 3
                        reduces error k-mers and improves assembly
                        quality.

  \--min-contig-len 500 Discards contigs shorter than 500 bp from the
                        final output. Short contigs are often assembly
                        artifacts and add noise to downstream analysis.

  \--kmin-1pass         Optional: use if the assembly stalls or consumes
                        excessive memory during the first k-mer graph
                        build. Trades some assembly quality for lower
                        memory usage.

  \--continue -o out    Resumes an interrupted assembly from the same
                        output directory. Useful if the laptop sleeps or
                        the run is killed mid-way.
  --------------------- -------------------------------------------------

Memory Management on a Laptop

MEGAHIT\'s fractional memory flag (-m 0.5) depends on it correctly
reading total available RAM, which can be unreliable on macOS and lead
to it claiming more memory than intended, causing segfaults. Using
explicit byte values is more predictable.

Reference values for the -m flag:

  ----------------------------------- -----------------------------------
  **Memory Cap**                      **Byte Value**

  3 GB (recommended)                  3000000000

  4 GB                                4000000000

  6 GB                                6000000000

  8 GB                                8000000000
  ----------------------------------- -----------------------------------

3 GB is the recommended default for a laptop with \~24 GB total RAM
running macOS, which typically has most RAM in use. For a single
microbial isolate (2-6 Mb genome), 3 GB is sufficient for graph
construction. If MEGAHIT fails with a graph-building error (not a
segfault), increase the value. If it segfaults, decrease it.

Checking Available RAM on macOS

Before running assembly, check the current memory state:

> top -l 1 \| grep PhysMem

This gives a summary such as: PhysMem: 23G used (3050M wired, 5613M
compressor), 327M unused. The compressor value represents memory that
macOS has compressed and can partially reclaim. Leave at least 2-3 GB
headroom beyond your -m value for the OS.

Tip: Close Chrome before running assembly. Chrome runs each tab as a
separate process and is significantly more RAM-intensive than Safari on
macOS. Switching to Safari or closing the browser entirely can free 1-2
GB before the assembly starts.

Coverage Check Before Assembly

A rough coverage estimate before assembly can save time. If coverage is
too low, assembly quality will suffer regardless of parameters.

Estimated coverage = (read pairs x read length x 2) / expected genome
size

Most bacteria are 2-10 Mb. As a quick example: 500,000 read pairs x 150
bp x 2 = 150 Mb of sequence. Against a 5 Mb genome that gives \~30x
coverage, which is workable. Below \~20x, assembly completeness degrades
noticeably.

If coverage is very high (\>200x), consider normalizing reads with
bbnorm before assembly to reduce memory consumption:

> bbnorm.sh in1=R1.fastq in2=R2.fastq out=normalized.fq target=100 min=3

Output and Quality Assessment

-   MEGAHIT output: assembled contigs are in
    megahit_output/final.contigs.fa

-   fastp QC report: review sample_fastp.html before assembly to confirm
    adapter removal and read quality distribution look reasonable.

-   Assembly QC: run QUAST on the final contigs to assess N50, total
    assembly length, and contig count. For a typical bacterial isolate,
    expect N50 in the hundreds of kb range and total assembly length
    close to the expected genome size.

> quast.py megahit_output/final.contigs.fa -o quast_output/

Troubleshooting

  --------------------- -------------------------------------------------
  **Parameter**         **Description**

  Segfault at start     Lower the -m byte value. The process is claiming
                        more RAM than the OS can provide. Try 2000000000.

  Graph build failure   MEGAHIT needs more memory. Increase -m
                        incrementally (e.g. to 4000000000).

  Assembly stalls at    Add \--kmin-1pass to reduce memory use during the
  k=21                  first k-mer graph step.

  Very few/short        Check coverage estimate. Also try removing
  contigs               \--min-count 3 and using the default of 2 if
                        coverage is below 40x.

  Run interrupted       Use \--continue -o megahit_output/ to resume from
                        where it stopped.
  --------------------- -------------------------------------------------
