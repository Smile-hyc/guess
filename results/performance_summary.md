# GPU Password Guessing Performance Summary

## Environment

- OS: Windows 11
- GPU: NVIDIA RTX 5060 Laptop GPU
- CUDA Toolkit: 12.8.1
- nvcc: 12.8.93
- CUDA target: `sm_120`
- Training input: first 3,010,000 lines of `Rockyou-singleLined-full.txt`
- Stop condition: more than 10,000,000 generated guesses

## Results

| Version | Guess time (s) | Hash time (s) | Train time (s) | Final guesses |
| --- | ---: | ---: | ---: | ---: |
| Serial baseline | 0.277775 | 2.18975 | 21.5854 | 10,106,852 |
| Basic single-PT GPU | 4.53492 | 2.44432 | 20.2038 | 10,106,852 |
| Single-PT GPU with timing | 4.50967 | 2.20927 | 20.4308 | 10,106,852 |

## Stage 1 Analysis

The basic GPU implementation preserves the original training, PT ordering,
priority-queue, and MD5 paths. It only moves the final-segment expansion in
`Generate()` to a CUDA kernel. Values are packed into contiguous characters
plus offset and length arrays, and results use a fixed 128-byte stride.

The generated count and progress sequence match the serial baseline. The basic
GPU version is about 16.33 times slower in guess generation because every PT,
including very small PTs, pays allocation, copy, kernel-launch, synchronization,
copy-back, and CPU string-rebuild overhead. Stage 2 measures these costs and
Stage 3 avoids sending small PTs to the GPU.

## Stage 2 Timing Breakdown

| Metric | Value |
| --- | ---: |
| GPU calls | 516 |
| GPU generated guesses | 10,106,852 |
| Average guesses per GPU call | 19,586.92 |
| Host packing | 0.287989 s |
| Device allocation | 2.803778 s |
| H2D copies | 0.300006 s |
| CUDA kernels | 0.032941 s |
| D2H copies | 0.136960 s |
| CPU string rebuild | 0.512353 s |
| Device free | 0.187788 s |
| Total `GenerateGPU` | 4.282755 s |

The kernel itself is fast, but repeated device allocation dominates the measured
GPU path. Allocation accounts for about 65% of `GenerateGPU` time, while kernel
execution accounts for less than 1%. This result motivates both threshold-based
scheduling and batching or buffer reuse.
