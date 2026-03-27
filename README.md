# LiteQG: Lightweight Quantized Graph for Approximate Nearest Neighbor Search

A high-performance C++ library for Approximate Nearest Neighbor Search (ANNS) that combines adaptive quantization with graph-based indexing. LiteQG achieves high recall with low memory footprint through PCA-driven dimension segmentation and SIMD-accelerated distance computation.

## Key Features

- **Two-Segment Asymmetric Quantization (SAQ)**: Segments dimensions by PCA variance — 4-bit codes for high-variance dimensions, 1-bit sign codes for low-variance dimensions — maximizing information per bit
- **FastScan Distance Computation**: Batch-processes 32 neighbors at once using AVX-512/AVX2 lookup tables for fast approximate distance scoring
- **Column-Oriented PCA Rotation**: Eliminates horizontal SIMD reductions during query rotation
- **Dynamic Quantization Planning**: DP algorithm allocates bit widths (1/2/4/8-bit) per segment to minimize quantization error under a fixed bit budget

## Requirements

- C++17 compiler
- AVX-512 (recommended) or AVX2
- OpenMP
- Python 3.10+ (for Python bindings)

## Directory Structure

```
├── symqglib/              # Core C++ library
│   ├── qg/                # Quantized graph (index, builder, scanner)
│   ├── quantization/      # SAQ, CAQ, FastScan implementations
│   ├── space/             # Distance metrics (L2, IP)
│   ├── utils/             # PCA, scalar quantization, IO, memory
│   └── third/             # Third-party deps (Eigen, Faiss, SVS)
├── python/                # Python bindings (pybind11)
├── reproduce/             # Benchmarking and reproduction scripts
├── test/                  # C++ test and benchmark code
└── data/                  # Datasets (see data/README.md)
```

## Python Bindings (Recommended)

### Installation

```bash
cd python/
pip install -r requirements.txt
pip install --no-build-isolation .
```

### API

```python
import symphonyqg
import numpy as np

D = 128
N = 1000000
data = np.random.random((N, D)).astype('float32')

# Build index
index = symphonyqg.Index("QG", "L2", num_elements=N, dimension=D, degree_bound=32)
index.build_index(data, EF=200, num_iter=3)

# Query
index.set_ef(100)
query = data[0]
neighbors = index.search(query, k=10)  # returns k neighbor IDs

# Persistence
index.save("my_index.bin")

index2 = symphonyqg.Index("QG", "L2", num_elements=N, dimension=D, degree_bound=32)
index2.load("my_index.bin")
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `index_type` | Index type, currently `'QG'` |
| `metric` | Distance metric, currently `'L2'` |
| `num_elements` | Number of vectors |
| `dimension` | Vector dimensionality |
| `degree_bound` | Max out-degree of graph, must be a multiple of 32 (default: 32) |
| `EF` | Beam size — controls speed/recall trade-off for both build and search |

## C++ Build

```bash
mkdir bin/ build/
cd build
cmake ..
make
```

## Datasets

See [`data/README.md`](data/README.md) for dataset download and preprocessing instructions. Supported datasets include SIFT, GIST, Deep, ImageNet, and more.

## Benchmarking

Refer to [`reproduce/README.md`](reproduce/README.md) for instructions on building indices and evaluating query performance (QPS-recall trade-off).

## Acknowledgments

This project builds upon ideas from [SymphonyQG](https://arxiv.org/abs/2411.12229) and incorporates components from Faiss, Eigen, and other open-source libraries.
