# SAQ-Graph Integration Plan

## Background

### Current State (lite-caq branch)
- Graph build: exact L2 with raw float vectors
- Graph search: CAQ 8-bit codes + rescaled IP estimator
- Layout: `[caq_code (D bytes) | neighbor_IDs]` per node, `CaqFactors` separate array
- Problem: symmetric quantization `[-vmax, vmax]` wastes half codebook on SIFT (non-negative data)
- Result: recall drops ~1.5% vs SQ8 at all EFS levels

### Original SymphonyQG Design (already in codebase)
- FHT random rotation (`rotator.hpp`) + RaBitQ 1-bit binarization (`rabitq.hpp`)
- 6-bit query quantization → 4-bit LUT fastscan (`fastscan_impl.hpp`, `qg_scanner.hpp`)
- Per-node layout: packed 4-bit codes of ALL neighbors + per-neighbor factors + neighbor IDs
- Distance: `dist = triple_x + sqr_y + fac_dq * width * (2*result - sumq) + fac_vq * vl`

### SAQ Paper Key Ideas
1. **PCA rotation** → concentrate variance in leading dimensions
2. **Dimension segmentation** → DP allocates variable bits per segment (high variance → more bits)
3. **CAQ code adjustment** → coordinate descent maximizing cos(o, o_a), better than plain rounding
4. **Multi-stage filtering** → variance bound → 1-bit fast → full-bit accurate
5. **Rescale estimator** → `<o,q> = (||o||^2 / <o,o_a>) * <o_a, q>`, unbiased

---

## Architecture Overview

```
Build:
  raw vectors ──→ graph construction (exact L2, unchanged)
       │
       ├──→ PCA fit ──→ rotation matrix R
       │
       └──→ rotate all vectors ──→ segment (DP) ──→ CAQ encode per segment
                                                          │
                                    ┌─────────────────────┤
                                    │                     │
                              4-bit codes            per-vector factors
                              (fastscan pack)        (rescale, norms)
                                    │                     │
                                    └────────┬────────────┘
                                             │
                                     colocate per-node
                                     (pack neighbor codes)

Search:
  query ──→ PCA rotate ──→ per-segment LUT build ──→ graph traversal
                                                          │
                                              ┌───────────┤
                                              │           │
                                         fastscan     candidate
                                         batch 32     filtering
                                         neighbors    (multi-stage)
                                              │           │
                                              └─────┬─────┘
                                                    │
                                              update heap
```

---

## Phase 0: Fix Asymmetric Quantization (Quick Win)

**Goal**: Fix the 1.5% recall gap vs SQ8 without changing architecture.

**Root Cause**: `caq_encode_single` uses symmetric range `[-vmax, vmax]`. SIFT vectors are
non-negative, so half the codebook (codes 0-127) maps to negative values that never occur.

**Change**: Use per-vector `[vmin, vmax]` asymmetric range.
- `delta = (vmax - vmin) / 256`
- Fold `delta` and `vmin` into precomputed factors:
  ```
  fac_dot = rescale * delta
  fac_sum = rescale * (0.5 * delta + vmin)
  ```
- `CaqFactors` grows from 2 to 3 floats: `{o_l2sqr, fac_dot, fac_sum}`
- Search hot path unchanged: `est_ip = fac_dot * dot(code, q) + fac_sum * sum_q`

**Files**: `caq.hpp` only (struct + encoder)

**Expected**: recall >= SQ8 at same memory, slightly better due to code adjustment.

---

## Phase 1: PCA Rotation

**Goal**: Reorder dimensions by variance for Phase 2 segmentation.

**Design**:
- At build time: compute covariance matrix → eigendecomposition → store rotation matrix `R` (D x D)
- Rotate all vectors: `x' = R^T * x` (project onto principal components)
- Eigenvalues = per-dimension variances in rotated space (needed for DP in Phase 2)
- Query rotation: `q' = R^T * q` → O(D^2) per query, ~16K FLOPs for D=128, negligible

**Implementation**:
- New class `PCARotator` in `symqglib/utils/pca_rotator.hpp`
  - `fit(data, num_points, dim)` → compute R, store eigenvalues
  - `rotate(src, dst)` → SIMD matrix-vector multiply
  - `save/load` → persist R matrix
- Replace `FHTRotator` usage (FHT is random rotation; PCA is data-adaptive, strictly better for SAQ)
- Store eigenvalues for segmentation DP

**Note**: PCA rotation preserves L2 distance (orthogonal transform), so build-time exact L2
can operate on either raw or rotated vectors.

**Files**: new `pca_rotator.hpp`, modify `qg.hpp` (add rotator member, rotate in copy_vectors and search)

---

## Phase 2: Dimension Segmentation + Variable Bit Allocation

**Goal**: Allocate bits optimally across dimensions. High-variance dims get more bits,
low-variance dims get fewer. Total bit budget can be < 8 bits/dim for memory savings.

**Design** (from SAQ paper, Algorithm 2):
- Input: per-dim variances (eigenvalues from Phase 1), average bit budget B_avg
- DP state: `f[num_segments][dim_offset][bits_used]` → minimum quantization error
- Error model: `error(seg) = sum(variance[d]) / 2^bits` for segment with `bits` per dim
- Constraint: segment dims must be multiples of `kDimPaddingSize` (for SIMD alignment)
- Output: `QuantPlan = [(dim_len_1, bits_1), (dim_len_2, bits_2), ...]`

**Example** (SIFT 128D, B_avg=4):
```
Segment 1: dims  0-31  (highest variance) → 8 bits → 32 bytes
Segment 2: dims 32-63  (medium variance)  → 4 bits → 16 bytes
Segment 3: dims 64-95  (lower variance)   → 2 bits →  8 bytes
Segment 4: dims 96-127 (lowest variance)  → 1 bit  →  4 bytes
Total: 60 bytes/vector vs 128 bytes (uniform 8-bit)
```

**Implementation**:
- Port `SaqDataMaker::dynamic_programming()` from SAQ's `saq_data.hpp:147-227`
- Adapt for graph use: segments produce independent quantizers
- Each segment has its own CAQ encoder (independent delta, rescale, code adjustment)

**Files**: new `symqglib/quantization/saq_plan.hpp`, modify `caq.hpp` (per-segment encode)

---

## Phase 3: Fastscan Layout + Batch Neighbor Scan

**Goal**: Use LUT-based 4-bit fastscan to batch-compute distances for all 32 neighbors
simultaneously, replacing the per-neighbor sequential dot product.

**Key Insight**: Graph degree_bound = 32 = kBatchSize. When visiting a node, we scan
exactly one batch of 32 neighbors — perfect for fastscan.

### 3a. Data Layout Redesign

Current per-node row in qdata_:
```
[this_node's_caq_code (D bytes) | neighbor_IDs (degree * 4)]
```

New per-node row in qdata_ (neighbor codes colocated):
```
[packed_fastscan_codes_of_32_neighbors | per_neighbor_factors | neighbor_IDs]
```

For each segment with B bits:
- If B <= 4: pack directly as 4-bit fastscan format (existing `pack_codes_helper`)
  - Size: `(seg_dim / 2) * 32` bytes per node (interleaved nibbles)
- If B > 4: split into high nibble (fast path) + low nibble (accurate path)
  - Fast codes: `(seg_dim / 2) * 32` bytes (top 4 bits)
  - Accurate codes: `(seg_dim / 2) * 32` bytes (bottom 4 bits)

Per-neighbor factors (3 floats x 32 neighbors = 384 bytes):
```
[o_l2sqr[32] | fac_dot[32] | fac_sum[32]]
```

Neighbor IDs (32 x 4 = 128 bytes).

**Memory estimate** (SIFT 128D, degree=32, uniform 4-bit):
```
Fastscan codes:    (128/2) * 32 = 2048 bytes
Factors:           3 * 32 * 4   =  384 bytes
Neighbor IDs:      32 * 4       =  128 bytes
Total per node:                   2560 bytes
vs current:        128 + 128    =  256 bytes (10x increase)
```

This is the standard SymphonyQG tradeoff: neighbor code duplication for cache locality + SIMD.

### 3b. Query Preparation

Per query (once):
1. PCA rotate query: `q' = R^T * q`
2. Per segment: scalar-quantize rotated query to 6-bit (existing `scalar::quantize`)
3. Build 4-bit LUT per segment (existing `pack_lut_impl`)
4. Precompute: `q_l2sqr = ||q||^2`, `sum_q = sum(q')`

### 3c. Neighbor Scan (Hot Path)

For each visited node, scan all 32 neighbors:
```cpp
// Step 1: fastscan accumulate (4-bit LUT, SIMD batch of 32)
for (each segment) {
    accumulate_impl(seg_dim, seg_packed_codes, seg_LUT, result);
}

// Step 2: convert fastscan result to approximate L2
for (i = 0..31) {
    raw = 2 * result[i] - sumq;         // signed conversion
    est_ip = fac_dot[i] * raw + fac_sum[i] * vl;
    dist[i] = o_l2sqr[i] + q_l2sqr - 2 * est_ip;
}
```

This replaces 32 sequential `caq_l2_estimate()` calls with one batched fastscan.

**Expected speedup**: 3-5x for the distance computation portion (SIMD batch + LUT vs sequential dot product).

**Files**: major rewrite of `qg.hpp` (layout, search), restore `qg_scanner.hpp` + `qg_query.hpp` integration

---

## Phase 4: Multi-Stage Filtering

**Goal**: Skip full distance computation for clearly distant neighbors using cheap lower bounds.

### Stage 0: Variance Bound (nearly free)

Before any code computation, use per-segment variance statistics:
```
lb = sum_over_segments(sigma_seg * m)   where m is confidence parameter
```
From Chebyshev's inequality: P(|est - true| > m*sigma) <= 1/m^2

If `lb > threshold`, skip this neighbor entirely. Cost: ~1 float multiply per segment.

### Stage 1: Top-Segment Fast Scan (cheap)

Only compute fastscan for the highest-variance segment (e.g., first 32 dimensions).
This captures the dominant distance contribution.

```
fast_dist = fastscan(seg_0_only) + correction
if (fast_dist > threshold * alpha)
    skip this neighbor
```

Cost: ~1/4 of full fastscan for 4-segment plan. Expected to filter 50-70% of neighbors.

### Stage 2: Full Distance (remaining neighbors only)

Run full multi-segment fastscan + factor correction only for survivors.

**Net effect**: Compute full distance for only 30-50% of neighbors, significant QPS boost.

**Files**: modify search loop in `qg.hpp`, add stage logic

---

## Phase 5: Encoding Optimization

### 5a. Progressive CAQ Codes

SAQ paper Lemma: taking the first `b` bits of a `B`-bit CAQ code forms a valid `b`-bit quantization.
This enables natural multi-resolution:
- Store 8-bit CAQ codes
- Fast path: use top 4 bits (fastscan)
- Accurate path: use all 8 bits (full precision)
- No separate encoding needed for different stages

### 5b. Build-Time Quantization

Graph construction still uses exact L2 on raw (unrotated) float vectors — unchanged.
After construction:
1. Compute PCA rotation
2. Rotate all vectors
3. Run segmentation DP
4. CAQ encode per segment
5. Pack fastscan codes per-node (colocate with neighbor IDs)

The CAQ encoding loop (coordinate descent) is O(r * D) per vector, r ~ 6 rounds.
For 1M vectors: ~6 * 128 * 1M = ~768M operations, parallelized with OpenMP, < 1 second.

---

## Implementation Order & Dependencies

```
Phase 0 ──→ Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4
(asym fix)  (PCA)       (segment)   (fastscan)  (filter)
                                         │
                                    Phase 5a
                                   (progressive)
```

Each phase is independently testable on SIFT-1M:

| Phase | Recall@10 target | QPS target | Memory  | Key metric        |
|-------|------------------|------------|---------|-------------------|
| 0     | >= 98.9% (=SQ8)  | ~same      | +4B/vec | Fix recall gap    |
| 1     | ~same            | -5% (rot)  | +D^2*4  | Enable Phase 2    |
| 2     | same @4bit avg   | +cache     | -50%    | Half memory       |
| 3     | same             | +3-5x dist | +10x    | Fastscan batch    |
| 4     | same             | +30-50%    | same    | Skip distant nbrs |

**Recommended priority**: Phase 0 → benchmark → Phase 1+2 → Phase 3 → Phase 4

Phase 0 is a minimal fix (one file change). If it achieves parity with SQ8, we have a solid
baseline to build the full SAQ pipeline on top of.

---

## Key Design Decisions

### Q: Per-vector codes vs per-node neighbor codes?

**Per-vector** (current): each vector's code stored once, look up by ID.
- Pro: minimal memory, simple layout
- Con: random access pattern, can't use fastscan (need batch of 32 consecutive)

**Per-node** (fastscan): each node stores packed codes of all 32 neighbors.
- Pro: sequential access, fastscan-friendly, perfect cache locality
- Con: 32x code duplication (each vector appears as neighbor of ~32 nodes)

**Decision**: Phase 0-2 use per-vector (simple, smaller memory). Phase 3 switches to per-node.

### Q: PCA vs FHT rotation?

**FHT** (current infrastructure): random rotation, preserves distance in expectation.
- Pro: O(D log D) rotation, already implemented
- Con: doesn't concentrate variance, segmentation can't exploit structure

**PCA** (SAQ approach): data-adaptive rotation, orders dims by variance.
- Pro: enables segmentation, eigenvalues directly give optimal bit allocation
- Con: O(D^2) rotation per query, O(N*D^2) to compute once

**Decision**: PCA. The O(D^2) query overhead is negligible for D=128 (~16K FLOPs vs millions
for graph traversal). The segmentation benefit is the core of SAQ's advantage.

### Q: What bit budget?

At 8 bits/dim uniform: same memory as SQ8, better recall from code adjustment + rescale.
At 4 bits/dim with segmentation: half the memory, comparable recall from intelligent allocation.
At 4 bits/dim + fastscan: half memory + much higher QPS from SIMD batch.

**Decision**: Start with 8-bit (Phase 0-1), then 4-bit variable (Phase 2-3) for the sweet spot.

### Q: degree_bound = 32 or 64?

Fastscan batch size = 32. If degree_bound = 64, we need 2 batches per node visit.
This still works but doubles the fastscan work. The original SymphonyQG supports this via
the loop `for (i = 0; i < degree_bound; i += kBatchSize)` in `scan_neighbors()`.

**Decision**: Support both. Default to 32 for best fastscan efficiency.
