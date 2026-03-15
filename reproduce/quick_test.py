"""Quick search test with per-EFS progress output."""
import sys
import symphonyqg
import numpy as np
from time import time
from utils.io import fvecs_read, ivecs_read

DATASET = sys.argv[1] if len(sys.argv) > 1 else "sift"
DEGREE = int(sys.argv[2]) if len(sys.argv) > 2 else 32
TOPK = 10
EFS_LIST = [10, 20, 30, 50, 80, 100, 150, 200, 300, 500]

base = fvecs_read(f"./data/{DATASET}/{DATASET}_base.fvecs")
query = fvecs_read(f"./data/{DATASET}/{DATASET}_query.fvecs")
gt = ivecs_read(f"./data/{DATASET}/{DATASET}_groundtruth.ivecs")
N, D = base.shape
NQ = query.shape[0]

index_path = f"./data/{DATASET}/symphonyqg_{DEGREE}.index"
print(f"Loading {index_path} ...", flush=True)
index = symphonyqg.Index(
    index_type="QG", metric="L2",
    num_elements=N, dimension=D, degree_bound=DEGREE,
)
index.load(index_path)
print(f"Loaded. N={N} D={D} NQ={NQ}\n")

# Warmup: 1 query
index.set_ef(10)
index.search(query[0], TOPK)

print(f"{'EFS':>6} {'Recall@'+str(TOPK):>10} {'QPS':>10} {'Avg ms':>10}")
print("-" * 42)

for ef in EFS_LIST:
    index.set_ef(ef)
    total_time = 0
    total_correct = 0

    for i in range(NQ):
        t1 = time()
        pred = index.search(query[i], TOPK)
        t2 = time()
        total_time += t2 - t1
        gt_set = set(gt[i][:TOPK])
        total_correct += sum(1 for p in pred if p in gt_set)

    recall = total_correct / (NQ * TOPK) * 100
    qps = NQ / total_time
    avg_ms = total_time / NQ * 1000
    print(f"{ef:>6} {recall:>9.2f}% {qps:>10.0f} {avg_ms:>9.3f}", flush=True)

    if recall > 99.5:
        break
