"""Generate IVF index with int8 quantization from references.json.gz - streaming."""

import gzip
import struct
import random
from pathlib import Path

import ijson
import numpy as np


def pack():
    """Pack references.json.gz into IVF index with int8 quantization."""
    resources_path = Path(__file__).parent.parent / "resources"
    data_path = Path(__file__).parent.parent / "data"
    data_path.mkdir(exist_ok=True)
    
    refs_file = resources_path / "references.json.gz"
    
    # First pass: count vectors and compute quantization params
    print("First pass: computing quantization parameters...")
    n_vectors = 0
    dim_min = np.full(14, np.inf, dtype=np.float32)
    dim_max = np.full(14, -np.inf, dtype=np.float32)
    
    with gzip.open(refs_file, "rb") as f:
        for ref in ijson.items(f, "item"):
            vec = ref["vector"]
            n_vectors += 1
            # Update min/max (handle -1 sentinel for dims 5, 6)
            for i in range(14):
                v = vec[i]
                if v >= 0 or i not in [5, 6]:
                    dim_min[i] = min(dim_min[i], v)
                    dim_max[i] = max(dim_max[i], v)
            if n_vectors % 500000 == 0:
                print(f"  Scanned {n_vectors}")
    
    print(f"Total vectors: {n_vectors}")
    
    # Compute scale
    dim_range = dim_max - dim_min
    dim_range = np.where(dim_range < 1e-6, 1.0, dim_range)
    dim_scale = 254.0 / dim_range
    
    # Sample vectors for clustering
    print("Sampling vectors for clustering...")
    sample_size = min(100000, n_vectors)
    sample_indices = set(random.sample(range(n_vectors), sample_size))
    
    sample_vectors = []
    idx = 0
    with gzip.open(refs_file, "rb") as f:
        for ref in ijson.items(f, "item"):
            if idx in sample_indices:
                vec = np.array(ref["vector"], dtype=np.float32)
                vec_q = np.clip((vec - dim_min) * dim_scale - 127, -128, 127).astype(np.float32)
                sample_vectors.append(vec_q)
            idx += 1
    sample_vectors = np.array(sample_vectors, dtype=np.float32)
    
    # K-means clustering on sample
    n_centroids = 64
    print(f"Clustering {sample_size} samples into {n_centroids} centroids...")
    random.seed(42)
    np.random.seed(42)
    
    indices = random.sample(range(sample_size), n_centroids)
    centroids = sample_vectors[indices].copy()
    
    for iteration in range(20):
        diff = sample_vectors[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        distances = np.sum(diff ** 2, axis=2)
        assignments = np.argmin(distances, axis=1)
        
        new_centroids = np.zeros_like(centroids)
        for j in range(n_centroids):
            mask = assignments == j
            if np.any(mask):
                new_centroids[j] = sample_vectors[mask].mean(axis=0)
            else:
                new_centroids[j] = centroids[j]
        centroids = new_centroids
    
    # Second pass: assign all vectors and write directly to arrays
    print("Second pass: assigning vectors to centroids...")
    lists = [[] for _ in range(n_centroids)]
    labels = np.zeros(n_vectors, dtype=np.int32)
    vectors_q = np.zeros((n_vectors, 14), dtype=np.int8)
    
    idx = 0
    with gzip.open(refs_file, "rb") as f:
        for ref in ijson.items(f, "item"):
            vec = np.array(ref["vector"], dtype=np.float32)
            labels[idx] = 1 if ref["label"] == "fraud" else 0
            
            # Quantize
            vec_q = np.clip((vec - dim_min) * dim_scale - 127, -128, 127).astype(np.int8)
            vectors_q[idx] = vec_q
            
            # Assign to centroid
            diff = centroids - vec_q.astype(np.float32)
            dist = np.sum(diff ** 2, axis=1)
            cluster = np.argmin(dist)
            lists[cluster].append(idx)
            
            idx += 1
            if idx % 500000 == 0:
                print(f"  Processed {idx}/{n_vectors}")
    
    n_fraud = np.sum(labels)
    print(f"Fraud: {n_fraud} ({100*n_fraud/n_vectors:.2f}%)")
    
    list_sizes = [len(lst) for lst in lists]
    print(f"Cluster sizes: min={min(list_sizes)}, max={max(list_sizes)}, avg={sum(list_sizes)//len(list_sizes)}")
    
    # Write binary index
    output_file = data_path / "rinha.idx"
    print(f"Writing {output_file}...")
    
    with open(output_file, "wb") as f:
        # Header
        f.write(struct.pack("<I", 0x52494E48))  # "RINH"
        f.write(struct.pack("<I", 5))  # version 5 = IVF + int8
        f.write(struct.pack("<I", n_vectors))
        f.write(struct.pack("<I", n_centroids))
        f.write(struct.pack("<I", 14))  # dimensions
        
        # Quantization params
        f.write(dim_min.astype(np.float32).tobytes())
        f.write(dim_scale.astype(np.float32).tobytes())
        
        # Centroids
        f.write(centroids.astype(np.float32).tobytes())
        
        # Labels
        f.write(labels.tobytes())
        
        # Quantized vectors
        f.write(vectors_q.tobytes())
        
        # Inverted lists
        for lst in lists:
            f.write(struct.pack("<I", len(lst)))
            for i in lst:
                f.write(struct.pack("<I", i))
    
    file_size = output_file.stat().st_size / 1024 / 1024
    print(f"Index written: {file_size:.2f} MB")
    
    # Memory estimate
    vectors_mem = n_vectors * 14 / 1024 / 1024  # int8
    labels_mem = n_vectors * 4 / 1024 / 1024  # int32
    centroids_mem = n_centroids * 14 * 4 / 1024 / 1024  # float32
    print(f"Runtime memory: ~{vectors_mem + labels_mem + centroids_mem:.1f} MB")


if __name__ == "__main__":
    pack()
