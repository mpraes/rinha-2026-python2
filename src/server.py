"""ASGI HTTP server for fraud detection API with IVF + int8."""

import struct
from datetime import datetime
from pathlib import Path

import numpy as np
import orjson


def clamp(x: float) -> float:
    """Clamp value to [0.0, 1.0]."""
    return max(0.0, min(1.0, x))


def load_index(data_path: Path):
    """Load IVF index with int8 quantization."""
    index_file = data_path / "rinha.idx"
    
    print(f"Loading index from {index_file}...")
    with open(index_file, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != 0x52494E48:
            raise ValueError(f"Invalid magic: {hex(magic)}")
        
        version = struct.unpack("<I", f.read(4))[0]
        if version != 5:
            raise ValueError(f"Unsupported version: {version}")
        
        n_vectors = struct.unpack("<I", f.read(4))[0]
        n_centroids = struct.unpack("<I", f.read(4))[0]
        n_dims = struct.unpack("<I", f.read(4))[0]
        
        # Quantization params
        dim_min = np.frombuffer(f.read(n_dims * 4), dtype=np.float32).copy()
        dim_scale = np.frombuffer(f.read(n_dims * 4), dtype=np.float32).copy()
        
        # Centroids
        centroids = np.frombuffer(f.read(n_centroids * n_dims * 4), dtype=np.float32).copy()
        centroids = centroids.reshape(n_centroids, n_dims)
        
        # Labels
        labels = np.frombuffer(f.read(n_vectors * 4), dtype=np.int32).copy()
        
        # Quantized vectors
        vectors_q = np.frombuffer(f.read(n_vectors * n_dims), dtype=np.int8).copy()
        vectors_q = vectors_q.reshape(n_vectors, n_dims)
        
        # Inverted lists
        lists = []
        for _ in range(n_centroids):
            count = struct.unpack("<I", f.read(4))[0]
            indices = struct.unpack(f"<{count}I", f.read(count * 4))
            lists.append(np.array(indices, dtype=np.int32))
    
    print(f"Index loaded: {n_vectors} vectors, {n_centroids} centroids")
    return {
        "n_vectors": n_vectors,
        "n_centroids": n_centroids,
        "dim_min": dim_min,
        "dim_scale": dim_scale,
        "centroids": centroids,
        "labels": labels,
        "vectors_q": vectors_q,
        "lists": lists,
    }


def load_json(path: Path):
    """Load JSON file."""
    with open(path, "rb") as f:
        return orjson.loads(f.read())


class FraudDetector:
    """Fraud detection engine using IVF with int8."""
    
    def __init__(self, resources_path: Path, data_path: Path):
        self.normalization = load_json(resources_path / "normalization.json")
        self.mcc_risk = load_json(resources_path / "mcc_risk.json")
        self.data = load_index(data_path)
    
    def vectorize(self, payload: dict) -> np.ndarray:
        """Convert payload to 14-dimensional vector."""
        tx = payload["transaction"]
        customer = payload["customer"]
        merchant = payload["merchant"]
        terminal = payload["terminal"]
        last_tx = payload.get("last_transaction")
        
        norm = self.normalization
        
        vec = np.zeros(14, dtype=np.float32)
        
        vec[0] = clamp(tx["amount"] / norm["max_amount"])
        vec[1] = clamp(tx["installments"] / norm["max_installments"])
        vec[2] = clamp((tx["amount"] / customer["avg_amount"]) / norm["amount_vs_avg_ratio"])
        
        dt = datetime.fromisoformat(tx["requested_at"].replace("Z", "+00:00"))
        vec[3] = dt.hour / 23.0
        vec[4] = dt.weekday() / 6.0
        
        if last_tx is None:
            vec[5] = -1.0
            vec[6] = -1.0
        else:
            last_dt = datetime.fromisoformat(last_tx["timestamp"].replace("Z", "+00:00"))
            minutes = (dt - last_dt).total_seconds() / 60.0
            vec[5] = clamp(minutes / norm["max_minutes"])
            vec[6] = clamp(last_tx["km_from_current"] / norm["max_km"])
        
        vec[7] = clamp(terminal["km_from_home"] / norm["max_km"])
        vec[8] = clamp(customer["tx_count_24h"] / norm["max_tx_count_24h"])
        vec[9] = 1.0 if terminal["is_online"] else 0.0
        vec[10] = 1.0 if terminal["card_present"] else 0.0
        
        known_merchants = set(customer["known_merchants"])
        vec[11] = 1.0 if merchant["id"] not in known_merchants else 0.0
        vec[12] = self.mcc_risk.get(merchant["mcc"], 0.5)
        vec[13] = clamp(merchant["avg_amount"] / norm["max_merchant_avg_amount"])
        
        return vec
    
    def quantize(self, vec: np.ndarray) -> np.ndarray:
        """Quantize float32 vector to int8."""
        d = self.data
        return np.clip(
            (vec - d["dim_min"]) * d["dim_scale"] - 127,
            -128, 127
        ).astype(np.int8)
    
    def search(self, query: np.ndarray, k: int = 5) -> int:
        """Search IVF index, return fraud count among k neighbors."""
        d = self.data
        
        # Quantize query
        query_q = self.quantize(query).astype(np.float32)
        
        # Find nearest centroid
        diff = d["centroids"] - query_q
        dists = np.sum(diff ** 2, axis=1)
        top_centroid = np.argmin(dists)
        
        # Get candidates from that centroid
        candidates = d["lists"][top_centroid]
        
        # Limit candidates for speed
        max_candidates = 100
        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]
        
        # Compute distances (int8 arithmetic)
        candidate_vecs = d["vectors_q"][candidates].astype(np.float32)
        dists = np.sum((candidate_vecs - query_q) ** 2, axis=1)
        
        # Get top-k
        if len(dists) < k:
            return 0
        
        top_k_local = np.argpartition(dists, k)[:k]
        top_k_indices = candidates[top_k_local]
        
        # Count frauds
        return int(np.sum(d["labels"][top_k_indices]))
    
    def detect(self, payload: dict) -> dict:
        """Detect fraud for a transaction."""
        vec = self.vectorize(payload)
        fraud_count = self.search(vec, k=5)
        fraud_score = fraud_count / 5.0
        approved = fraud_score < 0.6
        return {"approved": approved, "fraud_score": fraud_score}


# Initialize detector at module load
detector: FraudDetector = None

# Pre-encoded responses
RESPONSE_READY = orjson.dumps({"status": "ready"})
RESPONSE_SAFE = orjson.dumps({"approved": True, "fraud_score": 0.0})

# Initialize detector
resources_path = Path(__file__).parent.parent / "resources"
data_path = Path(__file__).parent.parent / "data"
print("Loading index...")
detector = FraudDetector(resources_path, data_path)
print("Index loaded")


async def app(scope, receive, send):
    """ASGI application for fraud detection."""
    if scope["type"] == "http":
        path = scope["path"]
        method = scope["method"]
        
        if method == "GET" and path == "/ready":
            await send_json(send, RESPONSE_READY)
        elif method == "POST" and path == "/fraud-score":
            await handle_fraud_score(scope, receive, send)
        else:
            await send_json(send, RESPONSE_SAFE)


async def handle_fraud_score(scope, receive, send):
    """Handle POST /fraud-score request."""
    try:
        body = b""
        more_body = True
        while more_body:
            message = await receive()
            body += message.get("body", b"")
            more_body = message.get("more_body", False)
        
        data = orjson.loads(body)
        result = detector.detect(data)
        await send_json(send, orjson.dumps(result))
    except Exception:
        await send_json(send, RESPONSE_SAFE)


async def send_json(send, body: bytes, status: int = 200):
    """Send JSON response."""
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [
            [b"content-type", b"application/json"],
            [b"content-length", str(len(body)).encode()],
        ],
    })
    await send({
        "type": "http.response.body",
        "body": body,
    })


def main():
    """Run server (for local development)."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")


if __name__ == "__main__":
    main()
