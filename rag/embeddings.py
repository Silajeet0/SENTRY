"""Lazy semantic embeddings and an explicitly lexical offline test backend."""
import hashlib
import math
import re
import os
from pathlib import Path


class SentenceTransformerEmbedder:
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2", revision=None, device="cpu"):
        os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "data/model_cache"))
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, revision=revision, device=device)
        resolved = getattr(self.model[0].auto_model.config, "_commit_hash", None)
        if not resolved and not Path(model).is_dir():
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(model, "config.json", revision=revision, local_files_only=True)
            resolved = Path(config_path).parent.name
        if not resolved:
            digest = hashlib.sha256()
            for path in sorted(Path(model).rglob("*")):
                if path.is_file() and path.suffix in {".json", ".safetensors", ".bin", ".txt"}:
                    digest.update(str(path.relative_to(model)).encode())
                    with path.open("rb") as handle:
                        for block in iter(lambda: handle.read(1024*1024), b""):
                            digest.update(block)
            resolved = digest.hexdigest()
        self.identity = f"sentence-transformers:{model}@{resolved}"

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False).tolist()


class HashEmbedder:
    """Deterministic lexical vectors for offline tests; NOT semantic embeddings."""
    identity = "lexical-hash-v1:512"

    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * 512
            for token in re.findall(r"\w+", text.casefold()):
                digest = hashlib.sha256(token.encode()).digest()
                vector[int.from_bytes(digest[:4], "little") % 512] += 1 if digest[4] % 2 else -1
            norm = math.sqrt(sum(x*x for x in vector)) or 1
            vectors.append([x/norm for x in vector])
        return vectors
