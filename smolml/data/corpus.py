"""Byte-level corpora, deterministic splits, and seeded batching.

Vocabulary is the 256 byte values — no tokenizer choices to muddy comparisons.
Three sources:

- :func:`load_sample` — a tiny bundled English sample, committed in the repo so
  tests and the offline smoke run never touch the network.
- :func:`synthetic_text8` — a deterministic, scaled ``text8``-style clone
  (lowercase letters + space) for CI-scale runs without a download.
- :func:`prepare_enwik8` — the real corpus; an **opt-in** network download that
  tests never call.
"""

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

VOCAB_SIZE: int = 256
"""Byte-level vocabulary: every model emits a distribution over 256 byte values."""

_SAMPLE_PATH = Path(__file__).parent / "sample" / "sample.txt"
ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"


class ByteCorpus:
    """A raw byte corpus with a deterministic train/val split.

    Stored as ``uint8`` to stay compact; batches are materialized as ``int64``.
    """

    def __init__(self, data: bytes | np.ndarray):
        arr = np.frombuffer(data, dtype=np.uint8) if isinstance(data, bytes) else data
        if arr.dtype != np.uint8:
            raise ValueError(f"corpus must be uint8 bytes, got {arr.dtype}")
        self.data: np.ndarray = arr

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def split(self, val_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Split into (train, val). Deterministic: val is the final
        ``val_fraction`` of the stream, train is the prefix. No shuffling, so the
        split is identical on every machine and the tail never leaks into train.
        """
        if not 0.0 < val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
        n = len(self)
        n_val = max(1, int(n * val_fraction))
        if n_val >= n:
            raise ValueError("corpus too small for the requested val_fraction")
        return self.data[: n - n_val], self.data[n - n_val :]


def load_sample() -> ByteCorpus:
    """Load the bundled offline sample corpus (committed under the package)."""
    return ByteCorpus(_SAMPLE_PATH.read_bytes())


# Approximate English letter frequencies (a..z), used to give the synthetic
# corpus learnable order-0 structure so a model can score well below 8 bpb.
_EN_LETTER_WEIGHTS = np.array(
    [
        8.2,
        1.5,
        2.8,
        4.3,
        12.7,
        2.2,
        2.0,
        6.1,
        7.0,
        0.15,
        0.77,
        4.0,
        2.4,
        6.7,
        7.5,
        1.9,
        0.095,
        6.0,
        6.3,
        9.1,
        2.8,
        0.98,
        2.4,
        0.15,
        2.0,
        0.074,
    ],
    dtype=np.float64,
)


def synthetic_text8(n_bytes: int, seed: int = 0) -> ByteCorpus:
    """A deterministic, scaled ``text8``-style clone for CI-scale runs.

    Emits lowercase letters (frequency-weighted) grouped into random-length
    "words" separated by single spaces — enough order-0 and word-length structure
    to be compressible (well below 8 bpb) while needing no network. Same
    ``(n_bytes, seed)`` always yields the same bytes.
    """
    if n_bytes <= 0:
        raise ValueError("n_bytes must be positive")
    rng = np.random.default_rng(seed)
    probs = _EN_LETTER_WEIGHTS / _EN_LETTER_WEIGHTS.sum()
    letters = np.arange(26)
    out = np.empty(n_bytes, dtype=np.uint8)
    pos = 0
    space = ord(" ")
    base = ord("a")
    while pos < n_bytes:
        word_len = int(rng.integers(2, 10))
        chars = rng.choice(letters, size=word_len, p=probs)
        for c in chars:
            if pos >= n_bytes:
                break
            out[pos] = base + int(c)
            pos += 1
        if pos < n_bytes:
            out[pos] = space
            pos += 1
    return ByteCorpus(out)


def prepare_enwik8(cache_dir: str | Path = "data/cache", n_bytes: int | None = None) -> ByteCorpus:
    """Download (once) and load the real ``enwik8`` corpus, byte-level.

    **Opt-in and network-bound — tests never call this.** Downloads
    ``enwik8.zip`` into ``cache_dir`` if absent, extracts it, and returns the
    first ``n_bytes`` (or the whole 100 MB if ``None``). The deterministic
    train/val split is then taken via :meth:`ByteCorpus.split`.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    raw = cache / "enwik8"
    if not raw.exists():
        zip_path = cache / "enwik8.zip"
        if not zip_path.exists():
            urllib.request.urlretrieve(ENWIK8_URL, zip_path)  # noqa: S310 (opt-in)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("enwik8", cache)
    data = raw.read_bytes()
    if n_bytes is not None:
        data = data[:n_bytes]
    return ByteCorpus(data)


def get_batch(
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (input, target) sequences with next-byte targets.

    ``x`` and ``y`` are ``(batch_size, seq_len)`` ``int64`` tensors; ``y`` is
    ``x`` shifted one byte to the right (the next-byte prediction target). Start
    offsets are drawn from ``generator`` (kept on CPU for cross-device
    reproducibility), then moved to ``device``.
    """
    max_start = len(data) - seq_len - 1
    if max_start < 0:
        raise ValueError(f"corpus length {len(data)} too short for seq_len {seq_len}")
    ix = torch.randint(max_start + 1, (batch_size,), generator=generator)
    x = torch.empty((batch_size, seq_len), dtype=torch.long)
    y = torch.empty((batch_size, seq_len), dtype=torch.long)
    for b, i in enumerate(ix.tolist()):
        chunk = torch.from_numpy(data[i : i + seq_len + 1].astype(np.int64))
        x[b] = chunk[:-1]
        y[b] = chunk[1:]
    return x.to(device), y.to(device)
