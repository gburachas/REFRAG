#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REFRAG-style RAG (compress → sense/select → expand) — Single-file reference implementation

This script reconstructs a REFRAG-style retrieval-augmented generation pipeline based on the
first 11 pages of the provided paper (compress context with encoder-produced chunk embeddings,
project those to decoder token space, selectively re-expand informative chunks, and decode).
It includes:
  - Qdrant-based retrieval index (build + search)
  - Encoder-side chunk embeddings (CLS pooling) + projection to decoder embedding dimension
  - Selective expansion via a tiny policy net (REINFORCE) with a strong heuristic fallback
  - Continual pretraining (CPT) curricula: reconstruction → next-paragraph prediction
  - Generation with TTFT/TTIT/throughput measurements
  - Full CLI with subcommands

USAGE (examples):
  # 0) Install deps (adjust CUDA wheel index if needed)
  #    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  #    pip install transformers==4.57.3 accelerate datasets sentencepiece qdrant-client sacrebleu numpy

  # 1) Build a Qdrant collection from a text corpus (1 doc per line)  (requires Qdrant server, e.g. docker compose up -d qdrant)
  #    python refrag.py index --corpus data/wiki_lines.txt --qdrant_url http://localhost:6343 --collection refrag --embed_model ollama://mxbai-embed-large:335m --ollama_url http://localhost:8089

  # 2) Continual pretraining (CPT) phase A: Reconstruction curriculum (freeze decoder)
  #    python refrag.py cpt_recon --train_json data/cpt_train.jsonl --enc roberta-base --dec meta-llama/Llama-3.2-3B      //meta-llama/Llama-3.2-1B

  # 3) Continual pretraining (CPT) phase B: Next-paragraph prediction curriculum (unfreeze decoder)
  #    python refrag.py cpt_next --train_json data/cpt_train.jsonl --enc roberta-base --dec meta-llama/Llama-3.2-3B       //meta-llama/Llama-3.2-1B

  # 4) Optional: train the RL policy that decides selective expansion (REINFORCE, reward=-PPL)
  #    python refrag.py train_policy --rag_json data/rag_train.jsonl --qdrant_url http://localhost:6343 --collection refrag --topk 8

  # 5) RAG generate (with compression rate k and policy-driven expansion fraction p)
  #    python refrag.py generate --qdrant_url http://localhost:6343 --collection refrag --question "Who discovered penicillin?" --topk 8 --k 16 --p 0.25

Data formats:
  - cpt_* expects JSONL with fields:
      {"id":"...", "tokens":"<long text>", "split":{"s":2048,"o":256}}
  - rag_* expects JSONL with fields:
      {"id":"...", "question":"...", "answers":["..."]}  # answers optional
  - index corpus: plain text file with one passage per line (≤ ~200 words).

Notes:
  - Default model IDs use Hugging Face Hub; for offline use, point to local directories.
  - For long contexts, PyTorch 2.1+ torch.compile may improve speed.
  - This implementation is designed for clarity + completeness; tune as needed.

Author: Matthew Combatti - Simulanics Technologies
"""
import os, sys, json, math, time, random, argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

# Transformers cache API (tuple past_key_values is deprecated in newer versions)
try:
    from transformers.cache_utils import Cache, DynamicCache
except Exception:
    Cache = None
    DynamicCache = None

try:
    from qdrant_client import QdrantClient, models as qdrant_models  # pip install qdrant-client
except Exception:
    QdrantClient = None  # type: ignore[assignment,misc]
    qdrant_models = None  # type: ignore[assignment]

try:
    import urllib.request
    _HAS_URLLIB = True
except Exception:
    _HAS_URLLIB = False

try:
    from pypdf import PdfReader  # pip install pypdf
    _HAS_PYPDF = True
except Exception:
    _HAS_PYPDF = False

try:
    import tiktoken  # pip install tiktoken
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False


# ----------------------------
# Utilities
# ----------------------------

def seed_everything(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_device():
    # Prefer CUDA (includes ROCm builds), then Apple MPS, then CPU
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def ensure_qdrant():
    if QdrantClient is None:
        raise RuntimeError(
            "qdrant-client is not installed. Install with `pip install qdrant-client`."
        )


def safe_torch_load(path: str, map_location=None):
    """
    torch.load() now warns that the default weights_only=False is unsafe (pickle).
    Prefer weights_only=True when supported; fall back for older torch versions.

    If you *must* load a non-weights-only checkpoint and you fully trust the source,
    set REFRAG_ALLOW_UNSAFE_TORCH_LOAD=1 to allow a fallback to weights_only=False.
    """
    # Prefer safe weights-only loading when available.
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older torch: no weights_only kwarg.
        return torch.load(path, map_location=map_location)
    except Exception as e:
        # Optional unsafe fallback for trusted sources only.
        allow_unsafe = os.environ.get("REFRAG_ALLOW_UNSAFE_TORCH_LOAD", "").strip().lower() in ("1", "true", "yes", "y", "on")
        if allow_unsafe:
            try:
                return torch.load(path, map_location=map_location, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=map_location)
        raise


def maybe_torch_compile(module: nn.Module, enabled: bool, mode: str = "reduce-overhead", fullgraph: bool = False):
    """
    Optional torch.compile() wrapper controlled by a CLI flag.

    Notes:
      - torch.compile is available in PyTorch 2.0+.
      - Compilation may fail for some models/devices/backends; this function safely falls back.
      - We compile only the decoder module (the hottest path for CPT + generation).
    """
    if not enabled:
        return module
    if not hasattr(torch, "compile"):
        print("[torch_compile] torch.compile not available in this PyTorch version; continuing without compile.")
        return module
    try:
        return torch.compile(module, mode=mode, fullgraph=fullgraph)
    except Exception as e:
        print(f"[torch_compile] compile failed; continuing without compile. error={repr(e)}")
        return module


# ----------------------------
# PDF parsing & section-aware chunking
# ----------------------------

import re as _re
import glob as _glob

_SECTION_RE = _re.compile(
    r'^\s*'
    r'(?:(?:\d+\.?)+\s+)'          # numbered: "1 ", "2.1 ", "3.2.1 "
    r'|(?:abstract|introduction|conclusion|references|acknowledgment|appendix|related\s+work|background|methodology|methods|results|discussion|experiments|evaluation)'
    , _re.IGNORECASE
)


def _get_tiktoken_enc():
    """Return a tiktoken encoder for token counting (cl100k_base, used by most modern models)."""
    if not _HAS_TIKTOKEN:
        raise RuntimeError("tiktoken is not installed. Install with `pip install tiktoken`.")
    return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, enc=None) -> int:
    if enc is None:
        enc = _get_tiktoken_enc()
    return len(enc.encode(text))


def _looks_like_section_header(line: str) -> bool:
    """Heuristic: line is a section header if it matches common patterns and is short."""
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    if _SECTION_RE.match(stripped):
        return True
    # ALL-CAPS short lines (e.g. "ABSTRACT", "INTRODUCTION")
    if stripped.isupper() and len(stripped.split()) <= 6:
        return True
    return False


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from a PDF file using pypdf."""
    if not _HAS_PYPDF:
        raise RuntimeError("pypdf is not installed. Install with `pip install pypdf`.")
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def chunk_text_by_sections(
    text: str,
    chunk_min: int = 256,
    chunk_max: int = 512,
    source: str = "",
) -> List[str]:
    """
    Split text into chunks respecting section boundaries.
    1. Split by detected section headers.
    2. If a section > chunk_max tokens, split further by paragraphs.
    3. Merge small consecutive chunks until they hit chunk_min.
    Returns list of chunk strings.
    """
    enc = _get_tiktoken_enc()
    lines = text.split("\n")

    # --- Phase 1: split into raw sections ---
    sections: List[str] = []
    current: List[str] = []
    for line in lines:
        if _looks_like_section_header(line) and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))

    # --- Phase 2: split large sections by paragraphs, then hard-split ---
    raw_chunks: List[str] = []
    for sec in sections:
        sec_tok = _count_tokens(sec, enc)
        if sec_tok <= chunk_max:
            raw_chunks.append(sec.strip())
        else:
            # split by double-newline (paragraphs)
            paragraphs = _re.split(r'\n\s*\n', sec)
            buf = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                candidate = (buf + "\n\n" + para).strip() if buf else para
                if _count_tokens(candidate, enc) <= chunk_max:
                    buf = candidate
                else:
                    if buf:
                        raw_chunks.append(buf)
                    # If single paragraph still too big, hard-split by tokens
                    if _count_tokens(para, enc) > chunk_max:
                        tokens = enc.encode(para)
                        for i in range(0, len(tokens), chunk_max):
                            raw_chunks.append(enc.decode(tokens[i:i + chunk_max]))
                        buf = ""
                    else:
                        buf = para
            if buf:
                raw_chunks.append(buf)

    # --- Phase 3: merge tiny consecutive chunks ---
    merged: List[str] = []
    buf = ""
    for chunk in raw_chunks:
        if not chunk.strip():
            continue
        candidate = (buf + "\n\n" + chunk).strip() if buf else chunk
        if _count_tokens(candidate, enc) <= chunk_max:
            buf = candidate
        else:
            if buf:
                merged.append(buf)
            buf = chunk
    if buf:
        merged.append(buf)

    # Add source metadata prefix if given
    if source:
        merged = [f"[source: {source}]\n{c}" for c in merged]

    return [c for c in merged if c.strip()]


def load_passages_from_path(corpus_path: str, chunk_min: int = 256, chunk_max: int = 512) -> List[str]:
    """
    Load passages from a file or directory.
    - .txt file: one passage per line (legacy)
    - .pdf file: extract text, section-chunk
    - directory: process all .pdf and .txt files inside
    """
    passages: List[str] = []
    paths: List[str] = []

    if os.path.isdir(corpus_path):
        paths = sorted(_glob.glob(os.path.join(corpus_path, "**", "*"), recursive=True))
    else:
        paths = [corpus_path]

    for p in paths:
        if p.lower().endswith(".pdf"):
            print(f"[index] parsing PDF: {os.path.basename(p)}")
            text = extract_text_from_pdf(p)
            chunks = chunk_text_by_sections(text, chunk_min=chunk_min, chunk_max=chunk_max, source=os.path.basename(p))
            print(f"[index]   → {len(chunks)} chunks")
            passages.extend(chunks)
        elif p.lower().endswith(".txt"):
            with open(p, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            print(f"[index] loaded {len(lines)} lines from {os.path.basename(p)}")
            passages.extend(lines)
        # skip other file types silently

    return passages


# ----------------------------
# Retrieval (Qdrant + encoder)
# ----------------------------

class PassageEncoder(nn.Module):
    """Passage encoder that returns a fixed vector per passage using CLS pooling (HuggingFace model)."""
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device=None):
        super().__init__()
        self.device = device or now_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.out_dim = self.encoder.config.hidden_size

    @torch.no_grad()
    def encode_passages(self, texts: List[str], bs: int = 32) -> np.ndarray:
        self.encoder.eval()
        if not texts:
            return np.zeros((0, self.out_dim), dtype=np.float32)
        vecs = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            toks = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            out = self.encoder(**toks).last_hidden_state
            emb = out[:, 0, :]  # CLS
            emb = F.normalize(emb, dim=-1)
            vecs.append(emb.detach().cpu().float().numpy())
        return np.concatenate(vecs, axis=0)

    @torch.no_grad()
    def encode_query(self, text: str) -> np.ndarray:
        v = self.encode_passages([text], bs=1)
        return v[0] if len(v) else np.zeros((self.out_dim,), dtype=np.float32)


class OllamaEmbedEncoder:
    """Passage encoder that calls an Ollama embedding service over HTTP."""
    def __init__(self, model_name: str = "mxbai-embed-large:335m",
                 ollama_url: str = "http://localhost:8089"):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        # Probe dimension with a test call
        test = self._call_embed(["dim probe"])
        self.out_dim = len(test[0])
        print(f"[OllamaEmbedEncoder] model={model_name} url={ollama_url} dim={self.out_dim}")

    def _call_embed(self, texts: List[str]) -> List[List[float]]:
        """Call Ollama /api/embed endpoint."""
        import urllib.request
        payload = json.dumps({"model": self.model_name, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.ollama_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["embeddings"]

    def encode_passages(self, texts: List[str], bs: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.out_dim), dtype=np.float32)
        vecs = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            embs = self._call_embed(batch)
            vecs.extend(embs)
        arr = np.array(vecs, dtype=np.float32)
        # L2 normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        arr = arr / norms
        return arr

    def encode_query(self, text: str) -> np.ndarray:
        v = self.encode_passages([text], bs=1)
        return v[0] if len(v) else np.zeros((self.out_dim,), dtype=np.float32)


def make_passage_encoder(embed_model: str, ollama_url: str = "http://localhost:8089"):
    """Factory: returns OllamaEmbedEncoder if embed_model starts with 'ollama://', else PassageEncoder."""
    if embed_model.startswith("ollama://"):
        model_name = embed_model[len("ollama://"):]
        return OllamaEmbedEncoder(model_name=model_name, ollama_url=ollama_url)
    return PassageEncoder(model_name=embed_model)


def get_qdrant_client(url: str = "http://localhost:6333") -> 'QdrantClient':
    """Return a Qdrant client connected to the given URL."""
    ensure_qdrant()
    return QdrantClient(url=url, check_compatibility=False)


def build_qdrant_collection(
    client: 'QdrantClient',
    collection_name: str,
    embeddings: np.ndarray,
    texts: List[str],
    batch_size: int = 500,
    append: bool = False,
):
    """Create (or recreate) a Qdrant collection and upsert passage vectors + text payloads.
    If append=True, add to existing collection (IDs offset by current point count)."""
    dim = embeddings.shape[1]
    start_id = 0
    if append and client.collection_exists(collection_name):
        info = client.get_collection(collection_name)
        start_id = info.points_count or 0
        print(f"[index] appending to existing collection (current points: {start_id})")
    else:
        # Recreate collection
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=dim,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
    # Batch upsert
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        points = [
            qdrant_models.PointStruct(
                id=start_id + i,
                vector=embeddings[i].tolist(),
                payload={"text": texts[i]},
            )
            for i in range(start, end)
        ]
        client.upsert(collection_name=collection_name, points=points)


def search_qdrant(
    client: 'QdrantClient',
    collection_name: str,
    query_vec: np.ndarray,
    topk: int,
) -> Tuple[List[float], List[str]]:
    """Search Qdrant and return (scores, passage_texts)."""
    results = client.query_points(
        collection_name=collection_name,
        query=query_vec.astype(np.float32).tolist(),
        limit=topk,
        with_payload=True,
    ).points
    scores = [hit.score for hit in results]
    texts = [hit.payload["text"] for hit in results if hit.payload]  # type: ignore[index]
    return scores, texts


# ----------------------------
# REFRAG Core
# ----------------------------

@dataclass
class REFRAGConfig:
    encoder_name: str = "roberta-base"
    decoder_name: str = "meta-llama/Llama-3.2-3B"
    chunk_len_tokens: int = 64     # k
    max_q_tokens: int = 256
    max_ctx_tokens: int = 2048     # s (pre-chunked, before compression)
    max_out_tokens: int = 256      # o
    selective_p: float = 0.25      # fraction cap for expansions
    policy_hidden: int = 256
    lr: float = 2e-5
    wd: float = 0.0
    grad_clip: float = 1.0
    fp16: bool = True
    seed: int = 1337
    torch_compile: bool = False


class ChunkEncoder(nn.Module):
    """Encoder that returns one vector per text chunk via CLS pooling."""
    def __init__(self, name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        self.model = AutoModel.from_pretrained(name)
        self.out_dim = self.model.config.hidden_size

    def forward(self, texts: List[str], device=None) -> torch.Tensor:
        device = device or next(self.model.parameters()).device
        if len(texts) == 0:
            return torch.zeros((0, self.out_dim), device=device)
        toks = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        h = self.model(**toks).last_hidden_state[:, 0, :]  # [CLS]
        h = F.normalize(h, dim=-1)
        return h


class TokenProjector(nn.Module):
    """Projection ϕ: encoder-dim → decoder token-embedding dim."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.proj(x)


class SelectPolicy(nn.Module):
    """
    Tiny policy π(ci) that outputs expansion prob per chunk.
    Input: chunk embedding ci (encoder space) + scalar pos (normalized [0,1]).
    Output: logits ∈ R (Bernoulli).
    """
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, c: torch.Tensor, pos01: torch.Tensor) -> torch.Tensor:
        x = torch.cat([c, pos01], dim=-1)
        return self.net(x).squeeze(-1)  # [L]


class REFRAG(nn.Module):
    """
    Builds decoder inputs consisting of:
      - question token embeddings (normal)
      - per-chunk compressed embeddings (projected from encoder) OR full token embeddings (expanded)
    """
    def __init__(self, cfg: REFRAGConfig):
        super().__init__()
        self.cfg = cfg
        self.device = now_device()

        # Modules
        self.encoder = ChunkEncoder(cfg.encoder_name).to(self.device)
        self.decoder_tok = AutoTokenizer.from_pretrained(cfg.decoder_name, use_fast=True)

        # Load decoder in float16 when fp16 is set (or always on CUDA) to save VRAM.
        # device_map="auto" spreads layers across all visible GPUs (and CPU if needed).
        _dtype = torch.float16 if (cfg.fp16 or self.device.type == "cuda") else None
        self.decoder = AutoModelForCausalLM.from_pretrained(
            cfg.decoder_name,
            torch_dtype=_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        # Determine the device used by the decoder's first parameter (for
        # encoder/projector/policy placement – they are small enough for one GPU).
        _dec_param = next(self.decoder.parameters(), None)
        if _dec_param is not None:
            self.device = _dec_param.device
        self.encoder = self.encoder.to(self.device)

        # Optional: torch.compile for the decoder hot path.
        self.decoder = maybe_torch_compile(self.decoder, enabled=cfg.torch_compile)

        self.dec_embed_dim = self.decoder.get_input_embeddings().weight.shape[1]
        self.projector = TokenProjector(self.encoder.out_dim, self.dec_embed_dim).to(self.device)
        self.policy = SelectPolicy(self.encoder.out_dim, hidden=cfg.policy_hidden).to(self.device)

        self.eos_id = self.decoder_tok.eos_token_id
        self.pad_id = self.decoder_tok.pad_token_id or self.decoder_tok.eos_token_id

    def _new_cache(self):
        # Prefer the modern cache API when available; fall back to legacy tuple behavior otherwise.
        if DynamicCache is None:
            return None
        return DynamicCache()

    def _ensure_cache(self, past_key_values):
        # Convert legacy tuple past_key_values into a Cache instance to avoid deprecation warnings.
        if Cache is None or DynamicCache is None:
            return past_key_values
        if past_key_values is None:
            return DynamicCache()
        if isinstance(past_key_values, Cache):
            return past_key_values
        return DynamicCache.from_legacy_cache(past_key_values)

    def _tokenize(self, text: str, max_len: int) -> Dict[str, torch.Tensor]:
        return self.decoder_tok(text, truncation=True, max_length=max_len, padding=False, return_tensors="pt")

    @torch.no_grad()
    def _decoder_token_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.get_input_embeddings()(input_ids.to(self.device))

    def _chunk_text(self, text: str, k_tokens: int) -> Tuple[List[str], List[torch.Tensor]]:
        toks = self.decoder_tok(text, truncation=True, max_length=self.cfg.max_ctx_tokens, return_tensors="pt")
        ids = toks.input_ids[0]  # [S]
        id_chunks = [ids[i:i+k_tokens] for i in range(0, ids.size(0), k_tokens)]
        str_chunks = [self.decoder_tok.decode(ch, skip_special_tokens=True) for ch in id_chunks]
        return str_chunks, id_chunks

    def _encode_chunks(self, chunk_strs: List[str]) -> torch.Tensor:
        return self.encoder(chunk_strs, device=self.device)

    def _project_chunks(self, c: torch.Tensor) -> torch.Tensor:
        return self.projector(c)

    def _select_expand_mask(self, c: torch.Tensor, p_max: float) -> torch.Tensor:
        L = c.size(0)
        if L == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        pos01 = torch.linspace(0, 1, steps=L, device=c.device).unsqueeze(-1)
        logits = self.policy(c, pos01)          # [L]
        probs = torch.sigmoid(logits)
        sample = torch.bernoulli(probs).bool()
        if p_max > 0.0:
            max_expand = max(0, int(round(p_max * L)))
            if sample.sum().item() > max_expand:
                topk = torch.topk(logits, k=max_expand).indices
                mask = torch.zeros_like(sample)
                mask[topk] = True
                sample = mask.bool()
        return sample

    def _heuristic_select(self, chunk_ids: List[torch.Tensor], q_text: str, p_max: float) -> torch.Tensor:
        L = len(chunk_ids)
        if L == 0 or p_max <= 0:
            return torch.zeros(L, dtype=torch.bool, device=self.device)
        scores = []
        with torch.no_grad():
            for ch in chunk_ids:
                inp = torch.cat([
                    self._tokenize(q_text, self.cfg.max_q_tokens).input_ids[0].to(self.device),
                    ch.to(self.device)
                ], dim=0).unsqueeze(0)
                labels = inp.clone()
                out = self.decoder(input_ids=inp, labels=labels)
                ppl = torch.exp(out.loss).item()
                scores.append(ppl)
        scores = np.asarray(scores)
        k = max(1, int(round(p_max * L)))
        top_idx = scores.argsort()[::-1][:k]  # expand highest perplexity chunks
        mask = torch.zeros(L, dtype=torch.bool, device=self.device)
        mask[top_idx] = True
        return mask

    def _embed_text_tokens(self, text: str, max_len: int = 512) -> torch.Tensor:
        """Tokenize text and return decoder embedding vectors. Shape: [T, D]."""
        ids = self._tokenize(text, max_len).input_ids.to(self.device)
        return self._decoder_token_embeddings(ids).squeeze(0)  # [T, D]

    def build_decoder_inputs(self, question: str, passages: List[str], k: int, p: float, use_policy: bool = True) -> Tuple[torch.Tensor, Dict]:
        # ── Llama-3 Instruct chat structure ──
        # We build: [system_header] [system_msg] [system_end]
        #           [user_header] "Context:\n" [compressed/expanded chunks]
        #           "\n\nQuestion: " [question] [user_end]
        #           [assistant_header]
        # The assistant_header cues the model to generate the answer.

        sys_msg = (
            "You are a knowledgeable research assistant. Use the provided context "
            "passages to answer the user's question. Provide a detailed helpful "
            "response and be technically accurate."
        )

        # Token-level pieces for the chat template
        sys_header = self._embed_text_tokens(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n", max_len=32)
        sys_body = self._embed_text_tokens(sys_msg, max_len=128)
        sys_end = self._embed_text_tokens("<|eot_id|>", max_len=8)

        user_header = self._embed_text_tokens(
            "<|start_header_id|>user<|end_header_id|>\n\n", max_len=32)
        ctx_label = self._embed_text_tokens("Context:\n", max_len=16)

        # ── Context → chunk, encode, project, select ──
        ctx_text = "".join(passages)
        chunk_strs, chunk_ids = self._chunk_text(ctx_text, k_tokens=k)
        L = len(chunk_strs)

        with torch.no_grad():
            c = self._encode_chunks(chunk_strs)    # [L, D_enc]
            ecnk = self._project_chunks(c)         # [L, D_dec]

        if use_policy:
            expand_mask = self._select_expand_mask(c, p_max=p)
        else:
            expand_mask = self._heuristic_select(chunk_ids, q_text=question, p_max=p)

        # Build context embedding sequence (compressed + expanded chunks)
        ctx_embs = []
        seg_flags = []
        for i, ids in enumerate(chunk_ids):
            if expand_mask[i]:
                tok_emb = self._decoder_token_embeddings(ids.unsqueeze(0)).squeeze(0)  # [t_i, D]
                ctx_embs.append(tok_emb)
                seg_flags.extend([1] * tok_emb.size(0))
            else:
                ctx_embs.append(ecnk[i].unsqueeze(0))  # [1, D]
                seg_flags.append(0)

        # Question + tail tokens
        q_prefix = self._embed_text_tokens("\n\nQuestion: ", max_len=16)
        q_body = self._embed_text_tokens(question, max_len=self.cfg.max_q_tokens)
        user_end = self._embed_text_tokens("<|eot_id|>", max_len=8)
        asst_header = self._embed_text_tokens(
            "<|start_header_id|>assistant<|end_header_id|>\n\n", max_len=32)

        # ── Assemble full sequence ──
        all_parts = [
            sys_header, sys_body, sys_end,
            user_header, ctx_label,
            *ctx_embs,
            q_prefix, q_body, user_end,
            asst_header,
        ]
        final = torch.cat(all_parts, dim=0).unsqueeze(0)  # [1, T', D]

        # Match decoder dtype
        _dec_dtype = next(self.decoder.parameters()).dtype
        final = final.to(dtype=_dec_dtype)

        extras = {
            "expand_mask": expand_mask.detach().cpu().numpy().tolist(),
            "num_chunks": L,
            "token_positions_flag": seg_flags,
        }
        return final, extras

    @torch.no_grad()
    def generate(self, question: str, passages: List[str], k: int, p: float,
                 max_new_tokens: int = 128, temperature: float = 0.0, top_p: float = 1.0,
                 use_policy: bool = True) -> Dict:
        self.decoder.eval()
        emb_in, extras = self.build_decoder_inputs(question, passages, k=k, p=p, use_policy=use_policy)

        # Build stop-token set
        stop_ids = {self.eos_id}
        eot_id = self.decoder_tok.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id != self.decoder_tok.unk_token_id:
            stop_ids.add(eot_id)

        # Prefill → KV cache
        cache = self._new_cache()
        t0 = time.time()
        out = self.decoder(inputs_embeds=emb_in, use_cache=True, past_key_values=cache)
        past_key_values = self._ensure_cache(out.past_key_values)
        ttft = time.time() - t0

        generated = []
        ttit_list = []
        logits = out.logits[:, -1, :]  # start from prefill logits

        for _ in range(max_new_tokens):
            t1 = time.time()
            if temperature > 0.0:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = cumulative_probs > top_p
                    cutoff[..., 1:] = cutoff[..., :-1].clone()
                    cutoff[..., 0] = False
                    sorted_probs[cutoff] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_id = torch.multinomial(sorted_probs, num_samples=1)
                    next_id = sorted_indices.gather(-1, next_id)
                else:
                    next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            nid = next_id.item()
            if nid in stop_ids:
                break
            generated.append(nid)

            step_out = self.decoder(
                input_ids=next_id, use_cache=True, past_key_values=past_key_values
            )
            ttit_list.append(time.time() - t1)
            logits = step_out.logits[:, -1, :]
            past_key_values = self._ensure_cache(step_out.past_key_values)

        text = self.decoder_tok.decode(generated, skip_special_tokens=True)
        throughput = (len(generated) / max(sum(ttit_list), 1e-6)) if ttit_list else 0.0
        return {
            "answer": text.strip(),
            "TTFT_sec": ttft,
            "TTIT_avg_sec": float(np.mean(ttit_list)) if ttit_list else 0.0,
            "throughput_tok_per_sec": throughput,
            "meta": extras,
        }

    @torch.no_grad()
    def generate_stream(self, question: str, passages: List[str], k: int, p: float,
                        max_new_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0,
                        use_policy: bool = True):
        """Streaming generator that yields (token_text, extras_or_None) tuples.

        The *first* yield is ``("", extras)`` so the caller can display
        compression metadata before any tokens arrive.  Subsequent yields are
        ``(token_str, None)`` for each decoded token.
        """
        self.decoder.eval()
        emb_in, extras = self.build_decoder_inputs(question, passages, k=k, p=p, use_policy=use_policy)

        # -- Compute compression statistics and attach to extras --
        expand_mask = extras["expand_mask"]
        num_chunks = extras["num_chunks"]
        num_expanded = int(sum(expand_mask))
        num_compressed = num_chunks - num_expanded
        seg_flags = extras["token_positions_flag"]
        original_ctx_tokens = sum(len(ids) for _, ids in zip(
            *self._chunk_text("".join(passages), k_tokens=k)))
        compressed_seq_tokens = len(seg_flags)
        extras["num_compressed"] = num_compressed
        extras["num_expanded"] = num_expanded
        extras["original_ctx_tokens"] = original_ctx_tokens
        extras["compressed_seq_tokens"] = compressed_seq_tokens
        extras["compression_ratio"] = (
            compressed_seq_tokens / max(original_ctx_tokens, 1)
        )

        # Yield extras before any text so UI can render the compression badge
        yield "", extras

        # Build stop-token set (Llama-3 uses both eos and eot_id)
        stop_ids = {self.eos_id}
        eot_id = self.decoder_tok.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id != self.decoder_tok.unk_token_id:
            stop_ids.add(eot_id)

        # Prefill — sample first token from prefill logits directly
        cache = self._new_cache()
        out = self.decoder(inputs_embeds=emb_in, use_cache=True, past_key_values=cache)
        past_key_values = self._ensure_cache(out.past_key_values)
        logits = out.logits[:, -1, :]

        for _ in range(max_new_tokens):
            if temperature > 0.0:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = cumulative > top_p
                    cutoff[..., 1:] = cutoff[..., :-1].clone()
                    cutoff[..., 0] = False
                    sorted_probs[cutoff] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_id = torch.multinomial(sorted_probs, num_samples=1)
                    next_id = sorted_indices.gather(-1, next_id)
                else:
                    next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            nid = next_id.item()
            if nid in stop_ids:
                break
            token_str = self.decoder_tok.decode([nid], skip_special_tokens=True)
            if token_str:  # skip empty strings from special tokens
                yield token_str, None

            # Next step
            step_out = self.decoder(
                input_ids=next_id, use_cache=True, past_key_values=past_key_values
            )
            logits = step_out.logits[:, -1, :]
            past_key_values = self._ensure_cache(step_out.past_key_values)

    @torch.no_grad()
    def generate_stream_standard(self, question: str, passages: List[str],
                                  max_new_tokens: int = 256, temperature: float = 0.0,
                                  top_p: float = 1.0):
        """Standard RAG streaming generator (no compression).

        Formats passages as plain text context in a chat-style prompt and
        generates via ``input_ids`` – works with any decoder out of the box
        (no trained projector required).

        Yields ``(token_str, extras_or_None)`` – first yield is ``("", extras)``
        with metadata, subsequent yields are ``(token_str, None)``.
        """
        self.decoder.eval()

        # Build a proper Llama-3 Instruct chat prompt for standard RAG
        ctx = "\n\n".join(f"[Passage {i+1}]\n{p}" for i, p in enumerate(passages))
        sys_msg = (
            "You are a knowledgeable research assistant. Use the provided context "
            "passages to answer the user's question. Provide a detailed helpful "
            "response and be technically accurate."
        )
        user_msg = f"Context:\n{ctx}\n\nQuestion: {question}"

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]
        input_ids = self.decoder_tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)
        # Truncate if needed
        max_prompt = self.cfg.max_ctx_tokens + self.cfg.max_q_tokens
        if input_ids.shape[1] > max_prompt:
            input_ids = input_ids[:, :max_prompt]
        prompt_len = input_ids.shape[1]

        extras = {
            "mode": "standard_rag",
            "prompt_tokens": prompt_len,
            "num_passages": len(passages),
        }
        yield "", extras

        # Build stop-token set
        stop_ids = {self.eos_id}
        eot_id = self.decoder_tok.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id != self.decoder_tok.unk_token_id:
            stop_ids.add(eot_id)

        # Prefill
        cache = self._new_cache()
        out = self.decoder(input_ids=input_ids, use_cache=True, past_key_values=cache)
        past_key_values = self._ensure_cache(out.past_key_values)
        logits = out.logits[:, -1, :]  # sample from prefill

        for _ in range(max_new_tokens):
            if temperature > 0.0:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = cumulative > top_p
                    cutoff[..., 1:] = cutoff[..., :-1].clone()
                    cutoff[..., 0] = False
                    sorted_probs[cutoff] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_id = torch.multinomial(sorted_probs, num_samples=1)
                    next_id = sorted_indices.gather(-1, next_id)
                else:
                    next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            nid = next_id.item()
            if nid in stop_ids:
                break
            token_str = self.decoder_tok.decode([nid], skip_special_tokens=True)
            if token_str:
                yield token_str, None

            step_out = self.decoder(
                input_ids=next_id, use_cache=True, past_key_values=past_key_values
            )
            logits = step_out.logits[:, -1, :]
            past_key_values = self._ensure_cache(step_out.past_key_values)

    # ----------------------------
    # Losses for CPT & RL policy
    # ----------------------------
    def loss_reconstruction(self, ctx_text: str, k: int, num_chunks_cap: Optional[int] = None) -> torch.Tensor:
        """
        Train encoder+projector to reconstruct tokens chunk-by-chunk from a single projected vector.

        Implementation detail:
        For each chunk, we repeat the single projected vector across the chunk length so that
        inputs_embeds has shape [1, T_chunk, D] to match labels [1, T_chunk]. This resolves the
        batch/sequence mismatch raised by cross_entropy in HF's causal LM loss.
        """
        # 1) Chunk the context in decoder token space
        chunk_strs, chunk_ids = self._chunk_text(ctx_text, k_tokens=k)
        if num_chunks_cap is not None:
            chunk_strs = chunk_strs[:num_chunks_cap]
            chunk_ids = chunk_ids[:num_chunks_cap]
        L = len(chunk_strs)
        if L == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # 2) Encode chunks (encoder space) → project to decoder embedding space
        c = self._encode_chunks(chunk_strs)      # [L, D_enc]
        e = self._project_chunks(c)              # [L, D_dec]

        # 3) Per-chunk reconstruction loss
        loss_accum = 0.0
        for i, ids in enumerate(chunk_ids):
            # Labels: shape [1, T]
            labels = ids.unsqueeze(0).to(self.device)              # [1, T]
            T = labels.size(1)

            # Inputs: repeat the single compressed vector across T time steps → [1, T, D_dec]
            # (expand is fine and memory-light; make contiguous to be safe for certain backends)
            inp_emb = e[i].unsqueeze(0).unsqueeze(1).expand(1, T, -1).contiguous()  # [1, T, D]

            # Cast to decoder dtype (projector outputs fp32 but decoder may be fp16)
            _dec_dtype = next(self.decoder.parameters()).dtype
            inp_emb = inp_emb.to(dtype=_dec_dtype)

            # Optional: attention mask (all ones since we provide T tokens)
            attn_mask = torch.ones((1, T), dtype=torch.long, device=self.device)

            out = self.decoder(inputs_embeds=inp_emb, attention_mask=attn_mask, labels=labels)
            loss_accum = loss_accum + out.loss

        return loss_accum / max(L, 1)


    def loss_next_para(self, full_text: str, s: int, o: int, k: int, expand_frac: float = 0.0) -> torch.Tensor:
        """
        Feed up to s tokens (compressed/expanded context) and predict up to o tokens.
        Fix: treat s/o as maxima; clamp per-example so we always have target tokens when possible.

        Shapes:
        ctx_emb:   [1, T_ctx, D]
        tgt_emb:   [1, T_tgt, D]
        inputs:    [1, T_ctx + T_tgt, D]
        labels:    [1, T_ctx + T_tgt]   (labels[:, :T_ctx] = -100; labels[:, T_ctx:] = out_ids)
        """
        # Tokenize (cap for speed, but we can still clamp within what we got)
        toks = self.decoder_tok(full_text, truncation=True, max_length=s + o, return_tensors="pt")
        ids = toks.input_ids[0].to(self.device)
        N = int(ids.numel())

        # Need at least 2 tokens to have 1 context + 1 target
        if N < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Choose a minimum target length to supervise (treat o as a max)
        # If o is small, min_tgt shrinks with it.
        min_tgt = min(o, 32)  # tune (e.g. 8/16/32/64) depending on your data
        min_tgt = max(1, int(min_tgt))

        # If the sequence is too short to provide min_tgt, fall back to "at least 1" target token
        if N < (1 + min_tgt):
            min_tgt = 1
            if N < 2:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Clamp context length so we leave room for at least min_tgt target tokens
        ctx_len = min(int(s), N - min_tgt)
        ctx_len = max(1, int(ctx_len))  # ensure at least 1 ctx token

        # Target length is whatever remains, capped by o
        tgt_len = min(int(o), N - ctx_len)
        tgt_len = max(0, int(tgt_len))

        if tgt_len <= 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # (Optional) random windowing inside the available tokenized span
        total_len = ctx_len + tgt_len
        if N > total_len:
            # sample start in [0, N - total_len]
            start = int(torch.randint(0, N - total_len + 1, (1,), device="cpu").item())
        else:
            start = 0

        ctx_ids = ids[start:start + ctx_len]
        out_ids = ids[start + ctx_len:start + ctx_len + tgt_len]

        # Build (compressed/expanded) context embedding sequence
        ctx_str = self.decoder_tok.decode(ctx_ids, skip_special_tokens=True)
        chunk_strs, chunk_ids = self._chunk_text(ctx_str, k_tokens=k)

        # Encoder→projector path for compressed chunks
        c = self._encode_chunks(chunk_strs)  # [L, D_enc]
        e = self._project_chunks(c)          # [L, D_dec]

        L = len(chunk_ids)
        expand_mask = torch.zeros(L, dtype=torch.bool, device=self.device)
        if L > 0 and expand_frac > 0.0:
            top = max(1, int(round(expand_frac * L)))
            lengths = torch.tensor([len(ch) for ch in chunk_ids], device=self.device)
            top_idx = torch.topk(lengths, k=min(top, L)).indices
            expand_mask[top_idx] = True

        seq = []
        for i, ids_i in enumerate(chunk_ids):
            if expand_mask[i]:
                seq.append(self._decoder_token_embeddings(ids_i.unsqueeze(0)).squeeze(0))  # [t_i, D]
            else:
                seq.append(e[i].unsqueeze(0))  # [1, D]

        if len(seq) == 0:
            # fallback: if no chunks (short context), embed ctx_ids directly
            seq.append(self._decoder_token_embeddings(ctx_ids.unsqueeze(0)).squeeze(0))  # [ctx_len, D]

        ctx_emb = torch.cat(seq, dim=0).unsqueeze(0).contiguous()  # [1, T_ctx, D]
        T_ctx = int(ctx_emb.size(1))

        # Teacher-forced target embeddings
        if out_ids.numel() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        tgt_ids = out_ids.unsqueeze(0).to(self.device)            # [1, T_tgt]
        tgt_emb = self._decoder_token_embeddings(tgt_ids)         # [1, T_tgt, D]
        T_tgt = int(tgt_emb.size(1))

        # Concatenate context + target embeddings for a single forward
        inputs = torch.cat([ctx_emb, tgt_emb], dim=1)             # [1, T_ctx + T_tgt, D]

        # Cast to decoder dtype (projector outputs fp32 but decoder may be fp16)
        _dec_dtype = next(self.decoder.parameters()).dtype
        inputs = inputs.to(dtype=_dec_dtype)

        # Attention mask (all ones)
        attn_mask = torch.ones((1, T_ctx + T_tgt), dtype=torch.long, device=self.device)

        # Labels: ignore context positions, supervise only the target span
        labels = torch.full((1, T_ctx + T_tgt), -100, dtype=torch.long, device=self.device)
        labels[0, T_ctx:T_ctx + T_tgt] = out_ids

        out = self.decoder(inputs_embeds=inputs, attention_mask=attn_mask, labels=labels)
        return out.loss


    def policy_step(self, question: str, passages: List[str], k: int, max_expand_frac: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """One REINFORCE step: sample expansion mask, compute reward = -PPL of supervised continuation."""
        ctx_text = "\n".join(passages)
        chunk_strs, chunk_ids = self._chunk_text(ctx_text, k_tokens=k)
        if len(chunk_strs) == 0:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        # ---- build compressed/expanded context sequence (no grad) ----
        with torch.no_grad():
            c = self._encode_chunks(chunk_strs)  # [L, Denc]
        L = c.size(0)
        pos01 = torch.linspace(0, 1, steps=L, device=self.device).unsqueeze(-1)
        logits = self.policy(c, pos01)          # [L]
        probs = torch.sigmoid(logits)
        bern = torch.distributions.Bernoulli(probs=probs)
        sample = bern.sample()                   # [L]

        max_expand = max(1, int(round(max_expand_frac * L)))
        if sample.sum().item() > max_expand:
            top_idx = torch.topk(logits, k=max_expand).indices
            mask = torch.zeros_like(sample)
            mask[top_idx] = 1.0
            sample = mask
        log_prob = bern.log_prob(sample).sum()

        with torch.no_grad():
            e = self._project_chunks(c)          # [L, Ddec]
        seq = []
        for i, ids_i in enumerate(chunk_ids):
            if sample[i] > 0.5:
                seq.append(self._decoder_token_embeddings(ids_i.unsqueeze(0)).squeeze(0))  # expanded tokens
            else:
                seq.append(e[i].unsqueeze(0))  # one-slot compressed chunk
        ctx_emb = torch.cat(seq, dim=0).unsqueeze(0)  # [1, T_ctx, D]

        # ---- prepend question embeddings (no grad) ----
        q_ids = self._tokenize(question, self.cfg.max_q_tokens).input_ids.to(self.device)
        with torch.no_grad():
            q_emb = self._decoder_token_embeddings(q_ids)  # [1, Q, D]
        dec_in = torch.cat([q_emb, ctx_emb], dim=1)        # [1, T_ctx+Q, D]
        _dec_dtype = next(self.decoder.parameters()).dtype
        dec_in = dec_in.to(dtype=_dec_dtype)

        # ---- build a short "target" continuation to score (no grad) ----
        with torch.no_grad():
            # quick greedy rollout conditioned on dec_in to synthesize a target
            cache = self._new_cache()
            out = self.decoder(inputs_embeds=dec_in, use_cache=True, past_key_values=cache)
            past = self._ensure_cache(out.past_key_values)
            rollout = []
            last = torch.tensor([[self.eos_id]], device=self.device)
            for _ in range(32):
                step_emb = self.decoder.get_input_embeddings()(last)
                o2 = self.decoder(inputs_embeds=step_emb, use_cache=True, past_key_values=past)
                last = torch.argmax(o2.logits[:, -1, :], dim=-1, keepdim=True)
                nid = last.item()
                if nid == self.eos_id:
                    break
                rollout.append(nid)
                past = self._ensure_cache(o2.past_key_values)
            target = torch.tensor([rollout[:16] or [self.eos_id]], device=self.device, dtype=torch.long)  # [1, T_tgt]

        # ---- compute reward = -PPL with proper masked labels (no grad) ----
        with torch.no_grad():
            tgt_emb = self._decoder_token_embeddings(target)                        # [1, T_tgt, D]
            inputs = torch.cat([dec_in, tgt_emb], dim=1).to(dtype=_dec_dtype)       # [1, T_ctx+Q+T_tgt, D]
            labels = torch.full((1, inputs.size(1)), -100, dtype=torch.long, device=self.device)
            labels[0, dec_in.size(1):dec_in.size(1) + target.size(1)] = target[0]   # only supervise target span
            out2 = self.decoder(inputs_embeds=inputs, labels=labels)
            ppl = torch.exp(out2.loss.detach())

        reward = -ppl
        return log_prob, reward



# ----------------------------
# Optim / Training helpers
# ----------------------------

def setup_optim(params, lr, wd, total_steps):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sch = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)
    return opt, sch


# ----------------------------
# Index build / load helpers
# ----------------------------

def cmd_index(args):
    seed_everything()
    enc = make_passage_encoder(args.embed_model, getattr(args, 'ollama_url', 'http://localhost:8089'))
    passages = load_passages_from_path(
        args.corpus,
        chunk_min=getattr(args, 'chunk_min', 256),
        chunk_max=getattr(args, 'chunk_max', 512),
    )
    if not passages:
        print("[index] no passages found.")
        return
    embs = enc.encode_passages(passages, bs=32)
    client = get_qdrant_client(args.qdrant_url)
    build_qdrant_collection(client, args.collection, embs, passages, append=getattr(args, 'append', False))
    print(f"[index] built collection '{args.collection}' with {len(passages)} new passages on {args.qdrant_url}")


# ----------------------------
# CLI Commands
# ----------------------------

def curriculum_schedule(total_steps: int, max_chunks: int):
    """Simple linear curriculum over steps: 1 → max_chunks."""
    plan = []
    for t in range(total_steps):
        c = 1 + int((max_chunks - 1) * (t / max(1, total_steps - 1)))
        plan.append(c)
    return plan


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                yield json.loads(ln)


def cmd_cpt_recon(args):
    seed_everything()
    cfg = REFRAGConfig(
        encoder_name=args.enc,
        decoder_name=args.dec,
        chunk_len_tokens=args.k,
        lr=args.lr,
        fp16=False,
        torch_compile=args.torch_compile,
    )
    model = REFRAG(cfg).to(now_device())
    # Freeze decoder; train encoder+projector
    for p in model.decoder.parameters():
        p.requires_grad = False
    params = list(model.encoder.parameters()) + list(model.projector.parameters())
    steps = args.steps
    opt, sch = setup_optim(params, lr=cfg.lr, wd=cfg.wd, total_steps=steps)

    data = list(load_jsonl(args.train_json))
    if len(data) == 0:
        print("[cpt_recon] no data.")
        return

    model.train()
    for step in range(steps):
        ex = random.choice(data)
        text = ex["tokens"]
        chunk_strs, _ = model._chunk_text(text, k_tokens=cfg.chunk_len_tokens)
        max_chunks = max(1, len(chunk_strs))
        cap = curriculum_schedule(steps, max_chunks)[step]
        loss = model.loss_reconstruction(text, k=cfg.chunk_len_tokens, num_chunks_cap=cap)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step(); sch.step()
        if step % max(1, args.log_every) == 0:
            print(f"[cpt_recon] step {step}/{steps} loss={loss.item():.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.encoder.state_dict(), os.path.join(args.out_dir, "encoder.pt"))
    torch.save(model.projector.state_dict(), os.path.join(args.out_dir, "projector.pt"))
    print(f"[cpt_recon] saved to {args.out_dir}")


def cmd_cpt_next(args):
    seed_everything()
    cfg = REFRAGConfig(
        encoder_name=args.enc,
        decoder_name=args.dec,
        chunk_len_tokens=args.k,
        lr=args.lr,
        fp16=False,
        torch_compile=args.torch_compile,
    )
    model = REFRAG(cfg).to(now_device())
    # Load from recon phase if provided
    if args.load_dir:
        enc_p = os.path.join(args.load_dir, "encoder.pt")
        proj_p = os.path.join(args.load_dir, "projector.pt")
        if os.path.exists(enc_p):
            model.encoder.load_state_dict(safe_torch_load(enc_p, map_location=now_device()))
        if os.path.exists(proj_p):
            model.projector.load_state_dict(safe_torch_load(proj_p, map_location=now_device()))
        print("[cpt_next] loaded encoder/projector init.")

    params = list(model.parameters())  # unfreeze all
    steps = args.steps
    opt, sch = setup_optim(params, lr=cfg.lr, wd=cfg.wd, total_steps=steps)
    data = list(load_jsonl(args.train_json))
    if len(data) == 0:
        print("[cpt_next] no data.")
        return

    model.train()
    for step in range(steps):
        ex = random.choice(data)
        text = ex["tokens"]
        s = ex.get("split", {}).get("s", 2048)
        o = ex.get("split", {}).get("o", 256)
        loss = model.loss_next_para(text, s=s, o=o, k=cfg.chunk_len_tokens, expand_frac=args.expand_frac)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step(); sch.step()
        if step % max(1, args.log_every) == 0:
            print(f"[cpt_next] step {step}/{steps} loss={loss.item():.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "refrag_full.pt"))
    print(f"[cpt_next] saved full model to {args.out_dir}")


def cmd_train_policy(args):
    seed_everything()
    cfg = REFRAGConfig(
        encoder_name=args.enc,
        decoder_name=args.dec,
        chunk_len_tokens=args.k,
        lr=args.lr,
        fp16=False,
        policy_hidden=args.policy_hidden,
        torch_compile=args.torch_compile,
    )
    model = REFRAG(cfg).to(now_device())
    # Optional warm-start
    if args.load_dir:
        try:
            model.encoder.load_state_dict(safe_torch_load(os.path.join(args.load_dir, "encoder.pt"), map_location=now_device()))
            model.projector.load_state_dict(safe_torch_load(os.path.join(args.load_dir, "projector.pt"), map_location=now_device()))
            print("[train_policy] loaded encoder/projector init.")
        except Exception:
            pass

    # Train policy only
    for p in model.decoder.parameters():
        p.requires_grad = False
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.projector.parameters():
        p.requires_grad = False
    params = list(model.policy.parameters())
    steps = args.steps
    opt, sch = setup_optim(params, lr=cfg.lr, wd=cfg.wd, total_steps=steps)

    client = get_qdrant_client(args.qdrant_url)
    qenc = make_passage_encoder(args.embed_model, getattr(args, 'ollama_url', 'http://localhost:8089'))

    data = list(load_jsonl(args.rag_json))
    if len(data) == 0:
        print("[train_policy] no data.")
        return

    baseline = None
    beta = 0.9  # EMA

    model.train()
    for step in range(steps):
        ex = random.choice(data)
        q = ex["question"]
        qv = qenc.encode_query(q)
        _, passages = search_qdrant(client, args.collection, qv, args.topk)

        log_prob, reward = model.policy_step(q, passages, k=cfg.chunk_len_tokens, max_expand_frac=args.p)
        r = reward.item()
        baseline = r if baseline is None else (beta*baseline + (1-beta)*r)
        advantage = r - baseline

        loss = -(log_prob * advantage)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step(); sch.step()

        if step % max(1, args.log_every) == 0:
            print(f"[train_policy] step {step}/{steps} reward={r:.4f} baseline={baseline:.4f} advantage={advantage:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.policy.state_dict(), os.path.join(args.out_dir, "policy.pt"))
    print(f"[train_policy] saved policy to {args.out_dir}")


def cmd_generate(args):
    seed_everything()
    cfg = REFRAGConfig(
        encoder_name=args.enc,
        decoder_name=args.dec,
        chunk_len_tokens=args.k,
        max_q_tokens=256,
        max_ctx_tokens=args.ctx_max,
        max_out_tokens=args.max_new,
        selective_p=args.p,
        fp16=False,
        torch_compile=args.torch_compile,
    )
    model = REFRAG(cfg)
    # Optional load
    if args.load_dir:
        enc_p = os.path.join(args.load_dir, "encoder.pt")
        proj_p = os.path.join(args.load_dir, "projector.pt")
        pol_p = os.path.join(args.load_dir, "policy.pt")
        full_p = os.path.join(args.load_dir, "refrag_full.pt")
        if os.path.exists(full_p):
            model.load_state_dict(safe_torch_load(full_p, map_location=now_device()), strict=False)
            print("[generate] loaded full model weights.")
        else:
            if os.path.exists(enc_p):
                model.encoder.load_state_dict(safe_torch_load(enc_p, map_location=now_device()))
            if os.path.exists(proj_p):
                model.projector.load_state_dict(safe_torch_load(proj_p, map_location=now_device()))
            if os.path.exists(pol_p):
                model.policy.load_state_dict(safe_torch_load(pol_p, map_location=now_device()))
            print("[generate] loaded available component weights.")

    client = get_qdrant_client(args.qdrant_url)
    qenc = make_passage_encoder(args.embed_model, getattr(args, 'ollama_url', 'http://localhost:8089'))
    qv = qenc.encode_query(args.question)
    _, passages = search_qdrant(client, args.collection, qv, args.topk)

    out = model.generate(
        question=args.question,
        passages=passages,
        k=args.k,
        p=args.p,
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        top_p=args.top_p,
        use_policy=(not args.heuristic),
    )
    print(json.dumps({"question": args.question, "passages": passages, **out}, indent=2))


# ----------------------------
# Argparse
# ----------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="REFRAG-style RAG (compress → sense/select → expand)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # index
    sp = sub.add_parser("index", help="Build Qdrant collection from corpus")
    sp.add_argument("--corpus", type=str, required=True, help="Text file, PDF file, or directory of PDFs/text files")
    sp.add_argument("--qdrant_url", type=str, default="http://localhost:6343", help="Qdrant server URL")
    sp.add_argument("--collection", type=str, default="refrag", help="Qdrant collection name")
    sp.add_argument("--embed_model", type=str, default="ollama://mxbai-embed-large:335m", help="Embedding model (use ollama://MODEL for Ollama, else HuggingFace model name)")
    sp.add_argument("--ollama_url", type=str, default="http://localhost:8089", help="Ollama embedding service URL (used when --embed_model starts with ollama://)")
    sp.add_argument("--chunk_min", type=int, default=256, help="Minimum chunk size in tokens (for PDF chunking)")
    sp.add_argument("--chunk_max", type=int, default=512, help="Maximum chunk size in tokens (for PDF chunking)")
    sp.add_argument("--append", action="store_true", help="Append to existing collection instead of recreating it")
    sp.set_defaults(func=cmd_index)

    # cpt_recon
    sp = sub.add_parser("cpt_recon", help="Continual pretraining phase A: reconstruction curriculum")
    sp.add_argument("--train_json", type=str, required=True, help="JSONL with {'tokens':..., 'split':{}}")
    sp.add_argument("--enc", type=str, default="roberta-base")
    sp.add_argument("--dec", type=str, default="meta-llama/Llama-3.2-3B")
    sp.add_argument("--k", type=int, default=64, help="Chunk length in decoder tokens")
    sp.add_argument("--steps", type=int, default=1000)
    sp.add_argument("--lr", type=float, default=2e-5)
    sp.add_argument("--log_every", type=int, default=50)
    sp.add_argument("--out_dir", type=str, default="runs/cpt_recon")
    sp.add_argument("--torch_compile", action="store_true", help="Enable torch.compile for the decoder (PyTorch 2.0+)")
    sp.set_defaults(func=cmd_cpt_recon)

    # cpt_next
    sp = sub.add_parser("cpt_next", help="Continual pretraining phase B: next-paragraph prediction")
    sp.add_argument("--train_json", type=str, required=True, help="JSONL with {'tokens':..., 'split':{'s','o'}}")
    sp.add_argument("--enc", type=str, default="roberta-base")
    sp.add_argument("--dec", type=str, default="meta-llama/Llama-3.2-3B")
    sp.add_argument("--k", type=int, default=64)
    sp.add_argument("--steps", type=int, default=1000)
    sp.add_argument("--lr", type=float, default=2e-5)
    sp.add_argument("--expand_frac", type=float, default=0.25, help="Uniform expansion fraction during CPT-B")
    sp.add_argument("--log_every", type=int, default=50)
    sp.add_argument("--load_dir", type=str, default="", help="Optional: dir with encoder.pt/projector.pt")
    sp.add_argument("--out_dir", type=str, default="runs/cpt_next")
    sp.add_argument("--torch_compile", action="store_true", help="Enable torch.compile for the decoder (PyTorch 2.0+)")
    sp.set_defaults(func=cmd_cpt_next)

    # train_policy
    sp = sub.add_parser("train_policy", help="Train selective expansion policy with REINFORCE")
    sp.add_argument("--rag_json", type=str, required=True, help="JSONL with {'question':..., 'answers':...} (answers optional)")
    sp.add_argument("--qdrant_url", type=str, default="http://localhost:6343", help="Qdrant server URL")
    sp.add_argument("--collection", type=str, default="refrag", help="Qdrant collection name")
    sp.add_argument("--embed_model", type=str, default="ollama://mxbai-embed-large:335m", help="Embedding model (use ollama://MODEL for Ollama, else HuggingFace model name)")
    sp.add_argument("--ollama_url", type=str, default="http://localhost:8089", help="Ollama embedding service URL")
    sp.add_argument("--enc", type=str, default="roberta-base")
    sp.add_argument("--dec", type=str, default="meta-llama/Llama-3.2-3B")
    sp.add_argument("--k", type=int, default=64)
    sp.add_argument("--steps", type=int, default=1000)
    sp.add_argument("--lr", type=float, default=1e-4)
    sp.add_argument("--p", type=float, default=0.25, help="Max expansion fraction per example")
    sp.add_argument("--topk", type=int, default=8, help="#passages retrieved per query")
    sp.add_argument("--policy_hidden", type=int, default=256)
    sp.add_argument("--log_every", type=int, default=50)
    sp.add_argument("--load_dir", type=str, default="", help="Optional: dir with encoder.pt/projector.pt")
    sp.add_argument("--out_dir", type=str, default="runs/policy")
    sp.add_argument("--torch_compile", action="store_true", help="Enable torch.compile for the decoder (PyTorch 2.0+)")
    sp.set_defaults(func=cmd_train_policy)

    # generate
    sp = sub.add_parser("generate", help="RAG generate with compression/expansion")
    sp.add_argument("--qdrant_url", type=str, default="http://localhost:6343", help="Qdrant server URL")
    sp.add_argument("--collection", type=str, default="refrag", help="Qdrant collection name")
    sp.add_argument("--embed_model", type=str, default="ollama://mxbai-embed-large:335m", help="Embedding model (use ollama://MODEL for Ollama, else HuggingFace model name)")
    sp.add_argument("--ollama_url", type=str, default="http://localhost:8089", help="Ollama embedding service URL")
    sp.add_argument("--enc", type=str, default="roberta-base")
    sp.add_argument("--dec", type=str, default="meta-llama/Llama-3.2-3B")
    sp.add_argument("--question", type=str, required=True)
    sp.add_argument("--topk", type=int, default=8)
    sp.add_argument("--k", type=int, default=64, help="Chunk length in tokens")
    sp.add_argument("--p", type=float, default=0.25, help="Max expansion fraction")
    sp.add_argument("--ctx_max", type=int, default=2048)
    sp.add_argument("--max_new", type=int, default=256)
    sp.add_argument("--temperature", type=float, default=0.0)
    sp.add_argument("--top_p", type=float, default=1.0)
    sp.add_argument("--heuristic", action="store_true", help="Use heuristic expansion instead of policy")
    sp.add_argument("--load_dir", type=str, default="", help="Optional: dir with saved weights (encoder/projector/policy or refrag_full.pt)")
    sp.add_argument("--torch_compile", action="store_true", help="Enable torch.compile for the decoder (PyTorch 2.0+)")
    sp.set_defaults(func=cmd_generate)

    return p


def main():
    p = build_argparser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
