#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py – Train REFRAG encoder+projector+policy using ingested Qdrant chunks.

Three training phases (all keep the LLM decoder FROZEN in fp16):
  A) CPT Reconstruction: train encoder+projector to reconstruct chunks
  B) CPT Next-paragraph: train encoder+projector to predict continuation
  C) Policy RL: train selective expansion policy via REINFORCE

Usage (run inside Docker or with suitable GPU env):
  python train.py \
      --decoder /models/Meta-Llama-3-8B-Instruct \
      --qdrant_url http://localhost:6343 \
      --collection refrag_papers \
      --out_dir /workspace/runs/refrag_trained

Steps default to 1500 / 1500 / 800 for phases A/B/C respectively.
"""

import os, sys, json, re, random, time, argparse, math
from typing import List, Tuple, Dict

# ── make refrag.py importable from the same directory ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from refrag import (
    REFRAG, REFRAGConfig,
    seed_everything, now_device, safe_torch_load,
    get_qdrant_client, search_qdrant,
    make_passage_encoder,
    curriculum_schedule,
)
from transformers import get_linear_schedule_with_warmup


# ─────────────────────────────────
# 1.  Extract chunks from Qdrant
# ─────────────────────────────────

def extract_all_chunks(qdrant_url: str, collection: str) -> List[Dict]:
    """Scroll through every point in the collection and return payload dicts."""
    from qdrant_client import QdrantClient
    client = QdrantClient(url=qdrant_url, check_compatibility=False)
    chunks = []
    offset = None
    while True:
        pts, offset = client.scroll(
            collection, limit=256, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for p in pts:
            text = p.payload.get("text", "")
            # Parse out the [source: ...] tag we embedded during PDF ingestion
            m = re.match(r"\[source:\s*(.+?)\]\s*", text)
            source = m.group(1) if m else "unknown"
            body = text[m.end():] if m else text
            chunks.append({"source": source, "text": body})
        if offset is None:
            break
    return chunks


def build_cpt_data(chunks: List[Dict], min_chars: int = 200) -> List[Dict]:
    """
    Group chunks by source document and concatenate into long texts.
    Each document becomes one CPT training example with
    {"id": ..., "tokens": "<text>", "split": {"s": 2048, "o": 256}}.
    """
    from collections import defaultdict
    by_source = defaultdict(list)
    for ch in chunks:
        by_source[ch["source"]].append(ch["text"])

    data = []
    for src, texts in by_source.items():
        full = "\n\n".join(texts)
        if len(full) < min_chars:
            continue
        data.append({
            "id": f"doc_{len(data)}",
            "tokens": full,
            "split": {"s": 2048, "o": 256},
        })
    return data


def build_policy_data(chunks: List[Dict], n_questions: int = 500) -> List[Dict]:
    """
    Create synthetic question-like entries for policy training.
    Strategy: pick a random chunk, use its first sentence (up to 120 chars)
    as a 'question' (the embedding search will retrieve related passages).
    """
    data = []
    for i in range(n_questions):
        ch = random.choice(chunks)
        text = ch["text"].strip()
        # First sentence (heuristic: split on period)
        first_sent = text.split(".")[0].strip()
        if len(first_sent) < 15:
            first_sent = text[:200]
        q = first_sent[:200]
        data.append({"id": f"pq_{i}", "question": q, "answers": []})
    return data


# ─────────────────────────────────
# 2.  Optimizer helper
# ─────────────────────────────────

def make_optim(params, lr: float, wd: float, total_steps: int):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sch = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps,
    )
    return opt, sch


# ─────────────────────────────────
# 3.  Phase A  –  CPT Reconstruction
# ─────────────────────────────────

def train_cpt_recon(model: REFRAG, data: List[Dict], steps: int, lr: float,
                    log_every: int = 25, grad_clip: float = 1.0):
    """Freeze decoder; train encoder + projector to reconstruct chunk tokens."""
    print(f"\n{'='*60}")
    print(f"  Phase A: CPT Reconstruction  ({steps} steps)")
    print(f"{'='*60}")
    # Freeze decoder
    for p in model.decoder.parameters():
        p.requires_grad = False
    params = list(model.encoder.parameters()) + list(model.projector.parameters())
    trainable = sum(p.numel() for p in params if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    opt, sch = make_optim(params, lr=lr, wd=model.cfg.wd, total_steps=steps)
    k = model.cfg.chunk_len_tokens

    model.encoder.train()
    model.projector.train()

    losses_log = []
    t0 = time.time()
    for step in range(steps):
        ex = random.choice(data)
        text = ex["tokens"]
        # Curriculum: gradually increase #chunks per step
        chunk_strs, _ = model._chunk_text(text, k_tokens=k)
        max_chunks = max(1, len(chunk_strs))
        cap = curriculum_schedule(steps, max_chunks)[step]

        loss = model.loss_reconstruction(text, k=k, num_chunks_cap=cap)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        sch.step()

        lv = loss.item()
        losses_log.append(lv)
        if step % log_every == 0 or step == steps - 1:
            elapsed = time.time() - t0
            avg = sum(losses_log[-50:]) / len(losses_log[-50:])
            print(f"  [A] step {step:>5d}/{steps}  loss={lv:.4f}  avg50={avg:.4f}  "
                  f"chunks_cap={cap}  ({elapsed:.0f}s)")

    return losses_log


# ─────────────────────────────────
# 4.  Phase B  –  CPT Next-Paragraph
# ─────────────────────────────────

def train_cpt_next(model: REFRAG, data: List[Dict], steps: int, lr: float,
                   expand_frac: float = 0.25, log_every: int = 25,
                   grad_clip: float = 1.0):
    """
    Next-paragraph prediction: predict o tokens given s tokens of
    compressed/expanded context.  Decoder stays FROZEN – gradients flow
    through inputs_embeds back to the projector and encoder.
    """
    print(f"\n{'='*60}")
    print(f"  Phase B: CPT Next-Paragraph Prediction  ({steps} steps)")
    print(f"{'='*60}")
    # Keep decoder frozen, only train encoder + projector
    for p in model.decoder.parameters():
        p.requires_grad = False
    for p in model.encoder.parameters():
        p.requires_grad = True
    for p in model.projector.parameters():
        p.requires_grad = True

    params = list(model.encoder.parameters()) + list(model.projector.parameters())
    trainable = sum(p.numel() for p in params if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    opt, sch = make_optim(params, lr=lr, wd=model.cfg.wd, total_steps=steps)
    k = model.cfg.chunk_len_tokens

    model.encoder.train()
    model.projector.train()

    losses_log = []
    t0 = time.time()
    for step in range(steps):
        ex = random.choice(data)
        text = ex["tokens"]
        s = ex.get("split", {}).get("s", 2048)
        o = ex.get("split", {}).get("o", 256)

        loss = model.loss_next_para(text, s=s, o=o, k=k, expand_frac=expand_frac)
        if loss.item() == 0.0:
            continue  # skip degenerate examples
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        sch.step()

        lv = loss.item()
        losses_log.append(lv)
        if step % log_every == 0 or step == steps - 1:
            elapsed = time.time() - t0
            avg = sum(losses_log[-50:]) / len(losses_log[-50:])
            print(f"  [B] step {step:>5d}/{steps}  loss={lv:.4f}  avg50={avg:.4f}  "
                  f"({elapsed:.0f}s)")

    return losses_log


# ─────────────────────────────────
# 5.  Phase C  –  Policy RL
# ─────────────────────────────────

def train_policy(model: REFRAG, data: List[Dict], steps: int, lr: float,
                 qdrant_url: str, collection: str, embed_model: str,
                 ollama_url: str, topk: int = 8, max_expand_frac: float = 0.25,
                 log_every: int = 25, grad_clip: float = 1.0):
    """REINFORCE training of the selective-expansion policy."""
    print(f"\n{'='*60}")
    print(f"  Phase C: Policy RL (REINFORCE)  ({steps} steps)")
    print(f"{'='*60}")
    # Freeze everything except policy
    for p in model.decoder.parameters():
        p.requires_grad = False
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.projector.parameters():
        p.requires_grad = False
    for p in model.policy.parameters():
        p.requires_grad = True

    params = list(model.policy.parameters())
    trainable = sum(p.numel() for p in params if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    opt, sch = make_optim(params, lr=lr, wd=model.cfg.wd, total_steps=steps)
    k = model.cfg.chunk_len_tokens

    client = get_qdrant_client(qdrant_url)
    qenc = make_passage_encoder(embed_model, ollama_url)

    model.policy.train()

    baseline = None
    beta = 0.9  # EMA

    rewards_log = []
    t0 = time.time()
    for step in range(steps):
        ex = random.choice(data)
        q = ex["question"]
        qv = qenc.encode_query(q)
        _, passages = search_qdrant(client, collection, qv, topk)

        log_prob, reward = model.policy_step(q, passages, k=k, max_expand_frac=max_expand_frac)
        r = reward.item()
        baseline = r if baseline is None else (beta * baseline + (1 - beta) * r)
        advantage = r - baseline

        loss = -(log_prob * advantage)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        sch.step()

        rewards_log.append(r)
        if step % log_every == 0 or step == steps - 1:
            elapsed = time.time() - t0
            avg_r = sum(rewards_log[-50:]) / len(rewards_log[-50:])
            print(f"  [C] step {step:>5d}/{steps}  reward={r:.4f}  baseline={baseline:.4f}  "
                  f"avg50_reward={avg_r:.4f}  ({elapsed:.0f}s)")

    return rewards_log


# ─────────────────────────────────
# 6.  Save / Load helpers
# ─────────────────────────────────

def save_checkpoint(model: REFRAG, out_dir: str, tag: str = ""):
    os.makedirs(out_dir, exist_ok=True)
    pref = f"{tag}_" if tag else ""
    enc_p = os.path.join(out_dir, f"{pref}encoder.pt")
    proj_p = os.path.join(out_dir, f"{pref}projector.pt")
    pol_p = os.path.join(out_dir, f"{pref}policy.pt")
    torch.save(model.encoder.state_dict(), enc_p)
    torch.save(model.projector.state_dict(), proj_p)
    torch.save(model.policy.state_dict(), pol_p)
    print(f"  Saved: {enc_p}, {proj_p}, {pol_p}")


# ─────────────────────────────────
# 7.  Main
# ─────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train REFRAG encoder+projector+policy")
    p.add_argument("--decoder", type=str,
                   default="/data2/models/models--meta-llama--Meta-Llama-3-8B-Instruct"
                           "/snapshots/e5e23bbe8e749ef0efcf16cad411a7d23bd23298",
                   help="Path or HF name of the decoder LLM")
    p.add_argument("--encoder", type=str, default="roberta-base",
                   help="Chunk encoder model")
    p.add_argument("--qdrant_url", type=str, default="http://localhost:6343")
    p.add_argument("--collection", type=str, default="refrag_papers")
    p.add_argument("--embed_model", type=str, default="ollama://mxbai-embed-large:335m")
    p.add_argument("--ollama_url", type=str, default="http://localhost:8089")
    p.add_argument("--out_dir", type=str, default="runs/refrag_trained")
    p.add_argument("--k", type=int, default=64, help="Chunk length in decoder tokens")
    # Steps per phase
    p.add_argument("--steps_a", type=int, default=1500, help="Phase A steps")
    p.add_argument("--steps_b", type=int, default=1500, help="Phase B steps")
    p.add_argument("--steps_c", type=int, default=800, help="Phase C steps")
    # LR
    p.add_argument("--lr_a", type=float, default=2e-5, help="Phase A learning rate")
    p.add_argument("--lr_b", type=float, default=1e-5, help="Phase B learning rate")
    p.add_argument("--lr_c", type=float, default=1e-4, help="Phase C learning rate")
    # Misc
    p.add_argument("--expand_frac", type=float, default=0.25)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--n_policy_questions", type=int, default=500,
                   help="Number of synthetic questions for policy training")
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--skip_a", action="store_true", help="Skip Phase A")
    p.add_argument("--skip_b", action="store_true", help="Skip Phase B")
    p.add_argument("--skip_c", action="store_true", help="Skip Phase C")
    p.add_argument("--resume_dir", type=str, default="",
                   help="Dir with encoder.pt/projector.pt to resume from")
    args = p.parse_args()

    seed_everything(args.seed)
    print(f"REFRAG Training Pipeline")
    print(f"  Decoder:    {args.decoder}")
    print(f"  Encoder:    {args.encoder}")
    print(f"  Qdrant:     {args.qdrant_url} / {args.collection}")
    print(f"  Output:     {args.out_dir}")
    print(f"  Steps:      A={args.steps_a}  B={args.steps_b}  C={args.steps_c}")
    print(f"  Chunk k:    {args.k}")

    # ── Extract chunks from Qdrant ──
    print("\n[1/6] Extracting chunks from Qdrant ...")
    chunks = extract_all_chunks(args.qdrant_url, args.collection)
    print(f"  Got {len(chunks)} chunks from {len(set(c['source'] for c in chunks))} documents")

    # ── Build training data ──
    print("[2/6] Building CPT training data ...")
    cpt_data = build_cpt_data(chunks)
    print(f"  {len(cpt_data)} document-level examples")
    total_chars = sum(len(d["tokens"]) for d in cpt_data)
    print(f"  Total text: {total_chars:,} characters (~{total_chars//4:,} tokens)")

    print("[3/6] Building policy training data ...")
    policy_data = build_policy_data(chunks, n_questions=args.n_policy_questions)
    print(f"  {len(policy_data)} synthetic questions")

    # ── Save training data for reproducibility ──
    os.makedirs(args.out_dir, exist_ok=True)
    cpt_path = os.path.join(args.out_dir, "cpt_train.jsonl")
    pol_path = os.path.join(args.out_dir, "policy_train.jsonl")
    with open(cpt_path, "w") as f:
        for d in cpt_data:
            f.write(json.dumps(d) + "\n")
    with open(pol_path, "w") as f:
        for d in policy_data:
            f.write(json.dumps(d) + "\n")
    print(f"  Saved {cpt_path} and {pol_path}")

    # ── Build model ──
    print("\n[4/6] Loading REFRAG model ...")
    cfg = REFRAGConfig(
        encoder_name=args.encoder,
        decoder_name=args.decoder,
        chunk_len_tokens=args.k,
        max_ctx_tokens=2048,
        max_out_tokens=256,
        max_q_tokens=256,
        selective_p=args.expand_frac,
        fp16=True,  # load decoder in fp16 always
    )
    model = REFRAG(cfg)
    print(f"  Decoder device(s): {set(str(p.device) for p in model.decoder.parameters())}")
    print(f"  Encoder device: {next(model.encoder.parameters()).device}")

    # Optional resume — search for best available checkpoint with prefix priority
    if args.resume_dir:
        dev = model.device
        # Priority order: final > phaseC > phaseB > phaseA > bare
        prefixes = ["final_", "phaseC_", "phaseB_", "phaseA_", ""]
        for comp, attr in [("encoder.pt", "encoder"), ("projector.pt", "projector"), ("policy.pt", "policy")]:
            loaded = False
            for pref in prefixes:
                candidate = os.path.join(args.resume_dir, f"{pref}{comp}")
                if os.path.exists(candidate):
                    getattr(model, attr).load_state_dict(
                        safe_torch_load(candidate, map_location=dev))
                    print(f"  Resumed {attr} from {candidate}")
                    loaded = True
                    break
            if not loaded:
                print(f"  [warn] No checkpoint found for {attr} in {args.resume_dir}")

    # ══════════════════════════════
    # Phase A – reconstruction
    # ══════════════════════════════
    if not args.skip_a:
        train_cpt_recon(model, cpt_data, steps=args.steps_a,
                        lr=args.lr_a, log_every=args.log_every)
        save_checkpoint(model, args.out_dir, tag="phaseA")
    else:
        print("\n  [skip] Phase A")

    # ══════════════════════════════
    # Phase B – next-paragraph
    # ══════════════════════════════
    if not args.skip_b:
        train_cpt_next(model, cpt_data, steps=args.steps_b,
                       lr=args.lr_b, expand_frac=args.expand_frac,
                       log_every=args.log_every)
        save_checkpoint(model, args.out_dir, tag="phaseB")
    else:
        print("\n  [skip] Phase B")

    # ══════════════════════════════
    # Phase C – policy RL
    # ══════════════════════════════
    if not args.skip_c:
        train_policy(model, policy_data, steps=args.steps_c,
                     lr=args.lr_c, qdrant_url=args.qdrant_url,
                     collection=args.collection, embed_model=args.embed_model,
                     ollama_url=args.ollama_url, topk=args.topk,
                     max_expand_frac=args.expand_frac, log_every=args.log_every)
        save_checkpoint(model, args.out_dir, tag="phaseC")
    else:
        print("\n  [skip] Phase C")

    # ── Final checkpoint (all components at their final state) ──
    save_checkpoint(model, args.out_dir, tag="final")
    print(f"\n{'='*60}")
    print(f"  Training complete!  Weights in: {args.out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
