"""
REFRAG Streamlit Frontend – multi-turn RAG chat with streaming,
collapsible retrieved-chunks inspector, and compression metrics.
"""

import os
import sys
import time
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="REFRAG Chat", page_icon="🔬", layout="wide")

# ---------------------------------------------------------------------------
# Ensure refrag.py is importable
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
from refrag import (
    REFRAG,
    REFRAGConfig,
    get_qdrant_client,
    make_passage_encoder,
    search_qdrant,
    safe_torch_load,
    now_device,
    seed_everything,
)

# ---------------------------------------------------------------------------
# Helper: compression badge
# ---------------------------------------------------------------------------

def _render_compression_badge(c: dict):
    """Render a compact compression indicator."""
    ratio_pct = c.get("compression_ratio", 1.0) * 100
    n_comp = c.get("num_compressed", 0)
    n_exp = c.get("num_expanded", 0)
    n_total = n_comp + n_exp
    orig_tok = c.get("original_ctx_tokens", 0)
    comp_tok = c.get("compressed_seq_tokens", 0)
    saved_pct = max(0.0, 100.0 - ratio_pct)

    cols = st.columns([2, 3])
    with cols[0]:
        st.metric(
            "Context compression",
            f"{saved_pct:.0f}% saved",
            delta=f"{orig_tok} → {comp_tok} tokens",
            delta_color="inverse",
        )
    with cols[1]:
        st.progress(min(1.0, max(0.0, 1.0 - c.get("compression_ratio", 1.0))))
        st.caption(
            f"{n_comp}/{n_total} chunks compressed · "
            f"{n_exp}/{n_total} expanded"
        )


# ---------------------------------------------------------------------------
# Sidebar – tunables
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Qdrant")
    qdrant_url = st.text_input("Qdrant URL", value="http://localhost:6343")
    collection = st.text_input("Collection", value="refrag_papers")

    st.subheader("Embedding")
    embed_model = st.text_input("Embed model", value="ollama://mxbai-embed-large:335m")
    ollama_url = st.text_input("Ollama URL", value="http://localhost:8089")

    st.subheader("Decoder")
    decoder_name = st.text_input("Decoder model", value="/data2/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e5e23bbe8e749ef0efcf16cad411a7d23bd23298")
    encoder_name = st.text_input("Chunk encoder", value="roberta-base")
    load_dir = st.text_input("Weights dir (optional)", value="")

    st.subheader("Generation")
    rag_mode = st.radio(
        "RAG Mode",
        ["Standard RAG (baseline)", "REFRAG (compressed)"],
        index=0,
        help="Standard RAG passes full text to the LLM. "
             "REFRAG uses trained encoder+projector compression (experimental).",
    )
    topk = st.slider("Top-K passages", 1, 32, 8)
    k_chunk = st.slider("Chunk length (tokens)", 16, 256, 64)
    p_expand = st.slider("Max expansion fraction", 0.0, 1.0, 0.50, step=0.05,
                         help="Fraction of chunks expanded to full tokens. "
                              "Higher = better quality but less compression.")
    max_new = st.slider("Max new tokens", 32, 1024, 512)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.0, step=0.05)
    top_p_val = st.slider("Top-p", 0.0, 1.0, 1.0, step=0.05)
    use_policy = st.checkbox("Use learned policy", value=True)

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading REFRAG model …")
def load_model(enc_name: str, dec_name: str, wdir: str):
    seed_everything()
    cfg = REFRAGConfig(
        encoder_name=enc_name,
        decoder_name=dec_name,
        max_ctx_tokens=4096,
        max_out_tokens=512,
        fp16=True,
        torch_compile=False,
    )
    model = REFRAG(cfg)

    # Auto-detect weight directory: prefer explicit wdir, then check
    # /app/runs/refrag_trained for trained weights from train.py
    dirs_to_check = []
    if wdir:
        dirs_to_check.append(wdir)
    dirs_to_check.append("/app/runs/refrag_trained")
    dirs_to_check.append("runs/refrag_trained")

    loaded_any = False
    for d in dirs_to_check:
        if not os.path.isdir(d):
            continue
        # Try full model first
        full_p = os.path.join(d, "refrag_full.pt")
        if os.path.exists(full_p):
            model.load_state_dict(
                safe_torch_load(full_p, map_location=now_device()), strict=False
            )
            loaded_any = True
            break
        # Try final checkpoint from train.py
        for prefix in ("final_", "phaseC_", "phaseB_", "phaseA_", ""):
            enc_f = os.path.join(d, f"{prefix}encoder.pt")
            proj_f = os.path.join(d, f"{prefix}projector.pt")
            pol_f = os.path.join(d, f"{prefix}policy.pt")
            if os.path.exists(enc_f):
                model.encoder.load_state_dict(
                    safe_torch_load(enc_f, map_location=now_device())
                )
                loaded_any = True
            if os.path.exists(proj_f):
                model.projector.load_state_dict(
                    safe_torch_load(proj_f, map_location=now_device())
                )
                loaded_any = True
            if os.path.exists(pol_f):
                model.policy.load_state_dict(
                    safe_torch_load(pol_f, map_location=now_device())
                )
                loaded_any = True
            if loaded_any:
                break
        if loaded_any:
            break

    if loaded_any:
        print(f"[REFRAG] Loaded trained weights from {d}", flush=True)
    else:
        print("[REFRAG] WARNING: No trained weights found – using random init!", flush=True)
    return model


@st.cache_resource(show_spinner="Connecting to Qdrant …")
def load_qdrant(url: str):
    return get_qdrant_client(url)


@st.cache_resource(show_spinner="Loading passage encoder …")
def load_passage_encoder(em: str, ou: str):
    return make_passage_encoder(em, ou)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("🔬 REFRAG Chat")
st.caption(
    "Compress → Sense / Select → Expand · Multi-turn RAG"
    if not rag_mode.startswith("Standard")
    else "Standard RAG (baseline) · No compression"
)

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # Retrieved chunks (collapsible)
            if msg.get("chunks"):
                with st.expander(
                    f"📚 Retrieved chunks ({len(msg['chunks'])})", expanded=False
                ):
                    for i, (score, text) in enumerate(msg["chunks"]):
                        st.markdown(f"**#{i+1}** · score {score:.4f}")
                        st.code(
                            text[:600] + ("…" if len(text) > 600 else ""),
                            language=None,
                        )
            # Compression badge
            if msg.get("compression"):
                _render_compression_badge(msg["compression"])

        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask a question about the ingested papers …"):
    # ---- User message ----
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ---- Retrieve passages ----
    qenc = load_passage_encoder(embed_model, ollama_url)
    client = load_qdrant(qdrant_url)
    qv = qenc.encode_query(prompt)
    scores, passages = search_qdrant(client, collection, qv, topk)

    # ---- Assistant response ----
    with st.chat_message("assistant"):
        # Collapsible retrieved chunks
        with st.expander(
            f"📚 Retrieved chunks ({len(passages)})", expanded=False
        ):
            for i, (score, text) in enumerate(zip(scores, passages)):
                st.markdown(f"**#{i+1}** · score {score:.4f}")
                st.code(
                    text[:600] + ("…" if len(text) > 600 else ""),
                    language=None,
                )

        # Placeholders for compression badge and streamed text
        placeholder_comp = st.empty()
        placeholder_text = st.empty()

        # ---- Stream generation ----
        model = load_model(encoder_name, decoder_name, load_dir)
        is_standard = rag_mode.startswith("Standard")

        full_text = ""
        compression_info = None
        t_start = time.time()

        if is_standard:
            gen = model.generate_stream_standard(
                question=prompt,
                passages=passages,
                max_new_tokens=max_new,
                temperature=temperature,
                top_p=top_p_val,
            )
        else:
            gen = model.generate_stream(
                question=prompt,
                passages=passages,
                k=k_chunk,
                p=p_expand,
                max_new_tokens=max_new,
                temperature=temperature,
                top_p=top_p_val,
                use_policy=use_policy,
            )

        for token_str, extras in gen:
            if extras is not None:
                if not is_standard:
                    compression_info = {
                        "compression_ratio": extras.get("compression_ratio", 1.0),
                        "num_compressed": extras.get("num_compressed", 0),
                        "num_expanded": extras.get("num_expanded", 0),
                        "original_ctx_tokens": extras.get("original_ctx_tokens", 0),
                        "compressed_seq_tokens": extras.get("compressed_seq_tokens", 0),
                        "num_chunks": extras.get("num_chunks", 0),
                    }
                    with placeholder_comp.container():
                        _render_compression_badge(compression_info)
                else:
                    with placeholder_comp.container():
                        st.info(
                            f"📝 Standard RAG – {extras.get('num_passages', 0)} passages · "
                            f"{extras.get('prompt_tokens', 0)} prompt tokens"
                        )
            else:
                full_text += token_str
                placeholder_text.markdown(full_text + "▌")

        elapsed = time.time() - t_start
        placeholder_text.markdown(full_text)

        tok_count = len(full_text.split())
        st.caption(f"⏱ {elapsed:.1f}s · ~{tok_count} words")

    # ---- Persist to session ----
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_text,
            "chunks": list(zip(scores, passages)),
            "compression": compression_info,
        }
    )
