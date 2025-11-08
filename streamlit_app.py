#!/usr/bin/env python3
"""
streamlit_app.py

Streamlit chat UI for AskMyDocuments (RAG):
- Loads chunk embeddings from `chunk_embeddings.json`
- Loads original chunk texts from `chunks.json`
- Embeds user query (Bedrock or OpenAI)
- Retrieves top-k similar chunks
- Calls an LLM (Claude via Bedrock or OpenAI)
- Displays answers with citations and sources

Environment variables (can live in .env):
  # Embeddings (must match what you used to build embeddings)
  EMBEDDING_PROVIDER=bedrock|openai               (default: bedrock)
  EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0 (Bedrock default)
  AWS_REGION=us-east-1
  OPENAI_API_KEY=sk-...

  # I/O
  CHUNKS_PATH=chunks.json
  EMBEDDINGS_PATH=chunk_embeddings.json

  # LLM (defaults to Claude via Bedrock; set model ID explicitly if needed)
  LLM_PROVIDER=bedrock|openai                     (default: bedrock)
  LLM_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
  LLM_MAX_TOKENS=1024
  LLM_TEMPERATURE=0.1
"""

import os
import json
import math
import time
import streamlit as st
from typing import List, Dict, Any, Tuple

# Optional: load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- Config ----------
EMB_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "bedrock").lower().strip()
EMB_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0").strip()
AWS_REGION = os.getenv("AWS_REGION", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

CHUNKS_PATH = os.getenv("CHUNKS_PATH", "chunks.json").strip()
EMBED_PATH = os.getenv("EMBEDDINGS_PATH", "chunk_embeddings.json").strip()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "bedrock").lower().strip()
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0").strip()
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# UI defaults
TOP_K_DEFAULT = 5
CONTEXT_MAX_CHARS_DEFAULT = 8000  # rough guard to keep prompts manageable


# ---------- Math helpers ----------
def _l2norm(vec: List[float]) -> float:
    return math.sqrt(sum(v*v for v in vec)) or 1.0

def _normalize(vec: List[float]) -> List[float]:
    n = _l2norm(vec)
    return [v / n for v in vec]

def _cosine(u: List[float], v: List[float]) -> float:
    # If both normalized, cosine = dot product
    return sum(a*b for a, b in zip(u, v))


# ---------- Data loading ----------
@st.cache_data(show_spinner=False)
def load_embeddings(path: str) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
    """
    Returns:
      vectors_norm: List[List[float]] normalized vectors
      meta_list:    matching list of {id, source}
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    vectors = []
    meta = []
    for item in raw:
        vec = item.get("embedding")
        if not vec:
            continue
        vectors.append(_normalize(vec))
        meta.append({"id": item.get("id"), "source": item.get("source", "")})
    return vectors, meta

@st.cache_data(show_spinner=False)
def load_chunks(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Map chunk-id -> {"text":..., "source":...}
    """
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    out = {}
    for item in arr:
        if isinstance(item, dict) and "id" in item and "text" in item:
            out[item["id"]] = {"text": item["text"], "source": item.get("source", "")}
    return out


# ---------- Embedding (query) ----------
def embed_query(text: str) -> List[float]:
    if EMB_PROVIDER == "bedrock":
        import boto3, json as _json
        bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION or None)
        if EMB_MODEL_ID.startswith("amazon.titan-embed"):
            body = _json.dumps({"inputText": text})
            resp = bedrock.invoke_model(
                modelId=EMB_MODEL_ID,
                accept="application/json",
                contentType="application/json",
                body=body
            )
            payload = json.loads(resp["body"].read())
            vec = payload.get("embedding") or payload.get("vector") or payload.get("embeddings")
            if not vec:
                raise RuntimeError(f"No embedding in Titan response: {payload}")
            return vec
        elif EMB_MODEL_ID.startswith("cohere."):
            body = _json.dumps({"texts": [text]})
            resp = bedrock.invoke_model(
                modelId=EMB_MODEL_ID,
                accept="application/json",
                contentType="application/json",
                body=body
            )
            payload = json.loads(resp["body"].read())
            embs = payload.get("embeddings")
            if not embs:
                raise RuntimeError(f"No embeddings in Cohere response: {payload}")
            item = embs[0]
            return item["embedding"] if isinstance(item, dict) and "embedding" in item else item
        else:
            raise ValueError(f"Unsupported Bedrock embedding model: {EMB_MODEL_ID}")
    elif EMB_PROVIDER == "openai":
        from openai import OpenAI
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY required for provider=openai")
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.embeddings.create(model=EMB_MODEL_ID, input=[text])
        return resp.data[0].embedding
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMB_PROVIDER}")


# ---------- Retrieval ----------
def retrieve(query: str,
            vectors_norm: List[List[float]],
            meta: List[Dict[str, Any]],
            id_to_chunk: Dict[str, Dict[str, Any]],
            top_k: int):
    q_vec = _normalize(embed_query(query))
    scores = []
    for i, v in enumerate(vectors_norm):
        s = _cosine(q_vec, v)
        scores.append((s, i))

    scores.sort(key=lambda x: x[0], reverse=True)
    hits = []
    for s, idx in scores[:top_k]:
        cid = meta[idx]["id"]
        src = meta[idx]["source"]
        chunk = id_to_chunk.get(cid, {})
        hits.append({
            "id": cid,
            "source": src,
            "score": round(float(s), 4),
            "text": chunk.get("text", "")
        })
    return hits


# ---------- LLM calls ----------
def call_llm_bedrock(system_prompt: str, user_prompt: str) -> str:
    import boto3
    import json as _json
    br = boto3.client("bedrock-runtime", region_name=AWS_REGION or None)

    # Anthropic messages schema on Bedrock
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": system_prompt + "\n\n" + user_prompt}
            ]}
        ]
    }

    resp = br.invoke_model(
        modelId=LLM_MODEL_ID,
        accept="application/json",
        contentType="application/json",
        body=_json.dumps(body),
    )
    payload = json.loads(resp["body"].read())
    parts = payload.get("content", [])
    txt = ""
    for p in parts:
        if isinstance(p, dict) and p.get("type") == "text":
            txt += p.get("text", "")
    return txt.strip() or str(payload)


def call_llm_openai(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY required for provider=openai")
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()


def answer_with_context(question: str, contexts: List[Dict[str, Any]], style="short") -> str:
    # Build prompt with citations
    sources_block = "\n".join(
        [f"[{i+1}] {c['source']} (score {c['score']})" for i, c in enumerate(contexts)]
    )
    context_text = "\n\n---\n\n".join(
        [f"[{i+1}] {c['text']}" for i, c in enumerate(contexts)]
    )

    guidance = (
        "Be concise." if style == "short" else
        "Provide a thorough, structured answer with bullet points when useful."
    )

    system_prompt = (
        "You answer strictly using the provided context. "
        "If the answer isn't clearly found, say you don't have enough information. "
        "Cite sources inline like [1], [2]. "
        + guidance
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Context (top matches):\n{context_text}\n\n"
        f"Sources:\n{sources_block}\n\n"
        "Answer with citations like [1], [2]."
    )

    if LLM_PROVIDER == "bedrock":
        return call_llm_bedrock(system_prompt, user_prompt)
    elif LLM_PROVIDER == "openai":
        return call_llm_openai(system_prompt, user_prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


# ---------- Streamlit UI ----------
st.set_page_config(page_title="AskMyDocuments", page_icon="", layout="wide")
st.title("AskMyDocuments — Chat")

# Sidebar controls
with st.sidebar:
    st.subheader("Settings")
    st.caption("These mirror your .env values.")

    st.text(f"Embeddings: {EMB_PROVIDER} | {EMB_MODEL_ID}")
    st.text(f"LLM: {LLM_PROVIDER} | {LLM_MODEL_ID}")
    top_k = st.slider("Top‑K chunks", 1, 15, TOP_K_DEFAULT, 1)
    context_max_chars = st.slider("Max context characters", 1000, 20000, CONTEXT_MAX_CHARS_DEFAULT, 500)
    style = st.selectbox("Answer style", ["short", "detailed"], index=0)

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# Preflight checks
if not os.path.exists(EMBED_PATH) or not os.path.exists(CHUNKS_PATH):
    st.error("Required files not found. Run your pipeline first:\n\n"
             f"- `{CHUNKS_PATH}`\n- `{EMBED_PATH}`")
    st.stop()

# Load data
with st.spinner("Loading embeddings and chunks..."):
    vectors_norm, meta = load_embeddings(EMBED_PATH)
    id_to_chunk = load_chunks(CHUNKS_PATH)

if not vectors_norm:
    st.warning("No embeddings available. Generate them with `python3 generate_embeddings.py`.")
    st.stop()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "sources" in m and m["sources"]:
            with st.expander("Show sources"):
                for i, s in enumerate(m["sources"]):
                    st.markdown(f"**[{i+1}] {s['source']}** — score `{s['score']}`")
                    st.code(s["text"][:800] + ("..." if len(s["text"]) > 800 else ""), language="markdown")

# Chat input
prompt = st.chat_input("Ask a question about your documents...")
if prompt:
    # Render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context & answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                hits = retrieve(prompt, vectors_norm, meta, id_to_chunk, top_k)

                # Trim context to max chars budget (approx)
                total = 0
                trimmed = []
                for h in hits:
                    t = h["text"]
                    add_len = len(t)
                    if total + add_len > context_max_chars:
                        # add partial slice if it helps
                        remaining = max(0, context_max_chars - total)
                        if remaining > 200:  # only add if at least meaningful
                            h2 = dict(h)
                            h2["text"] = t[:remaining]
                            trimmed.append(h2)
                            total += remaining
                        break
                    trimmed.append(h)
                    total += add_len

                answer = answer_with_context(prompt, trimmed, style=style)
                st.markdown(answer)

                # Show sources
                with st.expander("Show sources"):
                    for i, h in enumerate(trimmed):
                        st.markdown(f"**[{i+1}] {h['source']}** — score `{h['score']}`")
                        st.code(h["text"][:1200] + ("..." if len(h["text"]) > 1200 else ""), language="markdown")

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": trimmed
                })
            except Exception as e:
                st.error(f"Error: {e}")