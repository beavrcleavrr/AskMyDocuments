#!/usr/bin/env python3
"""
generate_embeddings.py

Reads chunks from `chunks.json` and writes embeddings to `chunk_embeddings.json`.

Features:
- Supports AWS Bedrock (Titan v2 or Cohere via Bedrock) and OpenAI embeddings.
- Sends ONLY the text string to the model (fixes "expected type: String, found: JSONObject").
- Handles batching (where supported) and retries.
- Preserves {id, source} fields and outputs: [{"id","source","embedding": [...]}].

Configuration via environment variables:
  EMBEDDING_PROVIDER   = bedrock | openai          (default: bedrock)
  EMBEDDING_MODEL_ID   = model identifier string   (default: amazon.titan-embed-text-v2:0)
  AWS_REGION           = e.g., us-east-1           (required for bedrock)
  OPENAI_API_KEY       = <key>                     (required for openai)
  CHUNKS_PATH          = path to input chunks      (default: chunks.json)
  EMBEDDINGS_OUT       = path to output embeddings (default: chunk_embeddings.json)
  EMBED_BATCH_SIZE     = batch size for providers  (default: 32; Titan embeds 1-by-1)
  EMBED_MAX_RETRIES    = retry attempts            (default: 3)
  EMBED_RETRY_BACKOFF  = exponential base          (default: 1.5)
"""

import os
import json
import time
from typing import List, Dict, Any

# Optional: load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- Config via env ----------
PROVIDER = os.getenv("EMBEDDING_PROVIDER", "bedrock").lower().strip()  # bedrock | openai
MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "").strip()                 # e.g., amazon.titan-embed-text-v2:0 or cohere.embed-english-v3
AWS_REGION = os.getenv("AWS_REGION", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

CHUNKS_PATH = os.getenv("CHUNKS_PATH", "chunks.json").strip()
OUT_PATH = os.getenv("EMBEDDINGS_OUT", "chunk_embeddings.json").strip()
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "3"))
RETRY_BACKOFF = float(os.getenv("EMBED_RETRY_BACKOFF", "1.5"))

# Defaults
if not MODEL_ID:
    if PROVIDER == "bedrock":
        MODEL_ID = "amazon.titan-embed-text-v2:0"
    elif PROVIDER == "openai":
        MODEL_ID = "text-embedding-3-small"


# ---------- Provider helpers ----------
def _backoff_sleep(attempt: int):
    time.sleep(RETRY_BACKOFF ** attempt)


# ===== AWS Bedrock =====
def _get_bedrock_client():
    import boto3
    if not AWS_REGION:
        # Let boto3 fall back to its default resolution if region not provided,
        # but explicit region is recommended to avoid errors.
        return boto3.client("bedrock-runtime")
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def _embed_bedrock_titan(bedrock, texts: List[str], model_id: str) -> List[List[float]]:
    """
    Titan Text Embeddings v2 expects: {"inputText": "<string>"}
    One call per text (no array batching in a single request).
    """
    embs: List[List[float]] = []
    for t in texts:
        body = json.dumps({"inputText": t})
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = bedrock.invoke_model(
                    modelId=model_id,
                    accept="application/json",
                    contentType="application/json",
                    body=body,
                )
                payload = json.loads(resp["body"].read())
                vec = payload.get("embedding") or payload.get("vector") or payload.get("embeddings")
                if not vec:
                    raise ValueError(f"No embedding vector in Titan response: {payload}")
                embs.append(vec)
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    _backoff_sleep(attempt)
        if last_error:
            # Log and continue; do not crash the whole job
            print(f"❌ Error embedding chunk (Titan): {last_error}")
            # Append placeholder to keep ordering (or skip — but then counts mismatch)
            embs.append([])
    return embs


def _embed_bedrock_cohere(bedrock, texts: List[str], model_id: str) -> List[List[float]]:
    """
    Cohere via Bedrock expects: {"texts": ["t1", "t2", ...]}
    Supports batching in a single call.
    """
    out: List[List[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        body = json.dumps({"texts": batch})
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = bedrock.invoke_model(
                    modelId=model_id,
                    accept="application/json",
                    contentType="application/json",
                    body=body,
                )
                payload = json.loads(resp["body"].read())
                embeddings = payload.get("embeddings")
                if not embeddings:
                    raise ValueError(f"No embeddings in Cohere response: {payload}")

                # Normalize shape: could be [{"embedding":[...]}] or [[...]]
                norm = []
                for item in embeddings:
                    if isinstance(item, dict) and "embedding" in item:
                        norm.append(item["embedding"])
                    else:
                        norm.append(item)
                out.extend(norm)
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    _backoff_sleep(attempt)
        if last_error:
            print(f"❌ Error embedding batch (Cohere via Bedrock): {last_error}")
            # Fill placeholders to keep alignment
            out.extend([[] for _ in batch])
    return out


# ===== OpenAI =====
def _embed_openai(texts: List[str], model: str) -> List[List[float]]:
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for provider=openai")
    client = OpenAI(api_key=OPENAI_API_KEY)

    out: List[List[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                out.extend([d.embedding for d in resp.data])
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    _backoff_sleep(attempt)
        if last_error:
            print(f"❌ Error embedding batch (OpenAI): {last_error}")
            out.extend([[] for _ in batch])
    return out


# ---------- Local helpers ----------
def _load_chunks(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    norm: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        if isinstance(item, dict) and "text" in item:
            text = item.get("text")
            text = "" if text is None else str(text)
            # Normalize whitespace to reduce provider issues
            text = " ".join(text.split())
            if not text:
                continue
            norm.append({
                "id": item.get("id", f"chunk_{i}"),
                "text": text,
                "source": item.get("source", ""),
            })
        else:
            # If raw strings were saved, accept them
            text = "" if item is None else str(item)
            text = " ".join(text.split())
            if not text:
                continue
            norm.append({"id": f"chunk_{i}", "text": text, "source": ""})
    return norm


def _write_embeddings(records: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main():
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"{CHUNKS_PATH} not found. Run your chunking step first.")

    chunks = _load_chunks(CHUNKS_PATH)
    if not chunks:
        _write_embeddings([], OUT_PATH)
        print(f"✅ Generated embeddings for 0 chunks and saved to '{OUT_PATH}'.")
        return

    texts = [c["text"] for c in chunks]  # IMPORTANT: send only strings to the provider

    # Route to provider
    if PROVIDER == "bedrock":
        bedrock = _get_bedrock_client()
        if MODEL_ID.startswith("amazon.titan-embed"):
            vectors = _embed_bedrock_titan(bedrock, texts, MODEL_ID)
        elif MODEL_ID.startswith("cohere."):
            vectors = _embed_bedrock_cohere(bedrock, texts, MODEL_ID)
        else:
            raise ValueError(f"Unsupported Bedrock embedding model_id: {MODEL_ID}")
    elif PROVIDER == "openai":
        vectors = _embed_openai(texts, MODEL_ID)
    else:
        raise ValueError(f"Unknown provider '{PROVIDER}'. Use 'bedrock' or 'openai'.")

    # Ensure we have a vector per input (empty vectors mean failed items but keep alignment)
    if len(vectors) != len(chunks):
        print(f"⚠️ Embedding count mismatch: got {len(vectors)} for {len(chunks)} inputs")

    out_records: List[Dict[str, Any]] = []
    dropped = 0
    for c, vec in zip(chunks, vectors):
        if not vec:
            # Drop failed items rather than saving empty vectors into the index
            dropped += 1
            continue
        out_records.append({
            "id": c["id"],
            "source": c.get("source", ""),
            "embedding": vec
        })

    _write_embeddings(out_records, OUT_PATH)
    kept = len(out_records)
    print(f"✅ Generated embeddings for {kept} chunks (dropped {dropped}) and saved to '{OUT_PATH}'.")


if __name__ == "__main__":
    main()