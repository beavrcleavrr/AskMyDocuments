#!/usr/bin/env python3
import os
import json
import uuid
import boto3
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
PREFIX = os.getenv("S3_PREFIX", "documents/")
LOCAL_DIR = os.getenv("LOCAL_DIR", "downloaded_docs")
OUTPUT_PATH = os.getenv("CHUNKS_OUTPUT", "chunks.json")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))  # words

# Supported file types
SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".jpeg", ".jpg", ".png")

# Initialize S3 client
s3 = boto3.client("s3", region_name=AWS_REGION)

# Ensure local directory exists
os.makedirs(LOCAL_DIR, exist_ok=True)

def list_and_download_documents():
    """List supported files under PREFIX and download them locally."""
    print(f"Listing files in bucket '{BUCKET_NAME}' with prefix '{PREFIX}'...")
    file_paths = []
    continuation_token = None

    while True:
        kwargs = {"Bucket": BUCKET_NAME, "Prefix": PREFIX}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        resp = s3.list_objects_v2(**kwargs)
        contents = resp.get("Contents", [])
        if not contents and not continuation_token:
            print("No documents found.")
            break

        for obj in contents:
            key = obj["Key"]
            if key.endswith("/") or not key.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            filename = os.path.basename(key)
            local_path = os.path.join(LOCAL_DIR, filename)
            print(f"Downloading s3://{BUCKET_NAME}/{key} -> {local_path} ...")
            s3.download_file(BUCKET_NAME, key, local_path)
            file_paths.append(local_path)

        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break

    return file_paths

def extract_text_with_ocr(page):
    """OCR a PDF page rendered to bitmap."""
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x upscale helps OCR
    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if mode == "RGBA":
        img = img.convert("RGB")
    return pytesseract.image_to_string(img)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF; fallback to OCR when needed."""
    text_parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            t = page.get_text()
            if not t or not t.strip():
                t = extract_text_with_ocr(page)
            text_parts.append(t)
    return "\n".join(text_parts)

def extract_text_from_txt(txt_path):
    """Read plain text from .txt file."""
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_image(image_path):
    """Extract text from image using OCR."""
    img = Image.open(image_path)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return pytesseract.image_to_string(img)

def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Split text into word-based chunks."""
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def process_documents(file_paths):
    """Process each file into chunk records."""
    all_records = []
    for path in file_paths:
        filename = os.path.basename(path)
        ext = os.path.splitext(filename)[1].lower()
        print(f"Extracting text from {filename} ...")

        try:
            if ext == ".pdf":
                text = extract_text_from_pdf(path)
            elif ext == ".txt":
                text = extract_text_from_txt(path)
            elif ext in (".jpeg", ".jpg", ".png"):
                text = extract_text_from_image(path)
            else:
                print(f"   Skipping unsupported file type: {filename}")
                continue
        except Exception as e:
            print(f"   Skipping {filename}: {e}")
            continue

        chunks = chunk_text(text)
        if not chunks:
            print(f"  (no text found) {filename}")
            continue

        for idx, c in enumerate(chunks):
            rec = {
                "id": f"{filename}_{idx}",
                "text": c,
                "source": filename
            }
            all_records.append(rec)

        print(f"   {len(chunks)} chunks from {filename}")

    return all_records

def write_chunks_json(records, output_path=OUTPUT_PATH):
    """Write chunk records to JSON file."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f" Wrote {len(records)} chunks to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    if not BUCKET_NAME:
        print(" Error: S3_BUCKET_NAME is not set in the .env file.")
        exit(1)

    files = list_and_download_documents()
    if files:
        records = process_documents(files)
        write_chunks_json(records, OUTPUT_PATH)
        print(f" Downloaded {len(files)} files and created {len(records)} chunks.")
    else:
        write_chunks_json([], OUTPUT_PATH)
        print(" No supported files processed.")
