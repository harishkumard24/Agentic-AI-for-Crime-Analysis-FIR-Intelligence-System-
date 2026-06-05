import sys
import subprocess
import argparse
import os


def install_if_missing(packages):
    import importlib
    for pkg, import_name in packages:
        try:
            importlib.import_module(import_name)
        except ImportError:
            print(f"[INSTALL] Installing missing package: {pkg}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


install_if_missing([
    ("pandas", "pandas"),
    ("chromadb", "chromadb"),
    ("sentence-transformers", "sentence_transformers"),
])

import sqlite3
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


SQL_REQUIRED_COLS = {"record_id", "crime_description"}
VECTOR_REQUIRED_COLS = {"record_id", "fir_text"}

SQL_INDEX_COLS = [
    "record_id",
    "crime_type_normalized",
    "location_normalized",
    "year",
    "month",
]

VECTOR_META_COLS = [
    "crime_type",
    "location",
    "day_of_week_label",
    "time_of_day_label",
    "day",
    "month",
    "year",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Load FIR datasets into SQLite and ChromaDB.")
    parser.add_argument("--sql_csv", default="vaishnav25/Chatbot/outputs/sql_ready_fir.csv")
    parser.add_argument("--vector_csv", default="vaishnav25/Chatbot/outputs/vector_ready_fir.csv")
    parser.add_argument("--sqlite_db", default="vaishnav25/Chatbot/outputs/fir_relational.db")
    parser.add_argument("--chroma_dir", default="vaishnav25/Chatbot/outputs/chroma_store")
    parser.add_argument("--collection_name", default="fir_documents")
    parser.add_argument("--embedding_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--chunk_size", type=int, default=750)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--reset_chroma", action="store_true", help="Delete and recreate Chroma collection.")
    return parser.parse_args()


def validate_file(path, label):
    if not os.path.isfile(path):
        print(f"[ERROR] {label} not found: {path}")
        sys.exit(1)


def validate_columns(df, required, label):
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] {label} is missing required columns: {missing}")
        sys.exit(1)


def ensure_dir(path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)


def load_sqlite(sql_csv, sqlite_db):
    print(f"\n[SQLite] Reading: {sql_csv}")
    df = pd.read_csv(sql_csv, dtype=str, keep_default_na=False)
    validate_columns(df, SQL_REQUIRED_COLS, "sql_ready_fir.csv")
    if df.empty:
        print("[ERROR] SQL CSV is empty.")
        sys.exit(1)
    print(f"[SQLite] Rows loaded from CSV: {len(df)}")

    ensure_dir(sqlite_db)
    conn = sqlite3.connect(sqlite_db)
    df.to_sql("fir_cases", conn, if_exists="replace", index=False)
    print(f"[SQLite] Table 'fir_cases' written to: {sqlite_db}")

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM fir_cases")
    count = cursor.fetchone()[0]
    print(f"[SQLite] Verified row count: {count}")

    existing_cols = set(df.columns)
    indexes_created = []
    for col in SQL_INDEX_COLS:
        if col in existing_cols:
            idx_name = f"idx_fir_{col}"
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON fir_cases ({col})")
            indexes_created.append(col)
    conn.commit()
    conn.close()
    print(f"[SQLite] Indexes created on: {indexes_created}")
    return count


def chunk_text(text, chunk_size, chunk_overlap):
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - chunk_overlap
    return chunks


def sanitize_metadata(meta):
    clean = {}
    for k, v in meta.items():
        if isinstance(v, (int, float, str, bool)):
            clean[k] = v
        elif v is None or (isinstance(v, float) and v != v):
            clean[k] = ""
        else:
            clean[k] = str(v)
    return clean


def load_chromadb(vector_csv, chroma_dir, collection_name, embedding_model,
                  chunk_size, chunk_overlap, batch_size, reset_chroma):
    print(f"\n[Chroma] Reading: {vector_csv}")
    df = pd.read_csv(vector_csv, dtype=str, keep_default_na=False)
    validate_columns(df, VECTOR_REQUIRED_COLS, "vector_ready_fir.csv")
    if df.empty:
        print("[ERROR] Vector CSV is empty.")
        sys.exit(1)
    if df["fir_text"].str.strip().eq("").all():
        print("[ERROR] fir_text column is entirely empty.")
        sys.exit(1)
    print(f"[Chroma] Vector rows loaded: {len(df)}")

    os.makedirs(chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)

    if reset_chroma:
        try:
            client.delete_collection(name=collection_name)
            print(f"[Chroma] Existing collection '{collection_name}' deleted.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"[Chroma] Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)

    all_chunks = []
    all_ids = []
    all_metas = []

    for _, row in df.iterrows():
        record_id = str(row["record_id"]).strip()
        fir_text = str(row.get("fir_text", "")).strip()
        if not fir_text:
            continue

        chunks = chunk_text(fir_text, chunk_size, chunk_overlap)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"fir_{record_id}_chunk_{idx}"
            meta = {
                "record_id": record_id,
                "chunk_id": chunk_id,
                "chunk_index": idx,
            }
            for col in VECTOR_META_COLS:
                meta[col] = str(row.get(col, "")).strip()
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metas.append(sanitize_metadata(meta))

    total_chunks = len(all_chunks)
    print(f"[Chroma] Total chunks produced: {total_chunks}")
    print(f"[Chroma] Average chunks per FIR: {total_chunks / max(len(df), 1):.2f}")

    print(f"[Chroma] Encoding and inserting in batches of {batch_size}...")
    for i in range(0, total_chunks, batch_size):
        batch_texts = all_chunks[i:i + batch_size]
        batch_ids = all_ids[i:i + batch_size]
        batch_metas = all_metas[i:i + batch_size]
        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=batch_metas,
        )
        print(f"  Inserted batch {i // batch_size + 1} / {(total_chunks + batch_size - 1) // batch_size}")

    try:
        final_count = collection.count()
        print(f"[Chroma] Final collection document count: {final_count}")
    except Exception:
        print("[Chroma] Could not retrieve final count.")

    return total_chunks, all_ids[0] if all_ids else "N/A", all_metas[0] if all_metas else {}


def main():
    args = parse_args()

    validate_file(args.sql_csv, "SQL CSV")
    validate_file(args.vector_csv, "Vector CSV")

    sql_count = load_sqlite(args.sql_csv, args.sqlite_db)

    total_chunks, sample_id, sample_meta = load_chromadb(
        args.vector_csv,
        args.chroma_dir,
        args.collection_name,
        args.embedding_model,
        args.chunk_size,
        args.chunk_overlap,
        args.batch_size,
        args.reset_chroma,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  SQL rows loaded          : {sql_count}")
    print(f"  SQLite path              : {args.sqlite_db}")
    print(f"  Total chunks             : {total_chunks}")
    print(f"  Chroma collection        : {args.collection_name}")
    print(f"  Chroma persist dir       : {args.chroma_dir}")
    print(f"  Sample chunk ID          : {sample_id}")
    print(f"  Sample metadata          : {sample_meta}")
    print("=" * 60)


if __name__ == "__main__":
    main()
