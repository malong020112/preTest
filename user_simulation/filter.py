import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import List, Dict, Any
from transformers import GPT2TokenizerFast
from openai import OpenAI


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get('task'):
                items.append(obj)
    return items


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def batched(seq: List[Any], size: int) -> List[List[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def embed_texts(client: OpenAI, model: str, texts: List[str], batch_size: int = 128, max_retries: int = 5, base_delay: float = 1.0) -> List[List[float]]:
    vectors: List[List[float]] = []
    for chunk in batched(texts, batch_size):
        retries = 0
        delay = base_delay
        while True:
            try:
                resp = client.embeddings.create(model=model, input=chunk)
                # OpenAI v1 returns data in order
                for d in resp.data:
                    vectors.append(d.embedding)
                break
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(delay)
                delay = min(delay * 2, 30)
    return vectors


def greedy_filter_by_similarity(tasks: List[Dict[str, Any]], vectors: List[List[float]], threshold: float) -> List[int]:
    selected: List[int] = []
    for i, vec in enumerate(vectors):
        keep = True
        for j in selected:
            if cosine_sim(vec, vectors[j]) >= threshold:
                keep = False
                break
        if keep:
            selected.append(i)
    return selected


def process_file(client: OpenAI, model: str, in_path: Path, out_path: Path, threshold: float, batch_size: int) -> None:
    rows = load_jsonl(in_path)
    if not rows:
        # Create empty file
        write_jsonl(out_path, [])
        print(f"{in_path.name}: 0 -> 0 (no valid tasks)")
        return

    texts = [str(r.get('task', '')).strip() for r in rows]
    vectors = embed_texts(client, model, texts, batch_size=batch_size)

    keep_idx = greedy_filter_by_similarity(rows, vectors, threshold)
    filtered = [rows[i] for i in keep_idx]

    write_jsonl(out_path, filtered)
    print(f"{in_path.name}: {len(rows)} -> {len(filtered)} (threshold={threshold})")


def main():
    parser = argparse.ArgumentParser(description="Filter tasks per category by embedding similarity (cosine < threshold)")
    parser.add_argument('--input_dir', type=Path, default=Path('expand_task'), help='Directory of per-category JSONL files')
    parser.add_argument('--output_dir', type=Path, default=Path('expand_task_filtered'), help='Directory to write filtered JSONL files')
    parser.add_argument('--model', type=str, default='text-embedding-ada-002', help='Embedding model name')
    parser.add_argument('--threshold', type=float, default=0.8, help='Cosine similarity threshold (keep pairwise < threshold)')
    parser.add_argument('--batch_size', type=int, default=128, help='Embedding batch size')
    parser.add_argument('--base_url', type=str, default=os.environ.get('OPENAI_BASE_URL'), help='Optional custom base URL for OpenAI-compatible endpoint')
    args = parser.parse_args()

    # Initialize client (read API key from env OPENAI_API_KEY)
    if args.base_url:
        client = OpenAI(base_url=args.base_url)
    else:
        client = OpenAI()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for entry in sorted(args.input_dir.glob('*.json')):
        out_path = args.output_dir / entry.name
        try:
            process_file(client, args.model, entry, out_path, args.threshold, args.batch_size)
        except Exception as e:
            print(f"Error processing {entry.name}: {e}")


if __name__ == '__main__':
    main()

