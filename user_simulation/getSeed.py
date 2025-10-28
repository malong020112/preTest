import json
from collections import defaultdict
from pathlib import Path
import argparse


def build_data_seed(input_path: Path, output_path: Path) -> None:
    counts = defaultdict(int)
    samples = defaultdict(list)  # collect up to 5 items per category

    with input_path.open('r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines
                continue

            cat = obj.get('category')
            if not cat:
                # Skip records without category
                continue

            counts[cat] += 1
            if len(samples[cat]) < 5:
                samples[cat].append(obj)

    seeds = []
    for cat, items in samples.items():
        seeds.append({
            'category': cat,
            'num': counts[cat],
            'items': items  # store the five (or fewer) items together
        })

    # Ensure parent exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write as JSON Lines: one JSON object per line
    with output_path.open('w', encoding='utf-8') as f:
        for obj in seeds:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Print a simple summary
    print(f"Wrote {len(seeds)} category seeds (JSONL) to: {output_path}")
    for cat in sorted(counts.keys()):
        print(f"{cat}: {counts[cat]}")


def main():
    parser = argparse.ArgumentParser(description="Generate data_seed.json from JSONL by category (5 samples per category)")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("IN3") / "train.jsonl",
        help="Path to input JSONL (default: IN3/train.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_seed.json"),
        help="Path to output JSON (default: data_seed.json)",
    )
    args = parser.parse_args()

    build_data_seed(args.input, args.output)


if __name__ == "__main__":
    main()
