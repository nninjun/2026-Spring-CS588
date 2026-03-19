"""
Generate UKBench retrieval example figures (query + top-4) for papers/reports.

Reads retrieval_results.jsonl (from main.py) and writes PNGs under figures/:
  example_P4_4of4.png, example_P4_2of4.png, example_P4_1of4.png

Each P@4 bucket (4/4, 2/4, 1/4) picks one query at random among all rows with that score.
Use --seed N for reproducible picks.

Run from PA1 directory:
  python viz_retrieval_examples.py
  python viz_retrieval_examples.py --seed 42
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_FIGURES_DIR = ROOT / "figures"
DEFAULT_IMAGE_DIR = ROOT / "UKBench_small"
DEFAULT_JSONL = ROOT / "retrieval_results.jsonl"


def image_index_from_name(name: str) -> int:
    m = re.search(r"image_(\d+)\.jpg$", name, re.IGNORECASE)
    if not m:
        raise ValueError(name)
    return int(m.group(1))


def category_from_index(idx: int) -> int:
    return idx // 4


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pick_random_row(rows, p_at_k: str, rng: random.Random):
    candidates = [r for r in rows if r["P@4"] == p_at_k]
    if not candidates:
        return None
    return rng.choice(candidates)


def show_image(ax, img_path: str, title: str, subtitle: str = "", *, fontsize_title=9, fontsize_sub=8):
    ax.imshow(Image.open(img_path).convert("RGB"))
    ax.axis("off")
    ax.set_title(title, fontsize=fontsize_title, fontweight="bold")
    if subtitle:
        ax.text(
            0.5,
            -0.08,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=fontsize_sub,
            color="#374151",
        )


def _draw_one_row(
    axes,
    row,
    image_dir: str,
    query_dir: str,
    test_dir: str,
    *,
    compact: bool,
    rank_word: bool = False,
    mark_fontsize: float | None = None,
):
    qpath = os.path.join(image_dir, query_dir, row["query_file"])
    q_cat = int(row["cat"])

    fs_title = 8 if compact else 9
    fs_sub = 7 if compact else 8
    fs_mark = mark_fontsize if mark_fontsize is not None else (7 if compact else 8)
    y_mark = -0.18 if compact else -0.22

    show_image(axes[0], qpath, "Query", row["query_file"], fontsize_title=fs_title, fontsize_sub=fs_sub)

    for i, fname in enumerate(row["top4"]):
        ax = axes[i + 1]
        tpath = os.path.join(image_dir, test_dir, fname)
        r_idx = image_index_from_name(fname)
        r_cat = category_from_index(r_idx)
        ok = r_cat == q_cat
        if rank_word:
            mark = "✓ same cat" if ok else "✗ other cat"
        else:
            mark = "✓" if ok else "✗"
        color = "#166534" if ok else "#991B1B"
        rank_title = f"Rank {i + 1}" if rank_word else f"#{i + 1}"
        show_image(ax, tpath, rank_title, fname, fontsize_title=fs_title, fontsize_sub=fs_sub)
        ax.text(
            0.5,
            y_mark,
            mark,
            transform=ax.transAxes,
            ha="center",
            fontsize=fs_mark,
            color=color,
            fontweight="bold",
        )


def make_figure(row, image_dir: str, query_dir: str, test_dir: str, out_path: str):
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), dpi=160)
    fig.patch.set_facecolor("white")
    _draw_one_row(axes, row, image_dir, query_dir, test_dir, compact=False, rank_word=True)
    fig.suptitle(
        f"Precision@4 = {row['P@4']}  |  {row['line']}",
        fontsize=10,
        y=1.02,
        color="#111827",
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate retrieval example figures (query + top-4) from retrieval_results.jsonl."
    )
    ap.add_argument("--image-dir", type=str, default=str(DEFAULT_IMAGE_DIR))
    ap.add_argument("--query-dir", type=str, default="query")
    ap.add_argument("--test-dir", type=str, default="test")
    ap.add_argument("--jsonl", type=str, default=str(DEFAULT_JSONL))
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_FIGURES_DIR))
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for which query is sampled per P@4 bucket (omit for different samples each run)",
    )
    args = ap.parse_args()

    rng = random.Random()
    if args.seed is not None:
        rng.seed(args.seed)

    rows = load_jsonl(args.jsonl)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    targets = [("4/4", "example_P4_4of4.png"), ("2/4", "example_P4_2of4.png"), ("1/4", "example_P4_1of4.png")]

    for p_at_k, fname in targets:
        row = pick_random_row(rows, p_at_k, rng)
        if row is None:
            print(f"Skip {p_at_k}: no row found in jsonl")
            continue
        out_path = os.path.join(args.out_dir, fname)
        make_figure(row, args.image_dir, args.query_dir, args.test_dir, out_path)
        print(f"  (picked query index {row['query']}, {row['query_file']})")


if __name__ == "__main__":
    main()
