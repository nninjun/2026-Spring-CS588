"""
Plot Precision@4 distribution and per-query values from retrieval_results.jsonl.

Writes PNGs under figures/ by default:
  - precision_at4_distribution.png  (bar chart of P@4 counts)
  - precision_at4_by_query.png    (scatter + labels for P@4 = 0.25)

Run from PA1 directory:
  python viz_precision_at4.py
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DEFAULT_JSONL = ROOT / "retrieval_results.jsonl"
DEFAULT_FIGURES_DIR = ROOT / "figures"
DEFAULT_OUT_DIST = DEFAULT_FIGURES_DIR / "precision_at4_distribution.png"
DEFAULT_OUT_BY_QUERY = DEFAULT_FIGURES_DIR / "precision_at4_by_query.png"


def parse_p_at_k(s: str) -> float:
    a, b = s.split("/")
    return float(int(a)) / float(int(b))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot Precision@4 distribution from retrieval_results.jsonl."
    )
    ap.add_argument("--jsonl", type=str, default=str(DEFAULT_JSONL))
    ap.add_argument(
        "--out-dist",
        type=str,
        default=str(DEFAULT_OUT_DIST),
        help="output path for P@4 distribution (bar chart)",
    )
    ap.add_argument(
        "--out-by-query",
        type=str,
        default=str(DEFAULT_OUT_BY_QUERY),
        help="output path for P@4 vs query index (scatter)",
    )
    ap.add_argument("--title-prefix", type=str, default="UKBench VGG16 Retrieval")
    ap.add_argument(
        "--label-p",
        type=float,
        default=0.25,
        help="label query indices where Precision@4 equals this value (default: 0.25)",
    )
    args = ap.parse_args()

    queries = []
    pvals = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = int(obj["query"])
            p = parse_p_at_k(obj["P@4"])
            queries.append(q)
            pvals.append(p)

    Path(args.out_dist).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_by_query).parent.mkdir(parents=True, exist_ok=True)

    counts = Counter(pvals)
    xs = [0.0, 0.25, 0.5, 0.75, 1.0]
    ys = [counts.get(x, 0) for x in xs]

    plt.figure(figsize=(7.5, 4.5), dpi=160)
    plt.bar([str(x) for x in xs], ys, color="#2D6CDF")
    plt.xlabel("Precision@4")
    plt.ylabel("Number of queries")
    plt.title(f"{args.title_prefix}: Precision@4 distribution")
    ymax = max(ys) if ys else 1
    for i, y in enumerate(ys):
        plt.text(i, y + max(1, int(0.01 * ymax)), str(y), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out_dist, bbox_inches="tight", facecolor="white")
    plt.close()

    plt.figure(figsize=(7.5, 4.5), dpi=160)
    plt.scatter(queries, pvals, s=14, alpha=0.75, color="#111827")
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Query index")
    plt.ylabel("Precision@4")
    plt.title(f"{args.title_prefix}: Precision@4 by query")
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)

    label_eps = 1e-9
    labeled = 0
    for q, p in zip(queries, pvals):
        if abs(p - args.label_p) <= label_eps:
            plt.text(
                q,
                p + 0.03,
                str(q),
                fontsize=8,
                ha="center",
                va="bottom",
                color="#B91C1C",
            )
            labeled += 1

    if labeled > 0:
        plt.text(
            0.99,
            0.02,
            f"labeled P@4={args.label_p:.2f}: {labeled} queries",
            transform=plt.gca().transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#B91C1C",
        )
    plt.tight_layout()
    plt.savefig(args.out_by_query, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {args.out_dist}")
    print(f"Saved: {args.out_by_query}")


if __name__ == "__main__":
    main()
