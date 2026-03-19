# CS588 PA1 — UKBench image retrieval

VGG16 (ImageNet pretrained) CNN features, **Euclidean distance**, exhaustive **top-4** search over the test set for each query. Reports **Average Precision@4** and total runtime for 250 queries.

## Setup

```bash
cd PA1
conda activate cs588
pip install -r requirements.txt
```

If the environment does not exist yet, create it once (e.g. `conda create -n cs588 python=3.10`) before `pip install`.  
Install a **PyTorch** wheel that matches your OS/CUDA from [pytorch.org](https://pytorch.org/get-started/locally/) if needed.

## Data layout

Place the **UKBench_small** dataset next to `main.py`:

```
PA1/
  main.py
  UKBench_small/
    query/    # 250 images
    test/     # 1000 images
```

Do **not** commit large image folders or checkpoints if your course forbids it.

## Run retrieval

```bash
python main.py
```

- Prints **Average Precision@4**, **total time** (feature extraction + search for all queries), and average time per query.
- Writes **`retrieval_results.jsonl`** (one JSON object per line: query, top-4 filenames, `P@4`, etc.).

Edit paths in `main.py` under **Settings** if your data lives elsewhere (`image_dir`, `results_jsonl`).

## Optional: figures for the report

Requires `matplotlib` (listed in `requirements.txt`).

**Precision@4 statistics** (histogram + scatter):

```bash
python viz_precision_at4.py
```

Outputs under **`figures/`** by default:

- `precision_at4_distribution.png`
- `precision_at4_by_query.png`

**Example panels** (query + top-4 for random 4/4, 2/4, 1/4 cases):

```bash
python viz_retrieval_examples.py
python viz_retrieval_examples.py --seed 42   # reproducible random picks
```

Outputs under **`figures/`**:

- `example_P4_4of4.png`, `example_P4_2of4.png`, `example_P4_1of4.png`

## Hardware

- Uses **`cuda:0`** when CUDA is available; otherwise CPU.
- GPU model and driver: report what you get from `nvidia-smi` if required.