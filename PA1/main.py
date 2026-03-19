"""
UKBench_small image retrieval (exhaustive search) using VGG16 CNN features.
VGG16 features + Euclidean distance + NearestNeighbors.
"""

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = ROOT / "UKBench_small"
DEFAULT_RESULTS_JSONL = ROOT / "retrieval_results.jsonl"

# Settings (paths relative to this repo root by default)
image_dir = str(DEFAULT_IMAGE_DIR)
query_dirname = "query"
test_dirname = "test"
batch_size = 32
k = 4
print_examples = 3
num_workers = 2
results_jsonl = str(DEFAULT_RESULTS_JSONL)


def get_image_index(path):
    m = re.search(r"image_(\d+)\.jpg$", os.path.basename(path), re.IGNORECASE)
    if not m:
        raise ValueError("Expected filename like image_XXXXX.jpg: " + path)
    return int(m.group(1))


def category_from_index(index):
    return index // 4


class UKBench(torch.utils.data.Dataset):
    def __init__(self, image_dir, subdir):
        self.root = os.path.join(image_dir, subdir)
        self.filenames = []
        for f in os.listdir(self.root):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                self.filenames.append(os.path.join(self.root, f))
        self.filenames.sort(key=lambda p: get_image_index(p))
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img = Image.open(self.filenames[index]).convert("RGB")
        img = self.transform(img)
        return img, index

    def __len__(self):
        return len(self.filenames)


test_dataset = UKBench(image_dir, test_dirname)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
try:
    weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
    network = torchvision.models.vgg16(weights=weights).features.to(device)
except AttributeError:
    network = torchvision.models.vgg16(pretrained=True).features.to(device)
network.eval()
avgpool = torch.nn.AvgPool2d((7, 7))

feat_vec_list = []
t_total0 = time.perf_counter()
with torch.inference_mode():
    for image, _ in test_loader:
        image = image.to(device, non_blocking=True)
        feature_map = network(image)
        feature_vec = torch.flatten(avgpool(feature_map), 1)
        if device.type != "cpu":
            feature_vec = feature_vec.cpu()
        feat_vec_list.append(feature_vec.numpy())
feat_vec_list = np.concatenate(feat_vec_list, axis=0)

test_indices = np.array([get_image_index(p) for p in test_dataset.filenames])
label_list = np.array([category_from_index(i) for i in test_indices])

nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(feat_vec_list)

query_dataset = UKBench(image_dir, query_dirname)

precision_k_list = []
results_list = []
query_loader = torch.utils.data.DataLoader(
    dataset=query_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

seen = 0
with torch.inference_mode():
    for images, idxs in query_loader:
        images = images.to(device, non_blocking=True)
        feature_map = network(images)
        feature_vec = torch.flatten(avgpool(feature_map), 1)
        if device.type != "cpu":
            feature_vec = feature_vec.cpu()
        q_vecs = feature_vec.numpy()

        distances, indices = nbrs.kneighbors(q_vecs)
        for bi in range(q_vecs.shape[0]):
            qi = int(idxs[bi])
            nn_label = label_list[indices[bi]]
            query_idx = get_image_index(query_dataset.filenames[qi])
            query_cat = category_from_index(query_idx)
            hit = int((nn_label == query_cat).sum())
            precision_k_list.append(hit / k)

            top_names = [os.path.basename(test_dataset.filenames[i]) for i in indices[bi]]
            line = "Query {}: {} (cat={}) -> top-{}: {}  P@{}={}/{}".format(
                qi, os.path.basename(query_dataset.filenames[qi]), query_cat, k, top_names, k, hit, k)
            results_list.append({"query": qi, "query_file": os.path.basename(query_dataset.filenames[qi]),
                                "cat": query_cat, "top4": top_names, "P@4": "{}/{}".format(hit, k), "line": line})

            if seen < print_examples:
                print(line)
            seen += 1
t_total1 = time.perf_counter()

average_precision = np.mean(np.array(precision_k_list))
total_time = t_total1 - t_total0

print("Average Precision@{} = {}".format(k, average_precision))
print("Total processing time (all {} queries): {:.3f} s".format(len(query_dataset), total_time))
print("Average time per query: {:.4f} s".format(total_time / len(query_dataset)))

if results_jsonl:
    Path(results_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(results_jsonl, "w", encoding="utf-8") as f:
        for obj in results_list:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print("Saved {} query results to {}".format(len(results_list), results_jsonl))
