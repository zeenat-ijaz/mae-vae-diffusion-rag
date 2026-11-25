import os
import sys
import json
import time
import torch

sys.path.append("starter")

device = "cuda" if torch.cuda.is_available() else "cpu"
UHAT_ROOT = "starter/data/uhat"


def make_folders():
    os.makedirs("results", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    os.makedirs("cache", exist_ok=True)


def save_json(obj, path):
    file = open(path, "w")
    json.dump(obj, file, indent=1)
    file.close()


def load_json(path):
    file = open(path)
    obj = json.load(file)
    file.close()
    return obj


def mean(values):
    return sum(values) / len(values)


def count_params(model):
    total = 0
    for p in model.parameters():
        total = total + p.numel()
    return total


def gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def batches(n, batch_size, shuffle=True):
    if shuffle:
        order = torch.randperm(n)
    else:
        order = torch.arange(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def load_uhat():
    saved_file = "cache/uhat.pt"
    if os.path.exists(saved_file):
        return torch.load(saved_file)
    from uhat import load_uhat_characters
    x_train, y_train, names = load_uhat_characters(root=UHAT_ROOT, split="train")
    x_test, y_test, _ = load_uhat_characters(root=UHAT_ROOT, split="test", class_names=names)
    os.makedirs("cache", exist_ok=True)
    torch.save((x_train, y_train, x_test, y_test, names), saved_file)
    return x_train, y_train, x_test, y_test, names
