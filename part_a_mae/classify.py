import sys
sys.path.append(".")
sys.path.append("starter")
sys.path.append("part_a_mae")

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from eurosat import load_eurosat_split, few_label_indices, patchify, CLASS_NAMES
from mae_model import MAE

torch.manual_seed(2)
make_folders()

x_train, y_train = load_eurosat_split("train")
x_test, y_test = load_eurosat_split("test")
few = few_label_indices(y_train)
print("labelled images used:", len(few))

model = MAE().to(device)
model.load_state_dict(torch.load("weights/mae.pt", map_location=device))
model.eval()
for p in model.parameters():
    p.requires_grad = False


def get_features(images):
    patches = patchify(images)
    all_features = []
    with torch.no_grad():
        for idx in batches(patches.shape[0], 500, shuffle=False):
            all_features.append(model.features(patches[idx].to(device)))
    return torch.cat(all_features)


train_features = get_features(x_train[few])
train_labels = y_train[few].to(device)
test_features = get_features(x_test)
test_labels = y_test.to(device)

classifier = nn.Linear(192, 10).to(device)
optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(200):
    logits = classifier(train_features)
    loss = loss_fn(logits, train_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print("epoch", epoch + 1, "| loss", round(loss.item(), 4))

with torch.no_grad():
    predictions = classifier(test_features).argmax(dim=1)

accuracy = (predictions == test_labels).float().mean().item()
print("test accuracy:", round(accuracy, 4), "(chance is 0.1)")

confusion = torch.zeros(10, 10, dtype=torch.long)
for true, pred in zip(test_labels.tolist(), predictions.tolist()):
    confusion[true, pred] += 1

pairs = []
for i in range(10):
    for j in range(10):
        if i != j:
            pairs.append((confusion[i, j].item(), CLASS_NAMES[i], CLASS_NAMES[j]))
pairs.sort(reverse=True)
print("most confused (count, true class, predicted class):")
for pair in pairs[:5]:
    print(pair)

torch.save(classifier.state_dict(), "weights/mae_classifier.pt")
save_json({"test_accuracy": accuracy, "confusion": confusion.tolist(), "class_names": CLASS_NAMES,
           "test_images": len(y_test), "labelled_images": len(few),
           "most_confused": pairs[:5]}, "results/mae_classifier.json")

plt.figure(figsize=(7, 6))
plt.imshow(confusion.numpy(), cmap="Blues")
plt.xticks(range(10), CLASS_NAMES, rotation=90)
plt.yticks(range(10), CLASS_NAMES)
for i in range(10):
    for j in range(10):
        plt.text(j, i, str(confusion[i, j].item()), ha="center", va="center", fontsize=7)
plt.xlabel("predicted")
plt.ylabel("true")
plt.title("Confusion matrix (accuracy " + str(round(accuracy, 3)) + ")")
plt.tight_layout()
plt.savefig("results/fig_mae_confusion.png", dpi=150)
print("saved results/fig_mae_confusion.png")
