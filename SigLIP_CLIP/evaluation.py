"""Train and evaluate one or more one-shot runs, each described by a YAML config.

For every config:
  1. Run train.py's vector averaging to build the 16 class prototypes
     (training the projection first if the config turns it on).
  2. Embed every image in each evaluation split (the one-shot images are left out).
  3. Predict the class whose prototype is most similar (cosine similarity).
  4. Print and save the results to <output_dir>/<run_name>/:
       results.json                    accuracy per split and per class
       confusion_<split>.png           confusion matrix of the combined prototype

Accuracy is reported for three prototypes, so you can see what each half contributes:
  text-only   average of the 10 prompt vectors
  image-only  average of the augmented one-shot image vectors
  combined    the averaged image + text vector (the method itself)

Usage (from the repository root):
    python ConvNeXt_2800/evaluation.py ConvNeXt_2800/configs/oneshot_seed0.yaml
    python ConvNeXt_2800/evaluation.py ConvNeXt_2800/configs/*.yaml     # several runs + summary table
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")                   # save figures to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

from train import ROOT, apply_projection, embed_images, get_device, list_images, load_config, load_model, run_dir, train

PROTOTYPE_KINDS = {"text-only": "text_prototypes", "image-only": "image_prototypes", "combined": "prototypes"}


def load_split(split_dir, classes, exclude):
    """All (path, label) pairs in a split, minus the one-shot support images."""
    items = []
    for label, c in enumerate(classes):
        for p in list_images(split_dir / c):
            if str(p.relative_to(ROOT)) not in exclude:
                items.append((p, label))
    return items


@torch.no_grad()
def predict(model, preprocess, device, items, checkpoint, batch_size):
    """Returns the true labels and, for each prototype kind, the predicted labels."""
    protos = {kind: checkpoint[key].to(device) for kind, key in PROTOTYPE_KINDS.items()}
    projection = checkpoint.get("projection")
    preds = {kind: [] for kind in protos}
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        feats = embed_images(model, preprocess, [Image.open(p).convert("RGB") for p, _ in batch], device)
        if projection is not None:      # prototypes live in the projected space, so test images must too
            feats = apply_projection(feats, projection)
        for kind, P in protos.items():
            preds[kind].append((feats @ P.T).argmax(dim=-1).cpu())
    labels = np.array([label for _, label in items])
    return labels, {kind: torch.cat(v).numpy() for kind, v in preds.items()}


def save_confusion_matrix(labels, preds, classes, title, path):
    cm = confusion_matrix(labels, preds, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, xticks_rotation=90, colorbar=False, cmap="Blues")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return cm


def evaluate(cfg, model, preprocess, tokenizer, device):
    print(f"\n=== {cfg['run_name']} ===")
    print("training (vector averaging):")
    checkpoint = train(cfg, model, preprocess, tokenizer, device)
    classes = checkpoint["classes"]
    exclude = set(checkpoint["support_images"].values())
    out = run_dir(cfg)

    results = {"run_name": cfg["run_name"], "support_images": checkpoint["support_images"], "splits": {}}
    for split, split_path in cfg["data"]["eval_splits"].items():
        items = load_split(ROOT / split_path, classes, exclude)
        start = time.time()
        labels, preds = predict(model, preprocess, device, items, checkpoint, cfg["eval"]["batch_size"])
        print(f"evaluated {split}: {len(items)} images in {time.time() - start:.0f}s")

        cm = save_confusion_matrix(labels, preds["combined"], classes,
                                   f"{cfg['run_name']} | {split} | combined: {accuracy_score(labels, preds['combined']):.1%}",
                                   out / f"confusion_{split}.png")
        results["splits"][split] = {
            "n_images": len(items),
            "accuracy": {kind: float(accuracy_score(labels, p)) for kind, p in preds.items()},
            "per_class_accuracy": {c: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(classes)},
        }

    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print_run_results(results, classes)
    print(f"saved results to {out.relative_to(ROOT)}/")
    return results


def print_run_results(results, classes):
    splits = list(results["splits"])
    print(f"\n{'accuracy':12s}" + "".join(f"{s:>10s}" for s in splits))
    for kind in PROTOTYPE_KINDS:
        print(f"{kind:12s}" + "".join(f"{results['splits'][s]['accuracy'][kind]:10.1%}" for s in splits))

    print(f"\n{'per class (combined)':22s}" + "".join(f"{s:>10s}" for s in splits))
    for c in classes:
        print(f"{c:22s}" + "".join(f"{results['splits'][s]['per_class_accuracy'][c]:10.0%}" for s in splits))


def print_summary(all_results):
    splits = list(all_results[0]["splits"])
    print("\n=== summary: combined-prototype accuracy ===")
    print(f"{'run':24s}" + "".join(f"{s:>10s}" for s in splits))
    for r in all_results:
        print(f"{r['run_name']:24s}" + "".join(f"{r['splits'][s]['accuracy']['combined']:10.1%}" for s in splits))
    if len(all_results) > 1:
        accs = {s: [r["splits"][s]["accuracy"]["combined"] for r in all_results] for s in splits}
        print(f"{'mean ± std':24s}" + "".join(f"{np.mean(a):7.1%}±{np.std(a):.1%}" for a in accs.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", help="one or more YAML config files")
    args = parser.parse_args()

    device = get_device()
    print("device:", device)

    loaded, all_results = {}, []                # reuse the model when configs share it
    for path in args.configs:
        cfg = load_config(path)
        key = (cfg["model"]["name"], cfg["model"]["pretrained"])
        if key not in loaded:
            loaded[key] = load_model(cfg, device)
        all_results.append(evaluate(cfg, *loaded[key], device))

    print_summary(all_results)


if __name__ == "__main__":
    main()
