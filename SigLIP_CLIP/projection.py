import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train import (ROOT, apply_projection, embed_images, embed_texts, get_device, list_classes,
                   load_config, load_prompts, pick_support_images, train_projection)
import open_clip
from augmentations import AUGMENTATIONS
from evaluation import load_split

KINDS = ["text-only", "image-only", "combined"]


def load_model(cfg, device):
    name, pretrained = cfg["model"]["name"], cfg["model"].get("pretrained")
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    return model.to(device).eval(), preprocess, open_clip.get_tokenizer(name)


def mean_proto(vecs):
    return torch.stack([F.normalize(v.mean(dim=0), dim=-1) for v in vecs])


def maybe_project(vecs, projection):
    return vecs if projection is None else apply_projection(vecs, projection)


def build_prototypes(cfg, model, preprocess, tokenizer, device):
    seed = cfg["train"]["seed"]
    torch.manual_seed(seed)

    train_dir = ROOT / cfg["data"]["train_dir"]
    classes = list_classes(train_dir)
    support = pick_support_images(train_dir, classes, seed)
    if cfg.get("prompt_templates"):
        with open(ROOT / cfg["prompts_file"]) as f:
            names = {c: entry["name"] for c, entry in yaml.safe_load(f).items()}
        prompts = {c: [t.format(names[c]) for t in cfg["prompt_templates"]] for c in classes}
    else:
        groups = cfg.get("prompt_groups", ["specific", "general", "social", "social_general"])
        prompts = load_prompts(ROOT / cfg["prompts_file"], classes, groups)

    image_vecs, text_vecs = [], []
    for c in classes:
        img = Image.open(support[c]).convert("RGB")
        views = [img] if cfg["train"].get("include_original", True) else []
        if cfg["train"].get("augment", True):
            views += [aug(img) for aug in AUGMENTATIONS.values()]
        image_vecs.append(embed_images(model, preprocess, views, device).cpu())
        text_vecs.append(embed_texts(model, tokenizer, prompts[c], device).cpu())
        print(f"  {c:14s} {support[c].name}  ({len(views)} image views, {len(prompts[c])} prompts)")

    proj_cfg = cfg.get("projection") or {}
    projection = train_projection(image_vecs, text_vecs, proj_cfg) if proj_cfg.get("enabled", False) else None

    w = cfg["train"].get("image_weight", 0.5)
    image_protos = mean_proto([maybe_project(v, projection) for v in image_vecs])
    text_protos = mean_proto([maybe_project(v, projection) for v in text_vecs])
    checkpoint = {
        "classes": classes,
        "prototypes": F.normalize(w * image_protos + (1 - w) * text_protos, dim=-1),
        "image_prototypes": image_protos,
        "text_prototypes": text_protos,
        "projection": projection,
        "support_images": {c: str(p.relative_to(ROOT)) for c, p in support.items()},
        "prompts": prompts,
        "config": cfg,
    }
    out = ROOT / cfg["output_dir"] / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out / "prototypes.pt")
    return checkpoint


@torch.no_grad()
def predict(cfg, model, preprocess, device, items, ck):
    protos = {"text-only": ck["text_prototypes"], "image-only": ck["image_prototypes"], "combined": ck["prototypes"]}
    preds = {k: [] for k in KINDS}
    bs = cfg["eval"]["batch_size"]
    for i in range(0, len(items), bs):
        images = [Image.open(p).convert("RGB") for p, _ in items[i:i + bs]]
        feats = maybe_project(embed_images(model, preprocess, images, device).cpu(), ck["projection"])
        for k, P in protos.items():
            preds[k].append((feats @ P.T).argmax(dim=-1))
    labels = np.array([label for _, label in items])
    return labels, {k: torch.cat(v).numpy() for k, v in preds.items()}


def evaluate(cfg, model, preprocess, tokenizer, device):
    print(f"\n=== {cfg['run_name']} ===")
    print("building prototypes:")
    ck = build_prototypes(cfg, model, preprocess, tokenizer, device)
    classes = ck["classes"]
    exclude = set(ck["support_images"].values())

    results = {"run_name": cfg["run_name"], "model": cfg["model"], "projection": cfg.get("projection"),
               "support_images": ck["support_images"], "splits": {}}
    for split, split_path in cfg["data"]["eval_splits"].items():
        items = load_split(ROOT / split_path, classes, exclude)
        start = time.time()
        labels, preds = predict(cfg, model, preprocess, device, items, ck)
        print(f"evaluated {split}: {len(items)} images in {time.time() - start:.0f}s")

        cm = confusion_matrix(labels, preds["combined"], labels=range(len(classes)))
        results["splits"][split] = {
            "n_images": len(items),
            "accuracy": {k: float(accuracy_score(labels, preds[k])) for k in KINDS},
            "per_class_accuracy": {c: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(classes)},
        }

    print_run_results(results, classes)
    print(f"saved prototypes to {cfg['output_dir']}/{cfg['run_name']}/prototypes.pt")
    return results


def print_run_results(results, classes):
    splits = list(results["splits"])
    print(f"\n{'accuracy':12s}" + "".join(f"{s:>10s}" for s in splits))
    for k in KINDS:
        print(f"{k:12s}" + "".join(f"{results['splits'][s]['accuracy'][k]:10.1%}" for s in splits))

    print(f"\n{'per class (combined)':22s}" + "".join(f"{s:>10s}" for s in splits))
    for c in classes:
        print(f"{c:22s}" + "".join(f"{results['splits'][s]['per_class_accuracy'][c]:10.0%}" for s in splits))


def print_summary(all_results):
    splits = list(all_results[0]["splits"])
    print("\n=== summary: combined-prototype accuracy ===")
    print(f"{'run':24s}" + "".join(f"{s:>10s}" for s in splits))
    for r in all_results:
        print(f"{r['run_name']:24s}" + "".join(f"{r['splits'][s]['accuracy']['combined']:10.1%}" for s in splits))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", help="one or more YAML config files")
    args = parser.parse_args()

    device = get_device()
    print("device:", device)

    loaded, all_results = {}, []
    for path in args.configs:
        cfg = load_config(path)
        key = (cfg["model"]["name"], cfg["model"].get("pretrained"))
        if key not in loaded:
            loaded[key] = load_model(cfg, device)
        all_results.append(evaluate(cfg, *loaded[key], device))

    print_summary(all_results)


if __name__ == "__main__":
    main()
