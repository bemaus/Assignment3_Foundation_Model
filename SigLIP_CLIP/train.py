import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import open_clip
from augmentations import AUGMENTATIONS

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(cfg, device):
    name, pretrained = cfg["model"]["name"], cfg["model"]["pretrained"]
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(name)
    return model.to(device).eval(), preprocess, tokenizer


def list_classes(folder):
    return sorted(p.name for p in Path(folder).iterdir() if p.is_dir())


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def run_dir(cfg):
    return ROOT / cfg["output_dir"] / cfg["run_name"]


@torch.no_grad()
def embed_images(model, preprocess, pil_images, device):
    batch = torch.stack([preprocess(im) for im in pil_images]).to(device)
    return F.normalize(model.encode_image(batch), dim=-1)


@torch.no_grad()
def embed_texts(model, tokenizer, texts, device):
    return F.normalize(model.encode_text(tokenizer(texts).to(device)), dim=-1)


def pick_support_images(train_dir, classes, seed):
    rng = random.Random(seed)
    return {c: rng.choice(list_images(train_dir / c)) for c in classes}


def load_prompts(path, classes, groups=("specific", "general", "social", "social_general")):
    with open(path) as f:
        spec = yaml.safe_load(f)
    prompts = {}
    for c in classes:
        if c not in spec:
            raise KeyError(f"{path} has no prompts for class '{c}'")
        entry = spec[c]
        templates = [p for g in groups for p in entry.get(g, [])]
        prompts[c] = [p.format(entry["name"]) for p in templates]
    return prompts


def apply_projection(vecs, projection):
    return F.normalize(vecs @ projection.to(vecs.device).T, dim=-1)


def supervised_contrastive_loss(z, labels, temperature):
    sim = z @ z.T / temperature
    self_mask = torch.eye(len(z), dtype=torch.bool)
    sim = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    positives = (labels[:, None] == labels[None, :]) & ~self_mask
    return -(log_prob.masked_fill(~positives, 0).sum(dim=1) / positives.sum(dim=1)).mean()


def train_projection(image_vecs, text_vecs, proj_cfg):
    feats = torch.cat(image_vecs + text_vecs)
    labels = torch.cat([torch.full((len(v),), i) for i, v in enumerate(image_vecs)] +
                       [torch.full((len(v),), i) for i, v in enumerate(text_vecs)])
    dim = feats.shape[1]

    W = torch.nn.Parameter(torch.eye(dim))
    optimizer = torch.optim.Adam([W], lr=proj_cfg.get("lr", 1e-3))
    epochs = proj_cfg.get("epochs", 300)
    temperature = proj_cfg.get("temperature", 0.07)
    identity_weight = proj_cfg.get("identity_weight", 1.0)

    print(f"training projection ({dim}x{dim}) on {len(feats)} vectors for {epochs} epochs")
    for epoch in range(1, epochs + 1):
        z = F.normalize(feats @ W.T, dim=-1)
        contrastive = supervised_contrastive_loss(z, labels, temperature)
        stay_close = ((W - torch.eye(dim)) ** 2).sum()
        loss = contrastive + identity_weight * stay_close

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % max(1, epochs // 5) == 0:
            print(f"  epoch {epoch:4d}  loss {loss.item():.4f}  (contrastive {contrastive.item():.4f}, "
                  f"distance from identity {stay_close.item():.4f})")
    return W.detach()


def train(cfg, model=None, preprocess=None, tokenizer=None, device=None):
    device = device or get_device()
    if model is None:
        model, preprocess, tokenizer = load_model(cfg, device)

    seed = cfg["train"]["seed"]
    torch.manual_seed(seed)

    train_dir = ROOT / cfg["data"]["train_dir"]
    classes = list_classes(train_dir)
    support = pick_support_images(train_dir, classes, seed)
    groups = cfg.get("prompt_groups", ["specific", "general", "social", "social_general"])
    prompts = load_prompts(ROOT / cfg["prompts_file"], classes, groups)

    image_vecs, text_vecs = [], []
    for c in classes:
        img = Image.open(support[c]).convert("RGB")
        views = [img] if cfg["train"].get("include_original", True) else []
        views += [aug(img) for aug in AUGMENTATIONS.values()]

        image_vecs.append(embed_images(model, preprocess, views, device).cpu())
        text_vecs.append(embed_texts(model, tokenizer, prompts[c], device).cpu())
        print(f"  {c:14s} {support[c].name}  ({len(views)} image views, {len(prompts[c])} prompts)")

    proj_cfg = cfg.get("projection") or {}
    projection = train_projection(image_vecs, text_vecs, proj_cfg) if proj_cfg.get("enabled", False) else None
    if projection is not None:
        image_vecs = [apply_projection(v, projection) for v in image_vecs]
        text_vecs = [apply_projection(v, projection) for v in text_vecs]

    image_protos = torch.stack([F.normalize(v.mean(dim=0), dim=-1) for v in image_vecs])
    text_protos = torch.stack([F.normalize(v.mean(dim=0), dim=-1) for v in text_vecs])

    w = cfg["train"].get("image_weight", 0.5)
    combined = F.normalize(w * image_protos + (1 - w) * text_protos, dim=-1)

    checkpoint = {
        "classes": classes,
        "prototypes": combined,
        "image_prototypes": image_protos,
        "text_prototypes": text_protos,
        "projection": projection,
        "support_images": {c: str(p.relative_to(ROOT)) for c, p in support.items()},
        "prompts": prompts,
        "config": cfg,
    }
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out / "prototypes.pt")
    print(f"saved {out.relative_to(ROOT) / 'prototypes.pt'}")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="path to a YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"training run '{cfg['run_name']}' with {cfg['model']['name']}")
    train(cfg)


if __name__ == "__main__":
    main()
