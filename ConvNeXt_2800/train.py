"""One-shot "training" by vector averaging.

For each class:
  1. Pick ONE image from the training folder (chosen by the config's seed).
  2. Make 10 augmented versions of it (plus the original if include_original is set),
     embed each one with the image encoder and average them -> image vector.
  3. Embed the class's text prompts (the groups listed in the config's prompt_groups;
     all 20 by default) with the text encoder and average them -> text vector.
  4. Average the image and text vectors -> one prototype vector per class.

Step 4 is the k x j grid idea: averaging (image_i + text_j) / 2 over every pair
gives exactly 0.5 * mean(images) + 0.5 * mean(texts), so we compute the two means
directly (k + j encoder passes instead of k * j). `image_weight` sets the mix;
0.5 is the plain grid average.

All vectors are made unit length before and after averaging. CLIP itself is never
changed.

Optional projection (config `projection.enabled: true`): before step 4, a small
linear layer is trained on top of the frozen CLIP vectors, using all the augmented
views and prompts as labeled examples. It pulls each class's vectors together and
pushes the 16 classes apart, so directions in CLIP's space that don't help separate
them count for less. The averaging and evaluation then happen in the projected space.

Usage (from the repository root):
    python ConvNeXt_2800/train.py ConvNeXt_2800/configs/oneshot_seed0.yaml
"""
import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # repository root (holds data/ and open_clip/)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import open_clip                        # noqa: E402  (local copy in ./open_clip)
from augmentations import AUGMENTATIONS  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------- helpers shared with evaluation.py ----------

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


# ---------- training ----------

def pick_support_images(train_dir, classes, seed):
    """Choose the single labeled example for each class."""
    rng = random.Random(seed)
    return {c: rng.choice(list_images(train_dir / c)) for c in classes}


def load_prompts(path, classes, groups=("specific", "general", "social", "social_general")):
    """Prompts per class from the chosen groups, with {} replaced by the class name."""
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
    """Map CLIP vectors through the learned projection and make them unit length again."""
    return F.normalize(vecs @ projection.to(vecs.device).T, dim=-1)


def supervised_contrastive_loss(z, labels, temperature):
    """Pull vectors of the same class together, push vectors of other classes apart.

    Every vector is compared with every other one: same-class pairs are positives,
    all other pairs are negatives (Khosla et al., 2020, "Supervised Contrastive Learning").
    """
    sim = z @ z.T / temperature
    self_mask = torch.eye(len(z), dtype=torch.bool)
    sim = sim.masked_fill(self_mask, float("-inf"))           # never compare a vector with itself
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    positives = (labels[:, None] == labels[None, :]) & ~self_mask
    return -(log_prob.masked_fill(~positives, 0).sum(dim=1) / positives.sum(dim=1)).mean()


def train_projection(image_vecs, text_vecs, proj_cfg):
    """Learn a square linear projection on top of the frozen CLIP vectors.

    Training data: every augmented view of the one-shot images plus every text prompt,
    labeled by class. The matrix starts as the identity (no change to CLIP's space), and
    a penalty keeps it close to the identity so it reshapes the space without forgetting it.
    """
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
    """Build and save the prototypes. Returns the saved checkpoint dict."""
    device = device or get_device()
    if model is None:
        model, preprocess, tokenizer = load_model(cfg, device)

    seed = cfg["train"]["seed"]
    torch.manual_seed(seed)             # makes the random augmentations repeatable

    train_dir = ROOT / cfg["data"]["train_dir"]
    classes = list_classes(train_dir)
    support = pick_support_images(train_dir, classes, seed)
    groups = cfg.get("prompt_groups", ["specific", "general", "social", "social_general"])
    prompts = load_prompts(ROOT / cfg["prompts_file"], classes, groups)

    # 1. Embed every augmented view and every prompt with frozen CLIP.
    image_vecs, text_vecs = [], []           # one (n, dim) tensor per class
    for c in classes:
        img = Image.open(support[c]).convert("RGB")
        views = [img] if cfg["train"].get("include_original", True) else []
        views += [aug(img) for aug in AUGMENTATIONS.values()]

        image_vecs.append(embed_images(model, preprocess, views, device).cpu())     # (n_views, dim)
        text_vecs.append(embed_texts(model, tokenizer, prompts[c], device).cpu())   # (n_prompts, dim)
        print(f"  {c:14s} {support[c].name}  ({len(views)} image views, {len(prompts[c])} prompts)")

    # 2. Optionally learn a projection that clusters each class and pushes classes apart.
    proj_cfg = cfg.get("projection") or {}
    projection = train_projection(image_vecs, text_vecs, proj_cfg) if proj_cfg.get("enabled", False) else None
    if projection is not None:
        image_vecs = [apply_projection(v, projection) for v in image_vecs]
        text_vecs = [apply_projection(v, projection) for v in text_vecs]

    # 3. Vector averaging (in the projected space when the projection is on).
    image_protos = torch.stack([F.normalize(v.mean(dim=0), dim=-1) for v in image_vecs])   # (16, dim)
    text_protos = torch.stack([F.normalize(v.mean(dim=0), dim=-1) for v in text_vecs])     # (16, dim)

    w = cfg["train"].get("image_weight", 0.5)
    combined = F.normalize(w * image_protos + (1 - w) * text_protos, dim=-1)

    checkpoint = {
        "classes": classes,
        "prototypes": combined,           # what evaluation uses
        "image_prototypes": image_protos,  # kept so evaluation can also report image-only
        "text_prototypes": text_protos,    # and text-only accuracy
        "projection": projection,         # (dim, dim) matrix, or None when the projection is off
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
