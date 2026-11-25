# Foundation Models: One-Shot Scene Classification

## ITCS 6169 – Assignment 3

This project tests how pretrained foundation models (CLIP and SigLIP) classify scene images using only one example image per class plus text prompts. It compares a single prompt, multiple prompts with augmentation, and a small learned projection layer, all set in a config file. The results are compared against a fully trained EfficientNet-B0 benchmark.

## Setup

Install the requirements (a virtual environment is recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Pick a configuration from `SigLIP_CLIP/configs/`:

- `clip_general_noaug.yaml` / `siglip_general_noaug.yaml` – one prompt, no augmentation
- `clip_all_prompts_aug.yaml` / `siglip_all_prompts_aug.yaml` – all prompts with augmentation
- `clip_projection.yaml` / `siglip_projection.yaml` – all prompts with augmentation and a projection layer

You can also create your own by copying one of these and changing its parameters.

From the project root, run:

```bash
python SigLIP_CLIP/evaluation.py SigLIP_CLIP/configs/<CONFIG_NAME>.yaml
```

For example:

```bash
python SigLIP_CLIP/evaluation.py SigLIP_CLIP/configs/clip_projection.yaml
```

Several configs can be run at once by listing them one after another:

```bash
python SigLIP_CLIP/evaluation.py SigLIP_CLIP/configs/clip_projection.yaml SigLIP_CLIP/configs/siglip_projection.yaml
```

## Running evaluation.py

The file builds the class prototypes from one image per class and the text prompts, then tests them on the train and test images. Accuracy (text-only, image-only and combined) and per-class accuracy are printed in the terminal. The prototypes, confusion matrices and a `results.json` file are saved to `SigLIP_CLIP/models/<run_name>/`. No pop-up windows are opened.

Pretrained weights are downloaded automatically the first time a model is loaded.

## Final Results Can Be Found in model_comparison.ipynb

Run all cells in `model_comparison.ipynb` to see every CLIP and SigLIP config compared against the EfficientNet benchmark (`Benchmark/efficientnet_b0_best.pt`), including accuracy tables, per-class accuracy and confusion matrices.
