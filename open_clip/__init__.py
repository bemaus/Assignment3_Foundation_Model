"""Trimmed copy of open_clip (mlfoundations/open_clip): CLIP-style image-text models only.

Audio (CLAP), captioning (CoCa, MaMMUT), NaFlex/GenLIP models, training losses/tasks,
Hugging Face Hub upload and ImageNet zero-shot helpers were removed.
"""
from .version import __version__

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .factory import (
    create_model,
    create_model_and_transforms,
    create_model_from_pretrained,
    get_tokenizer,
    list_models,
    add_model_config,
    get_model_config,
    load_checkpoint,
)
from .model import (
    CLIP,
    CustomTextCLIP,
    CLIPTextCfg,
    CLIPVisionCfg,
    get_cast_dtype,
    get_input_dtype,
)
from .pretrained import (
    list_pretrained,
    list_pretrained_tags_by_model,
    get_pretrained_cfg,
    download_pretrained,
)
from .tokenizer import Tokenizer, SimpleTokenizer, tokenize, decode
from .transform import image_transform, AugmentationCfg
