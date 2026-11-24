from torchvision import transforms as T

AUGMENTATIONS = {
    "horizontal_flip":  T.RandomHorizontalFlip(p=1.0),
    "random_crop":      T.RandomResizedCrop(256, scale=(0.6, 0.9), ratio=(0.9, 1.1)),
    "center_zoom":      T.Lambda(lambda im: T.CenterCrop((int(im.height * 0.8), int(im.width * 0.8)))(im)),
    "rotation":         T.RandomRotation(degrees=15),
    "brightness":       T.ColorJitter(brightness=0.4),
    "contrast":         T.ColorJitter(contrast=0.4),
    "gaussian_blur":    T.GaussianBlur(kernel_size=7, sigma=(0.5, 2.0)),
    "perspective":      T.RandomPerspective(distortion_scale=0.3, p=1.0),
    "shift_and_scale":  T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    "sharpen":          T.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0),
}

assert len(AUGMENTATIONS) == 10
