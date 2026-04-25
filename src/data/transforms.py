"""
Image transforms for training and validation/test.

ImageNet stats are used since both ResNet50 and EfficientNet-B0
are pretrained on ImageNet. NIH X-rays are grayscale but converted
to 3-channel before normalization.
"""

from torchvision import transforms

# ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Training transforms with light augmentation.
    Augmentation is conservative for medical images — we don't want to
    distort diagnostic features like lung opacities or cardiomegaly borders.
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),  # slight oversize
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Validation / test transforms — deterministic, no augmentation.
    Also used during XAI heatmap generation.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Reverse ImageNet normalization for visualization.
    tensor: (C, H, W) or (B, C, H, W)
    """
    import torch
    mean = torch.tensor(mean).view(1, 3, 1, 1) if tensor.dim() == 4 else torch.tensor(mean).view(3, 1, 1)
    std  = torch.tensor(std).view(1, 3, 1, 1)  if tensor.dim() == 4 else torch.tensor(std).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
