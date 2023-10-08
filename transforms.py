import numpy as np
import torch
import torchvision.transforms as T

# augmentations

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB

train_transforms = T.Compose(
    [
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ConvertImageDtype(torch.float),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

test_transforms = T.Compose(
    [
        T.ConvertImageDtype(torch.float),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# https://arxiv.org/abs/1710.09412v2
def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    assert alpha > 0, "alpha should be larger than 0"
    assert x.size(0) > 1, "Mixup cannot be applied to a single instance."

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam