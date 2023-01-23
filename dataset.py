from torchvision.datasets import CIFAR10
import albumentations as A
from configs import base_config


classes10 = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class CIFARDataset(CIFAR10):
    class2idx = {classname: idx for idx, classname in enumerate(classes10)}
    idx2class = {idx: classname for idx, classname in enumerate(classes10)}

    def __init__(self, root=base_config.data_directory, train=True, download=True) -> None:

        self.trnsfrms_ = A.Compose([
            A.Resize(base_config.image_height, base_config.image_width),
            A.Normalize(mean=0, std=1)  # эквивалентно torchvision.transforms.Lambda(lambda z: z / 255.0)
            ])

        super().__init__(root=root, train=train, download=download)

        self.augmentations = None
        if self.train:
            self.augmentations = A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomBrightnessContrast(p=0.2),
                A.Rotate(p=0.3)
            ])
    
    def __getitem__(self, index: int):
        image, label = self.data[index], self.targets[index]

        image = self.trnsfrms_(image=image)["image"]

        if self.augmentations is not None:
            image = self.augmentations(image=image)["image"]
        
        return image.transpose(2, 0, 1).astype('float32'), label
    
    def __len__(self) -> int:
        return super().__len__()
