import math
import random
import numpy as np

import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Cifar10Classification(Dataset):
    """
        Implementation of Dataloader for Cifar10 Dataset Classification
    """

    def __init__(
        self,
        data_path,
        data_mode,
        data_download=False,
        input_normalization=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.228, 0.224, 0.225)
    ) -> None:
        """
            Dataloader for loading data.
        """
        super().__init__()

        assert data_mode in [
            "train", "val", "test"], "'data_mode' must be one of this element: ['train', 'val', 'test']."

        # Loading 'Cifar-10' data
        self.data_mode = data_mode
        self.cifar10 = torchvision.datasets.CIFAR10(root=data_path, train=False if self.data_mode == "test" else True,
                                                    download=data_download, transform=None)

        # Normalization
        self.input_normalization = input_normalization
        self.mean, self.std = mean, std

        self.transfrom = transforms.ToTensor()
        if self.input_normalization:
            self.transfrom = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        self.samples_index = [ix for ix in range(len(self.cifar10))]

    def __getitem__(self, index):
        sample_index = self.samples_index[index]
        image, label = self.cifar10[sample_index]
        image = self.transfrom(image)

        return image, label

    def __len__(self):
        """
            This function returns number of data.
        """
        return len(self.samples_index)
