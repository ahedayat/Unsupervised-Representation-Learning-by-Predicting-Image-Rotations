import random
import numpy as np

import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Cifar10Rotation(Dataset):
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
        std=(0.228, 0.224, 0.225),
        rotation_angles=(0, 90, 180, 270),
        show_all_rotations=False,
        shuffle=False
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

        # Rotation Angles
        self.rotation_angles = rotation_angles
        # self.show_all_possibilities = show_all_possibilities

        # Normalization
        self.input_normalization = input_normalization
        self.mean, self.std = mean, std

        self.transfrom = transforms.ToTensor()
        if self.input_normalization:
            self.transfrom = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        self.samples_index = list()

        for ix in range(len(self.cifar10)):
            angles_index = list()

            if not show_all_rotations:
                # Select randomly a rotation angle
                rotation_index = np.random.randint(len(self.rotation_angles))
                angles_index.append(rotation_index)
            else:
                # Select all rotation angles
                angles_index = [ix for ix in range(len(self.rotation_angles))]

            for angle_index in angles_index:
                self.samples_index.append(
                    (ix, angle_index)
                )

        if shuffle:
            random.shuffle(self.samples_index)
            random.shuffle(self.samples_index)

    def __getitem__(self, index):
        index, rotation_index = self.samples_index[index]
        image, _ = self.cifar10[index]

        rotation_angle = self.rotation_angles[rotation_index]

        image = transforms.functional.rotate(image, angle=rotation_angle)
        image = self.transfrom(image)

        return image, rotation_index

    def __len__(self):
        """
            This function returns number of data.
        """
        return len(self.samples_index)
