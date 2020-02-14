from torch.utils.data.dataset import Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image


class CervixDataset(VisionDataset):
    def __init__(self, root, csv_path, transform=None, target_transform=None):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        super(CervixDataset, self).__init__(root, transform=transform, target_transform=target_transform)

        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img = Image.open(single_image_name)

        # Get label(class) of the image based on the cropped pandas column
        label = self.label_arr[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return self.data_len
