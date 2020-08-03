import torch
import os
import pandas as pd
import matplotlib.image as mpimg
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class XrayImageDataset(Dataset):
    """"Hand Xray Dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to training csv
            root_dir (string): Path to training images
            transform(callable, optional): Optional transform to be applied on a sample
        """
        self.ages = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ages)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image path
        img_name = os.path.join(self.root_dir, str(self.ages.iloc[idx, 0]) + '.png')

        # Open image
        image = mpimg.imread(img_name)

        # Extract z_score for selected image
        image_age = self.ages.iloc[idx, 4]
        image_age = np.array([image_age])
        image_age = image_age.astype('double')

        # Extract gender of selected image
        image_gender = self.ages.iloc[idx, 3]

        # Format output dictionary
        sample = {'image': image, 'bone_age_z': image_age, 'gender': image_gender}

        if self.transform:
            image = Image.fromarray(sample['image']*255).convert('RGB')  # Convert image to RGB
            image_eq = ImageOps.equalize(image, mask=None)  # Equalize histogram of images
            sample['image'] = self.transform(image_eq)  # Execute transforms

        return sample
