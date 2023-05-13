import os
import torch
import numpy as np
import logging

from PIL import Image

class OilSpillDataset(torch.utils.data.Dataset):
    # List of classes
    CLASSES = ['oil spill', 'look-alike', 'ship', 'land']

    def __init__(self, root, mode="train", classes=None, transform=None):
        assert mode in {"train", "val", "test"}
        # Classes
        # Oil spill = 1 (cyan)
        # Look alike= 2 (red)
        # Ships     = 3 (brown)
        # Land      = 4 (green)
        # Sea       = 0 (background-black)

        self.root = root
        self.mode = mode
        self.transform = transform
        self.classes = classes
        self.classes_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.images_directory = os.path.join(self.root, self.mode, "images")
        self.masks_directory = os.path.join(self.root, self.mode, "labels_1D")

        # The train-val files are already separated by dirs, so we only need to get the images filenames
        self.filenames = [f.split(".")[0] for f in os.listdir(self.images_directory)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + '.jpg')
        mask_path = os.path.join(self.masks_directory, filename + '.png')

        # Need to expand
        image = np.array(Image.open(image_path))
        # convert to other format HWC -> CHW
        image = np.moveaxis(image, -1, 0)
        # mask
        mask = np.array(Image.open(mask_path))
        masks = [(mask == v) for v in self.classes_values]
        mask = np.stack(masks, axis=-1).astype('float')
        mask = np.moveaxis(mask, -1, 0)
        logging.debug(f"Get item index: {idx}, filename: {filename}, image shape: {image.shape}, mask shape: {mask.shape}")

        sample = dict(image=image, mask=mask)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample