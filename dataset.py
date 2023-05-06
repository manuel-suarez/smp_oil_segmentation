import os
import torch
import numpy as np

from PIL import Image

class OilSpillDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", class_to_mask='oil spill', transform=None):
        assert mode in {"train", "val", "test"}
        # Classes
        # Oil spill = 1 (cyan)
        # Look alike= 2 (red)
        # Ships     = 3 (brown)
        # Land      = 4 (green)
        # Sea       = 0 (background-black)
        assert class_to_mask in {'oil spill', 'look-alike', 'ship', 'land'}

        self.root = root
        self.mode = mode
        self.transform = transform
        self.class_to_mask = class_to_mask

        self.images_directory = os.path.join(self.root, self.mode, "images")
        self.masks_directory = os.path.join(self.root, self.mode, "labels_1D")

        # The train-val files are already separated by dirs so we only need to get the images filenames
        self.filenames = [f.split(".")[0] for f in os.listdir(self.images_directory)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + '.jpg')
        mask_path = os.path.join(self.masks_directory, filename + '.png')

        # Need to expand
        image = np.expand_dims(np.array(Image.open(image_path).convert('L')), axis=2)
        mask = self._preprocess_mask(np.array(Image.open(mask_path)), self.class_to_mask)

        # convert to other format HWC -> CHW
        image = np.moveaxis(image, -1, 0)
        mask = np.expand_dims(mask, 0)

        sample = dict(image=image, mask=mask)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def _preprocess_mask(mask, class_to_mask):
        # Depending on class_to_mask it's the value that we are letting on image-mask
        # Oil spill = 1 (cyan)
        # Look alike= 2 (red)
        # Ships     = 3 (brown)
        # Land      = 4 (green)
        # Sea       = 0 (background-black)
        mask = mask.astype(np.float32)
        if class_to_mask == 'oil spill':
            mask[mask == 2] = 0.0
            mask[mask == 3] = 0.0
            mask[mask == 4] = 0.00
        elif class_to_mask == 'look-alike':
            mask[mask == 1] = 0.0
            mask[mask == 2] = 1.0
            mask[mask == 3] = 0.0
            mask[mask == 4] = 0.0
        elif class_to_mask == 'ship':
            mask[mask == 1] = 0.0
            mask[mask == 2] = 0.0
            mask[mask == 3] = 1.0
            mask[mask == 4] = 0.0
        elif class_to_mask == 'land':
            mask[mask == 1] = 0.0
            mask[mask == 2] = 0.0
            mask[mask == 3] = 0.0
            mask[mask == 4] = 1.0

        return mask