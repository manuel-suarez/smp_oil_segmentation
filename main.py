import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from pprint import pprint
from torch.utils.data import DataLoader

from segmentation_models_pytorch import SimpleOxfordPetDataset

# download data
root = "."
SimpleOxfordPetDataset.download(root)

