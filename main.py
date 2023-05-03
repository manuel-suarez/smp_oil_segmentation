import os
import torch
import logging
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from pprint import pprint
from torch.utils.data import DataLoader

from segmentation_models_pytorch import SimpleOxfordPetDataset

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s %(levelname)s - %(message)s', level=logging.INFO)
# download data
root = "."
SimpleOxfordPetDataset.download(root)

logging.info('Done!')