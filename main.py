import os
import torch
import logging
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from pprint import pprint
from torch.utils.data import DataLoader

from dataset import OilSpillDataset
from model import OilModel
from utils import save_figure
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from pytorch_lightning.loggers import CSVLogger

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s: %(name)s %(levelname)s - %(message)s', level=logging.INFO)
logging.info("Start!")

# redirect lightning logging to file
logger = logging.getLogger("lightning.pytorch")
logger.addHandler(logging.FileHandler("core.log"))

# download data
data_dir = "/home/est_posgrado_manuel.suarez/data/oil-spill-dataset_256"
figures_dir = "figures"
results_dir = "results"
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# init train, val, test sets
logging.info("Dataset configuration")
train_dataset = OilSpillDataset(data_dir, "train")
valid_dataset = OilSpillDataset(data_dir, "val")
test_dataset = OilSpillDataset(data_dir, "test")

logging.info(f"Train size: {len(train_dataset)}")
logging.info(f"Valid size: {len(valid_dataset)}")
logging.info(f"Test size: {len(test_dataset)}")

n_cpu = os.cpu_count()
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=n_cpu)
valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)

# Samples
save_figure(train_dataset, "Train", "figure_01.png")
save_figure(valid_dataset, "Val", "figure_02.png")
save_figure(test_dataset, "Test", "figure_03.png")

logging.info("Model instantiation")
model = OilModel("FPN", "resnet34", in_channels=3, out_classes=1)

logging.info("Training")
logger = CSVLogger("logs", name="my_exp_name")
trainer = pl.Trainer(gpus=1, max_epochs=400, logger=logger)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

# run validation dataset
logging.info("Validation metrics")
valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
logging.info(valid_metrics)

# run test dataset
logging.info("Test metrics")
test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
logging.info(test_metrics)

logging.info("Result visualization")
batch = next(iter(test_dataloader))
with torch.no_grad():
    model.eval()
    logits = model(batch["image"])
pr_masks = logits.sigmoid()

for idx, (image, gt_mask, pr_mask) in enumerate(zip(batch["image"], batch["mask"], pr_masks)):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pr_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
    plt.title("Prediction")
    plt.axis("off")

    plt.savefig(os.path.join(results_dir, f"result_{str(idx).zfill(2)}.png"))

logging.info('Done!')