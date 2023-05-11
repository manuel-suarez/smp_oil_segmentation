import os
import torch
import logging
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from dataset import OilSpillDataset
from model import OilModel
from utils import save_figure
from pytorch_lightning.loggers import CSVLogger

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s: %(name)s %(levelname)s - %(message)s', level=logging.INFO)
logging.info("Start!")

# redirect lightning logging to file
logger = logging.getLogger("lightning.pytorch")
logger.addHandler(logging.FileHandler("core.log"))

# download data
data_dir = "/home/est_posgrado_manuel.suarez/data/oil-spill-dataset_256"

# init train, val, test sets
#for arch in ['unet', 'unetplusplus', 'manet', 'linknet', 'fpn', 'pspnet', 'deeplabv3', 'deeplabv3plus', 'pan']:
for arch in ['unet', 'linknet', 'fpn', 'pspnet', 'pan']:
    logging.info(f"Architecture: {arch}")
    logging.info("1.- Dataset configuration")
    classes = ['oil spill', 'look-alike', 'ship']
    train_dataset = OilSpillDataset(data_dir, "train", classes=classes)
    valid_dataset = OilSpillDataset(data_dir, "val", classes=classes)
    test_dataset = OilSpillDataset(data_dir, "test", classes=classes)

    logging.info(f"Train size: {len(train_dataset)}")
    logging.info(f"Valid size: {len(valid_dataset)}")
    logging.info(f"Test size: {len(test_dataset)}")

    n_cpu = os.cpu_count()
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=n_cpu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)

    figures_dir = f"{arch}_figures"
    results_dir = f"{arch}_results"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Samples
    save_figure(train_dataset, "Train", os.path.join(figures_dir, "figure_01.png"))
    save_figure(valid_dataset, "Val", os.path.join(figures_dir, "figure_02.png"))
    save_figure(test_dataset, "Test", os.path.join(figures_dir, "figure_03.png"))

    logging.info("2.- Model instantiation")
    model = OilModel(arch, "resnet34", in_channels=1, out_classes=len(classes))

    logging.info("3.- Training")
    logger = CSVLogger(f"{arch}_logs", name="my_exp_name")
    trainer = pl.Trainer(gpus=1, max_epochs=100, logger=logger)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    # run validation dataset
    logging.info("4.- Validation metrics")
    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    logging.info(valid_metrics)

    # run test dataset
    logging.info("5.- Test metrics")
    test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
    logging.info(test_metrics)

    logging.info("6.- Result visualization")
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