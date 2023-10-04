import os
import logging
import argparse
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from dataset import OilSpillDataset
from model import OilModel
from utils import save_figure, test_model
from pytorch_lightning.loggers import CSVLogger


cositas
pastelitos


def create_datasets(data_dir, classes):
    return (
        OilSpillDataset(data_dir, "train", classes=classes),
        OilSpillDataset(data_dir, "val", classes=classes),
        OilSpillDataset(data_dir, "test", classes=classes)
    )


def create_dataloaders(n_cpu, train_dataset, valid_dataset, test_dataset):
    return (
        DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=n_cpu),
        DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=n_cpu),
        DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)
    )


def process(res_dir, data_dir, arch):
    logging.info(f"Architecture: {arch}")
    logging.info("1.- Dataset configuration")
    classes = ['oil spill', 'look-alike', 'ship', 'land']
    train_dataset, valid_dataset, test_dataset = create_datasets(data_dir, classes)
    logging.info(f"\tTrain dataset size: {len(train_dataset)}, "
                 f"valid dataset size: {len(valid_dataset)}, "
                 f"test dataset size: {len(test_dataset)}")

    train_dataloader, valid_dataloader, test_dataloader = \
        create_dataloaders(os.cpu_count(), train_dataset, valid_dataset, test_dataset)

    figures_dir = f"{arch}_figures"
    results_dir = f"{arch}_results"
    logs_dir = f"{arch}_logs"
    os.makedirs(os.path.join(res_dir, figures_dir), exist_ok=True)
    os.makedirs(os.path.join(res_dir, results_dir), exist_ok=True)

    # Samples
    save_figure(train_dataset, "Train", os.path.join(res_dir, figures_dir, "figure_01.png"))
    save_figure(valid_dataset, "Val", os.path.join(res_dir, figures_dir, "figure_02.png"))
    save_figure(test_dataset, "Test", os.path.join(res_dir, figures_dir, "figure_03.png"))

    logging.info("2.- Model instantiation")
    model = OilModel(arch, "resnet34", in_channels=3, out_classes=len(classes))

    logging.info("3.- Training")
    logger = CSVLogger(os.path.join(res_dir, logs_dir), name="my_exp_name")
    trainer = pl.Trainer(gpus=1, max_epochs=100, logger=logger)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    # run validation dataset
    logging.info("4.- Validation metrics")
    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    logging.info(f"\tMetrics: {valid_metrics}")

    # run test dataset
    logging.info("5.- Test metrics")
    test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
    logging.info(f"\tMetrics: {test_metrics}")

    logging.info("6.- Result visualization")
    batch = next(iter(test_dataloader))
    test_model(model, batch, os.path.join(res_dir, results_dir))


def main(arch, res_dir, data_dir):
    process(res_dir, data_dir, arch)


parser = argparse.ArgumentParser(
    prog='Oil spill multiclass segmentation',
    description='Multiclass segmentation on Oil Spill Dataset',
    epilog='With a great power comes a great responsability'
)
parser.add_argument('architecture')
args = parser.parse_args()
arch = args.architecture
#for arch in ['unet', 'unetplusplus', 'manet', 'linknet', 'fpn', 'pspnet', 'deeplabv3', 'deeplabv3plus', 'pan']:
logging.basicConfig(filename=f"{arch}_app.log", filemode='w', format='%(asctime)s: %(name)s %(levelname)s - %(message)s', level=logging.INFO)

# redirect lightning logging to file
logger = logging.getLogger("lightning.pytorch")
logger.addHandler(logging.FileHandler("core.log"))

logging.info("Start!")
main(arch, "results", "/home/est_posgrado_manuel.suarez/data/oil-spill-dataset_256")
logging.info('Done!')
