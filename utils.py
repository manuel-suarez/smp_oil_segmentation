import matplotlib.pyplot as plt
import logging
import os

figures_dir = "figures"

def save_figure(dataset, name, figname):
    sample = dataset[5]

    plt.subplot(1, 2, 1)
    image = sample["image"]
    # logging.info(f"Image shape: {image.shape}")
    image = image.transpose(1, 2, 0)
    # logging.info(f"Image transpose shape: {image.shape}")
    plt.imshow(image)  # for visualization we have to transpose back to HWC
    plt.subplot(1, 2, 2)
    mask = sample["mask"]
    # logging.info(f"Mask shape: {mask.shape}")
    mask = mask.squeeze()
    # logging.info(f"Mask squeeze shape: {mask.shape}")
    # Display first mask only
    plt.imshow(mask[0])
    plt.savefig(figname)
    logging.info(f"{name} image shape: {sample['image'].shape}")
