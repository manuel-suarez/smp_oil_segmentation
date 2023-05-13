import matplotlib.pyplot as plt
import logging
import torch
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

def test_model(model, batch, results_dir):
    with torch.no_grad():
        model.eval()
        logits = model(batch["image"])
    pr_masks = logits.sigmoid()

    for idx, (image, gt_mask, pr_mask) in enumerate(zip(batch["image"], batch["mask"], pr_masks)):
        plt.figure(figsize=(10, 5))

        logging.info(
            f"Test model, image shape: {image.shape}, gt_mask shape: {gt_mask.shape}, prediction mask shape: {pr_mask.shape}")
        logging.info(
            f"Squeeze, image shape: {image.numpy().transpose(1, 2, 0).shape}, "
            f"gt_mask shape: {gt_mask.numpy().transpose(1, 2, 0).shape}, "
            f"prediction mask shape: {pr_mask.numpy().transpose(1, 2, 0).shape}")
        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask.numpy().transpose(1, 2, 0)) # just squeeze classes dim, because we have only one class
        plt.title("Ground truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask.numpy().transpose(1, 2, 0)) # just squeeze classes dim, because we have only one class
        plt.title("Prediction")
        plt.axis("off")

        plt.savefig(os.path.join(results_dir, f"result_{str(idx).zfill(2)}.png"))