import matplotlib.pyplot as plt
import logging

def save_figure(dataset, name, figname):
    sample = dataset[5]
    plt.subplot(1, 2, 1)
    plt.imshow(sample["image"].transpose(1, 2, 0))  # for visualization we have to transpose back to HWC
    plt.subplot(1, 2, 2)
    plt.imshow(sample["mask"].squeeze())  # for visualization we have to remove 3rd dimension of mask
    plt.savefig(os.path.join(figures_dir, figname))
    logging.info(f"{name} image shape: {sample['image'].shape}")
