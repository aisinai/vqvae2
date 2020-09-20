import os
import h5py
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
from sklearn.metrics.ranking import roc_auc_score


class ChestXrayHDF5(Dataset):
    def __init__(self, path):
        hdf5_database = h5py.File(path, 'r')
        self.path = path
        self.hdf5_database = hdf5_database

    def __getitem__(self, index):
        hdf5_image = self.hdf5_database["img"][index, ...]  # read image
        image = torch.from_numpy(hdf5_image)
        # returns numpy.ndarray
        hdf5_label = self.hdf5_database["labels"][index, ...]
        label = [int(i) for i in hdf5_label]
        return image, torch.FloatTensor(label)

    def __len__(self):
        return self.hdf5_database["img"].shape[0]


class CXRDataset(Dataset):
    def __init__(self, dataset, img_dir, image_list_file, img_size, num_label, view, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            mode: 'frontal' or 'lateral'
            transform: optional transform to be applied on a sample.
        """
        view = view.lower()
        image_names = []
        labels = []

        f = open(image_list_file, 'r')
        d = '\t'
        reader = csv.reader(f, delimiter=d)
        next(reader, None)
        for row in reader:
            if dataset == 'CheXpert':
                items = row[0].split(',')
                image_name = os.path.join(img_dir, items[0][20:])
                img_view = items[3].lower()
                label = items[5:]
            elif dataset == 'mimic':
                items = row[0].split(',')
                image_name = f'{img_dir}/{items[0]}'
                img_view = items[1].lower()
                label = items[2:]
            indices = [i for i, x in enumerate(label) if x == "1.0"]
            output = np.zeros(num_label)
            if img_view == view:
                for index in indices:
                    output[index] = 1
                    output[index] = int(output[index])
                labels.append(output)
                image_names.append(image_name)

        self.image_names = image_names
        self.labels = labels
        self.img_size = img_size
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its label
        """
        image_name = self.image_names[index]
        im = Image.open(image_name).convert('RGB')
        old_size = im.size
        ratio = float(self.img_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im = im.resize(new_size, Image.ANTIALIAS)
        # create pads
        image = Image.new("RGB", (self.img_size, self.img_size))
        # paste resized image in the middle of padding
        image.paste(im, ((self.img_size - new_size[0]) // 2,
                         (self.img_size - new_size[1]) // 2))
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_names)


def recon_image(n_row, original_img, model, save_path, epoch, Tensor):
    """Saves a grid of decoded / reconstructed digits."""
    model.eval()
    original_img = original_img[0:n_row**2, :]
    with torch.no_grad():
        out, _ = model(original_img)

    # remove normalization
    mean = torch.FloatTensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).type(Tensor)
    std = torch.FloatTensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).type(Tensor)
    original_img = original_img * std + mean
    out = out * std + mean

    save_image(original_img.data, f'{save_path}/sample/original.png',
               nrow=n_row, normalize=True, range=(0, 1))
    save_image(out.data, f'{save_path}/sample/{str(epoch + 1).zfill(4)}.png',
               nrow=n_row, normalize=True, range=(0, 1))
    save_image(torch.cat([original_img, out], 0).data, f'{save_path}/sample/flat_{str(epoch + 1).zfill(4)}.png',
               nrow=n_row**2, normalize=True, range=(0, 1))
    model.train()


def compute_AUCs(gt, pred, N_CLASSES):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return np.asarray(AUROCs)


def save_loss_plots(n_epochs, latest_epoch, losses, save_path):
    epochs = range(1, n_epochs + 1)
    fig = plt.figure(figsize=(15, 15))
    latest_epoch += 1
    # BCE Loss
    ax1 = fig.add_subplot(111)
    ax1.plot(epochs, losses[0, :, 0], '-')
    ax1.plot(epochs, losses[1, :, 0], '-')
    ax1.set_title('All Losses')
    ax1.set_xlabel('Epochs')
    ax1.axis(xmin=1, xmax=latest_epoch)
    ax1.legend(["Train Loss", "Validation Loss"], loc="upper right")

    plt.close(fig)
    fig.savefig(f'{save_path}/loss graphs.png')


def save_loss_AUROC_plots(n_epochs, latest_epoch, losses, AUROCs, save_path):
    epochs = range(1, n_epochs + 1)
    fig = plt.figure(figsize=(15, 15))
    latest_epoch += 1
    # BCE Loss
    ax1 = fig.add_subplot(121)
    ax1.plot(epochs, losses[0, :], '-')
    ax1.plot(epochs, losses[1, :], '-')
    ax1.legend(["Train", "Val"], loc="upper right")
    ax1.set_title('BCE Loss')
    ax1.set_xlabel('Epochs')
    ax1.axis(xmin=1, xmax=latest_epoch)

    ax2 = fig.add_subplot(122)
    ax2.plot(epochs, np.mean(AUROCs[0, :], axis=1), '-')
    ax2.plot(epochs, np.mean(AUROCs[1, :], axis=1), '-')
    ax2.legend(["Train", "Val"], loc="upper right")
    ax2.set_title('Average AUROC')
    ax2.set_xlabel('Epochs')
    ax2.axis(xmin=0, xmax=latest_epoch)

    plt.close(fig)
    fig.savefig(f'{save_path}/loss and AUROC graphs.png')


def rgb2gray(original_img):
    r, g, b = original_img[0, :], original_img[1, :], original_img[2, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
