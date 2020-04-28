import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from networks import VQVAE
from utilities import ChestXrayHDF5, recon_image, save_loss_plots

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=256)
parser.add_argument('--n_epochs', type=int, default=5600)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--first_stride', type=int, default=4, help="2, 4, 8, or 16")
parser.add_argument('--second_stride', type=int, default=2, help="2, 4, 8, or 16")
parser.add_argument('--embed_dim', type=int, default=64)
parser.add_argument('--data_path', type=str, default='/home/aisinai/work/HDF5_datasets')
parser.add_argument('--dataset', type=str, default='CheXpert', help="CheXpert or mimic")
parser.add_argument('--view', type=str, default='frontal', help="frontal or lateral")
parser.add_argument('--save_path', type=str, default='/home/aisinai/work/VQ-VAE2/20200422/vq_vae')
parser.add_argument('--train_run', type=str, default='0')
args = parser.parse_args()
torch.manual_seed(816)

save_path = f'{args.save_path}/{args.dataset}/{args.train_run}'
os.makedirs(save_path, exist_ok=True)
os.makedirs(f'{save_path}/checkpoint/', exist_ok=True)
os.makedirs(f'{save_path}/sample/', exist_ok=True)
with open(f'{save_path}/args.txt', 'w') as f:
    for key in vars(args).keys():
        f.write(f'{key}: {vars(args)[key]}\n')
        print(f'{key}: {vars(args)[key]}')

dataloaders = {}
dataloaders['train'] = DataLoader(ChestXrayHDF5(f'{args.data_path}/{args.dataset}_train_{args.size}_{args.view}.hdf5'),
                                  batch_size=128,
                                  shuffle=True,
                                  drop_last=True)
dataloaders['valid'] = DataLoader(ChestXrayHDF5(f'{args.data_path}/{args.dataset}_valid_{args.size}_{args.view}.hdf5'),
                                  batch_size=128,
                                  shuffle=True,
                                  drop_last=True)
for i, (img, targets) in enumerate(dataloaders['valid']):
    sample_img = Variable(img.type(Tensor))
    break


if cuda:
    model = VQVAE(first_stride=args.first_stride, second_stride=args.second_stride, embed_dim=args.embed_dim).cuda()
else:
    model = VQVAE(first_stride=args.first_stride, second_stride=args.second_stride, embed_dim=args.embed_dim)
n_gpu = torch.cuda.device_count()
if n_gpu > 1:
    device_ids = list(range(n_gpu))
    model = nn.DataParallel(model, device_ids=device_ids)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

losses = np.zeros((2, args.n_epochs, 3))  # [0,:,:] index for train, [1,:,:] index for valid

for epoch in range(args.n_epochs):
    for phase in ['train', 'valid']:
        model.train(phase == 'train')  # True when 'train', False when 'valid'
        criterion = nn.MSELoss()

        latent_loss_weight = 0.25
        n_row = 5
        loader = tqdm(dataloaders[phase])
        for i, (img, label) in enumerate(loader):
            img = Variable(img.type(Tensor))
            with torch.set_grad_enabled(phase == 'train'):
                optimizer.zero_grad()
                out, latent_loss = model(img)
                recon_loss = criterion(out, img)
                latent_loss = latent_loss.mean()
                loss = recon_loss + latent_loss_weight * latent_loss
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    losses[0, epoch, :] = [loss, recon_loss, latent_loss]
                else:
                    losses[1, epoch, :] = [loss, recon_loss, latent_loss]
                lr = optimizer.param_groups[0]['lr']

            loader.set_description((f'phase: {phase}; epoch: {epoch + 1}; total_loss: {loss.item():.5f}; '
                                    f'latent: {latent_loss.item():.5f}; mse: {recon_loss.item():.5f}; '
                                    f'lr: {lr:.5f}'))

            if i % 10 == 0:
                recon_image(n_row, sample_img, model, f'{save_path}', epoch, Tensor)

        save_loss_plots(args.n_epochs, epoch, losses, f'{save_path}')
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'{save_path}/checkpoint/vqvae_{str(epoch + 1).zfill(3)}.pt')
