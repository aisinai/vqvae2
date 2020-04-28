import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from networks import Densenet121
from utilities import ChestXrayHDF5, compute_AUCs, save_loss_AUROC_plots
from pytorch_memlab import profile

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=256)
parser.add_argument('--n_epochs', type=int, default=5600)
parser.add_argument('--n_classes', type=int, default=14)
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--weight_decay", type=float, default=0.1e-5, help="adam: weight decay (L2 penalty)")
parser.add_argument('--first_stride', type=int, default=4)
parser.add_argument('--second_stride', type=int, default=2)
parser.add_argument('--embed_dim', type=int, default=1)
parser.add_argument('--data_path', type=str, default='/home/aisinai/work/HDF5_datasets/recon_latent')
parser.add_argument('--dataset', type=str, default='mimic')
parser.add_argument('--view', type=str, default='frontal')
parser.add_argument('--save_path', type=str, default='/home/aisinai/work/VQ-VAE2/20200427/densenet121')
parser.add_argument('--vqvae_file', type=str, default='vqvae_040.pt')
parser.add_argument('--recon', type=str, default='latent', help="input type; 'orig', 'recon', or 'latent'")
args = parser.parse_args()
print(args)
torch.manual_seed(816)


CLASS_NAMES = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
               'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
               'Pleural Other', 'Fracture', 'Support Devices']

model = Densenet121(n_classes=args.n_classes, input_type=args.recon)
save_path = f'{args.save_path}/{args.recon}'
os.makedirs(save_path, exist_ok=True)
model = model.cuda() if cuda else model
n_gpu = torch.cuda.device_count()
if n_gpu > 1:
    device_ids = list(range(n_gpu))
    model = nn.DataParallel(model, device_ids=device_ids)

dataloaders = {}
dataloaders['train'] = DataLoader(ChestXrayHDF5(f'{args.data_path}/{args.dataset}_train_{args.size}_{args.recon}.hdf5'),
                                  batch_size=64,
                                  shuffle=True,
                                  drop_last=True)
dataloaders['valid'] = DataLoader(ChestXrayHDF5(f'{args.data_path}/{args.dataset}_valid_{args.size}_{args.recon}.hdf5'),
                                  batch_size=64,
                                  shuffle=True,
                                  drop_last=True)

lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(args.b1, args.b2), weight_decay=args.weight_decay)
criterion = nn.BCELoss()


@profile
def train(dataloaders, optimizer, criterion, model, n_epochs, n_classes, lr, b1, b2, weight_decay, save_path):
    losses = np.zeros((2, n_epochs))  # [0,:] for train, [1,:] for val
    aurocs = np.zeros((2, n_epochs, n_classes))  # [0,:] for train, [1,:] for val
    best_loss = 999999
    for epoch in range(n_epochs):
        for phase in ['train', 'valid']:
            gt = torch.FloatTensor().cuda() if cuda else torch.FloatTensor()
            pred = torch.FloatTensor().cuda() if cuda else torch.FloatTensor()
            model.train(phase == 'train')  # True when 'train', False when 'valid'
            loader = tqdm(dataloaders[phase])
            for i, (img, label) in enumerate(loader):
                real_img = Variable(img.type(Tensor))
                real_targets = Variable(label.cuda()) if cuda else Variable(label)
                with torch.set_grad_enabled(phase == 'train'):
                    output_C = model(real_img)
                    gt = torch.cat((gt, real_targets.cuda()), 0)
                    pred = torch.cat((pred, output_C.cuda()), 0)
                    # calculate gradient and update parameters in train phase
                    optimizer.zero_grad()
                    c_loss = criterion(output_C, real_targets)
                    if phase == 'train':
                        c_loss.backward()
                        optimizer.step()
                loader.set_description((f'phase: {phase}; epoch: {epoch + 1};'
                                        f'total_loss: {c_loss.item():.5f}; lr: {lr:.5f}'))

            auroc = compute_AUCs(gt, pred, n_classes)
            auroc_avg = np.array(auroc).mean()
            if phase == 'train':
                losses[0, epoch] = c_loss
                aurocs[0, epoch, :] = auroc
            elif phase == 'valid':
                losses[1, epoch] = c_loss
                aurocs[1, epoch, :] = auroc
                if c_loss > best_loss:
                    print(f'decay lr from {lr} to {lr/10} as not seeing improvement in val loss')
                    lr = lr / 10
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                                 betas=(b1, b2),
                                                 weight_decay=weight_decay)

                if c_loss < best_loss:
                    best_loss = c_loss
                    torch.save(model.state_dict(),
                               f'{save_path}/best_densenet_model_epoch_{str(epoch + 1).zfill(3)}.pt')
            print('-' * 10)
            print(f'{phase}: [Epoch {epoch + 1} / {n_epochs}]')
            print(f'[Classifier loss: {c_loss.item():.4f}; Classifier avg AUROC = {auroc_avg:.4f}%]')
            for i in range(n_classes):
                print(f'The AUROC of {CLASS_NAMES[i]} is {auroc[i]:.4f}')
            save_loss_AUROC_plots(n_epochs, epoch, losses, aurocs, save_path)
            print('END')
        torch.save(model.state_dict(), f'{save_path}/densenet_{str(epoch + 1).zfill(3)}.pt')
        torch.save(losses, f'{save_path}/losses.pt')
        torch.save(aurocs, f'{save_path}/aurocs.pt')


train(dataloaders, optimizer, criterion, model, 1,  # run only 1 epoch
      args.n_classes, args.lr, args.b1, args.b2, args.weight_decay, save_path)
