import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models
from tqdm import tqdm
from networks import VQVAE
from utilities import ChestXrayHDF5, compute_AUCs, save_loss_AUROC_plots

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=256)
parser.add_argument('--n_epochs', type=int, default=5600)
parser.add_argument('--n_classes', type=int, default=15)
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--weight_decay", type=float, default=0.1e-5, help="adam: weight decay (L2 penalty)")
parser.add_argument('--first_stride', type=int, default=4)
parser.add_argument('--second_stride', type=int, default=2)
parser.add_argument('--data_path', type=str, default='/home/aisinai/work/HDF5_datasets/uncropped')
parser.add_argument('--dataset', type=str, default='mimic')
parser.add_argument('--view', type=str, default='frontal')
parser.add_argument('--save_path', type=str, default='/home/aisinai/work/VQ-VAE2/20200317/densenet121')
parser.add_argument('--train_ID', type=str, default='0')
parser.add_argument('--vqvae_file', type=str, default='vqvae_031.pt')
args = parser.parse_args()
print(args)
torch.manual_seed(816)

save_path = f'{args.save_path}/{args.dataset}/{args.train_ID}'
os.makedirs(save_path, exist_ok=True)

CLASS_NAMES = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
               'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
               'Pleural Other', 'Fracture', 'Support Devices', 'No Positive Labels']

dataloaders = {}
dataloaders['train'] = DataLoader(ChestXrayHDF5(f'{args.data_path}/{args.dataset}_train_256_norm_True_{args.view}.hdf5'),
                                  batch_size=8,
                                  shuffle=True,
                                  drop_last=True)
dataloaders['valid'] = DataLoader(ChestXrayHDF5(f'{args.data_path}/{args.dataset}_valid_256_norm_True_{args.view}.hdf5'),
                                  batch_size=8,
                                  shuffle=True,
                                  drop_last=True)

if cuda:
    vqvae_pretrain = VQVAE(first_stride=args.first_stride, second_stride=args.second_stride).cuda()
else:
    vqvae_pretrain = VQVAE(first_stride=args.first_stride, second_stride=args.second_stride)

vqvae_path = f'/home/aisinai/work/VQ-VAE2/20200311/vq_vae/{args.train_ID}/checkpoint/{args.vqvae_file}'
vqvae_pretrain.load_state_dict(torch.load(vqvae_path))

model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(nn.Linear(num_ftrs, args.n_classes), nn.Sigmoid())
model = model.cuda() if cuda else model

n_gpu = torch.cuda.device_count()
if n_gpu > 1:
    device_ids = list(range(n_gpu))
    vqvae_pretrain = nn.DataParallel(vqvae_pretrain, device_ids=device_ids)
    model = nn.DataParallel(model, device_ids=device_ids)

lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(args.b1, args.b2), weight_decay=args.weight_decay)
criterion = nn.BCELoss()
# [0,:,:] index for train, [1,:,:] index for val
losses = np.zeros((2, args.n_epochs))
aurocs = np.zeros((2, args.n_epochs, args.n_classes))
best_loss = 999999

for epoch in range(args.n_epochs):
    for phase in ['train', 'valid']:
        gt = torch.FloatTensor().cuda() if cuda else torch.FloatTensor()
        pred = torch.FloatTensor().cuda() if cuda else torch.FloatTensor()
        model.train(phase == 'train')  # True when 'train', False when 'valid'
        loader = tqdm(dataloaders[phase])
        for i, (img, label) in enumerate(loader):
            real_img = Variable(img.type(Tensor))
            real_targets = Variable(label.cuda()) if cuda else Variable(label)
            decoded_img, _ = vqvae_pretrain(real_img)
            with torch.set_grad_enabled(phase == 'train'):
                output_C = model(decoded_img)
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
        auroc = compute_AUCs(gt, pred, args.n_classes)
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
                                             betas=(args.b1, args.b2),
                                             weight_decay=args.weight_decay)

            if c_loss < best_loss:
                best_loss = c_loss
                torch.save(model.state_dict(),
                           f'{save_path}/best_densenet_model_epoch_{str(epoch + 1).zfill(3)}.pt')
        print('-' * 10)
        print(f'{phase}: [Epoch {epoch + 1} / {args.n_epochs}]')
        print(f'[Classifier loss: {c_loss.item():.4f}; Classifier avg AUROC = {auroc_avg:.4f}%]')
        for i in range(args.n_classes):
            print(f'The AUROC of {CLASS_NAMES[i]} is {auroc[i]:.4f}')
        save_loss_AUROC_plots(args.n_epochs, epoch, losses, aurocs, save_path)
    torch.save(model.state_dict(),
               f'{save_path}/densenet_{str(epoch + 1).zfill(3)}.pt')
    torch.save(losses, f'{save_path}/losses.pt')
    torch.save(aurocs, f'{save_path}/aurocs.pt')
