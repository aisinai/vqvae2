import argparse
import os
import h5py
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from networks import VQVAE
from utilities import ChestXrayHDF5

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--first_stride', type=int, default=4)
parser.add_argument('--second_stride', type=int, default=2)
parser.add_argument('--embed_dim', type=int, default=64)
parser.add_argument('--data_path', type=str, default='/home/aisinai/work/HDF5_datasets')
parser.add_argument('--dataset', type=str, default='mimic')
parser.add_argument('--view', type=str, default='frontal')
parser.add_argument('--save_path', type=str, default='/home/aisinai/work/HDF5_datasets/recon_latent')
parser.add_argument('--train_run', type=str, default='0')
parser.add_argument('--vqvae_file', type=str, default='vqvae_040.pt')
args = parser.parse_args()
print(args)
torch.manual_seed(816)

os.makedirs(args.save_path, exist_ok=True)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Load model
vqvae_path = f'/home/aisinai/work/VQ-VAE2/20200422/vq_vae/CheXpert/{args.train_run}/checkpoint/{args.vqvae_file}'
if cuda:
    vqvae_pretrain = VQVAE(first_stride=args.first_stride, second_stride=args.second_stride,
                           embed_dim=args.embed_dim).cuda()
else:
    vqvae_pretrain = VQVAE(first_stride=args.first_stride, second_stride=args.second_stride, embed_dim=args.embed_dim)
vqvae_pretrain.load_state_dict(torch.load(vqvae_path))
n_gpu = torch.cuda.device_count()
if n_gpu > 1:
    device_ids = list(range(n_gpu))
    vqvae_pretrain = nn.DataParallel(vqvae_pretrain, device_ids=device_ids)
vqvae_pretrain.eval()

# Define dataset
dataset = {}
dataset['train'] = ChestXrayHDF5(f'{args.data_path}/{args.dataset}_train_{args.size}_frontal.hdf5')
dataset['valid'] = ChestXrayHDF5(f'{args.data_path}/{args.dataset}_valid_{args.size}_frontal.hdf5')
dataloaders = {}
dataloaders['train'] = DataLoader(dataset['train'], batch_size=8, shuffle=False, drop_last=False)
dataloaders['valid'] = DataLoader(dataset['valid'], batch_size=8, shuffle=False, drop_last=False)

for phase in ['train', 'valid']:
    # Create HDF5
    hdf5_recon = h5py.File(f'{args.save_path}/{args.dataset}_{phase}_{args.size}_recon.hdf5', 'w')
    hdf5_recon.create_dataset('img', (len(dataset[phase]), 3, args.size, args.size))
    hdf5_recon.create_dataset('labels', (len(dataset[phase]), 14))
    hdf5_latent = h5py.File(f'{args.save_path}/{args.dataset}_{phase}_{args.size}_latent.hdf5', 'w')
    hdf5_latent.create_dataset('img', (len(dataset[phase]), 2, 64, 64))
    hdf5_latent.create_dataset('labels', (len(dataset[phase]), 14))
    loader = tqdm(dataloaders[phase])
    for i, (img, targets) in enumerate(loader):
        loader.set_description(f'phase: {phase}')
        real_img = Variable(img.type(Tensor))
        decoded_img, _ = vqvae_pretrain(real_img)
        quant_t, quant_b, _, id_t, id_b = vqvae_pretrain.encode(real_img)
        upsample_t = vqvae_pretrain.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        for j in range(img.shape[0]):
            hdf5_recon['img'][i * args.batch_size + j, :] = decoded_img.cpu().detach().numpy()[j, :]
            hdf5_recon['labels'][i * args.batch_size + j] = targets.cpu().detach().numpy()[j]
            hdf5_latent['img'][i * args.batch_size + j, :] = quant.cpu().detach().numpy()[j, :]
            hdf5_latent['labels'][i * args.batch_size + j] = targets.cpu().detach().numpy()[j]
