import argparse
import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from networks import VQVAE
from utilities import ChestXrayHDF5, rgb2gray

save_path = '/home/aisinai/work/VQ-VAE2-Images'
os.makedirs(save_path, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default='/home/aisinai/work/VQ-VAE2-Images')
parser.add_argument('--data_path', type=str, default='/home/aisinai/work/HDF5_datasets')
parser.add_argument('--dataset', type=str, default='mimic')
parser.add_argument('--size', type=int, default=1024)
parser.add_argument('--view', type=str, default='frontal')
parser.add_argument('--model_name', type=str, default='A')
args = parser.parse_args()

# Define dataset
if args.model_name == 'D':
    dataset = ChestXrayHDF5('/home/aisinai/work/HDF5_datasets/recon_latent/mimic_valid_256_orig.hdf5')
else:
    dataset = ChestXrayHDF5(f'{args.data_path}/{args.dataset}_valid_{args.size}_{args.view}_normalized.hdf5')
batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ImageNet mean and std
mean = torch.FloatTensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).type(Tensor)
std = torch.FloatTensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).type(Tensor)

save_orig_path = f'{args.save_path}/{args.size}/{args.view}/original'
save_recon_path = f'{args.save_path}/{args.size}/{args.view}/{args.model_name}'
os.makedirs(save_orig_path, exist_ok=True)
os.makedirs(save_recon_path, exist_ok=True)
if args.model_name == 'A':
    model_dir = '/home/aisinai/work/VQ-VAE2/20200422/vq_vae/CheXpert/0/checkpoint/vqvae_040.pt'
    model = VQVAE(first_stride=4, second_stride=2).cuda() if cuda else VQVAE()
elif args.model_name == 'B':
    model_dir = '/home/aisinai/work/VQ-VAE2/20200422/vq_vae/CheXpert/1/checkpoint/vqvae_040.pt'
    model = VQVAE(first_stride=8, second_stride=4).cuda() if cuda else VQVAE()
elif args.model_name == 'C':
    model_dir = '/home/aisinai/work/VQ-VAE2/20200422/vq_vae/CheXpert/3/checkpoint/vqvae_040.pt'
    model = VQVAE(first_stride=16, second_stride=4).cuda() if cuda else VQVAE()
elif args.model_name == 'D':
    model_dir = '/home/aisinai/work/VQ-VAE2/20200422/vq_vae/CheXpert/embed1/checkpoint/vqvae_040.pt'
    model = VQVAE(first_stride=4, second_stride=2, embed_dim=1).cuda() if cuda else VQVAE()

model.load_state_dict(torch.load(model_dir))
n_gpu = torch.cuda.device_count()
if n_gpu > 1:
    device_ids = list(range(n_gpu))
    model = nn.DataParallel(model, device_ids=device_ids)
model.eval()

loader = tqdm(dataloader)
for i, (img, targets) in enumerate(loader):
    loader.set_description(f'Model {args.model_name}:')
    real_img = Variable(img.type(Tensor))
    decoded_img, _ = model(real_img)
    real_img = real_img * std + mean
    decoded_img = decoded_img * std + mean
    for j in range(img.shape[0]):
        save_image(real_img[j, :].data,
                   f'{save_orig_path}/{str(i * batch_size + j).zfill(4)}.png',
                   nrow=1, normalize=True, range=(0, 1))
        save_image(decoded_img[j, :].data,
                   f'{save_recon_path}/{str(i * batch_size + j).zfill(4)}.png',
                   nrow=1, normalize=True, range=(0, 1))
