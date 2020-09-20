import argparse
import os
import h5py
from torchvision import transforms
from utilities import CXRDataset

#########################
# GENERATE HDF5 DATASET #
# ########################

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=256)
parser.add_argument('--CXR_dataset', type=str, default='CheXpert')
args = parser.parse_args()

IMG_DIR = f'/home/aisinai/data/{args.CXR_dataset}'
DATA_DIR = f'/home/aisinai/data/{args.CXR_dataset}'
HDF5_DIR = '/home/aisinai/work/HDF5_datasets'
os.makedirs(HDF5_DIR, exist_ok=True)

num_label = 14
nc = 3  # number of channels; 3 for RGB, 1 for grayscale
mean = [0.485, 0.456, 0.406]  # ImageNet mean
std = [0.229, 0.224, 0.225]  # ImageNet std
normalization = transforms.Normalize(mean=mean, std=std)
transform_array = [transforms.Resize(args.img_size),
                   transforms.CenterCrop(args.crop_size),
                   transforms.ToTensor(),
                   normalization]

# Generate HDF5 dataset
for mode in ['valid', 'train']:
    # for view in ['frontal', 'lateral']:
    for view in ['frontal']:
        image_list_file = f'{DATA_DIR}/{mode}.csv'
        dataset = CXRDataset(dataset=args.CXR_dataset,
                             img_dir=IMG_DIR,
                             image_list_file=image_list_file,
                             img_size=args.img_size,
                             num_label=num_label,
                             view=view,
                             transform=transforms.Compose(transform_array))
        num_images = len(dataset)  # total number of images in train set
        shape = (num_images, nc, args.crop_size, args.crop_size)
        hdf5 = h5py.File(f'{HDF5_DIR}/{args.CXR_dataset}_{mode}_{args.crop_size}_{view}_normalized.hdf5', 'w')
        hdf5.create_dataset('img', shape)
        hdf5.create_dataset('labels', (num_images, num_label))

        for i in range(num_images):
            img, label = dataset[i]
            hdf5['img'][i, ...] = img
            hdf5['labels'][i] = label
            if (i + 1) % 100 == 0:
                print(f'{args.CXR_dataset}_{mode}_{args.crop_size}_{view}.hdf5:'
                      f'{i + 1}/{num_images} images completed')
            elif i + 1 == num_images:
                print(f'{args.CXR_dataset}_{mode}_{args.crop_size}_{view}.hdf5:'
                      f'{i + 1}/{num_images} images completed')
