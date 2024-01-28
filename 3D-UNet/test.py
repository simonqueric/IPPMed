from unet3d import *
from dataset import *
from transforms import *
import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import nibabel as nib  # Assuming you're working with medical imaging data, you may need nibabel for reading NIfTI files

# Path to the directory containing the data
data_dir = '/home/infres/ext-6398/data_challenge/train'

# Create an empty list to store file paths
file_list = []

# Iterate through the lung segmentation directory
lung_seg_dir = os.path.join(data_dir, 'lungs_seg')
for filename in os.listdir(lung_seg_dir):
    if filename.endswith('.nii.gz'):
        lung_seg_path = os.path.join(lung_seg_dir, filename)
        
        # Extract the corresponding tumor mask and scan volume filenames
        base_filename = filename.split('_lungseg.nii.gz')[0]
        tumor_mask_path = os.path.join(data_dir, 'seg', base_filename + '_seg.nii.gz')
        scan_volume_path = os.path.join(data_dir, 'volume', base_filename + '_vol.nii.gz')

        # Check if tumor mask and scan volume files exist
        if os.path.exists(tumor_mask_path) and os.path.exists(scan_volume_path):
            file_list.append({
                'lung_segmentation': lung_seg_path,
                'tumor_mask': tumor_mask_path,
                'scan_volume': scan_volume_path
            })

dataloader = CustomDataset(data_dir)

IN_CHANNELS = 1
NUM_CLASSES = 1
N_EPOCHS = 1
BATCH_SIZE = 1

model = UNet3D(in_channels = IN_CHANNELS, num_classes = NUM_CLASSES)

if torch.cuda.is_available():
    print("cuda available")
    model = model.cuda()
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda
    device = "cuda"
elif not torch.cuda.is_available():
    print('cuda not available! Training initialized on cpu ...')
    device = "cpu"


lung_segmentations, tumor_masks, scan_volumes =  dataloader[0]
size = 128

# Define the size of the subcube
m = 200

# Calculate starting and ending indices for extraction
start_idx = (512 - m) // 2
end_idx = start_idx + m


print("test forward")
print(scan_volumes.shape)
print(scan_volumes[start_idx:end_idx,start_idx:end_idx,:size].shape)
scan_volumes = scan_volumes[start_idx:end_idx,start_idx:end_idx,:size].unsqueeze(0).unsqueeze(0).to(device)
image = model(scan_volumes)
print(image.shape)
