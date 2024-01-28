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
data_dir = '/home/ids/ext-6398/Competition/data_challenge/train'

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
# Define loss function and optimizer
criterion = nn.MSELoss()  # Example loss function, replace with appropriate loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Example optimizer, adjust learning rate as needed

# Assuming you have a DataLoader named 'dataloader' as defined earlier
for epoch in range(N_EPOCHS):
    running_loss = 0.0

        # Iterate over the data loader
    for lung_segmentations, tumor_masks, scan_volumes in dataloader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Assuming data shape is [batch_size, depth, height, width]
        lung_segmentations = lung_segmentations.unsqueeze(0).unsqueeze(0).to(device)  # Add channel dimension
        tumor_masks = tumor_masks.unsqueeze(0).unsqueeze(0).to(device)
        scan_volumes = scan_volumes.unsqueeze(0).unsqueeze(0).to(device)

        # Forward pass
        outputs = model(scan_volumes)

        # Compute the loss
        loss = criterion(outputs, tumor_masks)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()


    # Print epoch statistics
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')

print('Finished Training')

torch.save(model.state_dict(), "/home/ids/ext-6398/Competition/IPPMed/3D-UNet/model_weights")
