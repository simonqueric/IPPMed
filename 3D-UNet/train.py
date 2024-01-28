from unet3d import *
from dataset import *
from transforms import *
import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import nibabel as nib  # Assuming you're working with medical imaging data, you may need nibabel for reading NIfTI files
import torch.nn.functional as F

def dice_loss(predicted, target, smooth=1e-6):
    intersection = torch.sum(predicted * target)
    cardinality = torch.sum(predicted) + torch.sum(target)
    dice = (2. * intersection + smooth) / (cardinality + smooth)
    return 1. - dice

def combined_loss(predicted, target, alpha=0.5, smooth=1e-6):
    bce = F.binary_cross_entropy_with_logits(predicted, target)
    dice = dice_loss(torch.sigmoid(predicted), target, smooth)
    combined = alpha * bce + (1 - alpha) * dice
    return combined
# Path to the directory containing the data
data_dir = '/home/infres/ext-6398/data_challenge/train'

dataloader = CustomDataset(data_dir)

IN_CHANNELS = 1
NUM_CLASSES = 1
N_EPOCHS = 30
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
optimizer = optim.Adam(model.parameters(), lr=0.002)  # Example optimizer, adjust learning rate as needed

# Assuming you have a DataLoader named 'dataloader' as defined earlier
for epoch in range(N_EPOCHS):
    running_loss = 0.0

        # Iterate over the data loader
    for i,(lung_segmentations, tumor_masks, scan_volumes) in enumerate(dataloader):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Assuming data shape is [batch_size, depth, height, width]
        w = np.min([tumor_masks.shape[2], lung_segmentations.shape[2], scan_volumes.shape[2]])
        tm_size = tumor_masks.shape[2]
        size = 2
        while(2*size <= w and size <= 128):
            size = 2*size
        print(size)
        if size<64:
            continue
        # Define the size of the subcube
        m = 320

        # Calculate starting and ending indices for extraction
        start_idx = (512 - m) // 2
        end_idx = start_idx + m
        

        lung_segmentations = lung_segmentations[start_idx:end_idx,start_idx:end_idx,:size].unsqueeze(0).unsqueeze(0).to(device)  # Add channel dimension
        tumor_masks = tumor_masks.to(device)
        scan_volumes = scan_volumes[start_idx:end_idx,start_idx:end_idx,:size].unsqueeze(0).unsqueeze(0).to(device)

        # Forward pass
        output = model(scan_volumes)
        print(output.shape)
        pred = torch.zeros((512, 512, tm_size)).to(device)
        pred[start_idx:end_idx,start_idx:end_idx,:size] = output[0,0,:,:,:]
        # Compute the loss
        loss = combined_loss(pred, tumor_masks)
        print(f"image {i} loss : {loss}")
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()


    # Print epoch statistics
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')
    torch.save(model.state_dict(), "/home/infres/ext-6398/3D-UNet/model_weights")
print('Finished Training')

torch.save(model.state_dict(), "/home/infres/ext-6398/3D-UNet/model_weights")
