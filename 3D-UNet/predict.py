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
data_dir = '/home/infres/ext-6398/data_challenge/test'

dataloader = CustomDataset(data_dir, predict=True)

IN_CHANNELS = 1
NUM_CLASSES = 1

model = UNet3D(in_channels = IN_CHANNELS, num_classes = NUM_CLASSES)
model_path = '/home/infres/ext-6398/3D-UNet/model_weights'

checkpoint = torch.load(model_path)

# Load the state dictionary of the model parameter by parameter
model_state_dict = model.state_dict()
for key in checkpoint.keys():
    if key in model_state_dict:
        model_state_dict[key] = checkpoint[key]

# Load the modified state dictionary into the model
model.load_state_dict(model_state_dict)

model.eval()

output_dir = "/home/infres/ext-6398/3D-UNet/Predictions"
os.makedirs(output_dir, exist_ok=True)

if torch.cuda.is_available():
    print("cuda available")
    model = model.cuda()
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda
    device = "cuda"
elif not torch.cuda.is_available():
    print('cuda not available! Training initialized on cpu ...')
    device = "cpu"

for i, scan_volumes in enumerate(dataloader):
    # Assuming data shape is [batch_size, depth, height, width]
    w = scan_volumes.shape[2]
    size = 2 
    while(2*size <= w and size < 64):
        size = 2*size
        
    # Define the size of the subcube
    m = 200 

    # Calculate starting and ending indices for extraction
    start_idx = (512 - m) // 2
    end_idx = start_idx + m 

    scan_volumes = scan_volumes[start_idx:end_idx, start_idx:end_idx, :size].unsqueeze(0).unsqueeze(0).to(device)

    # Make predictions
    pred = torch.zeros((512, 512, w)).to(device)
    pred[start_idx:end_idx, start_idx:end_idx, :size] = model(scan_volumes)

    # Convert tensor to numpy array
    prediction_array = pred.cpu().detach().numpy()

    # Create NIfTI image
    nii_img = nib.Nifti1Image(prediction_array, np.eye(4))

    # Save the NIfTI image with a filename based on the index
    filename = f"LUNG1-{str(i+1).zfill(3)}.nii.gz"
    filepath = os.path.join(output_dir, filename)
    nib.save(nii_img, filepath)

    

