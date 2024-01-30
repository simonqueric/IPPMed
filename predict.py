import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import shutil
from model import conv_block, encoder_block, decoder_block, UNet

if torch.cuda.is_available():
    print("cuda available")    
    device = torch.device("cuda")
else:
    print("not available")
    device = torch.device("cpu")

model = UNet()
model = model.to(device)
model.load_state_dict(torch.load("checkpoints/checkpoint_unet2d_relu_lr=1e-4_rescale_t=0.5_adam_50_epochs_batch=16_reduce_scheduler.pth", map_location=device))


# Path to the directory containing the data
test = "../data_challenge/test/volume/"
files = os.listdir(test)

output_dir = "./Predictions"

for i, file in enumerate(files) : 
    img = nib.load(test+file)
    img_v = img.get_fdata()
    x, y, z = img_v.shape
    prediction = np.zeros((x, y, z))
    img_vol = torch.from_numpy(img_v).to(torch.float32).unsqueeze(0).to(device)
    for idx in range(z):
        x = img_vol.unsqueeze(0)[:,:,:,:,idx]
        pred = model(x)
        prediction[:,:,idx] = (torch.sigmoid(pred).squeeze(0, 1).cpu().detach().numpy()>.5)
    #convert prediction to nib image
    prediction = nib.Nifti1Image(prediction, img.affine)
    #save file as .nii.gz file
    filename = f"LUNG1-{str(i+1).zfill(3)}.nii.gz"
    filepath = os.path.join(output_dir, filename)
    nib.save(prediction, filepath)

    

