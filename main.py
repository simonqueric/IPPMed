import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
from unet3d import UNet3D, Conv3DBlock, UpConv3DBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from loss import DiceLoss, DiceBCELoss
from model import conv_block, encoder_block, decoder_block, UNet
from dataset import IPPMedDataset_2D, rescale_image
from time import time
from sklearn.metrics import accuracy_score

#log_dir="logs2/"
#writer=SummaryWriter(log_dir=log_dir)

def train(model, loader, optimizer, loss_fn, device, epochs=1):
    epoch_loss = 0.0
    model.train()
    k=0
    for epoch in range(epochs) :
        epoch_loss = 0.0
        mean_accuracy = 0.0
        start = time() 
        for x, y in loader:
       	    x = x.to(device) #.half()
            y = y.to(device) #.half()
            optimizer.zero_grad()
            y_pred = model(x) #.half()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        #for name, param in model.named_parameters():
            #writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step=epoch)

        #writer.add_scalar("Training loss", epoch_acuracy/len(loader), global_step=epoch)
        if len(losses)>=1 and epoch_loss/len(loader) < min(losses) :
            print("Loss decreased")
            torch.save(model.state_dict(), file_checkpoint)
        losses.append(epoch_loss/len(loader))
        end = time()
        print("epoch ", epoch, "epoch loss :", epoch_loss/len(loader), "running time :", end-start)
    return epoch_loss

if __name__=="__main__" :
 
    if torch.cuda.is_available():
        print("cuda available")    
        device = torch.device("cuda")
    else:
        print("not available")
        device = torch.device("cpu")
    
   
    torch.manual_seed(42) #add seed for the dataloader shuffling    
    batch_size=16
    data_root = "data/train/"
    ippmed_dataset_2D = IPPMedDataset_2D(data_root, transform=rescale_image)
    data_train_loader = DataLoader(dataset=ippmed_dataset_2D, batch_size=batch_size, shuffle=True) 
	
    
    x, y = next(iter(data_train_loader))
    print(x.max())
    file_checkpoint = "checkpoints/checkpoint_unet2d_relu_lr=1e-4_rescale_t=0.5_adam_20_epochs_batch=16_reduce_scheduler.pth"
    print(file_checkpoint)
    #model = UNet3D(1, 1).to(device)    
    #print(model(x.to(device)).shape)
    print("Number of batches :", len(data_train_loader))

    in_channels = 1  # single-channel input
    out_channels = 1  # Number of segmentation classes
    model = UNet().to(device)
    
    
    #for name, param in model.named_parameters():
    #    writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step=0)

    lr = 1e-4
    beta_1 = 0.9
    beta_2 = 0.999
    weight_decay = 1e-4
    eps = 1e-8
    #optimizer = torch.optim.adam(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #, betas=(beta_1, beta_2), weight_decay=weight_decay)
    num_epochs=20
    #scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, verbose=True)
    loss_fn = DiceBCELoss() #nn.CrossEntropyLoss(reduction="mean") #DiceBCELoss()
    losses=[]
    accuracies=[]
    #print(torch.cuda.memory_summary(device=None, abbreviated=False))
    epoch_loss = train(model, data_train_loader, optimizer, loss_fn, device, epochs=num_epochs)
 
