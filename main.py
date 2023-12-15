import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from loss import DiceLoss, DiceBCELoss
from model import conv_block, encoder_block, decoder_block, UNet
from dataset import IPPMedDataset
from time import time

def train(model, loader, optimizer, loss_fn, device, epochs=1):
    epoch_loss = 0.0
    losses = []
    model.train()
    k=0
    for epoch in range(epochs) :
        epoch_loss = 0.0
        start = time() 
        for x, y in loader:
       	    x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            k+=1
            
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
    
    
    batch_size=16
    data_root = "data/train/"
    ippmed_dataset = IPPMedDataset(data_root, transform=None)
    data_train_loader = DataLoader(dataset=ippmed_dataset, batch_size=batch_size, shuffle=True) 
	
    file_checkpoint = "checkpoint2.pth"    

    print("Number of images :", len(data_train_loader))

    in_channels = 1  # single-channel input
    out_channels = 1  # Number of segmentation classes
    model = UNet().to(device)
    #model.load_state_dict(torch.load(file_checkpoint))
    lr =1e-4 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss() #nn.CrossEntropyLoss(reduction="mean") #DiceBCELoss()
    losses=[]
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    epoch_loss = train(model, data_train_loader, optimizer, loss_fn, device, epochs=10)
   

    torch.save(model.state_dict(), file_checkpoint)     
