# Code for IPPMed kaggle competition

IPPMed is a medical data challenge organized by the Institut Polytechnique de Paris and Télécom alumni.
The topic is finding the best model for the segmentation of lung tumours on CT scans.

All the code is written in Python. Use `pip install requirements.txt` to install needed libraries.  

## 1. Preprocessing

As a preprocessing step, we divide all the volume and segmentation images into slices of size $512\times 512$ and downsample them. We keep all the slices that contain a tumor and we keep randomly $50$%  of the slices where there is nothing to segment. The slices are stored in the files Xtrain and ytrain. To apply preprocessing use the command `python3 preprocessing.py`.

## 2. 2D UNet

The model is a classical <a href="https://arxiv.org/abs/1505.04597"> UNet architecture </a> with ReLU activations. We borrow the implementation of the following <a href="https://github.com/nikhilroxtomar/Retina-Blood-Vessel-Segmentation-in-PyTorch/tree/main"> github repo </a>. To train the model, you can either use the command `sbatch train.sh` or `python3 main.py`.

## 3. Prediction

To make the prediction, you just need to use the command `python3 predict.py`.

### Modify file paths to use with your environnement
