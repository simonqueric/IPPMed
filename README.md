# Code for IPPMed kaggle competition

IPPMed is a medical data challenge organized by the Institut Polytechnique de Paris and Télécom alumni.
The topic is finding the best model for the segmentation of lung tumours on CT scans.

## 1. Preprocessing

As a preprocessing step, we divide all the volume and segmentation images into slices of size $512\times 512$ and downsample them. We keep all the slices that contain a tumor and we keep randomly $50$ % of the slices where there is nothing to segment. The slices are stored in the files Xtrain and ytrain.

## 2. 2D UNet

The model is a classical UNet architecture with ReLU activations. To train the model, you can either use the command `sbatch train.sh` or `python3 main.py`.

## 3. Prediction

To make the prediction, you just need to use the command `python3 predict.py`.

### Modify file paths to use with your environnement
