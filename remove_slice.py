import numpy as np
import nibabel as nib
import shutil
import os

PATH_TO_TRAIN_FOLDER = "../data_challenge/train"
slice_dir = PATH_TO_TRAIN_FOLDER + "/slice_vol/"
liste_vol = os.listdir(slice_dir) 
liste_vol.sort()
seg_dir = PATH_TO_TRAIN_FOLDER + "/slice_seg/"
liste_seg = os.listdir(seg_dir) 
liste_seg.sort()

for i, (x, y) in enumerate(zip(liste_vol, liste_seg)):
    t = nib.load(seg_dir+y).get_fdata()
    if len(np.unique(t))>1:
        shutil.copy(slice_dir+x, PATH_TO_TRAIN_FOLDER + "/Xtrain/"+x)
        shutil.copy(seg_dir+y, PATH_TO_TRAIN_FOLDER + "/ytrain/"+y)
    else :
        u = np.random.uniform(0, 1)
        if u<=.5 :
            shutil.copy(slice_dir+x, PATH_TO_TRAIN_FOLDER + "/Xtrain/"+x)
            shutil.copy(seg_dir+y, PATH_TO_TRAIN_FOLDER + "/ytrain/"+y)
