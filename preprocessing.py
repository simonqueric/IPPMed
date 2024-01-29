import os
import shutil
import nibabel as nib
import tqdm
import numpy as np

np.random.seed(42)

PATH_TO_TRAIN_FOLDER = "WRITE THE PATH HERE"

def split_slices(files, dataset_dir, output_dir):
    for filename in tqdm(files):
        c = 0
        if filename.endswith('.nii.gz'):
            file_path = os.path.join(dataset_dir, filename)

            img = nib.load(file_path)
            volume_data = img.get_fdata()
            _, _, h = volume_data.shape

            for i in range(h):
                output_filename = f"{filename.split('.')[0]}_slice_{c + 1}.nii.gz"
                output_path = os.path.join(output_dir, output_filename)
                c+=1
                new_img = nib.Nifti1Image(volume_data[:,:,i], img.affine)
                nib.save(new_img, output_path)


os.mkdir(PATH_TO_TRAIN_FOLDER + "/slice_vol")
os.mkdir(PATH_TO_TRAIN_FOLDER + "/slice_seg")

os.mkdir(PATH_TO_TRAIN_FOLDER + "/Xtrain")
os.mkdir(PATH_TO_TRAIN_FOLDER + "/ytrain")


files = os.listdir(PATH_TO_TRAIN_FOLDER + "/volume/")
files.sort()

dataset_dir = PATH_TO_TRAIN_FOLDER + "/volume/"
output_dir = PATH_TO_TRAIN_FOLDER + "/slice_vol/"


split_slices(files, dataset_dir, output_dir)

files = os.listdir(PATH_TO_TRAIN_FOLDER + "/volume/")
files.sort()

dataset_dir = PATH_TO_TRAIN_FOLDER + "/lung_seg/"
output_dir = PATH_TO_TRAIN_FOLDER + "/slice_seg/"

split_slices(files, dataset_dir, output_dir)

#Remove empty slices randomly for data efficiency 

for i, (x, y) in enumerate(zip(liste_vol, liste_seg)):
    t = nib.load(seg_dir+y).get_fdata()
    if len(np.unique(t))>1:
        shutil.copy(slice_dir+x, PATH_TO_TRAIN_FOLDER + "/Xtrain/"+x)
        shutil.copy(seg_dir+y, PATH_TO_TRAIN_FOLDER + "/ytrain/"+y)
    else : 
        u = np.random.uniform(0, 1)
        if u<=.5 :
            shutil.copy(slice_dir+x, "data/train/Xtrain/"+x)
            shutil.copy(seg_dir+y, "data/train/ytrain/"+y)


