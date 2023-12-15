import nibabel as nib
import os
import torch
from torch.utils.data import Dataset



class IPPMedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        vol_dir = os.path.join(root_dir, 'slice_vol')
        self.vol_paths = [os.path.join(vol_dir, vol_path) for vol_path in os.listdir(vol_dir)]        
        seg_dir = os.path.join(root_dir, 'slice_seg')
        self.seg_paths = [os.path.join(seg_dir, seg_path) for seg_path in os.listdir(seg_dir)]        
        self.vol_paths.sort()
        self.seg_paths.sort()
        self.transform = transform
        
    def __len__(self):
        return len(self.vol_paths)

    def __getitem__(self, idx):
        vol_path = self.vol_paths[idx]
        seg_path = self.seg_paths[idx]
        vol_img= nib.load(vol_path).get_fdata()
        seg_img = nib.load(seg_path).get_fdata()

        if self.transform:
            vol_img = self.transform(vol_img)
            seg_img = self.transform(seg_img)

        return torch.from_numpy(vol_img).to(torch.float32).unsqueeze(0), torch.from_numpy(seg_img).to(torch.float32).unsqueeze(0)
