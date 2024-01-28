import torch
from torch.utils.data import Dataset
import os
import nibabel as nib

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.lung_seg_dir = os.path.join(data_dir, 'lungs_seg')
        self.tumor_mask_dir = os.path.join(data_dir, 'seg')
        self.scan_volume_dir = os.path.join(data_dir, 'volume')

        # List the files available in the lung segmentation directory
        self.file_list = os.listdir(self.lung_seg_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load lung segmentation, tumor mask, and scan volume for the given index
        lung_seg_path = os.path.join(self.lung_seg_dir, self.file_list[idx])
        base_filename = self.file_list[idx].split('_lungseg.nii.gz')[0]
        tumor_mask_path = os.path.join(self.tumor_mask_dir, base_filename + '_seg.nii.gz')
        scan_volume_path = os.path.join(self.scan_volume_dir, base_filename + '_vol.nii.gz')

        # Load data from NIfTI files
        lung_seg_data = nib.load(lung_seg_path).get_fdata()
        tumor_mask_data = nib.load(tumor_mask_path).get_fdata()
        scan_volume_data = nib.load(scan_volume_path).get_fdata()

        # Convert data to PyTorch tensors
        lung_seg_tensor = torch.tensor(lung_seg_data, dtype=torch.float32)
        tumor_mask_tensor = torch.tensor(tumor_mask_data, dtype=torch.float32)
        scan_volume_tensor = torch.tensor(scan_volume_data, dtype=torch.float32)

        return lung_seg_tensor, tumor_mask_tensor, scan_volume_tensor
