"""
data_utils.py
"""

import os
import glob
import logging
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset


def generate_file_list(data_dir):
    """Generate a list of image files.
    Args:
        data_dir (str):   
    Returns:

    """
    query = os.path.abspath(f'{data_dir}/**/*.jpg')
    image_files = glob.glob(query, recursive=True)

    if not image_files:
        logging.warning(f"No image file found in '{data_dir}'.")

    return sorted(image_files)


class MaskDataset(Dataset):
    def __init__(self, image_files, target, group_age=False, train=True, transform=None):
        self.image_files = image_files
        self.target = target
        self.group_age = group_age
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def _gender_to_cls(self, gender):
        assert gender in ['male', 'female']

        if gender == 'male':
            return 0
        elif gender == 'female':
            return 1
        
    def _age_to_cls(self, age):
        if age < 30:
            return 0
        elif 30 <= age < 60:
            return 1
        else:
            return 2

    def _mask_to_cls(self, mask):
        mask_types = {'incorrect_mask': 0, 'mask1': 1, 'mask2': 2,
                      'mask3': 3, 'mask4': 4, 'mask5': 5, 'normal': 6}
        
        return mask_types[mask]

    def __getitem__(self, idx):
        fpath = self.image_files[idx]
        img = Image.open(fpath)

        if self.transform:
            img = self.transform(img)

        img = transforms.ToTensor()(img)

        if self.train:
            label = {}

            fpath_split = os.path.split(fpath)
            mask = fpath_split[1].split('.')[0]
            identity = os.path.split(fpath_split[0])[1].split('_')
            age, gender = int(identity[3]), identity[1]

            if self.group_age:
                age = self._age_to_cls(age)

            label['age'] = age
            label['gender'] = self._gender_to_cls(gender)
            label['mask'] = self._mask_to_cls(mask)

            if self.target == 'all':
                return img, label
            else:
                return img, label[self.target]

        return img


def get_dataloader(dataset, batch_size=8, val_split=0.2, shuffle=True, drop_last=True):
    train_size = int(len(dataset) * (1 - val_split))
    file_indices = np.arange(len(dataset))
    
    train_dataloader = DataLoader(Subset(dataset, file_indices[:train_size]),
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   drop_last=drop_last,
                                   )
    
    valid_dataloader = None

    msg = 'Get dataloader...\n'
    if val_split:
        valid_dataloader = DataLoader(Subset(dataset, file_indices[train_size:]),
                                    batch_size=batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    )
        
        msg += f'\tTrain : Validation = {len(train_dataloader)} : {len(valid_dataloader)}\n'
        
    msg += f'\tTotal Dataset size: {len(dataset)}\n'
    msg += f'\tBatch size: {batch_size}, # iters: {len(train_dataloader)}'

    logging.info(msg)

    return train_dataloader, valid_dataloader
    