import os

import cv2
from pydicom import dcmread
from tqdm import tqdm

os.system('mkdir png_images')

files = os.listdir('stage_2_train_images')

import torch


class P():

    def __len__(self,):
        return len(files)

    def __getitem__(self, idx):
        file = files[idx]
        ds = dcmread(os.path.join('stage_2_train_images', file))
        img = ds.pixel_array
        cv2.imwrite(os.path.join('png_images', file[:-4] + '.png'), img)
        return 0


for _ in tqdm(
        torch.utils.data.DataLoader(P(),
                                    batch_size=16,
                                    num_workers=16,
                                    collate_fn=lambda x: x)):
    pass

files = os.listdir('stage_2_test_images')

import torch


class P():

    def __len__(self,):
        return len(files)

    def __getitem__(self, idx):
        file = files[idx]
        ds = dcmread(os.path.join('stage_2_test_images', file))
        img = ds.pixel_array
        cv2.imwrite(os.path.join('png_images', file[:-4] + '.png'), img)
        return 0


for _ in tqdm(
        torch.utils.data.DataLoader(P(),
                                    batch_size=16,
                                    num_workers=16,
                                    collate_fn=lambda x: x)):
    pass
