import os
import torch
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import re

train_labels = pd.read_csv('train.csv')
val_labels = pd.read_csv('valid.csv')
target_folder = 'all_images_symlink'
os.makedirs(target_folder, exist_ok=True)


def get_accno(labels, i):
    return re.sub('\W', '_', labels['Path'][i])[len('CheXpert_v1_0_'):-4]

def run(i):
    os.system(f"ln -s ../../{train_labels['Path'][i]} all_images_symlink/{get_accno(train_labels, i)}.jpg")

with Pool(16) as p:
    for _ in tqdm(p.imap(run, range(len(train_labels)))):
        pass

def run(i):
    os.system(f"ln -s ../../{val_labels['Path'][i]} all_images_symlink/{get_accno(val_labels, i)}.jpg")

with Pool(16) as p:
    for _ in tqdm(p.imap(run, range(len(val_labels)))):
        pass