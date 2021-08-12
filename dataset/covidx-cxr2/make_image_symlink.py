import os
import torch
from multiprocessing import Pool
from tqdm import tqdm

png_root_folder = '.'
target_folder = 'all_symlinks'
png_paths = []
for png_folder, _, files in os.walk(png_root_folder):
    for file in files:
        if file.endswith('.png'):
            png_paths.append(os.path.join(png_folder, file))


def make_link(source_path):
    os.system(f'ln -s "{source_path}" "{os.path.split(source_path)[1]}"')


os.makedirs(target_folder, exist_ok=True)
cwd = os.getcwd()
os.chdir(target_folder)
with Pool(16) as p:
    for _ in tqdm(p.imap(make_link, png_paths)):
        pass
os.chdir(cwd)