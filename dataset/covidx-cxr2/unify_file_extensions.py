import os
folders = ['train', 'test']

from multiprocessing import Pool
from tqdm import tqdm
import cv2
def run(path):
    name, _ = os.path.splitext(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    shape = img.shape[-3:-1]
    cv2.imwrite(name + '.png', img)
    return shape
with Pool(16) as pool:
    all_shapes = [shape for shape in tqdm(pool.imap(run, (os.path.join(folder, file) for folder in folders for file in os.listdir(folder))))]