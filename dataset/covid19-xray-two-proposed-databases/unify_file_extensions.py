exts = {'.jpeg', '.jpg', '.png'}

import cv2
paths = []
out_paths = []
for folder, _, files in os.walk('.'):
    for file in files:
        ext = os.path.splitext(file)[1]
        if ext in exts:
            path = os.path.join(folder, file)
            paths.append(path)
            out_path = os.path.join(folder, file.replace(ext, '.png'))
            out_paths.append(out_path)
            #img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            #shapes.append(img.shape[-3:-1])
            #cv2.imwrite(out_path, img)
            
from tqdm import tqdm
from multiprocessing import Pool
def run(i):
    path, out_path = paths[i], out_paths[i]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(out_path, img)
    return img.shape[-3:-1]
with Pool(16) as pool:
    shapes = [shape for shape in tqdm(pool.imap(run, range(len(paths))))]