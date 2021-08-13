## Adapted from https://www.kaggle.com/xhlulu/siim-covid-19-convert-to-jpg-256px?scriptVersionId=63196459

import os
import sys
from collections import Counter
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm
from tqdm.auto import tqdm


def read_xray(path, voi_lut=True, fix_monochrome=True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    max_value = 2**dicom.BitsStored - 1
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = max_value - data

    if max_value != 255:
        data = data.astype(np.float) / max_value
        data = (data * 255).astype(np.uint8)

    return data


def dicom2png_kaggler(dicom_path,
                      png_path=None,
                      output_width=None,
                      output_height=None):
    np_image = read_xray(dicom_path)
    if output_width is not None and output_height is not None:
        np_image = cv2.resize(np_image, (output_width, output_height),
                              interpolation=cv2.INTER_LINEAR)
    else:
        assert output_width is None and output_height is None
    if png_path is not None:
        cv2.imwrite(png_path, np_image)
    return np_image


### Transfer dcm to png 1024
source_root_folder = '.'
target_root_folder = './png_1024'
os.makedirs(target_root_folder, exist_ok=True)
source_paths = []
target_paths = []
for source_folder, folders, files in os.walk(source_root_folder,
                                             followlinks=True):
    for file in files:
        if file.endswith('.dcm'):
            source_paths.append(os.path.join(source_folder, file))
            target_folder = source_folder.replace(source_root_folder,
                                                  target_root_folder)
            target_file = file.replace('dcm', 'png')
            target_paths.append(os.path.join(target_folder, target_file))
print(f'Found {len(source_paths)} dcms.')


def transfer(info):
    i, (source_path, target_path) = info
    os.makedirs(os.path.split(target_path)[0], exist_ok=True)
    try:
        dicom2png_kaggler(source_path,
                          png_path=target_path,
                          output_width=1024,
                          output_height=1024)
        #dicom2png_kaggler(source_path, png_path=target_path) #, output_width=1024, output_height=1024)
        return i, ''
    except Exception:  # pylint: disable=broad-except
        exc_type, exc_obj, exc_tb = sys.exc_info()
        try:
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        except Exception:  # pylint: disable=broad-except
            fname = '?'
        return i, f'{exc_obj}, ({exc_type.__name__}, file: {fname}, line: {exc_tb.tb_lineno})'
        #fail_idxes.append(i)
        #fail_reasons.append(f'{exc_obj}, ({exc_type.__name__}, file: {fname}, line: {exc_tb.tb_lineno})')


with Pool(16) as pool:
    out_stat = [
        (idx, reason)
        for idx, reason in tqdm(
            pool.imap(transfer, enumerate(zip(source_paths, target_paths))))
        if reason != ''
    ]
if len(out_stat) > 0:
    fail_idxes, fail_reasons = tuple(zip(*out_stat))
else:
    fail_idxes, fail_reasons = [], []
