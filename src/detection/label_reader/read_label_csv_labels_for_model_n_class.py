import re

import pandas as pd
import torch
from tqdm import tqdm

FORCE_BINARY = False
ONLY_CLASS = None


def revise_bbox(bbox):
    """
        Args:
            bbox (list[float]): x1, y1, x2, y2
        Return:
            bbox (list[float]): x1, y1, x2, y2
            valid (bool)
    """
    x1, y1, x2, y2 = bbox
    if abs(x1 - x2) < 1 or abs(y1 - y2) < 1:
        return None, False

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2], True


def convert_bbox_str_to_bbox(bbox_str):
    """
        Return:
            bbox (list[float]): x1, y1, x2, y2
            class (int)
            valid (bool)
    """
    bbox_str_values = re.findall(r'\d+\.*\d*', bbox_str)
    bbox = [float(x) for x in bbox_str_values[:4]]
    bbox, valid = revise_bbox(bbox)
    if not valid:
        return None, None, False
    cls = int(bbox_str_values[4]) if not FORCE_BINARY else 1
    if ONLY_CLASS is not None:
        if cls == ONLY_CLASS:
            cls = 1
        else:
            valid = False

    return bbox, cls, valid


def read_label(label_file, split=None, force_binary=False, only_class=None):
    """
        Args:
            label_file (str): file path of the csv file
            split (str): Either 'train', 'val', 'test' or None
            force_binary (bool): Return class label 1 whatever.
            only_class (None or int): form a label data that has only one specific class.
    """
    global FORCE_BINARY, ONLY_CLASS
    FORCE_BINARY = force_binary
    ONLY_CLASS = only_class
    data = pd.read_csv(label_file, dtype=str)
    if split is not None:
        data = data[data['split'] == split]
    data = data.fillna('')
    labels = list()

    for idx, accno, label_str in tqdm(data[['ACCNO',
                                            'labels_for_model']].itertuples()):
        labels.append(get_label(idx, accno, label_str))

    return labels


def get_label(idx, accno, label_str):
    bbox_strs = re.findall(r'\(.+?\)', label_str)
    valid_labels = [
        (bbox, cls) for bbox, cls, valid in (convert_bbox_str_to_bbox(bbox_str)
                                             for bbox_str in bbox_strs) if valid
    ]
    bboxes, clses = tuple(zip(*valid_labels)) if valid_labels else ([], [])

    bboxes = torch.FloatTensor(bboxes).reshape(-1, 4)
    clses = torch.LongTensor(clses)
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    image_id = torch.LongTensor([idx])
    is_crowds = torch.zeros_like(clses)

    label = {
        'ACCNO': accno,
        'labels': {
            'boxes': bboxes,
            'labels': clses,
            'image_id': image_id,
            'area': areas,
            'iscrowd': is_crowds
        }
    }
    return label
