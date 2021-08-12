# pylint: disable=invalid-name
"""
    All utility functions for classification tasks
"""

import functools
import os
import pickle
import random
import math

import cv2
import numpy as np
import pandas as pd
import skimage
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import yaml
from collections import Iterable, OrderedDict
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
import elasticdeform
import timm
from multiprocessing import Pool

import utils

try:
    base_dir = os.path.dirname(os.path.realpath(__file__))
except:
    base_dir = '.'

with open(os.path.join(base_dir, 'metadata.yaml'), 'r') as stream:
    metadata = yaml.safe_load(stream)
metadata['root_path'] = base_dir


def read_label(path, column_name, split, label_only=False):
    """ Read label into a list

    Args:
        path (str): path of the data
        split (str): 'train', 'val' or 'test'
    Return:
        list: each item for the list is a list of [ACCNO (str), label (int)]
    """
    if path.endswith('xlsx'):
        if split is not None:
            data = pd.read_excel(path, sheet_name=split, dtype=str)
        else:
            data = pd.read_excel(path, dtype=str)
        data[column_name] = pd.to_numeric(data[column_name])
    elif path.endswith('csv'):
        data = pd.read_csv(path, dtype=str)
        data[column_name] = pd.to_numeric(data[column_name])
        if split is not None:
            if 'split' in data:
                data = data[data['split'] == split]
            elif 'set' in data:
                if split == 'val':
                    split = 'validation'
                data = data[data['set'] == split]
            else:
                raise ValueError('no split column presents in the data')
    if label_only:
        return data[column_name].tolist()
    
    return data[['ACCNO', column_name]].values.tolist()


def read_image(path):
    return torchvision.transforms.functional.to_tensor(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))


class ClassificationDataset(torch.utils.data.Dataset):
    """ classification dataset"""

    def __init__(self,
                 task_name,
                 blackout,
                 split,
                 transform=None,
                 blackout_margin=25,
                 lung_crop_fill=0,
                 multitask=False,
                 more_info=False,
                 equalize=False,
                 standardize=False,
                 return_idx=False,
                 exclude_accno=None,
                 removing_idxes=None,
                 preprocess_resize_1024=True):
        """
        Args:
            task_name: task name defined in metadata.yaml
            blackout: to blackout the area outside the lung.
                If set to True, blackout file must be provided and
                given in the metadata.
            split (str): 'train', 'val' or 'test'
            transform (callable, optional): Optional transform to be applied
                on an image.
            equalize: Redundant, please always set this to False. Use
                ClassificationDatasetWithEqualizeAndStandardize
                for equalization.
            standardize: Redundant, please always set this to False. Use
                ClassificationDatasetWithEqualizeAndStandardize
                for equalization.
            blackout_margin: the margin added to lung box to blackout
            multitask: if True, task name will be returned (so that one can
                distinguish which multi-task head to be trained when loading
                a batch of images)
            more_info: If True, image path and ACCNO will also be returned
        """
        if exclude_accno is None:
            exclude_accno = []
            
        self.more_info = more_info
        self.task_name = task_name
        self.transform = transform
        self.blackout_margin = blackout_margin
        self.lung_crop_fill = lung_crop_fill
        self.multitask = multitask
        self.return_idx = return_idx
        self.file_extension = metadata[task_name].get('file_extension', None) or 'png'

        image_folder = os.path.join(metadata['root_path'],
                                    metadata[task_name]['image_folder'])
        self.image_folder = image_folder

        label_file = os.path.join(metadata['root_path'],
                                  metadata[task_name]['label_file'])
        self.data = read_label(label_file,
                               metadata[task_name]['label_column_name'], split)
        

        ignore_set = check_all_images_exist(
            image_folder,
            self.data,
            ignore_not_found=('ignore_not_found' in metadata[task_name]),
            extension=self.file_extension)
        if 'ignore_not_found' in metadata[task_name]:
            self.data = [
                x for x in self.data if (x[0] + '.' + self.file_extension) not in ignore_set
            ]
        if removing_idxes is not None:
            self.data = [d for i, d in enumerate(self.data) if i not in removing_idxes]
        if exclude_accno:
            self.data = [x for x in self.data if x[0] not in exclude_accno]

        self.blackout = blackout
        print(f'len dataset: {len(self)}')
        self.preprocess_resize_1024 = preprocess_resize_1024
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        accno = self.data[idx][0]
        image_path = os.path.join(self.image_folder, accno + '.' + self.file_extension)
        img = read_image(image_path)

        if self.preprocess_resize_1024 and tuple(img.shape[-2:]) != (1024, 1024):
            img = torchvision.transforms.functional.resize(img, (1024, 1024))
            
        if self.transform is not None:
            img = self.transform(img)
        label = self.data[idx][1]
        output_list = [img, label]
        if self.multitask:
            output_list.append(self.task_name)
        if self.more_info:
            output_list += [image_path, accno]
        if self.return_idx:
            output_list.append(idx)

        return output_list


def get_single_dataset(arg):
    task, args, kwargs = arg
    print(f"Loading {task} {kwargs['split'] if 'split' in kwargs else args[1]} data")
    return ClassificationDataset(task, *args, **kwargs)

class MultiTaskAggregatedClassificationDataset(torch.utils.data.Dataset):
    """Aggregating multiple datasets and output multiple labels for each image"""
    def __init__(self, task_names, *args, **kwargs):
        self.return_idx = kwargs.get('return_idx', False)
        with Pool(8) as pool:
            self.datasets = list(pool.imap(get_single_dataset, ((task, args, kwargs) for task in task_names)))
        self.task_names = task_names
        accno2label = OrderedDict()
        accno2label_mask = OrderedDict()
        accno2dataset_idx = OrderedDict()
        for i, dataset in enumerate(self.datasets):
            print(f'Aggregating {dataset.task_name} {dataset.split} data.')
            for idx in range(len(dataset)):
                accno, label = dataset.data[idx]
                if accno not in accno2label:
                    accno2label[accno] = torch.zeros(len(self.datasets)).long()
                    accno2label_mask[accno] = torch.zeros(len(self.datasets)).bool()
                    accno2dataset_idx[accno] = torch.zeros(len(self.datasets)).long()
                accno2label[accno][i] = label
                accno2label_mask[accno][i] = True
                accno2dataset_idx[accno][i] = idx
        self.label = list(accno2label.values())
        self.label_mask = list(accno2label_mask.values())
        self.dataset_idx = list(accno2dataset_idx.values())
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        task_idx = torch.where(self.label_mask[idx])[0][0]
        out_item = self.datasets[task_idx][self.dataset_idx[idx][task_idx]]
        out_item[1] = self.label[idx]
        if self.return_idx:
            out_item[-1] = idx
        out_item.append(self.label_mask[idx])
        return out_item # img, labels, ..., label_mask


def collate_transforms(transform):
    """
        Return:
            color jitter transform if exist
            transforms other than color jitter if exist
    """
    if not isinstance(transform, torchvision.transforms.Compose):
        if transform is None:
            return (torchvision.transforms.Compose([]),
                    torchvision.transforms.Compose([]))
        if isinstance(transform, torchvision.transforms.ColorJitter):
            return transform, torchvision.transforms.Compose([])
        return torchvision.transforms.Compose([]), transform
    cj_trans = torchvision.transforms.Compose([
        x for x in transform.transforms
        if isinstance(x, torchvision.transforms.ColorJitter)
    ])
    other_trans = torchvision.transforms.Compose([
        x for x in transform.transforms
        if not isinstance(x, torchvision.transforms.ColorJitter)
    ])
    return cj_trans, other_trans


def random_scaled_crop_collate_decorator(org_collate=torch.utils.data._utils.collate.default_collate, scale=(0.08, 1)):
    """crop a patch for each image in the batch within specified scale range.
    
    Args:
        org_collate: collate function that returns a tuple like (images, *other_info),
            where images is a tensor with shape (batch, channel, height, width)
        scale (tuple): scale range.
    """
    def random_scaled_crop_collate(datalist):
        try:
            images, *other_info = org_collate(datalist)
            has_other_info = True
        except:
            images = org_collate(datalist)
            has_other_info = False
        
        new_images = []
        height, width = images.shape[-2:]
        target_ratio = random.uniform(*scale)
        w = int(round(width * target_ratio))
        h = int(round(height * target_ratio))
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)
        for image in images:
            new_images.append(image[:, i:i + h, j:j + w])
        if has_other_info:
            return [torch.stack(new_images)] + other_info
        return torch.stack(new_images)
    return random_scaled_crop_collate


def check_all_images_exist(image_folder, label_data, ignore_not_found=False, extension='png'):
    """
        To check if all accno in the dataset has the corresponding image.
    """
    label_images = {x[0] + '.' + extension for x in label_data}
    all_images = set(os.listdir(image_folder))
    if len(label_images) != len(set.intersection(label_images, all_images)):
        if not ignore_not_found:
            raise ValueError(
                f"There are some ACCNO in the label file which cannot find "
                f"corresponding images in image folder. "
                f"(please specify 'ignore_not_found' in the dataset metadata "
                f"if you want to ignore them.) \n "
                f"Unfound:{label_images-all_images}")
    return label_images - all_images


def repr_dict(config):
    """
        (Deprecated)To generate a string which represent a
            dictionary object for the purpose of constructing
            folder or file name.
    """
    out_list = []
    for k, v in config.items():
        if isinstance(v, dict):
            v_str = repr_dict(v)
        elif isinstance(v, (list, tuple, set)):
            v_str = '_'.join([str(x) for x in v])
        else:
            v_str = str(v)
            if '/' in v_str or '\\' in v_str:
                continue
        v_str = ''.join([ch if is_naming_char(ch) else '_' for ch in v_str])
        out_list.append(str(k) + '_' + v_str)
    return '_'.join(out_list)


def print_dict_readable(config):
    """
        To print the dictionary in the format that
            is ready to be recorded in gitlab
    """
    readable_str = ''
    name = config['name']
    for k, v in config.items():
        if not k == 'name':
            v_str = repr(v) if not isinstance(v, float) else f'{v:.4}'
            print(f"{name}['{k}'] = {v_str}")
            readable_str += f"{name}['{k}'] = {v_str}"
            readable_str += '\n'
    return readable_str


def is_naming_char(ch):
    """
        The function which can be replaced by re but
            I has no time to do so.
    """
    if '9' >= ch >= '0':
        return True
    if 'z' >= ch >= 'a':
        return True
    if 'Z' >= ch >= 'A':
        return True
    if ch == '_':
        return True
    return False


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')



def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank

    This code is copied and revised from https://github.com/pytorch/vision/blob/master/references/detection/utils.py
    """
    world_size = utils.get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda") # pylint: disable=not-callable
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)] # pylint: disable=not-callable
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(
            torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,),
                              dtype=torch.uint8,
                              device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list




import torch.distributed as dist
import functools
def synchronize_variables(*args):
    """
        Aggregate the values processes by each torch
            distributed process

        Argument:
            args: tuple of torch tensors to be aggregated,
                where the last element must be the indexes
                to discard the repetetions.
    """
    dist.barrier()
    is_tensor = [isinstance(x, torch.Tensor) for x in args]
    if [i for i, x in enumerate(args) if not is_tensor[i] and not isinstance(x, list)]:
        raise ValueError(f'Unable to gather {type(args[i])} datas ({i}-th data).')
    if is_tensor[-1]:
        all_img_ids = all_gather(args[-1])
        merged_img_ids = []
        for p in all_img_ids:
            merged_img_ids.extend(p.tolist())
    else:
        merged_img_ids = functools.reduce(lambda x, y: (x + y), utils.all_gather(args[-1]))
    merged_img_ids = np.array(merged_img_ids)
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    
    receives = []
    for i, arg in enumerate(args[:-1]):
        if is_tensor[i]:
            all_receive = all_gather(arg)
            receive = []
            for p in all_receive:
                receive.append(p)
            receive = torch.cat([x.to(receive[0].device) for x in receive])[idx]
        else:
            receive = functools.reduce(lambda x, y: (x + y), utils.all_gather(arg))
            receive = [receive[j] for j in idx]
        receives.append(receive)
        
    dist.barrier()
    return receives


def get_auc(scores, targets):
    """

    Args:
        scores (Tensor): `(N,C)`
        targets (Tensor): `(N,)`
    Return:
        tensor of size `(C,)`
    """
    C = scores.shape[1]
    e = torch.eye(C, dtype=torch.long)
    onehot = e[targets]
    scores = scores.cpu().numpy()
    onehot = onehot.cpu().numpy()
    auc = []
    for i in range(C):
        auc.append(roc_auc_score(onehot[:, i], scores[:, i]))
    return torch.tensor(auc) # pylint: disable=not-callable


def get_ap(scores, targets):
    """

    Args:
        scores (Tensor): `(N,C)` including background.
        targets (Tensor): `(N,)` where background is index 0.
    Return:
        tensor of size `(C,)`
    """
    C = scores.shape[1]
    e = torch.eye(C, dtype=torch.long)
    onehot = e[targets]
    scores = scores.cpu().numpy()
    onehot = onehot.cpu().numpy()
    ap = []
    for i in range(C):
        ap.append(average_precision_score(onehot[:, i], scores[:, i]))
    return torch.tensor(ap) # pylint: disable=not-callable


class MLPLayer(nn.Module):
    """
        The class that control each task head in multi-task model
    """

    def __init__(self, in_features, out_features, num_layers=1):
        super(MLPLayer, self).__init__()
        self.num_layers = num_layers
        self.mlp = []
        for _ in range(num_layers - 1):
            self.mlp.append(nn.Linear(in_features, in_features))
        self.mlp.append(nn.Linear(in_features, out_features))
        self.mlp = nn.ModuleList(self.mlp)

    def forward(self, x): # pylint: disable=arguments-differ
        feat = x
        for i in range(self.num_layers - 1):
            feat = F.relu(self.mlp[i](feat))
        feat = self.mlp[-1](feat)
        return feat


# reference: https://pytorch.org/docs/stable/_modules/torch/nn/modules/container.html#ModuleDict
class MultiTaskModel(nn.ModuleDict):
    """
        Build a module dict for a multitask classifier.
    """

    def __init__(self, modules, tasks=None):
        """
            Args:
                modules: A dictionary containing a backbone with key 'backbone'
                    and several task heads with keys the task names respectively
        """
        self.tasks = tasks
        super().__init__(modules)

    def forward(self, x, tasks=None, key=None): # pylint: disable=arguments-differ
        """
            Args:
                tasks: if given, must be one of the following format:
                    1. a list of task strings. the length is
                        expected to be x.shape[0]
                    2. a string
            Return: if tasks is given, return a list of output corresponding to
                the given tasks; if not given, return a dictionary of
                output tensors
        """
        if key is not None:
            return getattr(self, key)(x)

        if tasks is not None:
            self.tasks = tasks

        feat = self.backbone(x)
        if tasks is None:
            return {
                key: getattr(self, key)(feat)
                for key in self._modules
                if key != 'backbone'
            }

        if isinstance(self.tasks, str):
            return getattr(self, self.tasks)(feat)

        return [
            getattr(self, self.tasks[i])(feat[i:i + 1])
            for i in range(x.shape[0])
        ]


class RandomAspectRatio():
    """
        This is the implementation of using torchvision.transforms.functional to
            simulate changing aspect ratio of the image.

        Note that this transform assumes input image to be a gray-level,
        3-channel, square, and 8-bit PIL Image.

        Args:
            scale (float): How much to change the aspect ratio of the image.
                The actual aspect ratio is chosen from
                original_aspect_ratio * [(1+scale)^-1, 1+scale]
    """

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img):
        h, w = img.shape[-2:]
        scale = random.random() * self.scale
        # choose to be wider or thinner
        if random.random() >= 0.5:
            padding = random.randint(0, int(scale * w / 2))
            img = torchvision.transforms.functional.pad(img, (padding, 0), 0,
                                                        'constant')
        else:
            padding = random.randint(0, int(scale * h / 2))
            img = torchvision.transforms.functional.pad(img, (0, padding), 0,
                                                        'constant')
        return torchvision.transforms.functional.resize(img, (h, w))

    def __repr__(self):
        return self.__class__.__name__ + f'(scale={self.scale})'


class RandomShift():
    """
        Random shift the image

        Args:
            scale_x: scale to shift the image and bbox horizontally.
                The shift pixel will be chosen from -width * scale_x
                to width * scale_x
            scale_y: (similary to scale_x)
            pad_mode: 'black' or 'repeat'.
                repeat padding means to fill the blank area with the
                original image .

    """

    def __init__(self, scale_x=0.5, scale_y=0.5, pad_mode='black'):
        if not 0 <= scale_x <= 1 and 0 <= scale_y <= 1:
            raise ValueError('scales should be between 0 and 1')
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.pad_mode = pad_mode

    def __call__(self, image):
        paste_rel_x, paste_rel_y = random.random(), random.random()

    
        ## torch style
        shift_x = int(image.shape[-1] * self.scale_x * (2 * paste_rel_x - 1))
        shift_y = int(image.shape[-2] * self.scale_x * (2 * paste_rel_x - 1))
        if self.pad_mode == 'black':
            mosaic = torch.zeros(*(image.shape[:-2]), image.shape[-2] * 3, image.shape[-1] * 3, dtype=image.dtype)
            mosaic[..., image.shape[-2]: image.shape[-2] * 2, image.shape[-1]: image.shape[-1] * 2] = image
        elif self.pad_mode == 'repeat':
            mosaic = image.repeat(1, 3, 3)
        image = mosaic[...,
                      image.shape[-2] - shift_y:image.shape[-2] * 2 - shift_y,
                      image.shape[-1] - shift_x:image.shape[-1] * 2 - shift_x
                     ]
        return image


class RandomPadCrop():
    """
        Randomly pad the given PIL Image on all sides and
            crop an area containing the original image.
            This is the implementation of using
            torchvision.transforms.functional to
            simulate the transform of resize smaller and randomly placed on
            a black canva.

        Note that this transform assumes input image to be a
            gray-level, 3-channel, square, and 8-bit PIL Image.

        Args:
            scale (float): How much to pad the image.
                The actual padding value is chosen from
                [0, int(scale * img_size)]
    """

    def __init__(self, scale):
        self.scale = scale

    @staticmethod
    def get_params(img, scale):
        """Get parameters for ``pad`` and ``crop`` for a random padded crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size padded

        Returns:
            tuple: params (padding, i, j, h, w) to be passed to ``pad``
                and ``crop`` for a random sized pad and crop.
        """
        h, w = img.shape[-2:]
        if w != h:
            raise ValueError('Not implemented for non-square image.')
        img_size = w

        padding = random.randint(0, int(scale * img_size))
        i = random.randint(0, padding)
        j = random.randint(0, padding)
        th = img_size + padding
        tw = img_size + padding
        return padding, i, j, th, tw

    def __call__(self, img):
        padding, i, j, h, w = self.get_params(img, self.scale)
        img = torchvision.transforms.functional.pad(img, padding, 0, 'constant')
        return torchvision.transforms.functional.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + f'(scale={self.scale})'


class RandomBlur():
    """
        Gaussian blur the image
        Note that this transform assumes input image to be a
            gray-level, 3-channel, square, and 8-bit PIL Image.
    """

    def __init__(self, scale):
        raise ValueError('Please use torchvision.transforms.GaussianBlur instead.')
        self.scale = scale  # scale = 1e-3 ~ 1e-2
        self.ToPILImage = torchvision.transforms.ToPILImage(mode='RGB')

    def __call__(self, img):
        w, h = img.size
        if w != h:
            raise ValueError('Not implemented for non-square image.')
        img_size = w

        img = np.asarray(img)
        sigma = random.uniform(0, img_size * self.scale)
        img = gaussian_filter(img, sigma=sigma)
        img = self.ToPILImage(img)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'scale={0})'.format(self.scale)
        return format_string


class Compose():
    """
        Compose image transformations with execetion probabilities.
    """

    def __init__(self, transform, prob_multiplier=1.):
        self.transform = transform
        self.prob_multiplier = prob_multiplier

    def __call__(self, image):
        for t in self.transform:
            if random.random() < self.prob_multiplier:
                try:
                    image = t(image)
                except Exception as e:
                    raise Exception(t.__class__.__name__ + str(e)) from e
        return image


class RandomNoise():
    """
        add gaussian noise to the image
        Note that this transform assumes input image to be a
            gray-level, 3-channel, square, and 8-bit PIL Image.
    """

    def __init__(self, n):
        raise NotImplementedError('This is still the old version of PIL-style. Please re-implement with torch-style.')
        self.n = n  # n = 0.5~5
        std = 0.05 * n
        self.var = std**2
        self.ToPILImage = torchvision.transforms.ToPILImage(mode='RGB')

    def __call__(self, img):
        img = np.asarray(img)

        noisy_img = skimage.util.random_noise(img,
                                              mode='gaussian',
                                              mean=0,
                                              var=self.var,
                                              seed=1)
        temp = noisy_img[:, :, 0]
        result = np.dstack((temp, temp, temp))
        img = (result * 255).astype(np.uint8)

        return self.ToPILImage(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'n={0})'.format(self.n)
        return format_string

    def repr_oneline(self):
        return f'noise_{self.n}'

def load_state_dict_reviser_not_strict(func):
    def load_state_dict_not_strict(*args, **kwargs):
        if not 'strict' in kwargs:
            kwargs['strict'] = False
        return func(*args, **kwargs)
    return load_state_dict_not_strict


class CV2Resize():
    """Resize the input PIL Image to the given size with cv2.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR``
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        
        if not torchvision.transforms.functional._is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img))) 
            
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return Image.fromarray(cv2.resize(np.array(img), (ow, oh), interpolation = self.interpolation))
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return Image.fromarray(cv2.resize(np.array(img), (ow, oh), interpolation = self.interpolation))
        else:
            return Image.fromarray(cv2.resize(np.array(img), tuple(self.size[::-1]), interpolation = self.interpolation))

    def __repr__(self):
        interpolate_str = str(self.interpolation)
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class RandomScaledCrop(torchvision.transforms.RandomResizedCrop):
    """Crop a patch with a range of random scale without resizing after cropping.
    
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made.

    Args:
        size: No effect
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return torchvision.transforms.functional.crop(img, i, j, h, w)


class FrozenBatchNorm2dWithEpsilon(torchvision.ops.misc.FrozenBatchNorm2d):
    """This class aims to make the epsilon consistent with the default value of torch.nn.BatchNorm2d"""
    
    def __init__(self, *args, **kwargs):
        if 'eps' not in kwargs:
            kwargs['eps'] = 1e-5
        super().__init__(*args, **kwargs)


def change_layer(model, source_module, target_module, params_map={}, state_dict_map={}):
    """ To change specific type of layers to another one. e.g. BatchNorm2d to FrozenBatchNorm2d
    
    Args:
        model (nn.Module): The model to alter
        source_module (nn.Module Class): the module class to be replaced. It will be used by checking
            if any module in the model is its instance.
        target_module (nn.Module Class): The constructor that will replace all source_module
        params_map (Dict[str: str]): The dict keys are the attributes of the source_module and the
            dict values are the parameter name to construct target_module
        state_dict_map (Dict[str: str]): The dict keys are the state dict key name of the source_module
            and the dict values are the state dict key name of the target_module to be loaded.
    """
    def check_children(model):
        for name, module in model.named_children():
            if isinstance(module, source_module):
                state_dict_to_load = {(k if k not in state_dict_map else state_dict_map[k]): v for k, v in module.state_dict().items()}
                setattr(model, name, target_module(**{new_param: getattr(module, org_param) for org_param, new_param in params_map.items()}))
                getattr(model, name).load_state_dict(state_dict_to_load, strict=False)
            check_children(getattr(model, name))
    check_children(model)


def freeze_all(nn_module):
    for p in nn_module.parameters():
        p.requires_grad_(False)
    change_layer(nn_module,
                 torch.nn.BatchNorm2d,
                 torchvision.ops.misc.FrozenBatchNorm2d,
                 params_map={'num_features':'num_features', 'eps':'eps'}
                )


class SAMOptim(torch.optim.Optimizer):
    """Code credit: https://github.com/davda54/sam/blob/main/sam.py
    This is the implementation of the paper: "Sharpness-Aware Minimization for Efficiently Improving Generalization"
    Revision with weight normalization.
    """
    def __init__(self, params, base_optimizer, rho=0.05, weight_norm=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, weight_norm=weight_norm, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]['accumulate_grad'] = None

    @torch.no_grad()
    def step(self, *args, first_step=False, update=True, **kwargs):
        if first_step:
            self.first_step(*args, **kwargs)
        else:
            self.second_step(*args, update=update, **kwargs)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            if group["weight_norm"]:
                weight_norm = self._weight_norm()

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                if group["weight_norm"]:
                    e_w *= p / (weight_norm + 1e-12)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad=False, update=True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        if update:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None and self.state[p]['accumulate_grad'] is not None:
                        p.grad.add_(self.state[p]['accumulate_grad'])
            self.base_optimizer.step()  # do the actual "sharpness-aware" update
            for group in self.param_groups:
                for p in group["params"]:
                    self.state[p]['accumulate_grad'] = None
            
        else:
            for group in self.param_groups:
                for p in group["params"]:
                    if self.state[p]['accumulate_grad'] is None:
                        self.state[p]['accumulate_grad'] = p.grad
                    elif p.grad is not None:
                        self.state[p]['accumulate_grad'] += p.grad

        if zero_grad:
            self.zero_grad(set_to_none=True)

    #def step(self, closure=None):
    #    raise NotImplementedError("SAM doesn't work like the other optimizers, you should first call `first_step` and the `second_step`; see the documentation for more info.")

    def _grad_norm(self):
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
    def _weight_norm(self):
        norm = torch.norm(
                    torch.stack([
                        p.norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p is not None
                    ]),
                    p=2
               )
        return norm

class ElasticDeform():
    """
        Perform elastic deformation to the image and the target
        correspondingly.   
    """
    def __init__(self, relative_sigma=1/40, points=5, order=0):
        """
            Args:
                relative_sigma (float): This value * image size would be 
                    the sigma parameter of elasticdeform.deform_random_grid
                points (int): The parameter in elasticdeform.deform_random_grid
                order {0, 1, 2, 3, 4}: 
                    The parameter in elasticdeform.deform_random_grid
        """
        self.relative_sigma = relative_sigma
        self.points = points
        self.order = order
    
    def __call__(self, image):
        np_img = (image.permute(1, 2, 0) * 255).numpy().astype(np.uint8) # (H, W, C)
        height, width, channel = np_img.shape
        
        # deform!
        deformed_img = elasticdeform.deform_random_grid(np_img, sigma=width*self.relative_sigma, points=self.points, order=self.order, axis=(0,1))
        
        # deformed image
        np_img_deformed = deformed_img[:, :, :channel]
        torch_img_deformed = torchvision.transforms.functional.to_tensor(np_img_deformed)
        
        return torch_img_deformed

class RandAugmentPT():
    """A wrapper for RandAugment in timm"""
    
    def __init__(self, in_size, ra_str='rand-m9-mstd0.5', img_mean=timm.data.constants.IMAGENET_DEFAULT_MEAN):
        if isinstance(img_mean, torch.Tensor):
            img_mean = img_mean.tolist()
        aa_params = dict(
            translate_const=int(in_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in img_mean]),
        )
        self.ra = timm.data.auto_augment.rand_augment_transform(ra_str, aa_params)
        
    def __call__(self, pt_img):
        pil_img = torchvision.transforms.functional.to_pil_image(pt_img)
        return torchvision.transforms.functional.to_tensor(self.ra(pil_img))

class SoftCrossEntropyLoss():
    def __init__(self, p_0):
        self.p_0 = p_0
    
    def __call__(self, preds, targets):
        # preds (N, C, ...)
        # targets (N, ...)
        C = preds.shape[1]
        span_targets = torch.full(preds.shape, (1 - self.p_0) / (C - 1), device=targets.device)
        idxes = [torch.tensor(range(shape))[..., None].repeat(1, torch.tensor(targets.shape[i + 1:]).prod()).flatten() if i != len(targets.shape) - 1 else torch.tensor(range(shape)) for i, shape in enumerate(targets.shape)]
        idxes = [x.repeat(round(len(idxes[0]) / len(x))) for x in idxes]
        idxes.insert(1, targets.flatten())
        span_targets[idxes] = self.p_0
        return -(torch.nn.functional.log_softmax(preds, dim=1) * span_targets).sum(dim=1).mean()


class BCEWithLogitsCategoricalLoss():
    def __init__(self, p_0=1):
        """Args
            p_0: soft labeling. (default=1 for disabling)
        """
        self.p_0 = p_0
        
    def __call__(self, preds, targets):
        # preds (N, C, ...)
        # targets (N, ...)
        C = preds.shape[1]
        span_targets = torch.full(preds.shape, (1 - self.p_0) / (C - 1), device=targets.device)
        idxes = [torch.tensor(range(shape))[..., None].repeat(1, torch.tensor(targets.shape[i + 1:]).prod()).flatten() if i != len(targets.shape) - 1 else torch.tensor(range(shape)) for i, shape in enumerate(targets.shape)]
        idxes = [x.repeat(round(len(idxes[0]) / len(x))) for x in idxes]
        idxes.insert(1, targets.flatten())
        span_targets[idxes] = self.p_0
        return torch.nn.functional.binary_cross_entropy_with_logits(preds, span_targets)

class FocalLoss(torch.nn.Module):
    """Focal loss with sigmoid and softmax version

    Sigmoid focal loss is the focal loss in the original paper.
    We add class weights and background class on it.
    If sigmoid activation is used, you can set `background_class` to
    determine whether background class is included in your inpus tensor
    and given weights.
    the implementation refer to
    https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py

    softmax focal loss takes the softmaxed target class value as pt
    and do the docal loss. If softmax is used,
    the given inputs tensor and weight should include background class.
    Alpha is ignored in softmax focal loss.

    There is a little difference between the mean reduction in sigmoid
    and softmax activation loss. when the activation is softmax, the mean
    method will sum up the weighted loss and be devided by the sum of
    weights. In the case of sigmoid loss, the calculation will be
    the average over all weighted loss. This is because the sum of weights
    in every batch of softmax loss will vary, while it does not happen
    in the sigmoid case.

    Let `C` be number of classes including background.

    Args:
        activation: (string, optional): ``'softmax'`` | ``'sigmoid'``.
            Activation functions applied to the input logits.
            Default: ``'sigmoid'``
        background_class: (bool, optional): define whether the inputs to
            calculate loss include background class. When the activation
            is softmax, this parameter is ignored and the inputs should
            always include background class. Default: ``True``.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C'`. When the
            activation is sigmoid and the `background_class` flag is
            `False`, ``C'=C-1``, otherwise ``C'=C``.
        gamma: (float, optional): focusing parameter, indicating how much
            to focus on the hard examples.
        alpha: (float, optional): weighting factor, weighting between positive
            and negative samples. when the activation is softmax, this parameter
            is ignored.
        suppress: (str): 'easy' or 'hard' (default='easy')
        reduction (string, optional): Specifies the reduction to apply to the
            output: ``'none'`` | ``'mean'``. ``'none'``: no reduction will be
            applied. ``'mean'``: Average over classes (and samples if `mode`
            is `'sample'`). Default: ``'mean'``.
    """

    def __init__(self,
                 activation='sigmoid',
                 background_class=True,
                 weight=None,
                 gamma=2.0,
                 alpha=0.5,
                 suppress='easy',
                 reduction='mean'):
        super().__init__()
        self.activation = activation
        self.background_class = background_class
        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha
        self.suppress = suppress
        self.reduction = reduction

    def forward(self, inputs, targets):  # pylint: disable=arguments-differ, inconsistent-return-statements
        """
        Let `N` be batch size and `C` be number of classes including background.

        Args:
            inputs: Tensor of shape `(N, C', d_1, d_2, ..., d_K)`, where each
                value is the logit prediction for the corresponding class.
                When the activation is sigmoid and the `background_class`
                flag is `False`, ``C'=C-1``, otherwise ``C'=C``.
            targets: Tensor of shape `(N, d_1, d_2, ..., d_K)`. The index always
            starts from 0 as background. i.e. 0 <= targets[i] <= C-1
        """
        # collate inputs and targets to (N',C') and (N',),
        # where N' = N * d_1 * d_2 * ... * d_K
        inputs = inputs.permute(0, *tuple(range(2, len(inputs.shape))),
                                1)  # shift C' to the last dim
        inputs = inputs.flatten(end_dim=-2)
        targets = targets.flatten()
        if self.activation == 'sigmoid':  # pylint: disable=no-else-return
            if self.background_class:
                y = torch.eye(inputs.shape[1])
            else:
                y = torch.eye(inputs.shape[1] + 1)
            t = y[targets].to(inputs.device)
            if not self.background_class:
                t = t[:, 1:]  # exclude background
            p = inputs.sigmoid()
            if self.suppress == 'hard':
                p = 1 - p
            one_minus_pt = p * (1 - t) + (1 - p) * t  # pt = p if t > 0 else 1-p
            w = self.alpha * t + (1 - self.alpha) * (
                1 - t)  # w = alpha if t > 0 else 1-alpha
            if self.background_class:
                w[:, 0] = 1 - w[:, 0]
            if self.weight is not None:
                weight = self.weight.to(inputs.device)
                weight = weight.repeat(inputs.shape[0], 1)
                w = w * weight
            all_loss = F.binary_cross_entropy_with_logits(inputs,
                                                          t,
                                                          reduction='none')
            if self.reduction == 'mean':
                return (all_loss * w).mean()
            elif self.reduction == 'sum':
                return (all_loss * w).sum()
            else:
                raise ValueError('wrong redunction')

        elif self.activation == 'softmax':
            log_prob = F.log_softmax(inputs, dim=-1)
            prob = F.softmax(inputs, dim=-1)
            if self.weight is not None:
                weight = self.weight.to(inputs.device)
            else:
                weight = None
            if self.suppress == 'hard':
                prob = 1 - prob
            return F.nll_loss(((1 - prob)**self.gamma) * log_prob,
                              targets,
                              weight=weight,
                              reduction=self.reduction)
        else:
            print(
                f'Unsupported activation type of focal loss: {self.activation}')
            print('We only support "sigmoid" and "softmax" now.')
            assert False