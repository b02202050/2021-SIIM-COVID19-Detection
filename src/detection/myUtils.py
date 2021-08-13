import copy
import functools
import os
import pickle
import random
import sys

import albumentations
import cv2
import elasticdeform
import numpy as np
import torch
import torch.distributed as dist
import torchvision
import utils
import yaml
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = '.'
if base_dir.endswith('classification'):
    base_dir = base_dir.replace('classification', 'detection')

cwd = os.getcwd()
os.chdir(base_dir)

sys.path.append(base_dir)
os.chdir(cwd)

with open(os.path.join(base_dir, 'metadata.yaml'), 'r') as stream:
    metadata = yaml.load(stream, Loader=yaml.FullLoader)


class DetectionDataset():
    """
        This is the base dataset that basically provide an image and a detection target
        
        Args:
            read_label: A module that read the label file specified in the metadata.
                It will be used as
                'labels = read_label(metadata[task_name]['label_file'], split)'.
                This module normally is specified in the metadata too.
                The usage should be
                'exec(f'from {metadata[task_name]['read_label']} import read_label')'
                and pass `read_label` to this argument.
            task_name: the keys in the metadata file.
            split: one of 'train', 'val', or 'test'.
            transforms: This transform takes an PIL RGB image as input.
            more_info: If True, the accesion number will be appended in the return tuple.
            pos_multiplier: duplicate the positive samples for n times.
            return_idx: If True, the idx passed to __getitem__ will be appended in the return tuple.
            transforms2: Do the transform based on the original image.
                If not None, the result image and target will be appended to the return tuple.
            transforms2_1:Do the transform based on the image after doing transforms2.
                If not None, the result image and target will be appended to the return tuple.
            transforms2_2:Do the transform based on the image after doing transforms2.
                If not None, the result image and target will be appended to the return tuple.
            only_class (int): specifying the only class that will be treated as positive. Default is None.
    """

    def __init__(
        self,
        read_label,
        task_name,
        split,
        transforms=None,
        more_info=False,
        return_idx=False,
        preprocess_resize_1024=False,
    ):
        self.task_name = task_name
        self.split = split
        self.transforms = transforms
        self.more_info = more_info
        self.return_idx = return_idx
        self.preprocess_resize_1024 = preprocess_resize_1024
        self.file_extension = metadata[task_name].get('file_extension',
                                                      None) or 'png'
        self.labels = read_label(metadata[task_name]['label_file'], split)
        # self.labels:
        #     [{'ACCNO': 'RA01817105660010',
        #       'labels': {
        #           'boxes': tensor([[ 97., 569., 160., 621.]]),
        #           'labels': tensor([1]),
        #           'image_id': tensor([0]),
        #           'area': tensor([3276.]),
        #           'iscrowd': tensor([0])
        #                 }
        #      },
        #      ...
        #     ]
        check_all_images_exist(metadata[task_name]['image_folder'], self.labels)
        self.exist_file_names = {
            name[:name.find('.')]: name
            for name in os.listdir(metadata[task_name]['image_folder'])
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, target, *other_info = self.get_item_pure(idx)
        if self.preprocess_resize_1024:
            if img.size != (1024, 1024):
                target['boxes'][:, [0, 2]] = target['boxes'][:, [0, 2]] * (
                    1024 / img.size[0])
                target['boxes'][:, [1, 3]] = target['boxes'][:, [1, 3]] * (
                    1024 / img.size[1])
                img = img.resize((1024, 1024))

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return_tuple = tuple([img, target] + other_info)
        return return_tuple

    def get_item_pure(self, idx):
        accno = self.labels[idx]['ACCNO']
        img_file = os.path.join(metadata[self.task_name]['image_folder'],
                                self.exist_file_names[accno])
        img = cv2.imread(img_file)[:, :, ::-1]
        img = Image.fromarray(img)
        target = copy.deepcopy(self.labels[idx]['labels'])
        return_tuple = (img, target)
        if self.more_info:
            return_tuple = return_tuple + (accno,)
        if self.return_idx:
            return_tuple = return_tuple + (idx,)
        return return_tuple


class DetectionGTDataset():

    def __init__(self, detection_dataset, original_img_size, transforms=None):
        self.detection_dataset = detection_dataset
        self.transforms = transforms
        self.black = Image.new('RGB', (original_img_size, original_img_size))
        self.tensor = torch.zeros(3, original_img_size, original_img_size)

    def __len__(self):
        return len(self.detection_dataset)

    def __getitem__(self, idx):
        target = copy.deepcopy(self.detection_dataset.labels[idx]['labels'])
        if self.transforms is not None:
            if isinstance(self.transforms, ToTensor) or (
                    isinstance(self.transforms, Compose) and
                    len(self.transforms.transforms) == 1 and
                    isinstance(self.transforms.transforms[0], ToTensor)):
                return self.tensor, target
            else:
                img, target = self.transforms(self.black, target)
                return img, target
        else:
            return self.black, target


def check_all_images_exist(image_folder, label_data, file_extension='png'):
    label_images = set([x['ACCNO'] for x in label_data])
    all_images = set([x[:x.find('.')] for x in os.listdir(image_folder)])
    if len(label_images) != len(set.intersection(label_images, all_images)):
        raise ValueError(
            f"There are some ACCNO in the label file which cannot find corresponding images in image folder. (Ignoring these labels are not implemented now.) \n Unfound:{label_images-all_images}"
        )
    return


def collate_fn(batch):
    return [list(x) for x in list(zip(*batch))]


def repr_dict(config):
    out_list = []
    for k, v in config.items():
        if isinstance(v, dict):
            v_str = repr_dict(v)
        elif isinstance(v, list) or isinstance(v, tuple) or isinstance(v, set):
            v_str = '_'.join([str(x) for x in v])
        else:
            v_str = str(v)
            if '/' in v_str or '\\' in v_str:
                continue
        k_str = ''.join([ch if is_naming_char(ch) else '_' for ch in str(k)])
        v_str = ''.join([ch if is_naming_char(ch) else '_' for ch in v_str])
        out_list.append(str(k) + '_' + v_str)
    return '_'.join(out_list)


def print_dict_readable_no_br(config):
    readable_str = ''
    name = config['name']
    for k, v in config.items():
        if not (k == 'name'):
            v_str = repr(v) if not isinstance(v, float) else f'{v:.4}'
            print(f"{name}['{k}'] = {v_str}")
            readable_str += f"{name}['{k}'] = {v_str}"
            readable_str += '\n'
    return readable_str




def get_ap(scores, targets):
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
    ap = []
    for i in range(C):
        ap.append(average_precision_score(onehot[:, i], scores[:, i]))
    return torch.tensor(ap)




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
    return torch.tensor(auc)


class Compose():

    def __init__(self, transforms, prob_multiplier=1., aug_on_pos=False):
        self.transforms = transforms
        self.prob_multiplier = prob_multiplier
        self.aug_on_pos = aug_on_pos

    def __call__(self, image, target):
        for t in self.transforms:
            enable_transform = False
            if isinstance(t, ToTensor):
                enable_transform = True
            elif random.random() < self.prob_multiplier:
                if not self.aug_on_pos:
                    enable_transform = True
                elif len(target['boxes']) > 0:
                    enable_transform = True
            if enable_transform:
                image, target = t(image, target)
        return image, target


class ToTensor():

    def __call__(self, image, target):
        image = torchvision.transforms.functional.to_tensor(image)
        return image, target


class FlipH():

    def __call__(self, image, target):
        width, height = image.size
        if random.random() < 0.5:
            image = torchvision.transforms.functional.hflip(image)
            if len(target['boxes']) != 0:
                target['boxes'][:, [0, 2]] = width - target['boxes'][:, [2, 0]]
        return image, target


class FlipV():

    def __call__(self, image, target):
        width, height = image.size
        if random.random() < 0.5:
            image = torchvision.transforms.functional.vflip(image)
            if len(target['boxes']) != 0:
                target['boxes'][:, [1, 3]] = height - target['boxes'][:, [3, 1]]
        return image, target


class Colorjitter():

    def __init__(self, *args, **kwargs):
        self.torchvision_colorjitter = torchvision.transforms.ColorJitter(
            *args, **kwargs)

    def __call__(self, image, target):
        return self.torchvision_colorjitter(image), target


class Blur():

    def __init__(self, *args, **kwargs):
        self.blur_image = BlurClassification(*args, **kwargs)

    def __call__(self, image, target):
        return self.blur_image(image), target


class RandomRotationExpand(object):
    """
        rotate img and bbox
        
        args:
            degree_range: If fixed is False, then this is the value such that
                rotation degree performed each time is smapled from
                [-self.degree_range, self.degree_range]. If fixed is True,
                then this variable should be a list that rotation degree
                performed each time is smapled from this given list.
            fixed: as stated in degree_range.
    """

    def __init__(self, degree_range, fixed=False):
        self.degree_range = degree_range
        self.fixed = fixed

    def __call__(self, image, target):
        if not self.fixed:
            self.degree = random.uniform(-self.degree_range, self.degree_range)
        else:
            self.degree = random.choice(self.degree_range)
        results = self.call_dict_like({
            'image': np.array(image),
            'bboxes': np.array(target['boxes'])
        })
        target['boxes'] = torch.tensor(results['bboxes']).float().reshape(-1, 4)

        return Image.fromarray(results['image']), target

    def call_dict_like(self, results):
        """
        Rotates an image and expands image to avoid cropping
        """

        mat = results['image']
        img_shape = mat.shape
        assert img_shape[0] == img_shape[1]
        img_size = img_shape[0]

        height, width = mat.shape[:2]  # image shape has 3 dimensions
        #print(height, width)
        image_center = (
            width / 2, height / 2
        )  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, self.degree, 1.)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        #print(bound_w, bound_h)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_img = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

        #resize rotated image to img_size
        rotated_img = cv2.resize(
            rotated_img, (img_size, img_size),
            interpolation=cv2.INTER_LINEAR)  # resized, no border

        resize_ratio = img_size / bound_w
        results['image'] = rotated_img

        # rotate bboxes
        bboxes = []
        for dets_ in results['bboxes']:
            #dets_=x1y1x2y2
            upper_left = dets_[:2]
            upper_right = [dets_[2], dets_[1]]
            lower_left = [dets_[0], dets_[3]]
            lower_right = dets_[2:4]
            points = np.concatenate(
                ([upper_left], [upper_right], [lower_left], [lower_right]),
                axis=0)
            #print(points)

            ones = np.ones(shape=(len(points), 1))
            points_ones = np.hstack([points, ones])
            points_new = rotation_mat.dot(points_ones.T).T  # transform points
            points_new = points_new * resize_ratio
            points_new = points_new.flatten()
            #print('points_new:', points_new)

            # get coordinates of modified reference box
            x1_new = min(points_new[0], points_new[2], points_new[4],
                         points_new[6])
            x2_new = max(points_new[0], points_new[2], points_new[4],
                         points_new[6])
            y1_new = min(points_new[1], points_new[3], points_new[5],
                         points_new[7])
            y2_new = max(points_new[1], points_new[3], points_new[5],
                         points_new[7])

            bboxes.append([x1_new, y1_new, x2_new, y2_new])
        bboxes = np.array(bboxes)
        results['bboxes'] = bboxes

        return results

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'degree={0})'.format(self.degree)
        return format_string

    def repr_oneline(self):
        return (f'rtc_{self.degree}')


class RandomShift(torch.nn.Module):
    """
        Random shift the image
        
        Args:
            scale_x: scale to shift the image and bbox horizontally.
                The shift pixel will be chosen from -width * scale_x
                to width * scale_x
            scale_y: (similary to scale_x)
            pad_mode: 'black'/'CONSTANT' or 'repeat'/'WRAP'.
                repeat padding means to fill the blank area with the
                original image .
        
    """

    def __init__(self, scale_x=0.5, scale_y=0.5, pad_mode='CONSTANT'):
        super().__init__()
        if not 0 <= scale_x <= 1 and 0 <= scale_y <= 1:
            raise ValueError('scales should be between 0 and 1')
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.pad_mode = pad_mode

    def forward(self, image, label=None):
        if label is None and isinstance(image, torch.Tensor):
            return self.forward_tensor_image(image)
        return self.forward_pil_with_label(image, label)

    def forward_pil_with_label(self, image, label=None):
        new_image = Image.new(image.mode, image.size)
        paste_rel_x, paste_rel_y = random.random(), random.random()
        shift_x = int(image.size[0] * self.scale_x * (2 * paste_rel_x - 1))
        shift_y = int(image.size[1] * self.scale_y * (2 * paste_rel_y - 1))

        if self.pad_mode in ['black', 'CONSTANT']:
            new_image.paste(image, (shift_x, shift_y))
            # shift the bbox
            if label is not None and len(label['boxes']):
                label['boxes'][:, [0, 2]] += shift_x
                label['boxes'][:, [1, 3]] += shift_y

        elif self.pad_mode in ['repeat', 'WRAP']:
            shift_x_inside = shift_x % image.size[0]
            shift_y_inside = shift_y % image.size[1]
            new_image.paste(image, (shift_x_inside, shift_y_inside))
            new_image.paste(image,
                            (shift_x_inside, shift_y_inside - image.size[1]))
            new_image.paste(image,
                            (shift_x_inside - image.size[0], shift_y_inside))
            new_image.paste(image, (shift_x_inside - image.size[0],
                                    shift_y_inside - image.size[1]))
            # shift the bbox
            if label is not None:
                len_boxes = len(label['boxes'])
                if len_boxes:
                    label['boxes'] = label['boxes'].repeat(4, 1)
                    label['labels'] = label['labels'].repeat(4)
                    label['area'] = label['area'].repeat(4)
                    label['iscrowd'] = label['iscrowd'].repeat(4)

                    label['boxes'][:, [0, 2]] += shift_x_inside
                    label['boxes'][:, [1, 3]] += shift_y_inside
                    label['boxes'][len_boxes:len_boxes * 2,
                                   [0, 2]] -= image.size[0]
                    label['boxes'][len_boxes * 2:len_boxes * 3,
                                   [1, 3]] -= image.size[1]
                    label['boxes'][len_boxes * 3:, [0, 2]] -= image.size[0]
                    label['boxes'][len_boxes * 3:, [1, 3]] -= image.size[1]

        # clamp the outsider bbox
        if label is not None:
            if len(label['boxes']):
                label['boxes'][:, [0, 2]] = label['boxes'][:, [0, 2]].clamp(
                    0, image.size[0])
                label['boxes'][:, [1, 3]] = label['boxes'][:, [1, 3]].clamp(
                    0, image.size[1])
                label['area'] = ((label['boxes'][:, 2] - label['boxes'][:, 0]) *
                                 (label['boxes'][:, 3] - label['boxes'][:, 1]))
                remain_list = [
                    i for i in range(len(label['boxes']))
                    if label['area'][i] >= 10.
                ]
                for k in ['boxes', 'labels', 'area', 'iscrowd']:
                    label[k] = label[k][remain_list]
            return new_image, label
        return new_image

    def forward_tensor_image(self, image):
        paste_rel_x, paste_rel_y = random.random(), random.random()

        ## torch style
        shift_x = int(image.shape[-1] * self.scale_x * (2 * paste_rel_x - 1))
        shift_y = int(image.shape[-2] * self.scale_x * (2 * paste_rel_x - 1))
        if self.pad_mode in ['black', 'CONSTANT']:
            mosaic = torch.zeros(*(image.shape[:-2]),
                                 image.shape[-2] * 3,
                                 image.shape[-1] * 3,
                                 dtype=image.dtype)
            mosaic[..., image.shape[-2]:image.shape[-2] * 2,
                   image.shape[-1]:image.shape[-1] * 2] = image
        elif self.pad_mode in ['repeat', 'WRAP']:
            mosaic = image.repeat(*[1 for _ in image.shape[:-2]], 3, 3)
        image = mosaic[...,
                       image.shape[-2] - shift_y:image.shape[-2] * 2 - shift_y,
                       image.shape[-1] - shift_x:image.shape[-1] * 2 - shift_x]
        return image


class ElasticDeform():
    """
        Perform elastic deformation to the image and the target
        correspondingly.   
    """

    def __init__(self, relative_sigma=1 / 40, points=5, order=0):
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

    def __call__(self, image, label):
        np_img = np.array(image)  # (H, W, C)
        height, width, channel = np_img.shape

        # construct label images to be deformed together with the real images.
        n_label = len(label['labels'])
        if n_label > 0:
            label_img = np.zeros((height, width, n_label), dtype=np.uint8)
            for i, box in enumerate(label['boxes']):
                box_list = box.int().tolist()
                label_img[box_list[1]:box_list[3], box_list[0]:box_list[2],
                          i] = 255
            img_to_deform = np.concatenate((np_img, label_img), axis=2)
        else:
            img_to_deform = np_img

        # deform!
        deformed_img = elasticdeform.deform_random_grid(img_to_deform,
                                                        sigma=width *
                                                        self.relative_sigma,
                                                        points=self.points,
                                                        order=self.order,
                                                        axis=(0, 1))

        # deformed image
        np_img_deformed = deformed_img[:, :, :channel]
        pil_img_deformed = Image.fromarray(np_img_deformed)

        # deformed label
        if n_label > 0:

            # Note that we should consider that some label deforms to the place
            # which is totally outside the image.
            label_removal_list = []
            for i in range(n_label):
                np_label_deformed = deformed_img[:, :, channel + i]
                label_coordinates = np.where(np_label_deformed > 0)
                if len(label_coordinates[0]) >= 10:
                    label['boxes'][i, 0] = int(label_coordinates[1].min())
                    label['boxes'][i, 2] = int(label_coordinates[1].max() + 1)
                    label['boxes'][i, 1] = int(label_coordinates[0].min())
                    label['boxes'][i, 3] = int(label_coordinates[0].max() + 1)
                else:
                    label_removal_list.append(i)
            if label_removal_list:
                label_retain_list = [
                    i for i in range(n_label) if not i in label_removal_list
                ]
                label['boxes'] = label['boxes'][label_retain_list]
                label['labels'] = label['labels'][label_retain_list]
                label['iscrowd'] = label['iscrowd'][label_retain_list]
                label['area'] = label['area'][label_retain_list]

        return pil_img_deformed, label


class AlbumentationTransforms():
    """
        Quikly apply transformations of the package albumentations.
    """

    def __init__(self, trnasform_name, transform_parameters_dict=None):
        """
            Args:
                trnasform_name (str): The class name of albumentations
                    transformation.
                transform_parameters_dict (dict): The parameters to be
                    passed to the albumentations transformation.
        """
        if transform_parameters_dict is None:
            transform_parameters_dict = dict()
        self.transform = albumentations.Compose(
            [
                getattr(albumentations,
                        trnasform_name)(**transform_parameters_dict)
            ],
            bbox_params=albumentations.BboxParams(format='pascal_voc',
                                                  label_fields=['field_list'],
                                                  min_area=1.),
        )

    def __call__(self, image, label):
        # Prepare input image
        w, h = image.size
        np_img = np.array(image)

        # Prepare input label
        list_label = {
            k: v.tolist()
            for k, v in label.items()
            if k not in ['image_id', 'boxes']
        }
        field_list = [{
            k: v[i:i + 1]
            for k, v in label.items()
            if k not in ['image_id', 'boxes']
        }
                      for i in range(len(label['labels']))]

        # Perform transformation
        bbox = label['boxes']
        if len(bbox):
            bbox[:, ::2] = bbox[:, ::2].clamp(0., w)
            bbox[:, 1::2] = bbox[:, 1::2].clamp(0., h)
        transformed = self.transform(image=np_img,
                                     bboxes=bbox.tolist(),
                                     field_list=field_list)

        # Collate output image
        transformed_image = Image.fromarray(transformed['image'])

        # Collate output label
        if len(transformed['field_list']):
            transformed_label = {
                k: torch.cat(
                    [each_label[k] for each_label in transformed['field_list']])
                for k in transformed['field_list'][0].keys()
            }
            transformed_label['boxes'] = torch.tensor(
                transformed['bboxes']).reshape(-1, 4)
            transformed_label['image_id'] = label['image_id']
        else:
            transformed_label = create_empty_label(label['image_id'])

        return transformed_image, transformed_label


def is_naming_char(ch):
    if ch >= '0' and ch <= '9':
        return True
    if ch >= 'a' and ch <= 'z':
        return True
    if ch >= 'A' and ch <= 'Z':
        return True
    if ch == '_':
        return True
    return False


def load_state_dict_reviser_not_strict(func):

    def load_state_dict_not_strict(*args, **kwargs):
        if not 'strict' in kwargs:
            kwargs['strict'] = False
        return func(*args, **kwargs)

    return load_state_dict_not_strict


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
    local_size = torch.tensor([tensor.numel()], device="cuda")  # pylint: disable=not-callable
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]  # pylint: disable=not-callable
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
    if [
            i for i, x in enumerate(args)
            if not is_tensor[i] and not isinstance(x, list)
    ]:
        raise ValueError(
            f'Unable to gather {type(args[i])} datas ({i}-th data).')
    if is_tensor[-1]:
        all_img_ids = all_gather(args[-1])
        merged_img_ids = []
        for p in all_img_ids:
            merged_img_ids.extend(p.tolist())
    else:
        merged_img_ids = functools.reduce(lambda x, y: (x + y),
                                          utils.all_gather(args[-1]))
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
            receive = functools.reduce(lambda x, y: (x + y),
                                       utils.all_gather(arg))
            receive = [receive[j] for j in idx]
        receives.append(receive)

    dist.barrier()
    return receives


class FrozenBatchNorm2dWithEpsilon(torchvision.ops.misc.FrozenBatchNorm2d):
    """This class aims to make the epsilon consistent with the default value of torch.nn.BatchNorm2d"""

    def __init__(self, *args, **kwargs):
        if 'eps' not in kwargs:
            kwargs['eps'] = 1e-5
        super().__init__(*args, **kwargs)


class SAMOptim(torch.optim.Optimizer):
    """Code credit: https://github.com/davda54/sam/blob/main/sam.py
    This is the implementation of the paper: "Sharpness-Aware Minimization for Efficiently Improving Generalization"
    Revision with weight normalization.
    """

    def __init__(self,
                 params,
                 base_optimizer,
                 rho=0.05,
                 weight_norm=False,
                 **kwargs):
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
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                if group["weight_norm"]:
                    e_w *= p / (weight_norm + 1e-12)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad=False, update=True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        if update:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None and self.state[p][
                            'accumulate_grad'] is not None:
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
        norm = torch.norm(torch.stack([
            p.grad.norm(p=2)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]),
                          p=2)
        return norm

    def _weight_norm(self):
        norm = torch.norm(torch.stack([
            p.norm(p=2)
            for group in self.param_groups
            for p in group["params"]
            if p is not None
        ]),
                          p=2)
        return norm


def create_empty_label(idx):
    labels = {}
    bbox_tensor = torch.tensor([]).reshape(-1, 4)
    labels['boxes'] = bbox_tensor
    labels['labels'] = torch.tensor([]).long()
    labels['image_id'] = torch.tensor([idx])
    labels['area'] = torch.tensor([])
    labels['iscrowd'] = torch.tensor([]).long()
    return labels
