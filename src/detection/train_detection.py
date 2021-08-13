""" File to train a FasterRCNN object detector. """
import os
import argparse
import datetime
import math
import re
import subprocess
import sys
import time
import copy
from tqdm import tqdm
import functools
import pprint
from collections import OrderedDict

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import (BackboneWithFPN,
                                                         resnet_fpn_backbone)
from torchvision.models.detection.faster_rcnn import (FastRCNNPredictor,
                                                      model_urls)
from torchvision.models.utils import load_state_dict_from_url
from torchvision.ops import misc as misc_nn_ops

import deterministic_setting # pylint: disable=unused-import
import myUtils
from module import fpn as my_fpn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
import timm
from timm_backbone_wrapper import timm_fpn_backbone

class FasterRCNNCustomizedForward(FasterRCNN):
    """ 
        This class aims to provide a convenient forward funciton for gradient
        synchronization when distributed training is used.
    """
    def forward(self, *args, **kwargs):
        if not 'key' in kwargs:
            return super().forward(*args, **kwargs)
        else:
            key = kwargs.pop('key')
            if isinstance(key, str):
                return getattr(self, key)(*args, **kwargs)
            elif isinstance(key, list):
                return functools.reduce(getattr, [self] + key)(*args, **kwargs)


def init_cuda_and_distributed(args_dict):
    # config cuda and distributed training
    os.environ["CUDA_VISIBLE_DEVICES"] = args_dict['CUDA_VISIBLE_DEVICES']
    parser = argparse.ArgumentParser()
    # distributed training parameters
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url',
                        default='env://',
                        help='url used to set up distributed training')

    args = parser.parse_args('')
    utils.init_distributed_mode(args)
    return args


def create_args_dict():
    args_dict = dict()
    # environment configs
    args_dict['CUDA_VISIBLE_DEVICES'] = '0'
    args_dict['tag'] = ''
    
    # data config
    args_dict['task_name'] = 'pneumonia_RSNA'
    args_dict['output_dir'] = f"work_dir/{args_dict['task_name']}_run1" # work_dir, debug
    args_dict['input_nc'] = 3
    
    # model config
    args_dict['input_size'] = 800 # 800
    args_dict['normalize_mean'] = None # None
    args_dict['normalize_std'] = None # None
    args_dict['backbone_body'] = 'tf_efficientnet_b7_ns' 
    args_dict['rpn_bg_iou_thresh'] = None
    args_dict['norm_layer'] = myUtils.FrozenBatchNorm2dWithEpsilon 
    args_dict['RoI_size_feature'] = False
    args_dict['rpn_nms_thresh'] = None 
    args_dict['relu_activation'] = 'relu' 
    args_dict['ACFPN'] = True
    args_dict['iAFF'] = True
    args_dict['RoI_type'] = 'fc' 
    args_dict['rpn_dropout'] = False
    args_dict['dropout_fc_layer'] = False
    args_dict['probabilistic_class'] = False
    args_dict['loss_type_uncertainty'] = False
    
    # # ! train_from_scratch, finetune and backbone_pretrained are multually exclusive !
    args_dict['train_from_scratch'] = False
    args_dict['fix_backbone_body'] = False
    args_dict['unfix_backbone_body'] = False
    args_dict['fine_tune'] = False
    args_dict['load_optim'] = False
    args_dict['fine_tune_model_file'] = ''
    args_dict['fine_tune_load_without_roi_head'] = False
    args_dict['fine_tune_load_without_rpn_head'] = False
    args_dict['fine_tune_load_multitask_head'] = False
    args_dict['fine_tune_non_strict'] = False
    args_dict['fine_tune_ratio'] = 1.
    args_dict['user_specified_backbone_pretrained'] = True
    args_dict['user_specified_backbone_pretrained_with_FPN'] = False
    args_dict['backbone_pretrained_file'] = '../classification/pretrained/pretrained_run1_transferred.pth'
    args_dict['use_imagenet_backbone'] = False
    args_dict['sync_bn'] = False
    args_dict['AGC'] = False # Adaptive gradient clipping
    
    
    # training config    
    args_dict['num_epochs'] = 20
    args_dict['batch_size'] = 8 
    args_dict['aug_sample_multiplier'] = 1
    args_dict['accumulate_grad_batches'] = 1
    args_dict['optim'] = 'torch.optim.Adam' 
    args_dict['amsgrad'] = True
    args_dict['learning_rate'] = 1e-4
    args_dict['weight_decay'] = 0 
    args_dict['multistep_lr_scheduler'] = False
    args_dict['lr_steps'] = [20,]
    args_dict['amp'] = True
    args_dict['num_workers'] = 4
    args_dict['SWA'] = True
    args_dict['SWA_lr_divider'] = 1. 
    args_dict['SWA_start_epoch'] = 0 
    args_dict['SWA_anneal_epoch'] = 20
    args_dict['SAM'] = True
    args_dict['fast_evaluate'] = True # need to ensure the same size of each testing image
    args_dict['img_size'] = 1024

    ## augmentation config
    args_dict['augmentation_prob_multiplier'] = 0.5
    args_dict['train_positive_multiplier'] = 1
    args_dict['aug_on_pos'] = False
    args_dict['random_window'] = False
    args_dict['blur'] = False
    args_dict['color_jitter'] = True
    args_dict['color_jitter_brightness'] = 0.4
    args_dict['color_jitter_contrast'] = 0.4
    args_dict['flip'] = True
    args_dict['flip_vertical'] = True
    args_dict['rotation'] = True
    args_dict['rotation_fixed_angle'] = False
    args_dict['rotation_degree'] = 25
    args_dict['rotation_mode'] = 'WRAP'
    args_dict['shuffle_box'] = False
    args_dict['flip_box'] = False
    args_dict['remove_box'] = False
    args_dict['shift_box_outward'] = False
    args_dict['shift'] = True
    args_dict['shift_mode'] = 'WRAP' 
    args_dict['elastic_deformation'] = True
    args_dict['elastic_deformation_sigma'] = 1/40
    args_dict['random_perspective'] = True
    args_dict['RandomResizedCrop'] = True
    
    try:
        args_dict['execution_file'] = os.path.basename(__file__)
    except:
        pass
    
    return args_dict


def get_loss_dict(model, images, targets, args_dict):
    loss_dict = model(images, targets)
    return loss_dict


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, # pylint: disable=redefined-outer-name
                    scaler, args_dict, caller_vars={}): # pylint: disable=redefined-outer-name
    """
        Train a model for one epoch.
        This function is copied and revised from
        github.com/pytorch/vision/blob/master/references/detection/engine.py
    """
        
    safe_state = True
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler_in_epoch = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler_in_epoch = utils.warmup_lr_scheduler(optimizer, warmup_iters,
                                                          warmup_factor)

    for n, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if n >= len(data_loader):
            break
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args_dict['SAM']:
            # First Step
            with torch.cuda.amp.autocast(enabled=args_dict['amp']):
                loss_dict = get_loss_dict(model, images, targets, args_dict)
                losses = sum(loss for loss in loss_dict.values()) 
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            if args_dict['AGC']:
                timm.utils.agc.adaptive_clip_grad([param for module in model.modules() if not isinstance(module, torch.nn.Linear) for param in module.parameters()])            
            scaler.step(optimizer, first_step=True)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # Second Step
            with torch.cuda.amp.autocast(enabled=args_dict['amp']):
                loss_dict = get_loss_dict(model, images, targets, args_dict)
                losses = sum(loss for loss in loss_dict.values())
                losses /= args_dict['accumulate_grad_batches']
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            if args_dict['AGC']:
                timm.utils.agc.adaptive_clip_grad(
                    [param for module in model.modules() if not isinstance(module, torch.nn.Linear) for param in module.parameters()],
                )                
            scaler.step(optimizer, first_step=False, update=(n % args_dict['accumulate_grad_batches'] == 0))
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            with torch.cuda.amp.autocast(enabled=args_dict['amp']):
                loss_dict = get_loss_dict(model, images, targets, args_dict)
                losses = sum(loss for loss in loss_dict.values())
                losses /= args_dict['accumulate_grad_batches']
            scaler.scale(losses).backward()
            if n % args_dict['accumulate_grad_batches'] == 0:
                scaler.unscale_(optimizer)
                if args_dict['AGC']:
                    timm.utils.agc.adaptive_clip_grad([param for module in model.modules() if not isinstance(module, torch.nn.Linear) for param in module.parameters()])                                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            assert False, "Loss is NaN"
        
        if args_dict['SWA'] and epoch >= args_dict['SWA_start_epoch']:
            caller_vars['swa_model'].update_parameters(caller_vars['model_without_ddp'])

        if lr_scheduler_in_epoch is not None:
            lr_scheduler_in_epoch.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    return safe_state


@torch.no_grad()
def evaluate(model, data_loader, device, args_dict): # pylint: disable=redefined-outer-name
    """
        To evaluate the model with a data loader.
        This function is copied and revised from
        github.com/pytorch/vision/blob/master/references/detection/engine.py
    """
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if args_dict['fast_evaluate']:
        coco = get_coco_api_from_dataset(
            myUtils.DetectionGTDataset(data_loader.dataset,
                                       args_dict['img_size'], data_loader.dataset.transforms))
    else:
        coco = get_coco_api_from_dataset(data_loader.dataset)
    #iou_types = _get_iou_types(model)
    iou_types = ['bbox']
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 10, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, outputs)
        }
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time,
                             evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


def train(args_dict, args):
    
    if args_dict['SWA']:
        from torch.optim.swa_utils import AveragedModel, SWALR
        
    args_dict['color_jitter_factor'] = {'brightness': args_dict['color_jitter_brightness'], 'contrast': args_dict['color_jitter_contrast']}
    
    if args_dict['ACFPN'] or args_dict['iAFF']:
        torchvision.models.detection.backbone_utils.FeaturePyramidNetwork = my_fpn.ModulerFPN
        timm_fpn_backbone.FeaturePyramidNetwork = my_fpn.ModulerFPN
        
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    gpu_info = subprocess.run(
        'nvidia-smi --query-gpu=index,name,driver_version --format=csv',
        shell=True,
        stdout=subprocess.PIPE,
        check=True).stdout

    if args_dict['user_specified_backbone_pretrained']:
        if args_dict['fine_tune']:
            raise ValueError(
                'fine_tune means loading a whole pretrained network instead of backbone.'
            )
        checkpoint = torch.load(args_dict['backbone_pretrained_file'],
                                map_location='cpu')
        if not checkpoint.get('strict_load', False):
            torch.nn.Module.load_state_dict = myUtils.load_state_dict_reviser_not_strict(
                torch.nn.Module.load_state_dict)
        print(checkpoint.keys())
        args_dict['normalize_mean'] = checkpoint['normalize_mean'] if 'normalize_mean' in checkpoint else checkpoint['norm_mean']
        args_dict['normalize_std'] = checkpoint['normalize_std'] if 'normalize_std' in checkpoint else checkpoint['norm_std']
    elif args_dict['fine_tune']:
        checkpoint = torch.load(args_dict['fine_tune_model_file'],
                                map_location='cpu')
        try:
            args_dict['input_nc'] = checkpoint['args_dict']['input_nc']
        except:
            args_dict['input_nc'] = 3
        try:
            args_dict['normalize_mean'] = checkpoint['args_dict']['normalize_mean']
            args_dict['normalize_std'] = checkpoint['args_dict']['normalize_std']
        except:
            args_dict['normalize_mean'] = None
            args_dict['normalize_std'] = None
        try:
            args_dict['input_size'] = checkpoint['args_dict']['input_size']
        except:
            pass

    if args_dict['normalize_mean'] is not None and len(
            args_dict['normalize_mean']) != args_dict['input_nc']:
        raise ValueError(
            f"length of args_dict['normalize_mean'] {len(args_dict['normalize_mean'])} and args_dict['input_nc'] {args_dict['input_nc']} do not match!"
        )

    output_dir = args_dict['output_dir']
    if utils.is_main_process():
        utils.mkdir(output_dir)

    # prepare log file
    log_mAP_file = os.path.join(output_dir, 'log_mAP.txt') # pylint: disable=invalid-name
    config_file = os.path.join(output_dir, 'config.txt')

    # print info
    args_dict['name'] = 'args_dict'
    print(f'{args_dict["name"]}:  ')
    print('\n<details><summary>Training configs</summary><pre>')
    #readable_str = pprint.pformat(args_dict, sort_dicts=False)
    #print(readable_str)
    readable_str = myUtils.print_dict_readable_no_br({k: (v if k != 'norm_layer' else str(v)) for k, v in args_dict.items()})
    print('\nwork directory: \n' + output_dir + '\n')
    print('gpu_info:')
    print(gpu_info.decode('utf-8'))
    print('</pre></details>')
    print('<details><summary>Learning curve</summary>\n')
    print('</details>')
    if utils.is_main_process():
        with open(config_file, 'w') as file:
            file.write(readable_str)
    
    # setup transforms
    transform_train_list = []
    if args_dict['remove_box']:
        transform_train_list.append(myUtils.RandomRemoveBox(prob=.2))

    if args_dict['flip_box']:
        transform_train_list.append(myUtils.RandomFlipBoxes())
    if args_dict['shuffle_box']:
        transform_train_list.append(myUtils.RandomShuffleGT())
    if args_dict['shift_box_outward']:
        transform_train_list.append(myUtils.RandomShiftBoxOutward())
    if args_dict['RandomResizedCrop']:
        transform_train_list.append(myUtils.AlbumentationTransforms('RandomResizedCrop', dict(height=1024, width=1024)))
    if args_dict['shift']:
        transform_train_list.append(
            myUtils.RandomShift(pad_mode=args_dict['shift_mode'],
                                **({'scale_x': args_dict['shift_scale'], 'scale_y': args_dict['shift_scale']} if 'shift_scale' in args_dict else {})
                               )
        )
    if args_dict['random_window']:
        transform_train_list.append(myUtils.RandomWindow(scale=0.7))
    if args_dict['blur']:
        transform_train_list.append(myUtils.Blur(scale=3e-3))
    if args_dict['color_jitter']:
        transform_train_list.append(
            myUtils.Colorjitter(
                brightness=args_dict['color_jitter_factor']['brightness'],
                contrast=args_dict['color_jitter_factor']['contrast'],
                saturation=0,
                hue=0))
    if args_dict['flip']:
        transform_train_list.append(myUtils.FlipH())
    if args_dict['flip_vertical']:
        transform_train_list.append(myUtils.FlipV())
    if args_dict['rotation']:
        if args_dict['rotation_mode'] == 'CONSTANT':
            transform_train_list.append(
                myUtils.RandomRotationExpand(args_dict['rotation_degree'],
                                             fixed=args_dict['rotation_fixed_angle']))
        elif args_dict['rotation_mode'] == 'WRAP':
            assert args_dict['rotation_fixed_angle'] == False, "fixed angle rotation with WRAP border mode is not implemented."
            transform_train_list.append(myUtils.AlbumentationTransforms('Rotate', dict(limit=args_dict['rotation_degree'], border_mode=3)))
    if args_dict['random_perspective']:
        transform_train_list.append(myUtils.AlbumentationTransforms('Perspective', {'p': 1.}))
    if args_dict['elastic_deformation']:
        transform_train_list.append(myUtils.ElasticDeform(relative_sigma=args_dict['elastic_deformation_sigma']))
    transform_train = myUtils.Compose(
        transform_train_list,
        prob_multiplier=args_dict['augmentation_prob_multiplier'],
        aug_on_pos=args_dict['aug_on_pos'],
    )
    transform_train_list = [transform_train, myUtils.ToTensor()]
    transform_train = myUtils.Compose(transform_train_list)

    transform_test_list = [myUtils.ToTensor()]
    transform_test = myUtils.Compose(transform_test_list)

    # load data
    print("Loading data")
    exec( # pylint: disable=exec-used
        f"from {myUtils.metadata[args_dict['task_name']]['read_label']} import read_label",
        globals()
    )

    dataset_train = myUtils.DetectionDataset(
        read_label,
        args_dict['task_name'],
        'train',
        transforms=transform_train,
        pos_multiplier=args_dict['train_positive_multiplier'],
    )
    if args_dict['fine_tune'] and args_dict['fine_tune_ratio'] != 1:
        torch.manual_seed(42)
        dataset_train = torch.utils.data.Subset(
            dataset_train,
            torch.randperm(len(dataset_train))
            [:int(len(dataset_train) * args_dict['fine_tune_ratio'])].tolist())

    dataset_val = myUtils.DetectionDataset(
        read_label,
        args_dict['task_name'],
        'val',
        transforms=transform_test,
    )
    dataset_train_eval = myUtils.DetectionDataset(
        read_label,
        args_dict['task_name'],
        'train',
        transforms=transform_test,
    )
    

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val,
                                                                      shuffle=False)
        train_eval_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train_eval, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        val_sampler = torch.utils.data.SequentialSampler(dataset_val)
        train_eval_sampler = torch.utils.data.SequentialSampler(dataset_train_eval)
    if args_dict['aug_sample_multiplier'] != 1:
        train_sampler = myUtils.sample_multiplier(train_sampler, args_dict['aug_sample_multiplier'])
    
    train_collate = myUtils.collate_fn
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args_dict['batch_size'],
        sampler=train_sampler,
        num_workers=args_dict['num_workers'],
        collate_fn=train_collate)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args_dict['batch_size'],
        sampler=val_sampler,
        num_workers=args_dict['num_workers'],
        collate_fn=myUtils.collate_fn)
    data_loader_train_eval = torch.utils.data.DataLoader(
        dataset_train_eval,
        batch_size=args_dict['batch_size'],
        sampler=train_eval_sampler,
        num_workers=args_dict['num_workers'],
        collate_fn=myUtils.collate_fn)

    # create model
    if args_dict['normalize_mean'] is None:
        args_dict['normalize_mean'] = [0.485, 0.456, 0.406]
        args_dict['normalize_std'] = [0.229, 0.224, 0.225]
    if args_dict['input_nc'] == 1:
        if len(args_dict['normalize_mean']) == 1:
            org_norm_mean = args_dict['normalize_mean'] * 3
            org_norm_std = args_dict['normalize_std'] * 3
        elif len(args_dict['normalize_mean']) == 3:
            org_norm_mean = args_dict['normalize_mean']
            org_norm_std = args_dict['normalize_std']
            args_dict['normalize_mean'] = [0.5]
            args_dict['normalize_std'] = [0.5]
    else:
        org_norm_mean, org_norm_std = None, None
    print("Creating model")
    if args_dict['backbone_body'] in timm_fpn_backbone.timm_backbone_body_default_params:
        pretrained=(not args_dict['train_from_scratch'] and
                    not args_dict['fine_tune'] and
                    not args_dict['user_specified_backbone_pretrained']
                   )
        backbone = timm_fpn_backbone.timm_fpn_backbone(
            args_dict['backbone_body'],
            pretrained=pretrained,
            input_nc=args_dict['input_nc'],
            weights=checkpoint['model']
            if args_dict['user_specified_backbone_pretrained'] and not args_dict['user_specified_backbone_pretrained_with_FPN'] else None,
            unfix_backbone_body=args_dict['unfix_backbone_body'],
            norm_layer=args_dict['norm_layer'] or misc_nn_ops.FrozenBatchNorm2d,
            org_norm_mean=org_norm_mean, 
            org_norm_std=org_norm_std,
            new_norm_mean=args_dict['normalize_mean'],
            new_norm_std=args_dict['normalize_std'],
        )
        if pretrained and args_dict['input_nc'] == 3 and not (args_dict['user_specified_backbone_pretrained'] or args_dict['user_specified_backbone_pretrained_with_FPN'] or args_dict['fine_tune']):
            args_dict['normalize_mean'] = list(backbone.body.timm_model.default_cfg['mean'])
            args_dict['normalize_std'] = list(backbone.body.timm_model.default_cfg['std'])
    if args_dict['user_specified_backbone_pretrained'] and args_dict['user_specified_backbone_pretrained_with_FPN']:
        msg = backbone.load_state_dict(checkpoint['model'])
        print(f"Backbone with FPN weights loading status: {msg}")
    
    model_kwargs = {}
    model_kwargs['min_size'] = args_dict['input_size']
    model_kwargs['max_size'] = args_dict['input_size']
    model_kwargs['image_mean'] = args_dict['normalize_mean']
    model_kwargs['image_std'] = args_dict['normalize_std']
    if args_dict['rpn_nms_thresh'] is not None:
        model_kwargs['rpn_nms_thresh'] = args_dict['rpn_nms_thresh']
    if model_kwargs.get('rpn_bg_iou_thresh', False):
        model_kwargs['rpn_bg_iou_thresh'] = args_dict['rpn_bg_iou_thresh']
    if args_dict.get('rpn_focal_loss', False):
        torchvision.models.detection.rpn.RegionProposalNetwork.compute_loss = myUtils.my_rpn_compute_loss
        if args_dict.get('rpn_box_focal_loss', False):
            myUtils.rpn_box_focal_loss = True
    if args_dict.get('roi_focal_loss', False):
        torchvision.models.detection.roi_heads.fastrcnn_loss = myUtils.fastrcnn_loss_focal_loss
        if args_dict.get('roi_box_focal_loss', False):
            myUtils.roi_box_focal_loss = True


    if not args_dict['use_imagenet_backbone'] and not args_dict[
            'fine_tune'] and not args_dict[
                'user_specified_backbone_pretrained'] and not args_dict[
                    'train_from_scratch'] and args_dict[
                        'input_nc'] == 3 and args_dict[
                            'backbone_body'] == 'resnet50':
        model_kwargs['num_classes'] = 91  # in order to load coco pretrained weight
        model = FasterRCNNCustomizedForward(backbone, **model_kwargs)
        state_dict = load_state_dict_from_url(
            model_urls['fasterrcnn_resnet50_fpn_coco'], progress=True)
        model.load_state_dict(state_dict)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, myUtils.metadata[args_dict['task_name']]['num_classes'])
    else:
        model_kwargs['num_classes'] = (
            myUtils.metadata[args_dict['task_name']]['num_classes'])
        model = FasterRCNNCustomizedForward(backbone, **model_kwargs)
    
    if args_dict['ACFPN']:
        model.backbone.fpn.setup_ACFPN()
    if args_dict['iAFF']:
        model.backbone.fpn.setup_iAFF()
        
    print(model)
    if args_dict['fine_tune']:
        state_dict_to_load = checkpoint['model']
        if args_dict['fine_tune_load_without_roi_head']:
            state_dict_to_load = {k: v for k, v in state_dict_to_load.items() if 'roi_heads.box_predictor' not in k}
        if args_dict['fine_tune_load_without_rpn_head']:
            state_dict_to_load = {k: v for k, v in state_dict_to_load.items() if 'rpn.head.cls_logits' not in k and 'rpn.head.bbox_pred' not in k}
        if args_dict['fine_tune_load_multitask_head']:
            state_dict_to_load = {k.replace('_multi_task.' + args_dict['task_name'], ''): v for k, v in state_dict_to_load.items() if not '_multi_task' in k or ('_multi_task.' + args_dict['task_name']) in k}
        print('Model loading status:', model.load_state_dict(state_dict_to_load, strict=not args_dict['fine_tune_non_strict']))       
    model.to(device)
    
    # create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if args_dict['optim'] == 'torch.optim.SGD':
        optim_class = torch.optim.SGD
        optim_params = dict(lr=args_dict['learning_rate'],
                            momentum=.9,
                            weight_decay=args_dict['weight_decay'])
    elif args_dict['optim'] == 'torch.optim.Adam':
        optim_class = torch.optim.Adam
        optim_params = dict(lr=args_dict['learning_rate'],
                            weight_decay=args_dict['weight_decay'],
                            amsgrad=args_dict['amsgrad'])

    if args_dict['SAM']:
        optimizer = myUtils.SAMOptim(params, optim_class, **optim_params)
    else:
        optimizer = optim_class(params, **optim_params)
        
    lr_scheduler = None
    if args_dict['multistep_lr_scheduler']:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args_dict['lr_steps'], gamma=.1)
    if args_dict['fine_tune'] and args_dict['load_optim']:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.distributed and args_dict['sync_bn']:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    scaler = torch.cuda.amp.GradScaler(enabled=args_dict['amp'])

    # SWA
    if args_dict['SWA']:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, anneal_epochs=args_dict['SWA_anneal_epoch'], swa_lr=args_dict['learning_rate'] / args_dict['SWA_lr_divider'])
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.gpu])
        model_without_ddp = model.module

    # training
    print("Start training")
    start_time = time.time()
    
    for epoch in range(args_dict['num_epochs']):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        safe_state = train_one_epoch(model, optimizer, data_loader_train, device, epoch, 10, scaler, args_dict, caller_vars=locals())
        if not safe_state:
            return 0.
        
        if args_dict['SWA'] and epoch >= args_dict['SWA_start_epoch']:
            coco_evaluator_val = evaluate(model, data_loader_val, device=device, args_dict=args_dict)
            val_mAP = coco_evaluator_val.coco_eval['bbox'].eval['precision'][0, :, :, 0, 2].mean()
            swa_model.update_parameters(model_without_ddp, val_mAP)
            swa_scheduler.step()
        elif lr_scheduler is not None:
            lr_scheduler.step()
        print("Start evaluation")
        if args_dict['SWA'] and epoch >= args_dict['SWA_start_epoch']:
            coco_evaluator_val = evaluate(swa_model, data_loader_val, device=device, args_dict=args_dict)
        else:
            coco_evaluator_val = evaluate(model, data_loader_val, device=device, args_dict=args_dict)
        # T:iou(0.5:0.95:0.05), R:recall(0:1:0.01), K:cls, A:area(all,s,m,l),M:maxdet(1,10,100)
        val_mAP = coco_evaluator_val.coco_eval['bbox'].eval['precision'][0, :, :, 0,
                                                                         2].mean()
        print(f'val_mAP: {val_mAP:.7f}')


        if epoch == 0 or epoch % 5 == 4:
            if args_dict['SWA'] and epoch >= args_dict['SWA_start_epoch']:
                coco_evaluator_train_eval = evaluate(swa_model,
                                                     data_loader_train_eval,
                                                     device=device,
                                                     args_dict=args_dict)
            else:
                coco_evaluator_train_eval = evaluate(model,
                                                     data_loader_train_eval,
                                                     device=device,
                                                     args_dict=args_dict)
            train_eval_mAP = coco_evaluator_train_eval.coco_eval['bbox'].eval[
                'precision'][0, :, :, 0, 2].mean()
            print(f'train_eval_mAP: {train_eval_mAP:.7f}')

        if utils.is_main_process():
            checkpoint = {
                'model':
                    model_without_ddp.state_dict() if (not args_dict['SWA'] or epoch < args_dict['SWA_start_epoch']) else swa_model.module.state_dict(),
                'lr_scheduler':
                    lr_scheduler.state_dict() if lr_scheduler is not None else None,
                'epoch':
                    epoch,
                'args_dict':
                    {k: (v if k != 'norm_layer' else str(v)) for k, v in args_dict.items()},
                'val_mAP':
                    val_mAP,
                'training_time':
                    str(datetime.timedelta(seconds=int(time.time() - start_time))),
                'env': {
                    'gpu_info': gpu_info.decode('ascii'),
                    'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
                }
            }
            metric_value = val_mAP
            if epoch == 0 or metric_value > best_metric_value:
                utils.save_on_master(checkpoint, os.path.join(output_dir, 'model_best.pth'))
                best_metric_value = metric_value
                best_epoch = epoch
    
            utils.save_on_master(checkpoint,
                                 os.path.join(output_dir, 'checkpoint.pth'))

            # write log
            if utils.is_main_process():
                with open(log_mAP_file, 'a') as file:
                    line_string = f'{val_mAP:.4f}'
                    if epoch == 0 or epoch % 5 == 4:
                        line_string += f' {train_eval_mAP:.4f}'
                    line_string += '\n'
                    file.write(line_string)
                    
    return best_metric_value

args_dict = create_args_dict()
distributed_args = init_cuda_and_distributed(args_dict)
best_val = train(args_dict, distributed_args)