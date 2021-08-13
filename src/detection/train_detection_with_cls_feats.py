""" File to train a FasterRCNN object detector. """
import argparse
import copy
import datetime
import math
import os
import time
from collections import OrderedDict

import deterministic_setting  # pylint: disable=unused-import
import myUtils
import timm
import torch
import torchvision
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from module import fpn as my_fpn
from timm_backbone_wrapper import timm_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torchvision.ops import misc as misc_nn_ops


def get_cls_model(device, cls_model_folder):
    ckpt_path = os.path.join(cls_model_folder, 'model_best.pth')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = timm.create_model(
        ckpt['args_dict']['model_name'],
        features_only=True,
    )
    timm_fpn_backbone.change_layer(model,
                                   torch.nn.BatchNorm2d,
                                   torchvision.ops.misc.FrozenBatchNorm2d,
                                   params_map={
                                       'num_features': 'num_features',
                                       'eps': 'eps'
                                   })
    print('classification model load_state_dict:',
          model.load_state_dict(ckpt['model'], strict=False))
    return model.to(device), ckpt['args_dict']['model_input_size'], ckpt[
        'norm_mean'], ckpt['norm_std'], ckpt['args_dict']['model_name']


class FuseClsFeatsBackboneBody(torch.nn.Module):

    def __init__(
        self,
        det_backbone_body,
        cls_model,
        cls_size,
        cls_norm_mean,
        cls_norm_std,
        cls_feat_name=[1, 2, 3, 4],
        det_feat_ch=[],
        cls_feat_ch=[],
    ):
        super().__init__()
        self.det_backbone_body = det_backbone_body
        self.cls_model = cls_model
        self.cls_feat_name = cls_feat_name
        self.att_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_ch, out_ch, 1)
            for in_ch, out_ch in zip(cls_feat_ch, det_feat_ch)
        ])
        self.cls_normalize = torchvision.transforms.Normalize(
            cls_norm_mean, cls_norm_std)
        self.cls_resize = torchvision.transforms.Resize((cls_size, cls_size))

    def forward(self, image):
        det_feats = self.det_backbone_body(image)
        det_feats_keys = det_feats.keys()
        det_feats = list(det_feats.values())
        cls_input_image = torch.stack([
            self.cls_normalize(self.cls_resize(img)) for img in self.org_image
        ])
        with torch.no_grad():
            cls_feats = self.cls_model(cls_input_image)
        cls_feats = [cls_feats[k] for k in self.cls_feat_name]
        det_feats = OrderedDict([
            (det_feats_key,
             torch.nn.functional.interpolate(
                 conv(cls_feat).sigmoid(), det_feat.shape[2:]) * det_feat)
            for cls_feat, conv, det_feat, det_feats_key in zip(
                cls_feats, self.att_convs, det_feats, det_feats_keys)
        ])
        return det_feats


def add_backup_org_image(model):
    model.org_forward = model.forward

    def forward_with_backup(self, image, target=None):
        self.backbone.body.org_image = copy.deepcopy(image)
        return self.org_forward(image, target)

    model.forward = forward_with_backup.__get__(model, model.__class__)


def get_loss_dict(model, images, targets, args_dict):
    loss_dict = model(images, targets)
    return loss_dict


def train_one_epoch(
        model,
        optimizer,
        data_loader,
        device,
        epoch,
        print_freq,  # pylint: disable=redefined-outer-name
        scaler,
        args_dict,
        caller_vars={}):  # pylint: disable=redefined-outer-name
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

        lr_scheduler_in_epoch = utils.warmup_lr_scheduler(
            optimizer, warmup_iters, warmup_factor)

    for n, (images, targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args_dict['SAM']:
            # First Step
            with torch.cuda.amp.autocast(enabled=args_dict['amp']):
                loss_dict = get_loss_dict(model, images, targets, args_dict)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer, first_step=True)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # Second Step
            with torch.cuda.amp.autocast(enabled=args_dict['amp']):
                loss_dict = get_loss_dict(model, images, targets, args_dict)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer, first_step=False, update=True)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            with torch.cuda.amp.autocast(enabled=args_dict['amp']):
                loss_dict = get_loss_dict(model, images, targets, args_dict)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
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
            caller_vars['swa_model'].update_parameters(
                caller_vars['model_without_ddp'])

        if lr_scheduler_in_epoch is not None:
            lr_scheduler_in_epoch.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return safe_state


@torch.no_grad()
def evaluate(model, data_loader, device, args_dict):  # pylint: disable=redefined-outer-name
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
                                       args_dict['img_size'],
                                       data_loader.dataset.transforms))
    else:
        coco = get_coco_api_from_dataset(data_loader.dataset)
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


parser = argparse.ArgumentParser()
parser.add_argument('config_file', type=str)
with open(parser.parse_args().config_file, 'r') as f:
    exec(f.read(), globals())

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

if args_dict['SWA']:
    from torch.optim.swa_utils import SWALR, AveragedModel

args_dict['color_jitter_factor'] = {
    'brightness': args_dict['color_jitter_brightness'],
    'contrast': args_dict['color_jitter_contrast']
}

if args_dict['ACFPN'] or args_dict['iAFF']:
    torchvision.models.detection.backbone_utils.FeaturePyramidNetwork = my_fpn.ModulerFPN
    timm_fpn_backbone.FeaturePyramidNetwork = my_fpn.ModulerFPN

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)

cls_model, cls_size, cls_norm_mean, cls_norm_std, cls_model_name = get_cls_model(
    device, args_dict['cls_model_folder'])

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
    args_dict['normalize_mean'] = checkpoint[
        'normalize_mean'] if 'normalize_mean' in checkpoint else checkpoint[
            'norm_mean']
    args_dict['normalize_std'] = checkpoint[
        'normalize_std'] if 'normalize_std' in checkpoint else checkpoint[
            'norm_std']
elif args_dict['fine_tune']:
    checkpoint = torch.load(args_dict['fine_tune_model_file'],
                            map_location='cpu')
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

output_dir = args_dict['output_dir']
if utils.is_main_process():
    utils.mkdir(output_dir)

# prepare log file
log_mAP_file = os.path.join(output_dir, 'log_mAP.txt')  # pylint: disable=invalid-name
config_file = os.path.join(output_dir, 'config.txt')

# print info
args_dict['name'] = 'args_dict'
print(f'{args_dict["name"]}:  ')
readable_str = myUtils.print_dict_readable_no_br(
    {k: (v if k != 'norm_layer' else str(v)) for k, v in args_dict.items()})
print('\nwork directory: \n' + output_dir + '\n')
if utils.is_main_process():
    with open(config_file, 'w') as file:
        file.write(readable_str)

# setup transforms
transform_train_list = []
if args_dict['RandomResizedCrop']:
    transform_train_list.append(
        myUtils.AlbumentationTransforms('RandomResizedCrop',
                                        dict(height=1024, width=1024)))
if args_dict['shift']:
    transform_train_list.append(
        myUtils.RandomShift(pad_mode=args_dict['shift_mode'],
                            **({
                                'scale_x': args_dict['shift_scale'],
                                'scale_y': args_dict['shift_scale']
                            } if 'shift_scale' in args_dict else {})))
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
            myUtils.RandomRotationExpand(
                args_dict['rotation_degree'],
                fixed=args_dict['rotation_fixed_angle']))
    elif args_dict['rotation_mode'] == 'WRAP':
        assert args_dict[
            'rotation_fixed_angle'] == False, "fixed angle rotation with WRAP border mode is not implemented."
        transform_train_list.append(
            myUtils.AlbumentationTransforms(
                'Rotate', dict(limit=args_dict['rotation_degree'],
                               border_mode=3)))
if args_dict['random_perspective']:
    transform_train_list.append(
        myUtils.AlbumentationTransforms('Perspective', {'p': 1.}))
if args_dict['elastic_deformation']:
    transform_train_list.append(
        myUtils.ElasticDeform(
            relative_sigma=args_dict['elastic_deformation_sigma']))
transform_train = myUtils.Compose(
    transform_train_list,
    prob_multiplier=args_dict['augmentation_prob_multiplier'],
)
transform_train_list = [transform_train, myUtils.ToTensor()]
transform_train = myUtils.Compose(transform_train_list)

transform_test_list = [myUtils.ToTensor()]
transform_test = myUtils.Compose(transform_test_list)

# load data
print("Loading data")
exec(  # pylint: disable=exec-used
    f"from {myUtils.metadata[args_dict['task_name']]['read_label']} import read_label",
    globals())

dataset_train = myUtils.DetectionDataset(read_label,
                                         args_dict['task_name'],
                                         'train',
                                         transforms=transform_train)

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

data_loader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=args_dict['batch_size'],
    sampler=train_sampler,
    num_workers=args_dict['num_workers'],
    collate_fn=myUtils.collate_fn)
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
print("Creating model")
if args_dict[
        'backbone_body'] in timm_fpn_backbone.timm_backbone_body_default_params:
    pretrained = (not args_dict['fine_tune'] and
                  not args_dict['user_specified_backbone_pretrained'])
    backbone = timm_fpn_backbone.timm_fpn_backbone(
        args_dict['backbone_body'],
        pretrained=pretrained,
        weights=checkpoint['model']
        if args_dict['user_specified_backbone_pretrained'] else None,
        norm_layer=args_dict['norm_layer'] or misc_nn_ops.FrozenBatchNorm2d,
    )
    if pretrained:
        args_dict['normalize_mean'] = list(
            backbone.body.timm_model.default_cfg['mean'])
        args_dict['normalize_std'] = list(
            backbone.body.timm_model.default_cfg['std'])

model_kwargs = {}
model_kwargs['min_size'] = args_dict['input_size']
model_kwargs['max_size'] = args_dict['input_size']
model_kwargs['image_mean'] = args_dict['normalize_mean']
model_kwargs['image_std'] = args_dict['normalize_std']
model_kwargs['num_classes'] = (
    myUtils.metadata[args_dict['task_name']]['num_classes'])
model = FasterRCNN(backbone, **model_kwargs)

if args_dict['ACFPN']:
    model.backbone.fpn.setup_ACFPN()
if args_dict['iAFF']:
    model.backbone.fpn.setup_iAFF()

print(model)
if args_dict['fine_tune']:
    state_dict_to_load = checkpoint['model']
    print(
        'Model loading status:',
        model.load_state_dict(state_dict_to_load,
                              strict=not args_dict['fine_tune_non_strict']))

cls_feat_name = [1, 2, 3, 4]
model.backbone.body = FuseClsFeatsBackboneBody(
    model.backbone.body,
    cls_model,
    cls_size,
    cls_norm_mean,
    cls_norm_std,
    cls_feat_name=cls_feat_name,
    det_feat_ch=[
        model.backbone.body.timm_model.feature_info.info[name]['num_chs']
        for name in [1, 2, 3, 4]
    ],
    cls_feat_ch=[
        cls_model.feature_info.info[name]['num_chs'] for name in cls_feat_name
    ],
)
add_backup_org_image(model)
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

if args.distributed and args_dict['sync_bn']:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

scaler = torch.cuda.amp.GradScaler(enabled=args_dict['amp'])

# SWA
if args_dict['SWA']:
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer,
                          anneal_epochs=args_dict['SWA_anneal_epoch'],
                          swa_lr=args_dict['learning_rate'] /
                          args_dict['SWA_lr_divider'])

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
    safe_state = train_one_epoch(model,
                                 optimizer,
                                 data_loader_train,
                                 device,
                                 epoch,
                                 10,
                                 scaler,
                                 args_dict,
                                 caller_vars=locals())
    if not safe_state:
        raise

    if args_dict['SWA'] and epoch >= args_dict['SWA_start_epoch']:
        coco_evaluator_val = evaluate(model,
                                      data_loader_val,
                                      device=device,
                                      args_dict=args_dict)
        val_mAP = coco_evaluator_val.coco_eval['bbox'].eval['precision'][
            0, :, :, 0, 2].mean()
        swa_model.update_parameters(model_without_ddp, val_mAP)
        swa_scheduler.step()
    print("Start evaluation")
    if args_dict['SWA'] and epoch >= args_dict['SWA_start_epoch']:
        coco_evaluator_val = evaluate(swa_model,
                                      data_loader_val,
                                      device=device,
                                      args_dict=args_dict)
    else:
        coco_evaluator_val = evaluate(model,
                                      data_loader_val,
                                      device=device,
                                      args_dict=args_dict)
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
                model_without_ddp.state_dict() if
                (not args_dict['SWA'] or epoch < args_dict['SWA_start_epoch'])
                else swa_model.module.state_dict(),
            'epoch':
                epoch,
            'args_dict': {
                k: (v if k != 'norm_layer' else str(v))
                for k, v in args_dict.items()
            },
            'val_mAP':
                val_mAP,
            'training_time':
                str(datetime.timedelta(seconds=int(time.time() - start_time))),
        }
        metric_value = val_mAP
        if epoch == 0 or metric_value > best_metric_value:
            utils.save_on_master(checkpoint,
                                 os.path.join(output_dir, 'model_best.pth'))
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
