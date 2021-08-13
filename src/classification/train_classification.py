"""
    Code for training a classifier.
    This code is forked from
        vision/references/classification/train.py of pytorch github
"""

from __future__ import print_function

import argparse
import datetime
import os
import subprocess
import sys
import time
import copy
from collections import OrderedDict

import torch
import torch.utils.data
import torchvision
from torch import nn
from torch.optim.swa_utils import AveragedModel
from torchvision import transforms
import timm

import deterministic_setting  # pylint: disable=unused-import
import myUtils
import utils

parser = argparse.ArgumentParser()
parser.add_argument('config_file', type=str)

with open(parser.parse_args().config_file, 'r') as f:
    exec(f.read(), globals())

deterministic_setting.set_deterministic(seed=args_dict['random_seed'])

# config cuda
os.environ["CUDA_VISIBLE_DEVICES"] = args_dict['CUDA_VISIBLE_DEVICES']
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)

output_dir = args_dict['output_dir']
utils.mkdir(output_dir)

# prepare log file
log_ap_file = os.path.join(output_dir, 'log_ap.txt')
if os.path.isfile(log_ap_file):
    os.remove(log_ap_file)
log_auc_file = os.path.join(output_dir, 'log_auc.txt')
if os.path.isfile(log_auc_file):
    os.remove(log_auc_file)
config_file = os.path.join(output_dir, 'config.txt')
if os.path.isfile(config_file):
    os.remove(config_file)

def train_one_epoch(
        model,  # pylint: disable=redefined-outer-name
        criterion,  # pylint: disable=redefined-outer-name
        optimizer,  # pylint: disable=redefined-outer-name
        data_loader,
        device,  # pylint: disable=redefined-outer-name
        epoch,  # pylint: disable=redefined-outer-name
        print_freq,
        scaler):
    """
        Train one epoch for a classifier, refering to
        https://github.com/pytorch/vision/blob/131ba1320b8208f10eb58d5feb7416c90ed839bb/references/detection/engine.py#L13
    """
    model.train()

    # set metric logger to log info. to monitor the training process
    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr',
                            utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s',
                            utils.SmoothedValue(window_size=10, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)

    if args_dict['warmup_lr'] and epoch == 0:
        base_lr = optimizer.param_groups[0]['lr']
        warmup_iter = min(round(len(data_loader) * args_dict['dataset_shrinking_factor']), 1000)
        
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, print_freq,header)):
        if i > round(len(data_loader) * args_dict['dataset_shrinking_factor']):
            break
        if args_dict['warmup_lr'] and epoch == 0 and i < warmup_iter:
            optimizer.param_groups[0]['lr'] = base_lr * (i + 1) / warmup_iter
        
        start_time_loop = time.time()
        image, target = image.to(device), target.to(device)
        if args_dict['amp']:
            image = image.to(memory_format=torch.channels_last)
        
        if args_dict['SAM']:
            with torch.cuda.amp.autocast(enabled=args_dict['amp']):
                output = model(image)
                loss = criterion(output, target) / args_dict['accumulate_grad_batches']
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer, first_step=True)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=args_dict['amp']):
                output = model(image)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer, first_step=False, update=(i % args_dict['accumulate_grad_batches'] == 0))
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
        else:
            with torch.cuda.amp.autocast(enabled=args_dict['amp']):
                output = model(image)
                loss = criterion(output, target) / args_dict['accumulate_grad_batches']
            scaler.scale(loss).backward()
            if i % args_dict['accumulate_grad_batches'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(),
                             lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['img/s'].update(round(batch_size / (time.time() - start_time_loop), 3))


def evaluate(model, criterion, data_loader, device, print_freq=10):  # pylint: disable=redefined-outer-name
    """
        Evaluate a classifier, refering to
        https://github.com/pytorch/vision/blob/131ba1320b8208f10eb58d5feb7416c90ed839bb/references/detection/engine.py#L70

        Args:
            model: A PyTorch model that given an input of dimension
                [batch, channel, height, width], it will output a
                logit tensor of dimension [batch, n_classes]
            criterion: The loss calculation that will be used as
                criterion(output, target)
            data_loader: An iterator that will yields (image, target)
                when iterating
            device: either cpu or gpu
    """

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # to record all prediction scores and target
    scores = torch.zeros(
        0, myUtils.metadata[args_dict['task_name']]['num_classes'])
    scores = scores.to(device)
    targets = torch.zeros(0, dtype=torch.long).to(device)
    all_idx = []

    with torch.no_grad():
        for image, target, idxes in metric_logger.log_every(
                data_loader, print_freq, header):
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            scores = torch.cat((scores, output))
            targets = torch.cat((targets, target))
            all_idx.append(idxes)

            loss = criterion(output, target)
            acc1, = utils.accuracy(output, target, topk=(1,))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        all_idx = torch.cat(all_idx)

    # gather the stats from all processes (on different GPUs)
    metric_logger.synchronize_between_processes()
    if utils.is_dist_avail_and_initialized():
        scores, targets = myUtils.synchronize_variables(scores, targets,
                                                        all_idx)

    scores_sm = scores.softmax(dim=1)
    auc = myUtils.get_auc(scores_sm, targets)
    ap = myUtils.get_ap(scores_sm, targets)
    print(f'* Auc: {auc}')
    print(f'* AP: {ap}')
    if args_dict['metric_class'] == -1:
        final_ap = ap.mean().item()
        final_auc = auc.mean().item()
    else:
        final_ap = ap[args_dict['metric_class']].item()
        final_auc = auc[args_dict['metric_class']].item()

    print(' * Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.acc1))
    return metric_logger.acc1.global_avg, final_ap, final_auc


# +
# distributed training parameters
parser = argparse.ArgumentParser(description='PyTorch Classification Training')

parser.add_argument('--world-size',
                    default=1,
                    type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url',
                    default='env://',
                    help='url used to set up distributed training')
args = parser.parse_args('')
utils.init_distributed_mode(args)

# +
# print info
# The style of printing is used to set hidden content on GitLab.
args_dict['name'] = 'args_dict'
print(f'{args_dict["name"]}:  ')
print('<details><summary>Training config</summary><pre>')
readable_str = myUtils.print_dict_readable(args_dict)
if utils.is_main_process():
    with open(config_file, 'w') as file:
        file.write(readable_str)
print('')
print('output_dir: ')
print(output_dir)
print('')
print('gpu_info:')
print(gpu_info.decode('utf-8'))
print('</pre></details>')
print('<details><summary>Learning curve</summary>')
print('</details>')
print('')
print('all configs:  ')
print(args)

print("Loading data")

assert not (args_dict['pretrained'] is not None and args_dict['finetune']), "Finetune and pretrained can be applied exclusively."

if args_dict['finetune']:
    checkpoint = torch.load(args_dict['finetune_model_file'], map_location='cpu')
    norm_mean = checkpoint.get('norm_mean', [0.485, 0.456, 0.406])
    norm_std = checkpoint.get('norm_std', [0.229, 0.224, 0.225])
    args_dict['channel'] = checkpoint['args_dict'].get('channel', 3)
elif args_dict['pretrained'] is not None and args_dict['pretrained'] != 'imagenet':
    checkpoint = torch.load(args_dict['pretrained'], map_location='cpu')
    norm_mean = checkpoint.get('normalize_mean', [0.485, 0.456, 0.406])
    norm_std = checkpoint.get('normalize_std', [0.229, 0.224, 0.225])
    args_dict['channel'] = checkpoint.get('input_nc', 3)
    torch.nn.Module.load_state_dict = myUtils.load_state_dict_reviser_not_strict(
        torch.nn.Module.load_state_dict)
elif args_dict['pretrained'] == 'imagenet':
    if args_dict['model_name'] in timm.list_models():
        m = timm.create_model(args_dict['model_name'])
        norm_mean = list(m.default_cfg['mean'])
        norm_std = list(m.default_cfg['std'])
    else:
        raise ValueError(f'Not known how to set normalizations '
                         f'for {args_dict["model_name"]} models.')
    if args_dict['channel'] != 3:
        assert args_dict['channel'] == 1
        org_norm_mean = norm_mean
        org_norm_std = norm_std
        norm_mean = [0.5]
        norm_std = [0.5]
elif args_dict['pretrained'] is None:
    if args_dict['channel'] == 3:
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
    else:
        norm_mean = [0.5 for _ in range(args_dict['channel'])]
        norm_std = [0.5 for _ in range(args_dict['channel'])]
    
normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

# +
if args_dict['restrict_square_resize']:
    args_dict['model_input_size'] = (args_dict['model_input_size'],
                                     args_dict['model_input_size'])

# Set Training transforms
transform_train_list = []
if args_dict['color_jitter']:
    transform_train_list.append(
        transforms.ColorJitter(
            brightness=args_dict['color_jitter_factor']['brightness'],
            contrast=args_dict['color_jitter_factor']['contrast'],
            saturation=0,
            hue=0))
if args_dict['blur']:
    transform_train_list.append(myUtils.RandomBlur(scale=3e-3))
if args_dict['noise']:
    transform_train_list.append(myUtils.RandomNoise(n=0.5))
if args_dict['flip']:
    transform_train_list.append(transforms.RandomHorizontalFlip(p=args_dict.get('flip_p', 0.5)))
if args_dict['flip_vertical']:
    transform_train_list.append(transforms.RandomVerticalFlip(p=args_dict.get('flip_p', 0.5)))
if args_dict['shift']:
    transform_train_list.append(
        myUtils.RandomShift(scale_x=0.2,
                            scale_y=0.2,
                            pad_mode=args_dict['shift_mode']))
if args_dict['RandomResizedCrop']:
    transform_train_list.append(transforms.RandomResizedCrop(args_dict['model_input_size']))
if args_dict['RandAugment']:
    transform_train_list.append(myUtils.RandAugmentPT(args_dict['model_input_size'], img_mean=norm_mean))
if args_dict['random_perspective']:
    transform_train_list.append(transforms.RandomPerspective(p=0.5))
if args_dict['rotation']:
    transform_train_list.append(
        transforms.RandomRotation(degrees=args_dict['rotation_degree'],
                                  expand=True))
if args_dict['elastic_deformation']:
    transform_train_list.append(myUtils.ElasticDeform())
if args_dict['aspect_ratio']:
    transform_train_list.append(myUtils.RandomAspectRatio(scale=0.2))
if args_dict['resize']:
    transform_train_list.append(myUtils.RandomPadCrop(scale=0.1))

train_augmentations_transforms = myUtils.Compose(
    transform_train_list,
    prob_multiplier=args_dict['augmentation_prob_multiplier'])
transform_train_list = [
    train_augmentations_transforms,
    transforms.Resize(args_dict['model_input_size'])
]

transform_test_list = [
    transforms.Resize(args_dict['model_input_size'])
]

transform_train_list.append(normalize)
transform_test_list.append(normalize)

transform_test = transforms.Compose(transform_test_list)
transform_train = transforms.Compose(transform_train_list)

# Load Data
print("Loading training data")
if args_dict['shapleys'] is not None:
    shapleys = torch.load(args_dict['shapleys'], map_location='cpu')
    if isinstance(args_dict['removing_by_shapley'], float):
        removing_idxes = set(shapleys.argsort()[:round(len(shapleys) * args_dict['removing_by_shapley'])].tolist())
    elif args_dict['removing_by_shapley'] == 'neg':
        removing_idxes = set((shapleys < 0).nonzero(as_tuple=True)[0].tolist())
    
dataset_train = myUtils.ClassificationDataset(
    task_name=args_dict['task_name'],
    blackout=args_dict['train_blackout'],
    split='train',
    transform=transform_train,
    equalize=args_dict['equalize'],
    standardize=args_dict['standardize'],
    removing_idxes=removing_idxes if args_dict['shapleys'] else None,
)

print("Loading validation data")
dataset_val = myUtils.ClassificationDataset(
    task_name=args_dict['task_name'],
    blackout=False,
    split='val',
    transform=transform_test,
    equalize=args_dict['equalize'],
    standardize=args_dict['standardize'],
    return_idx=True,
)
    
print("Loading training data for evaluation")
dataset_train_evaluate = myUtils.ClassificationDataset(
    task_name=args_dict['task_name'],
    blackout=False,
    split='train',
    transform=transform_test,
    equalize=args_dict['equalize'],
    standardize=args_dict['standardize'],
    return_idx=True,
    removing_idxes=removing_idxes if args_dict['shapleys'] else None,
)
if args_dict['dataset_shrinking_factor'] != 1.:
    torch.manual_seed(42)
    dataset_train_evaluate = torch.utils.data.Subset(
        dataset_train_evaluate,
        torch.randperm(len(dataset_train_evaluate))
        [:int(len(dataset_train_evaluate) * args_dict['dataset_shrinking_factor'])].tolist())

print("Create data loaders")
if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)
    train_evaluate_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train_evaluate)
else:
    train_sampler = torch.utils.data.RandomSampler(dataset_train)
    val_sampler = torch.utils.data.SequentialSampler(dataset_val)
    train_evaluate_sampler = torch.utils.data.SequentialSampler(
        dataset_train_evaluate)
if args_dict['aug_sample_multiplier'] != 1:
    train_sampler = myUtils.sample_multiplier(train_sampler, args_dict['aug_sample_multiplier'])

data_loader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=args_dict['batch_size'],
    sampler=train_sampler,
    num_workers=min(args_dict['batch_size'] // 2, 16),
    pin_memory=False)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=args_dict['batch_size'],
    sampler=val_sampler,
    num_workers=min(args_dict['batch_size'] // 2, 16),
    pin_memory=False)

data_loader_train_evaluate = torch.utils.data.DataLoader(
    dataset_train_evaluate,
    batch_size=args_dict['batch_size'],
    sampler=train_evaluate_sampler,
    num_workers=min(args_dict['batch_size'] // 2, 16),
    pin_memory=False)


print("Create model")
# create model
if args_dict['model_name'] in timm.list_models():
    model = timm.create_model(
        args_dict['model_name'],
        pretrained=(args_dict['pretrained'] == 'imagenet'),
        num_classes=myUtils.metadata[args_dict['task_name']]['num_classes']
    )
    
if args_dict['pretrained'] is not None and args_dict['pretrained'] != 'imagenet':
    msg = model.load_state_dict({k: v for k, v in checkpoint['model'].items() if k not in ['fc.weight', 'fc.bias', 'classifier.weight', 'classifier.bias']})
    print(f'weight loading state: {msg}')

print(model.to(device))
if args.distributed and args_dict['sync_bn']:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
if args_dict['amp']:
    model = model.to(memory_format=torch.channels_last)

criterion = nn.CrossEntropyLoss()
if args_dict['use_focal_loss']:
    criterion = myUtils.FocalLoss(activation=('softmax' if not args_dict['use_sigmoid'] else 'sigmoid'),
                                  gamma=args_dict['FL_gamma'],
                                  alpha=args_dict['FL_alpha'],
                                  suppress=args_dict['FL_suppress'],
                                 )
elif args_dict['class_weight'] is not None:
    criterion = nn.CrossEntropyLoss(weight=torch.tensor( # pylint: disable=not-callable
        args_dict['class_weight']).to(device))
elif args_dict['soft_label'] != 1.0:
    criterion = myUtils.SoftCrossEntropyLoss(args_dict['soft_label'])
elif args_dict['use_sigmoid']:
    criterion = myUtils.BCEWithLogitsCategoricalLoss(args_dict['soft_label'])

if args_dict['optimizer'] == 'SGD':
    optim_class = torch.optim.SGD
    optim_params = dict(lr=args_dict['lr'],
                        momentum=0.9,
                        weight_decay=args_dict['weight_decay']
                       )
elif args_dict['optimizer'] == 'Adam':
    optim_class = torch.optim.Adam
    optim_params = dict(lr=args_dict['lr'],
                        amsgrad=True,
                        weight_decay=args_dict['weight_decay']
                       )

if args_dict['SAM']:
    optimizer = myUtils.SAMOptim(model.parameters(), optim_class, **optim_params)
else:
    optimizer = optim_class(model.parameters(), **optim_params)
    
scaler = torch.cuda.amp.GradScaler(enabled=args_dict['amp'])

model_without_ddp = model
if args.distributed:
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.gpu])
    model_without_ddp = model.module

start_epoch = 0

if args_dict['cosine_lr']:
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args_dict['epochs'])

if args_dict['finetune']:
    model_without_ddp.load_state_dict(checkpoint['model'])
    if args_dict['load_optim']:
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('scaler', None) is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        if checkpoint.get('lr_scheduler', None) is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
print("Start training")
start_time = time.time()
for epoch in range(start_epoch, args_dict['epochs']):
    if epoch in args_dict['epoch_step_lr']:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10
    if args.distributed:
        train_sampler.set_epoch(epoch)
    train_one_epoch(model, criterion, optimizer, data_loader_train, device,
                    epoch, 10, scaler)
    if args_dict['cosine_lr']:
        lr_scheduler.step()
    val_acc1, val_ap, val_auc = evaluate(model,
                                criterion,
                                data_loader_val,
                                device=device)
    if epoch == start_epoch or (epoch + start_epoch) % 5 == 4:
        train_acc1, train_ap, train_auc = evaluate(model,
                                        criterion,
                                        data_loader_train_evaluate,
                                        device=device)
    if output_dir:
        # save model
        checkpoint = {
            'model':model_without_ddp.state_dict(),
            'lr_scheduler':
                lr_scheduler.state_dict() if args_dict['cosine_lr'] else None,
            'scaler':
                scaler.state_dict(),
            'epoch':
                epoch,
            'args_dict':
                args_dict,
            'args':
                args,
            'val_acc':
                val_acc1,
            'val_ap':
                val_ap,
            'val_auc':
                val_auc,
            'training_time':
                str(datetime.timedelta(seconds=int(time.time() - start_time))),
            'norm_mean': norm_mean,
            'norm_std': norm_std,
        }
        if args_dict['choose_best'] == 'acc':
            metric_value = val_acc1
        elif args_dict['choose_best'] == 'ap':
            metric_value = val_ap
        elif args_dict['choose_best'] == 'auc':
            metric_value = val_auc
        if epoch == 0 or metric_value > best_metric_value:
            utils.save_on_master(checkpoint,
                                 os.path.join(output_dir, 'model_best.pth'))
            best_metric_value = metric_value
        utils.save_on_master(checkpoint,
                             os.path.join(output_dir, 'checkpoint.pth'))

        # write log
        if utils.is_main_process():
            with open(log_ap_file, 'a') as file:
                if epoch == start_epoch or (epoch + start_epoch) % 5 == 4:
                    file.write(' '.join([str(val_ap), str(train_ap)]) + '\n')
                else:
                    file.write(str(val_ap) + '\n')
            with open(log_auc_file, 'a') as file:
                if epoch == start_epoch or (epoch + start_epoch) % 5 == 4:
                    file.write(' '.join([str(val_auc), str(train_auc)]) + '\n')
                else:
                    file.write(str(val_auc) + '\n')

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))
