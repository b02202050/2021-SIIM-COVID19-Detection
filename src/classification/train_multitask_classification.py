# -*- coding: utf-8 -*-
"""
    Multitask classification training.
    Each classifier share the same backbone.
    When data with several tasks included in a batch,
        each data will only pass through its task branch
        (usually one-layered MLP). The loss is then
        averaged by the batch size.
"""
from __future__ import print_function

import argparse
import datetime
import os
import time
from collections import OrderedDict

import deterministic_setting  # pylint: disable=unused-import
import myUtils
import timm
import torch
import torch.utils.data
import utils
from torch import nn
from torchvision import transforms

args_dict = dict()

# ## Mode Config

DEBUG = False

# ## Train Config

## Kaggle permitted datasets
args_dict['tasks_to_train'] = {
    'NIH_Pneumothorax',
    'NIH_Atelectasis',
    'NIH_Cardiomegaly',
    'NIH_Consolidation',
    'NIH_Edema',
    'NIH_Effusion',
    'NIH_Emphysema',
    'NIH_Fibrosis',
    'NIH_Hernia',
    'NIH_Infiltration',
    'NIH_Mass',
    'NIH_Nodule',
    'NIH_Pleural_Thickening',
    'NIH_Pneumonia',
}
args_dict['tasks_to_train'] |= {
    'CheXpert_Pneumothorax', 'CheXpert_Lung_Lesion', 'CheXpert_Atelectasis',
    'CheXpert_Enlarged_Cardiomediastinum', 'CheXpert_Pneumonia',
    'CheXpert_Lung_Opacity', 'CheXpert_Pleural_Other', 'CheXpert_Edema',
    'CheXpert_Cardiomegaly', 'CheXpert_Fracture', 'CheXpert_Pleural_Effusion',
    'CheXpert_Consolidation', 'CheXpert_Support_Devices'
}
args_dict['tasks_to_train'] |= {
    'kaggle_chest_xray_covid19_pneumonia',
    'kaggle_covidx_cxr2',
    'kaggle_chest_xray_pneumonia',
    'kaggle_curated_chest_xray_image_dataset_for_covid19',
    'kaggle_covid19_xray_two_proposed_databases_3_classes',
    'kaggle_covid19_xray_two_proposed_databases_5_classes',
    'kaggle_ricord_covid19_xray_positive_tests',
}
args_dict['tasks_not_to_val'] = args_dict['tasks_to_train'] - set([
    task for task in args_dict['tasks_to_train'] if 'kaggle' in task
]) - {'NIH_Pneumonia', 'CheXpert_Pneumonia'}

args_dict['task_name_mapping_files'] = {
    '../../dataset/CheXpert-v1.0/CheXpert_task_map_to_NIH.txt',
}
args_dict['task_heads'] = {}
args_dict['default_mlp_num'] = 1
args_dict['CUDA_VISIBLE_DEVICES'] = '0'
args_dict['pretrained'] = 'imagenet'  # None, 'imagenet', path
args_dict['sync_bn'] = True
args_dict['optimizer'] = 'Adam'
args_dict['weight_decay'] = 0.
args_dict['batch_size'] = 16
args_dict['lr'] = 1e-4
args_dict['num_workers'] = 8
args_dict['epochs'] = 20
args_dict['amp'] = True
args_dict['backbone'] = 'tf_efficientnet_b7_ns'
args_dict['model_input_size'] = 512
args_dict['output_dir'] = 'work_dir/multitask/pretrained_run1'
args_dict['preprocess_resize_1024'] = True
args_dict['tags']: ''

# Data augmentations
args_dict['augmentation_prob_multiplier'] = 0.25
args_dict['flip'] = True
args_dict['flip_vertical'] = True
args_dict['shift'] = True
args_dict['shift_mode'] = 'repeat'
args_dict['aspect_ratio'] = True
args_dict['resize'] = True
args_dict['rotation'] = True
args_dict['rotation_degree'] = 15
args_dict['color_jitter'] = True
args_dict['color_jitter_factor'] = {'brightness': 0.3, 'contrast': 0.3}

tasks_to_val = args_dict['tasks_to_train'] - args_dict['tasks_not_to_val']

# config cuda
os.environ["CUDA_VISIBLE_DEVICES"] = args_dict['CUDA_VISIBLE_DEVICES']
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()
parser.add_argument('--world-size',
                    default=1,
                    type=int,
                    help='number of distributed processes')
args = parser.parse_args('')
args.dist_url = 'env://'
utils.init_distributed_mode(args)

# load checkpoint
heads_not_to_load = set()
all_heads = dict()
for task in args_dict['tasks_to_train']:
    if task in args_dict['task_heads']:
        all_heads[task] = args_dict['task_heads'][task]
    else:
        all_heads[task] = args_dict['default_mlp_num']

# collate task name
task_mapping = {}
for task_name_mapping_file in args_dict['task_name_mapping_files']:
    with open(task_name_mapping_file, 'r') as f:
        task_mapping_list = [
            (x if x[-1] != '\n' else x[:-1]).split() for x in f.readlines()
        ]
    task_mapping.update({x[0]: x[1] for x in task_mapping_list if x[0] != x[1]})

heads_not_to_load = heads_not_to_load | set(task_mapping.keys())
for task in task_mapping:
    if task in all_heads:
        del all_heads[task]
        if task_mapping[task] not in all_heads:
            all_heads[task_mapping[task]] = args_dict['task_heads'].get(
                task, args_dict['default_mlp_num'])

output_dir = args_dict['output_dir']

# prepare log file
log_ap_file = os.path.join(output_dir, 'log_ap.txt')
log_auc_file = os.path.join(output_dir, 'log_auc.txt')
config_file = os.path.join(output_dir, 'config.txt')
log_ap_file_tasks = {}
for task in tasks_to_val:
    log_ap_file_tasks[task] = os.path.join(output_dir, f'log_ap_{task}.txt')
log_auc_file_tasks = {}
for task in tasks_to_val:
    log_auc_file_tasks[task] = os.path.join(output_dir, f'log_auc_{task}.txt')


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
        Train one epoch for a multitask classifier, copied and modified from
        https://github.com/pytorch/vision/blob/131ba1320b8208f10eb58d5feb7416c90ed839bb/references/detection/engine.py#L13

        Args:
            model (myUtils.MultiTaskModel)
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Epoch: [{}]'.format(epoch)
    for c_iter, (images, targets, label_masks) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        this_batch_size = len(images)
        start_time_loop = time.time()
        images, targets = images.to(device), targets.to(device)
        with torch.cuda.amp.autocast(enabled=args_dict['amp']):
            feature = model(images, key='backbone')
            alpha = 0.

            # This forward method is not only faster, but less probable to produce error of gradient synchronization
            # due to head branch forwarding not necessarily happens in each process
            loss = 0.
            num_pseudo = 0
            for head_task in sorted(all_heads):
                task_idxes = [
                    i for i, task in enumerate(
                        sorted(args_dict['tasks_to_train']))
                    if head_task == task_mapping.get(task, task)
                ]
                output = model(feature, key=head_task)
                task_mask = label_masks[:, task_idxes].sum(dim=1).sign()
                task_mask = torch.stack((1 - task_mask, task_mask),
                                        dim=1).to(device)
                head_target = targets[:, task_idxes].max(dim=1).values

                pseudo_labels = output.argmax(dim=1)
                combined_label = (task_mask[:, 0] * pseudo_labels +
                                  task_mask[:, 1] * head_target).long()

                if not isinstance(criterion, torch.nn.CrossEntropyLoss):
                    raise ValueError(
                        'Only cross entropy loss is implemented now.')
                head_loss = torch.nn.functional.cross_entropy(output,
                                                              combined_label,
                                                              reduction='none')
                head_loss = head_loss * ((1 - alpha) * task_mask[:, 1] + alpha)
                head_loss = head_loss.sum()
                loss += head_loss
                num_pseudo += task_mask[:, 0].sum().item()
            if criterion.reduction == 'mean':
                loss /= (this_batch_size * len(all_heads) +
                         (alpha - 1) * num_pseudo)
            elif criterion.reduction == 'sum':
                loss *= this_batch_size

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        metric_logger.update(loss=loss.item(),)


@torch.no_grad()
def evaluate(model,
             tasks_to_val,
             criterion,
             data_loader,
             device,
             print_freq=10):  # pylint: disable=redefined-outer-name
    """
        Evaluate the multitask classifier.

        Args:
            model (myUtils.MultiTaskModel)
            tasks_to_val: name of the tasks to evaluate. (please refer to metadata.yaml)
                must be a subset of tasks_to_train
            criterion: The loss calculation that will be used as
                criterion(output, target)
            data_loader: An iterator that will yields (image, target)
                when iterating
            device: either cpu or gpu


    """
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    scores = {
        task: torch.empty(0,
                          myUtils.metadata[task]['num_classes'],
                          device=device) for task in tasks_to_val
    }
    targets = {
        task: torch.empty(0, dtype=torch.long, device=device)
        for task in tasks_to_val
    }
    all_idx = {task: torch.empty(0, dtype=torch.long) for task in tasks_to_val}
    all_union_idx = torch.empty(0, dtype=torch.long)

    for image, target, idxes, label_masks in metric_logger.log_every(
            data_loader, print_freq, header):
        image = image.to(device)
        target = target.to(device)
        feature = model(image, key='backbone')
        for task in tasks_to_val:
            effective_label_idx = torch.where(
                label_masks[:, data_loader.dataset.task_names.index(task)])[0]
            output = model(feature, key=task_mapping.get(task, task))
            scores[task] = torch.cat(
                (scores[task], output[effective_label_idx]))
            targets[task] = torch.cat(
                (targets[task],
                 target[effective_label_idx,
                        data_loader.dataset.task_names.index(task)]))
            all_idx[task] = torch.cat(
                (all_idx[task], idxes[effective_label_idx]))
        metric_logger.update()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    aps = {}
    aucs = {}

    for task in tasks_to_val:
        if args.distributed:
            scores[task], targets[task] = myUtils.synchronize_variables(
                scores[task], targets[task], all_idx[task])

        scores_sm = scores[task].softmax(dim=1)
        auc = myUtils.get_auc(scores_sm, targets[task])
        ap = myUtils.get_ap(scores_sm, targets[task])
        print(f'* Auc({task}): {auc}')
        final_ap = ap[1].item(
        )  # I found a bug when I refactor my code. If you mind, use ap.mean().item()
        final_auc = auc[1].item(
        )  # I found a bug when I refactor my code. If you mind, use auc.mean().item()
        aps[task] = final_ap
        aucs[task] = final_auc

    return aps, aucs


if utils.is_main_process():
    utils.mkdir(output_dir)

# print info
args_dict['name'] = 'args_dict'
print(f'{args_dict["name"]}:  ')
readable_str = myUtils.print_dict_readable(args_dict)
if utils.is_main_process():
    with open(config_file, 'w') as file:
        file.write(readable_str)
print('')
print('all tasks and head fc number:')
print(all_heads)
print('')
print('output_dir:')
print(output_dir)
print('')

device = torch.device(device_str)

# Set normalization
if args_dict['pretrained'] is not None and args_dict['pretrained'] != 'imagenet':
    checkpoint = torch.load(args_dict['pretrained'], map_location='cpu')
    norm_mean = checkpoint.get('normalize_mean', [0.485, 0.456, 0.406])
    norm_std = checkpoint.get('normalize_std', [0.229, 0.224, 0.225])
    args_dict['channel'] = checkpoint.get('input_nc', 3)
else:
    if args_dict['pretrained'] == 'imagenet':
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

# Prepare Data transforms
transform_train_list = []
if args_dict['flip']:
    transform_train_list.append(transforms.RandomHorizontalFlip(p=0.5))
if args_dict['flip_vertical']:
    transform_train_list.append(transforms.RandomVerticalFlip(p=0.5))
if args_dict['shift']:
    transform_train_list.append(
        myUtils.RandomShift(scale_x=0.2,
                            scale_y=0.2,
                            pad_mode=args_dict['shift_mode']))
if args_dict['aspect_ratio']:
    transform_train_list.append(myUtils.RandomAspectRatio(scale=0.2))
if args_dict['resize']:
    transform_train_list.append(myUtils.RandomPadCrop(scale=0.1))
if args_dict['rotation']:
    transform_train_list.append(
        transforms.RandomRotation(degrees=args_dict['rotation_degree'],
                                  expand=True))
if args_dict['color_jitter']:
    transform_train_list.append(
        transforms.ColorJitter(
            brightness=args_dict['color_jitter_factor']['brightness'],
            contrast=args_dict['color_jitter_factor']['contrast'],
            saturation=0,
            hue=0))
train_augmentations_transforms = myUtils.Compose(
    transform_train_list,
    prob_multiplier=args_dict['augmentation_prob_multiplier'])

transform_list = [transforms.Resize(args_dict['model_input_size'])]
transform_list.append(normalize)
transform_test = transforms.Compose(transform_list)
transform_train = transforms.Compose([train_augmentations_transforms] +
                                     transform_list)

# Creating the model
print("Creating model")

model = OrderedDict()

if args_dict['backbone'] in timm.list_models():
    model['backbone'] = timm.create_model(
        args_dict['backbone'],
        pretrained=(args_dict['pretrained'] == 'imagenet'),
        num_classes=0)
    dim_feats = model['backbone'].num_features

if args.distributed and args_dict['sync_bn']:
    model['backbone'] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        model['backbone'])
for task, fc_num in sorted(all_heads.items()):
    model[task] = myUtils.MLPLayer(dim_feats,
                                   myUtils.metadata[task]['num_classes'],
                                   num_layers=fc_num)

model = myUtils.MultiTaskModel(model)
model.to(device)

# Load Training Data
print("Create datasets")
dataset_train = myUtils.MultiTaskAggregatedClassificationDataset(
    task_names=sorted(args_dict['tasks_to_train']),
    split='train',
    transform=transform_train,
    preprocess_resize_1024=args_dict['preprocess_resize_1024'])

dataset_val = myUtils.MultiTaskAggregatedClassificationDataset(
    task_names=sorted(tasks_to_val),
    split='val',
    transform=transform_test,
    return_idx=True,
    preprocess_resize_1024=args_dict['preprocess_resize_1024'])
dataset_train_evaluate = myUtils.MultiTaskAggregatedClassificationDataset(
    task_names=sorted(tasks_to_val),
    split='train',
    transform=transform_test,
    return_idx=True,
    preprocess_resize_1024=args_dict['preprocess_resize_1024'])
print("Creating data loaders")
if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val,
                                                                  shuffle=False)
    train_evaluate_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train_evaluate, shuffle=False)
else:
    train_sampler = torch.utils.data.RandomSampler(dataset_train)
    val_sampler = torch.utils.data.SequentialSampler(dataset_val)
    train_evaluate_sampler = torch.utils.data.SequentialSampler(
        dataset_train_evaluate)

data_loader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=args_dict['batch_size'],
    sampler=train_sampler,
    num_workers=args_dict['num_workers'],
    pin_memory=False)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=args_dict['batch_size'],
    sampler=val_sampler,
    num_workers=args_dict['num_workers'],
    pin_memory=False)

data_loader_train_evaluate = torch.utils.data.DataLoader(
    dataset_train_evaluate,
    batch_size=args_dict['batch_size'],
    sampler=train_evaluate_sampler,
    num_workers=args_dict['num_workers'],
    pin_memory=False)

# Prepare for the training
criterion = nn.CrossEntropyLoss()

if args_dict['optimizer'] == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args_dict['lr'],
                                momentum=0.9,
                                weight_decay=args_dict['weight_decay'])
elif args_dict['optimizer'] == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args_dict['lr'],
                                 amsgrad=True,
                                 weight_decay=args_dict['weight_decay'])

scaler = torch.cuda.amp.GradScaler(enabled=args_dict['amp'])

model_without_ddp = model
if args.distributed:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu],
    )
    model_without_ddp = model.module

# Start training
print("Start training")
start_time = time.time()
for epoch in range(args_dict['epochs']):
    if args.distributed:
        train_sampler.set_epoch(epoch)
    train_one_epoch(model, criterion, optimizer, data_loader_train, device,
                    epoch, 10, scaler)

    print(f'Validating...')
    out = evaluate(model,
                   sorted(tasks_to_val),
                   criterion,
                   data_loader_val,
                   device=device)
    val_aps, val_aucs = out
    print('')
    val_ap_mean = torch.mean(torch.tensor(list(val_aps.values()))).item()  # pylint: disable=not-callable
    val_auc_mean = torch.mean(torch.tensor(list(val_aucs.values()))).item()  # pylint: disable=not-callable

    if epoch == 0 or epoch % 5 == 4:
        print(f'Validating the training set...')
        out = evaluate(model,
                       sorted(tasks_to_val),
                       criterion,
                       data_loader_train_evaluate,
                       device=device)
        train_aps, train_aucs = out
        train_ap_mean = torch.mean(torch.tensor(list(
            train_aps.values()))).item()
        train_auc_mean = torch.mean(torch.tensor(list(
            train_aucs.values()))).item()
    # save model
    checkpoint = {
        'model':
            model_without_ddp.state_dict(),
        'optimizer':
            optimizer.state_dict(),
        'scaler':
            scaler.state_dict(),
        'epoch':
            epoch,
        'args_dict':
            args_dict,
        'task_mapping':
            task_mapping,
        'norm_mean':
            norm_mean,
        'norm_std':
            norm_std,
        'all_heads':
            all_heads,
        'val_ap':
            val_ap_mean,
        'each_val_ap':
            val_aps,
        'val_auc':
            val_auc_mean,
        'each_val_auc':
            val_aucs,
        'training_time':
            str(datetime.timedelta(seconds=int(time.time() - start_time))),
    }
    if epoch == 0 or val_ap_mean > best_ap:
        utils.save_on_master(checkpoint,
                             os.path.join(output_dir, 'model_best_ap.pth'))
        best_ap = val_ap_mean
    if epoch == 0 or val_auc_mean > best_auc:
        utils.save_on_master(checkpoint,
                             os.path.join(output_dir, 'model_best_auc.pth'))
        best_auc = val_auc_mean
    utils.save_on_master(checkpoint, os.path.join(output_dir, 'checkpoint.pth'))

    # write log
    if utils.is_main_process():
        with open(log_ap_file, 'a') as file:
            if epoch == 0 or epoch % 5 == 4:
                file.write(' '.join([str(val_ap_mean),
                                     str(train_ap_mean)]) + '\n')
            else:
                file.write(str(val_ap_mean) + '\n')
        with open(log_auc_file, 'a') as file:
            if epoch == 0 or epoch % 5 == 4:
                file.write(' '.join([str(val_auc_mean),
                                     str(train_auc_mean)]) + '\n')
            else:
                file.write(str(val_auc_mean) + '\n')

        for task in tasks_to_val:
            with open(log_ap_file_tasks[task], 'a') as file:
                if epoch == 0 or epoch % 5 == 4:
                    file.write(
                        ' '.join([str(val_aps[task]),
                                  str(train_aps[task])]) + '\n')
                else:
                    file.write(str(val_aps[task]) + '\n')
            with open(log_auc_file_tasks[task], 'a') as file:
                if epoch == 0 or epoch % 5 == 4:
                    file.write(
                        ' '.join([str(val_aucs[task]),
                                  str(train_aucs[task])]) + '\n')
                else:
                    file.write(str(val_aucs[task]) + '\n')

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))
