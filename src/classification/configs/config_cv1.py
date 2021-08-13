args_dict = dict()

args_dict['task_name'] = 'covid19_kaggle_train_cv_5_1'
args_dict['choose_best'] = 'ap'
args_dict['CUDA_VISIBLE_DEVICES'] = '0'
args_dict['random_seed'] = None  # None

args_dict['pretrained'] = 'pretrained/pretrained_run1_transferred.pth'

args_dict['sync_bn'] = False
args_dict['optimizer'] = 'Adam'
args_dict['warmup_lr'] = True
args_dict['weight_decay'] = 0
args_dict['batch_size'] = 16
args_dict['lr'] = 1e-3 / 256 * args_dict['batch_size'] * len(
    args_dict['CUDA_VISIBLE_DEVICES'].split(','))

args_dict['epochs'] = 20
args_dict['amp'] = True
args_dict['model_name'] = 'tf_efficientnet_b7_ns'
args_dict['model_input_size'] = 512
args_dict['output_dir'] = f"work_dir/{args_dict['task_name']}_run1"

args_dict['use_focal_loss'] = True
args_dict['FL_suppress'] = 'hard'
args_dict['FL_alpha'] = 0.5
args_dict['FL_gamma'] = 0.5
args_dict['use_sigmoid'] = True

args_dict['SAM'] = True

args_dict['augmentation_prob_multiplier'] = 0.5
args_dict['flip'] = True
args_dict['flip_vertical'] = True
args_dict['shift'] = True
args_dict['shift_mode'] = 'repeat'
args_dict['color_jitter'] = True
args_dict['color_jitter_factor'] = {'brightness': 0.4, 'contrast': 0.4}
args_dict['rotation'] = True
args_dict['rotation_degree'] = 25
args_dict['resize'] = False
args_dict['aspect_ratio'] = False
args_dict['blur'] = False
args_dict['noise'] = False
args_dict['RandomResizedCrop'] = True
args_dict['random_perspective'] = True
args_dict['elastic_deformation'] = True
args_dict['RandAugment'] = True
