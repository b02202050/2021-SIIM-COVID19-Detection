args_dict = dict()

args_dict['task_name'] = 'covid19_kaggle_train_cv_5_0'
args_dict['choose_best'] = 'ap'
args_dict['metric_class'] = -1
args_dict['CUDA_VISIBLE_DEVICES'] = '0'
args_dict['channel'] = 3 # 3
args_dict['random_seed'] = None # None

args_dict['pretrained'] = 'pretrained/pretrained_run1_transferred.pth'

args_dict['sync_bn'] = False
args_dict['optimizer'] = 'Adam'
args_dict['epoch_step_lr'] = []
args_dict['warmup_lr'] = True
args_dict['cosine_lr'] = False
args_dict['weight_decay'] = 0  # 0
args_dict['batch_size'] = 16
args_dict['aug_sample_multiplier'] = 1 # 1
args_dict['accumulate_grad_batches'] = 1
args_dict['lr'] = 1e-3 / 256 * args_dict['batch_size'] * len(args_dict['CUDA_VISIBLE_DEVICES'].split(',') * args_dict['accumulate_grad_batches'])

args_dict['epochs'] = 20
args_dict['amp'] = True
args_dict['model_name'] = 'tf_efficientnet_b7_ns'
args_dict['model_input_size'] = 512
args_dict['output_dir'] = f"work_dir/{args_dict['task_name']}_run1"

# Exclusive args (Please note that some are mutually exclusive)
args_dict['class_weight'] = None
args_dict['use_focal_loss'] = True
args_dict['FL_suppress'] = 'hard'
args_dict['FL_alpha'] = 0.5
args_dict['FL_gamma'] = 0.5
args_dict['soft_label'] = 1.0
args_dict['use_sigmoid'] = True


args_dict['train_blackout'] = False
args_dict['dataset_shrinking_factor'] = 1.0
args_dict['fix_backbone'] = False
args_dict['SWA'] = False
args_dict['SWAG_diag'] = False
args_dict['SAM'] = True
args_dict['shapleys'] = None # None, 'work_dir/calc_shapley/trial_2021_05_08_12_05_54_872714/unbiased_shapleys.pth'
args_dict['removing_by_shapley'] = 0.1 # ratios or 'neg'
args_dict['final_pool'] = 'avg' # avg, max, multipcam, pcam, softmax

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
args_dict['equalize'] = False
args_dict['standardize'] = False
args_dict['restrict_square_resize'] = False
args_dict['RandomResizedCrop'] = True
args_dict['random_perspective'] = True
args_dict['elastic_deformation'] = True
args_dict['RandAugment'] = True

args_dict['finetune'] = False
args_dict['load_optim'] = False
args_dict['finetune_model_file'] = ''
args_dict['tag'] = ''