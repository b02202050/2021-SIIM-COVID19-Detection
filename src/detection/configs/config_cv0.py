import myUtils

args_dict = dict()
# environment configs
args_dict['CUDA_VISIBLE_DEVICES'] = '0'

# data config
args_dict['task_name'] = 'covid19_kaggle_train_cv_5_0'
args_dict['cls_model_folder'] = '../classification/work_dir/covid19_kaggle_train_cv_5_0_run1'
args_dict['output_dir'] = f"work_dir/{args_dict['task_name']}_run1"

# model config
args_dict['input_size'] = 800
args_dict['backbone_body'] = 'tf_efficientnet_b7_ns' 
args_dict['norm_layer'] = myUtils.FrozenBatchNorm2dWithEpsilon 
args_dict['ACFPN'] = True
args_dict['iAFF'] = True

args_dict['fine_tune'] = True
args_dict['fine_tune_model_file'] = 'work_dir/pneumonia_RSNA_run1/model_best.pth'
args_dict['fine_tune_non_strict'] = True
args_dict['user_specified_backbone_pretrained'] = False
args_dict['backbone_pretrained_file'] = ''
args_dict['sync_bn'] = False

# training config    
args_dict['num_epochs'] = 50
args_dict['batch_size'] = 8 
args_dict['optim'] = 'torch.optim.Adam' 
args_dict['amsgrad'] = True
args_dict['learning_rate'] = 1e-4
args_dict['weight_decay'] = 0
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
args_dict['shift'] = True
args_dict['shift_mode'] = 'WRAP' 
args_dict['elastic_deformation'] = True
args_dict['elastic_deformation_sigma'] = 1/40
args_dict['random_perspective'] = True
args_dict['RandomResizedCrop'] = True