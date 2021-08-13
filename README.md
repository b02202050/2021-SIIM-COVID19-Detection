# 2021-SIIM-COVID19-Detection
6th place solution for [SIIM-FISABIO-RSNA COVID-19 Detection Challenge](https://www.kaggle.com/c/siim-covid19-detection).


## Installation


* Run this docker container: nvcr.io/nvidia/pytorch:20.10-py3 (Please see [the release note](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_20-10.html#rel_20-10) for packages detail if you like to install them by yourself.)
* `pip install -r requirements.txt`

<!-- #region -->
## Data preparation
#### SIIM covid 19 dataset
* Download [the main competition dataset](https://www.kaggle.com/c/siim-covid19-detection/data) then extract to ./dataset/siim-covid19-detection

```bash
$ cd dataset/siim-covid19-detection
$ python dcm2png.py
$ python make_image_symlink.py
```

#### External datasets
* Download [kaggle_chest_xray_covid19_pneumonia](https://www.kaggle.com/prashant268/chest-xray-covid19-pneumonia) then extract to ./dataset/chest-xray-covid19-pneumonia
* Download [kaggle_covidx_cxr2](https://www.kaggle.com/andyczhao/covidx-cxr2) then extract to ./dataset/covidx-cxr2
* Download [kaggle_chest_xray_pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) then extract to ./dataset/chest-xray-pneumonia
* Download [kaggle_curated_chest_xray_image_dataset_for_covid19](https://www.kaggle.com/unaissait/curated-chest-xray-image-dataset-for-covid19) then extract to ./dataset/curated-chest-xray-image-dataset-for-covid19
* Download [kaggle_covid19_xray_two_proposed_databases](https://www.kaggle.com/edoardovantaggiato/covid19-xray-two-proposed-databases) then extract to ./dataset/covid19-xray-two-proposed-databases
* Download [kaggle_ricord_covid19_xray_positive_tests](https://www.kaggle.com/raddar/ricord-covid19-xray-positive-tests) then extract to ./dataset/ricord-covid19-xray-positive-tests
* Download [CXR14](https://nihcc.app.box.com/v/ChestXray-NIHCC) then extract to ./dataset/NIHCC_CXR14/NIHCC_CXR14
* Download [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) then extract to ./dataset/CheXpert-v1.0/
* Download [pneumonia_RSNA](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) then extract to ./dataset/rsna-pneumonia-detection-challenge/

```bash
$ cd dataset/chest-xray-covid19-pneumonia && python make_image_symlink.py && cd -
$ cd dataset/covidx-cxr2 && python unify_file_extensions.py && python make_image_symlink.py && cd -
$ cd dataset/chest-xray-pneumonia && python make_image_symlink.py && cd -
$ cd dataset/curated-chest-xray-image-dataset-for-covid19 && python make_image_symlink.py && cd -
$ cd datasetcovid19-xray-two-proposed-databases && python unify_file_extensions.py && python make_image_symlink.py && cd -
$ cd dataset/curated-chest-xray-image-dataset-for-covid19 && python make_image_symlink.py && cd -
$ cd dataset/CheXpert-v1.0 && python make_image_symlink.py && cd -
$ cd dataset/rsna-pneumonia-detection-challenge && python dcm2png.py && cd - # I did not notice that this preprocessing may not be correct. You can use more correct processing (But may not reproduce my results.)
```
<!-- #endregion -->

dataset structure should be [./dataset_structure.txt](dataset_structure.txt)

<!-- #region -->
## Processed label description (The processing of labels is trivial and is done by the author)
For clarity, the SIIM Covid class index is assigned as the following.
* Class 0: Negative for Pneumonia
* Class 1: Typical Appearance
* Class 2: Indeterminate Appearance
* Class 3: Atypical Appearance

For the class indexing of the external dataset, please refer to the comment in `metadata.yaml`.  
If the patient ID is present, we split the dataset by it.


#### Classification label column
  * ACCNO: file name without file extension
  
#### Detection label column
  * ACCNO: file name without file extension
  * labels_for_model: label string where each paranthesis is made up of (x1, y1, x2, y2, 1). The coordinates are corresponding to the image resised to 1024. 
<!-- #endregion -->

## Solution summary
![Alt text](./images/summary.png?raw=true "Optional Title")
* Multi-task classification:
  * Shared-backbone multi-head classifier
* Sharpness-aware minimization:
  * Foret, Pierre, et al. "Sharpness-aware minimization for efficiently improving generalization." arXiv preprint arXiv:2010.01412 (2020).
* Inverse focal loss:
  * replace `(1-p_t) ** gamma` with `p_t ** gamma` in the original focal loss to suppress outlier samples.
* Stochastic weight averaging:
  * Izmailov, Pavel, et al. "Averaging weights leads to wider optima and better generalization." arXiv preprint arXiv:1803.05407 (2018).
* Attentional-guided context FPN:
  * Cao, Junxu, et al. "Attention-guided context feature pyramid network for object detection." arXiv preprint arXiv:2005.11475 (2020).
* Attentional feature fusion:
  * Dai, Yimian, et al. "Attentional feature fusion." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2021.
* Fixed feature attention:
  * We use feature pyramid from the classification model for attention.
![Alt text](./images/attention.png?raw=true "Optional Title")

We ensemble 5-fold CV study-level and image-level models respectively for our final submission.

<!-- #region -->
## Training
*Let `$SIIM_root` denote the root folder of this package.*

#### Ploting learning curve
When training each model, you can use *plot_metrics.ipynb* to plot the learning curve by simply modifying the `model_folder` with the work_dir of each training run.

#### Multi-task classification pretraining
```bash
$ cd $SIIM_root/src/classification
$ python train_multitask_classification.py
$ python transfer_multitask_pretrained_backbone.py
```
The average AUROC of all validation task is 0.9054

#### SIIM Covid19 study-level training
```bash
$ cd $SIIM_root/src/classification
$ python train_classification.py configs/config_cv0.py
$ python train_classification.py configs/config_cv1.py
$ python train_classification.py configs/config_cv2.py
$ python train_classification.py configs/config_cv3.py
$ python train_classification.py configs/config_cv4.py
```

|                             | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
| --------------------------- | ------ | ------ | ------ | ------ | ------ |
| validation mAP of 4 classes | 0.5735 | 0.5905 | 0.5841 | 0.5825 | 0.5942 |

The final trained study-level model will be saved at `$SIIM_root/src/classification/work_dir/covid19_kaggle_train_cv_5_*_run1/model_best.pth` where * should be replaced with 0 to 4.

#### RSNA pneumonia detection pretraining
```bash
$ cd $SIIM_root/src/detection
$ python train_detection.py configs/config_RSNA_pna_pretraining.py
```

The validation mAP\@0.5 is 0.4352


#### SIIM Covid19 image-level training
```bash
$ cd $SIIM_root/src/detection
$ python train_detection_with_cls_feats.py configs/config_cv0.py
$ python train_detection_with_cls_feats.py configs/config_cv1.py
$ python train_detection_with_cls_feats.py configs/config_cv2.py
$ python train_detection_with_cls_feats.py configs/config_cv3.py
$ python train_detection_with_cls_feats.py configs/config_cv4.py
```

validation mAP\@0.5

|                              | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
| ---------------------------- | ------ | ------ | ------ | ------ | ------ |
| w/o fixed feature attention  | 0.5504 | 0.5412 | 0.5845 | 0.5446 | 0.5952 |
| with fixed feature attention | 0.5547 | 0.5456 | 0.5841 | 0.5472 | 0.6047 |

The final trained image-level model will be saved at `$SIIM_root/src/detection/work_dir/covid19_kaggle_train_cv_5_*_run1/model_best.pth` where * should be replaced with 0 to 4.
<!-- #endregion -->

## Inference
Use [our Kaggle kernel](https://www.kaggle.com/terenceythsu/siim-covid19-2021-6th-place) for model inference.
Simply replace `args_dict['cls']['ckpt_paths']` and `args_dict['det']['ckpt_paths']` with your trained model.

The LB scores of ensemble models are shown below. For each TTA, only horizontal flip is performed.

| Study-level ensemble + Image-level ensemble | Public LB | Private LB |
| ------------------------------------------- | --------- | ---------- |
| w/o TTA w/o fixed feature attention         | 0.633     | 0.621      |
| w/o TTA                                     | 0.633     | 0.625      |
| study-level TTA                             | 0.634     | 0.626      |
| image-level TTA                             | 0.635     | 0.627      |
| both TTA                                    | 0.636     | 0.628      |


## Wish you have fun and save more lives in the world! :hospital:
