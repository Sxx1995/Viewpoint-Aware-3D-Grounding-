# Viewpoint-Aware Visual Grounding in 3D Scenes
### [OpenReview](https://openreview.net/forum?id=LX9gTkDbqE&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3Dthecvf.com%2FCVPR%2F2024%2FConference%2FAuthors%23your-submissions))
This is the official implement of CVPR paper "Viewpoint-Aware Visual Grounding in 3D Scenes".
This project is adapted from [BUTD_DETR](https://github.com/nickgkan/butd_detr).

## Install
Install the Python environment with Anaconda:

```conda env create -f environment.yml```

Same as [BUTD_DETR](https://github.com/nickgkan/butd_detr), our model adapts [PointNet++](http://arxiv.org/abs/1706.02413) as our visual backbone. 

To enable the PointNet++ libary, please compile the CUDA layers with `sh init.sh`.

## Data Preparation
To parepare ScanRefer, Sr3D, and Nr3D dataset, please follow the [Data Preparation](https://github.com/nickgkan/butd_detr?tab=readme-ov-file#data-preparation) of BUTD_DETR.

To generate the synthetic dataset, please run the following code:

```sh prepare_synthetic_data.sh```

The process will result in 2 files which are `syn_ScanRefer_filtered_train.json` and `scanrefer_pred_spans_train.json`.

The structure of dataset folder should be:

```
DATA_ROOT
│
├── scanrefer
│   ├── syn_ScanRefer_filtered_train.json
│   └── syn_ScanRefer_filtered_train.txt
│
└── scanrefer_pred_spans_train.json
```

## Train
With the syntethic data, you can run train a model from scratch with the following code:

```
python -m torch.distributed.launch --nproc_per_node 2 --master_port $RANDOM \
    train_dist_mod_2stage.py --num_decoder_layers 6 --num_encoder_layers 6 \
    --use_color --use_global_obj \
    --weight_decay 0.0005 --viewpoint_weight 10.0\
    --data_root DATA_ROOT/ \
    --start_epoch 105 --val_freq 1 --batch_size 20 --save_freq 3 --print_freq 1 --num_workers 0\
    --lr_backbone=2e-3 --lr=2e-4 \
    --dataset syn_scanrefer scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/bdetr \
    --pp_checkpoint DATA_ROOT/gf_detector_l6o256.pth \
    --butd --self_attend --augment_det
```

To pretrain the model with only synthetic data, you can use the following code:

```
python -m torch.distributed.launch --nproc_per_node 2 --master_port $RANDOM \
    train_dist_mod_2stage.py --num_decoder_layers 6 --num_encoder_layers 6\
    --use_color \
    --weight_decay 0.0005 \
    --data_root DATA_ROOT/ \
    --val_freq 10 --batch_size 28 --save_freq 5 --print_freq 10 --num_workers 2\
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset syn_scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/pretrain \
    --pp_checkpoint DATA_ROOT/gf_detector_l6o256.pth \
    --butd --self_attend --augment_det
```
