


# HRFormer
#  prepare the models pre-trained on ImageNet classificaiton following [tools/README.md](tools/README.md). Finally run:
#  8 GPUs
python plain_train_net.py --num-gpus 8 \
  --config-file configs/cityscapes/hgformer_swin_tiny_bs16_20k.yaml OUTPUT_DIR path_to_output

python plain_train_net.py \
  --config-file configs/cityscapes/hgformer_swin_tiny_bs16_20k.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH SET_TO_SOME_REASONABLE_VALUE SOLVER.BASE_LR SET_TO_SOME_REASONABLE_VALUE


# HRDA-dg
### Step 1
sh tools/download_checkpoints.sh  # No  Needed

### Step 2
extract them to `data/cityscapes`

### Step 3
python run_experiments.py --exp 50


### Step 3
python -m tools.test path/to/config_file path/to/checkpoint_file --eval mIoU --dataset BDD100K
python -m tools.test path/to/config_file path/to/checkpoint_file --eval mIoU --dataset Mapillary --eval-option efficient_test=True




# SAN-SAW
### Step 1
wget http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth .
mv DeepLab_resnet_pretrained_init-f81d91e8.pth pretrained_model/DeepLab_resnet_pretrained_init-f81d91e8.pth

### Step 2
tools\train.py    # 640×640 resolution










