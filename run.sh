


# HRFormer
#  prepare the models pre-trained on ImageNet classificaiton following [tools/README.md](tools/README.md). Finally run:
#  8 GPUs
#python plain_train_net.py --num-gpus 8 \
#  --config-file configs/cityscapes/hgformer_swin_tiny_bs16_20k.yaml OUTPUT_DIR path_to_output
cd HGFormer-main/
conda activate hg

wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
python tools/convert-pretrained-swin-model-to-d2.py swin_tiny_patch4_window7_224.pth swin_tiny_patch4_window7_224.pkl


python plain_train_net.py \
  --config-file configs/cityscapes/hgformer_swin_tiny_bs16_20k.yaml \
  --num-gpus 1   #  --batch_size 16 --output_dir ./output


  #SOLVER.IMS_PER_BATCH SET_TO_SOME_REASONABLE_VALUE SOLVER.BASE_LR SET_TO_SOME_REASONABLE_VALUE


# HRDA-dg
### Step 1
mamba activate hrda
sh tools/download_checkpoints.sh  # No  Needed
# or https://huggingface.co/IMvision12/mit-model/blob/03d2d4dd1828f797fd75f3bd1c9b227744daf050/mit_b5.pth?
mv /home/cbtil3/Downloads/mit_b5.pth pretrained/mit_b5.pth

mamba activate flow
python tools/REFUGE.py   # generate the sample.json

# CP Data
#bash ./tools/mv_refuge_data.sh


### Step 3  TRAIN
mamba activate hrda
python run_experiments.py --exp 50

### Step 3
python -m tools.test path/to/config_file path/to/checkpoint_file --eval mIoU --dataset BDD100K
python -m tools.test path/to/config_file path/to/checkpoint_file --eval mIoU --dataset Mapillary --eval-option efficient_test=True

#  mkdir -p data/gta/images


# SAN-SAW
### Step 1
wget http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth
mv DeepLab_resnet_pretrained_init-f81d91e8.pth \
   pretrained_model/DeepLab_resnet_pretrained_init-f81d91e8.pth



mkdir -p /home/cbtil3/hao/repo/DG_ICCV_comparison/SAN-SAW-main/log/gta5_pretrain_2
### Step 2
cd DG_ICCV_comparison/
cd SAN-SAW-main
mamba activate hrda
python  tools/train.py    # 640×640 resolution





