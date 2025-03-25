# ------------------ SHADE ------------------
cd SHADE-master/
conda activate flow

pip install ipdb

# Remember to change the datasets.num_classes = 3, if changing dataset

python train.py \
  --dataset brain \
  --val_dataset brain \
  --arch network.deepv3.DeepR50V3PlusD \
  --city_mode 'train' \
  --lr_schedule poly \
  --lr 0.01 \
  --poly_exp 0.9 \
  --max_cu_epoch 10000 \
  --class_uniform_pct 0.5 \
  --class_uniform_tile 1024 \
  --crop_size 768 \
  --scale_min 0.5 \
  --scale_max 2.0 \
  --rrotate 0 \
  --max_iter 40000 \
  --bs_mult 4 \
  --gblur \
  --color_aug 0.5 \
  --date 2206 \
  --exp r50os16_gtav \
  --ckpt ./logs/ \
  --tb_path ./logs/    --max_iter 100000

# Test,  -m torch.distributed.launch --nproc_per_node=1
snapshot=./logs/2206/r50os16_gtav/03_25_09/last_None_epoch_199_mean-iu_0.00000.pth

python  valid.py \
  --val_dataset brain \
  --arch network.deepv3.DeepR50V3PlusD \
  --date 2207 \
  --bs_mult_val 12 \
  --exp r50os16_val \
  --snapshot $snapshot


# --local_rank 0

#   --wandb_name r50os16_gtav
#  --val_dataset cityscapes bdd100k mapillary \



# --------------------------------  HRFormer --------------------------------
#  prepare the models pre-trained on ImageNet classificaiton following [tools/README.md](tools/README.md). Finally run:
#  8 GPUs
#python plain_train_net.py --num-gpus 8 \
#  --config-file configs/cityscapes/hgformer_swin_tiny_bs16_20k.yaml OUTPUT_DIR path_to_output
cd HGFormer-main/
conda activate hg

wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
python tools/convert-pretrained-swin-model-to-d2.py swin_tiny_patch4_window7_224.pth swin_tiny_patch4_window7_224.pkl


cd HGFormer-main/
conda activate hg

# Train
python plain_train_net.py \
  --config-file configs/cityscapes/hgformer_swin_tiny_bs16_20k.yaml \
  --num-gpus 1 --gpu 0  #  --batch_size 16 --output_dir ./output

# Test
conda activate hg

checkpoint=./output/model_0019999.pth
path_to_output=./output

python demo/inference.py --config-file configs/cityscapes/hgformer_swin_tiny_bs16_20k.yaml \
--input datasets/acdc/rgb_anon/all/test --output $path_to_output \
--opts MODEL.WEIGHTS $checkpoint


HRFormer
# mIoU | mDice |
#+-------+-------+
# 88.43 | 93.96 |




# ---------------------------------- HRDA-dg ----------------------------------
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

### Test
23100 epoch


#TEST_ROOT=$1
#CONFIG_FILE="${TEST_ROOT}/*${TEST_ROOT: -1}.json"

mamba activate hrda
TEST_ROOT=work_dirs/local-exp50/250303_1416_gtaCAug2cs_dgdacs_fdthings_srconly_rcs001_shade_shb_daformer_sepaspp_mitb5_poly10warm_s0_f2b81
CONFIG_FILE=${TEST_ROOT}/250303_1416_gtaCAug2cs_dgdacs_fdthings_srconly_rcs001_shade_shb_daformer_sepaspp_mitb5_poly10warm_s0_f2b81.json
SHOW_DIR="$./preds"
CHECKPOINT_FILE="${TEST_ROOT}/iter_18000.pth"


python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} \
        --eval mIoU mDice --show-dir ${SHOW_DIR} --opacity 1

# hrda
# aAcc   |  mIoU |  mAcc | mDice |
#+-------+-------+-------+-------+
#| 99.76 | 87.68 | 93.59 | 93.22 |





# --------------------------- SAN-SAW ---------------------------
### Step 1
wget http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth
mv DeepLab_resnet_pretrained_init-f81d91e8.pth \
   pretrained_model/DeepLab_resnet_pretrained_init-f81d91e8.pth



mkdir -p /home/cbtil3/hao/repo/DG_ICCV_comparison/SAN-SAW-main/log/gta5_pretrain_2

### Step 2
cd DG_ICCV_comparison/
cd SAN-SAW-main
mamba activate hrda
python  tools/train.py  # Test:  --validate_only   --pth    log/gta5_pretrain_2/718.pt
# Test
python  tools/train.py  --validate_only   --pth    log/gta5_pretrain_2/718.pt
# SAN-SAW
# Val Epoch:0.000, MIoU1:0.809, Dice:0.888







