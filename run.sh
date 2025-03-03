


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

# Test
python demo/inference.py --config-file configs/cityscapes/hgformer_swin_tiny_bs16_20k.yaml \
--input datasets/acdc/rgb_anon/all/test --output path_to_output \
--opts MODEL.WEIGHTS path_to_checkpoint

python plain_train_net.py --num-gpus 8 --config-file configs/cityscapes/hgformer_swin_tiny_bs16_20k.yaml \
--eval-only MODEL.WEIGHTS path_to_checkpoint OUTPUT_DIR path_to_output



 ./output/model_0004999.pth


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
#Val Epoch:0.000,  PA1:0.996, MPA1:0.924, MIoU1:0.809, FWIoU1:0.992, PC:0.858, Dice:0.888



TEST_ROOT=$1
CONFIG_FILE="${TEST_ROOT}/*${TEST_ROOT: -1}.json"
CHECKPOINT_FILE="${TEST_ROOT}/latest.pth"
SHOW_DIR="${TEST_ROOT}/preds"
echo 'Config File:' $CONFIG_FILE
echo 'Checkpoint File:' $CHECKPOINT_FILE
echo 'Predictions Output Directory:' $SHOW_DIR
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --show-dir ${SHOW_DIR} --opacity 1






