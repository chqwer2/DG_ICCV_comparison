


# HRFormer
### Example conda environment setup
conda create --name hgformer python=3.8 -y
conda activate hgformer
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

# under your working directory
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

#pip install git+https://github.com/mcordts/cityscapesScripts.git

cd ..
git clone https://github.com/dingjiansw101/HGFormer.git
cd HGFormer
pip install -r requirements.txt
cd hgformer/modeling/pixel_decoder/ops
sh make.sh




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










