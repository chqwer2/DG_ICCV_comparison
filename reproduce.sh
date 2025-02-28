

# HRFormer
### Example conda environment setup
conda create --name hgformer python=3.8  -y
conda activate hgformer
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade matplotlib
pip install -U opencv-python

# under your working directory
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

#pip install git+https://github.com/mcordts/cityscapesScripts.git

cd HGFormer-main
pip install -r requirements.txt
cd hgformer/modeling/pixel_decoder/ops
sh make.sh



# HRDA-dg
cd HRDA-dg
conda create -n hrda python=3.12 -y
conda activate hrda
pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade matplotlib
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7


