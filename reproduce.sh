# Download Dataset


https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets

/bask/projects/j/jiaoj-rep-learn/Hao/datasets



# HRFormer
### Example mamba environment setup
mamba create --name hg python=3.12  -y
mamba activate hg
pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade matplotlib   #detectron2
pip install -U opencv-python
mamba install pytorch torchvision cudatoolkit torchaudio pytorch-cuda  -c pytorch -c nvidia  -y

mamba install nvidia::cuda-nvcc  -y
cd /home/cbtil3/hao/repo/DG_ICCV_comparison/HGFormer-main/hgformer/modeling/pixel_decoder/ops
_USE_CXX11_ABI=0
sh make.sh

#mamba install nvidia/label/cuda-11.3.0::cuda -y
#conda install pytorch torchvision torchaudio -c pytorch

# under your working directory
#python -m pip install detectron2 -f \
#  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

pip install timm  einops  scipy   scikit-image


pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 torchaudio==2.4.0+cu124       \
 --extra-index-url https://download.pytorch.org/whl/cu124

#pip install git+https://github.com/mcordts/cityscapesScripts.git

mamba install -c nvidia cuda-toolkit  -y
pip install cityscapesScripts



#  ---------------------------------------------

cd HGFormer-main
pip install -r requirements.txt
cd hgformer/modeling/pixel_decoder/ops
sh make.sh


# --------------------------------------------------
# HRDA-dg
cd HRDA-dg
mamba create -n hrda python=3.12 -y
mamba activate hrda
pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade matplotlib
pip install -r requirements.txt  -f https://download.pytorch.org/whl/torch_stable.html
mamba install -c nvidia cuda-toolkit -y
pip install monai['all']



# --------------------------------------------------
cd SAN_SAW-main
mamba create -n san python=3.12 -y
mamba activate san
pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade matplotlib
# pip install -r requirements.txt  -f https://download.pytorch.org/whl/torch_stable.html
mamba install -c nvidia cuda-toolkit -y

mamba install pytorch torchvision cudatoolkit torchaudio pytorch-cuda  -c pytorch -c nvidia  -y
pip install tensorboardX   kmeans1d  imageio


