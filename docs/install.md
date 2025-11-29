# Set up a environment for DuOcc

### Set up a new virtual environment
```bash
conda create -n duocc python=3.8 -y
conda activate duocc 
```

### Install packpages using pip3
```bash
duocc_path="path/to/duocc"
cd ${duocc_path}
conda install -y -c conda-forge cudatoolkit-dev=11.7
pip install --upgrade pip
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch_geometric==2.5.3 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
python -m pip install setuptools==69.5.1
pip install pyg_lib torch_scatter torch_sparse torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install einops
pip3 install -r requirement.txt
apt-get update && apt-get install -y ninja-build
```

### Compile the deformable_aggregation and others CUDA op
```bash
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../deform_attn_3d 
python setup.py build_ext --inplace
cd ../projects/mmdet3d_plugin/models/bev_pool_v2
python setup.py build_ext --inplace
cd ../../../..
```



---
-> Next Page: [Prepare The Dataset](./prepare_data.md)