### Prepare the data

### Dowload the nuScenes dataset
Download the [nuScenes dataset](https://www.nuscenes.org/nuscenes#download) and create symbolic links.
```bash
cd ${duocc_path}
mkdir data
ln -s path/to/nuscenes ./data
```

### Dowload the Occ3D-nuScenes dataset
For Occupancy Prediction task, download Occ3D-nuScenes from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) and place it in data/nuscenes/gts 

### Prepare pkl files
Pack the meta-information and labels of the dataset, and generate the required .pkl files.
```bash
python3 tools/create_data_duocc.py
```

### Generate anchors by K-means
```bash
python3 tools/anchor_generator.py --ann_file ./data/nuscenes_anno_pkls/nuscenes_occ_infos_aug_train.pkl
```

### Download pre-trained weights
Download the required backbone [BEVDet serires](https://github.com/HuangJunJie2017/BEVDet) and place it in ckpts folder.



### The Overall Structure
Please make sure the structure of DuOcc is as follows:
```shell script
DuOcc
├── projects/
├── tools/
├── ckpts/
│   ├── bevdet-r50-4d-depth-cbgs.pth
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   │   ├── nusceness_occ_infos_train.pkl
│   │   ├── nusceness_occ_infos_val.pkl
│   ├── nuscenes_anno_pkls/
│   │   ├── nuscenes_occ_infos_aug_train.pkl
│   │   ├── nuscenes_occ_infos_aug_val.pkl
├── nuscenes_kmeans900_42m.npy
└── others
```
→ Back to: [Training & Inference](/#%EF%B8%8F-training--inference)
