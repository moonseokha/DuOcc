import random
import math
import os
from os import path as osp
import cv2
import tempfile
import copy
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.eval.detection.config import config_factory as det_configs
from nuscenes.eval.common.config import config_factory as track_configs
from pyquaternion import Quaternion
from .occ_metrics import Metric_mIoU, Metric_FScore, Metric_mIoU_surround
# from .ray_metrics import main as calc_rayiou
# from .ego_pose_dataset import EgoPoseDataset
from torch.utils.data import DataLoader
import mmcv
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from .utils import (
    draw_lidar_bbox3d_on_img,
    draw_lidar_bbox3d_on_bev,
)
import pickle
import copy

colors_map = np.array(
    [
        [  0,   0,   0, 255],       # other                
        [255, 120,  50, 255],       # barrier              
        [255, 192, 203, 255],       # bicycle              
        [255, 255,   0, 255],       # bus                  
        [  0, 150, 245, 255],       # car                  
        [  0, 255, 255, 255],       # construction_vehicle 
        [200, 180,   0, 255],       # motorcycle           
        [255,   0,   0, 255],       # pedestrian           
        [255, 140, 150, 255],       # traffic_cone         
        [135,  60,   0, 255],       # trailer              
        [160,  32, 240, 255],       # truck                
        [255,   0, 255, 255],       # driveable_surface     
        [139, 137, 137, 255],       # other_flat            
        [ 75,   0,  75, 255],       # sidewalk             
        [150, 240,  80, 255],       # terrain                        
        [222, 184, 135, 255],       # manmade              
        [  0, 175,   0, 255],       # vegetation
        [255, 255, 255, 0],         # empty
    ])



@DATASETS.register_module()
class NuScenes3DDetTrackDataset(Dataset):
    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }
    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }
    CLASSES = (
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    )
    occ_classes = ['barrier','bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
                'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
                'other_flat', 'sidewalk', 'terrain', 'manmade','vegetation']
    
    ID_COLOR_MAP = [
        (245, 150, 0), # car
        (240, 32, 160), # truck
        (255, 255, 0),  # C.V
        (0, 255, 255),  # bus
        (0, 60, 135), # trailer
        (50, 120, 255), # barrier
        (0, 180, 200), # motorcycle
        (203, 192, 255), # bicycle
        (0, 0, 255), # pedestrian
        (150, 240, 255), # traffic cone
    ]

    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        occ_classes = None,
        load_interval=1,
        num_cams = 6,
        with_velocity=True,
        modality=None,
        test_mode=False,
        det3d_eval_version="detection_cvpr_2019",
        track3d_eval_version="tracking_nips_2019",
        version="v1.0-trainval",
        use_valid_flag=False,
        vis_score_threshold=0.3,
        data_aug_conf=None,
        sequences_split_num=1,
        with_seq_flag=False,
        keep_consistent_seq_aug=True,
        tracking=False,
        tracking_threshold=0.2,
        use_semantic = False,
        use_different_aug = False,
        valid_for_detection = False,
        surround_occ = False,
        **kwargs,
    ):
        self.use_different_aug = use_different_aug
        self.version = version
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__()
        self.valid_for_detection = valid_for_detection
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.box_mode_3d = 0
        self.num_cams = num_cams
        self.use_semantic = use_semantic
        self.surround_occ = surround_occ
        if 'save_results' in kwargs:
            self.save_results = kwargs['save_results']
        else:
            self.save_results = None
        if classes is not None:
            self.CLASSES = classes
        if occ_classes is not None:
            self.occ_classes = occ_classes
        else:
            self.occ_classes = occ_classes
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.ann_file)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        self.num_empty_iter = 0
        self.with_velocity = with_velocity
        self.det3d_eval_version = det3d_eval_version
        self.det3d_eval_configs = det_configs(self.det3d_eval_version)
        self.track3d_eval_version = track3d_eval_version
        self.track3d_eval_configs = track_configs(self.track3d_eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.vis_score_threshold = vis_score_threshold

        self.data_aug_conf = data_aug_conf
        self.tracking = tracking
        self.tracking_threshold = tracking_threshold
        self.sequences_split_num = sequences_split_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug
        self.current_aug = None
        self.last_id = None
        if with_seq_flag:
            self._set_sequence_group_flag()

    def __len__(self):
        return len(self.data_infos)

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]["sweeps"]) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)
        if self.sequences_split_num != 1:
            if self.sequences_split_num == "all":
                self.flag = np.array(
                    range(len(self.data_infos)), dtype=np.int64
                )
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(
                                    bin_counts[curr_flag]
                                    / self.sequences_split_num
                                ),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )

                    for sub_seq_idx in (
                        curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert (
                    len(np.bincount(new_flags))
                    == len(np.bincount(self.flag)) * self.sequences_split_num
                )
                self.flag = np.array(new_flags, dtype=np.int64)


    def get_augmentation_3d(self):
        if self.data_aug_conf is None:
            return None
       
        if not self.test_mode:
            rotate_3d = np.random.uniform(*self.data_aug_conf["rot3d_range"])
            if self.data_aug_conf["bda_aug"] is not None:
                rotate_bda = np.random.uniform(*self.data_aug_conf["bda_aug"]['rot_lim'])
                scale_bda = np.random.uniform(*self.data_aug_conf["bda_aug"]['scale_lim'])
                flip_dx = np.random.uniform() < self.data_aug_conf["bda_aug"]['flip_dx_ratio']
                flip_dy = np.random.uniform() < self.data_aug_conf["bda_aug"]['flip_dy_ratio']
            else:
                rotate_bda = 0
                scale_bda = 1.0
                flip_dx = False
                flip_dy = False
        else:
            rotate_3d = 0
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
            
        aug_config = {
            "rotate_3d": rotate_3d,
            "rotate_bda" : rotate_bda,
            "scale_bda": scale_bda,
            "flip_dx": flip_dx,
            "flip_dy": flip_dy,
            "test_mode": self.test_mode,
        }
        return aug_config

    def get_augmentation_2d(self):
        if self.data_aug_conf is None:
            return None
        # print("self.bda_aug_conf : ",self.bda_aug_conf )
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])

        else:
            resize = float(fW) / float(W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0

        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "test_mode": self.test_mode,
        }
        return aug_config

    def get_augmentation(self):
        if self.use_different_aug:
            aug_config_2d = [self.get_augmentation_2d() for _ in range(self.num_cams)]
        else:
            aug_config_2d = [self.get_augmentation_2d()] * self.num_cams
        aug_config_3d = self.get_augmentation_3d()
        aug_config = {"2d": aug_config_2d, "3d": aug_config_3d}
        
        return aug_config
        
    def __getitem__(self, idx):
        while True:
            if isinstance(idx, dict):
                aug_config = idx["aug_config"]
                idx = idx["idx"]
            else:
                aug_config = self.get_augmentation()
            data = self.get_data_info(idx)
            data["aug_config"] = aug_config
            data = self.pipeline(data)
            if not self.test_mode and len(data['gt_bboxes_3d']) == 0:
                idx = self._rand_another(idx)
                self.num_empty_iter += 1
                continue
            self.num_empty_iter = 0
            return data

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        if self.num_empty_iter > 10:
            pool = np.where(self.flag != self.flag[idx])[0]
        else:
            pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def get_cat_ids(self, idx):
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format="pkl")
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        print(self.metadata)
        return data_infos

    def get_data_info(self, index):
        info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"] / 1e6,
            lidar2ego_translation=info["lidar2ego_translation"],
            lidar2ego_rotation=info["lidar2ego_rotation"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
        )
        if 'occ_path' in info:
            input_dict['occ_gt_path'] = info['occ_path']
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = pyquaternion.Quaternion(
            info["lidar2ego_rotation"]
        ).rotation_matrix
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])
        ego2global = np.eye(4)
        ego2global[:3, :3] = pyquaternion.Quaternion(
            info["ego2global_rotation"]
        ).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])
        input_dict["lidar2ego"] = lidar2ego.copy()
        if 'gt_boxes_ego' in info:
            input_dict["box2global"] = ego2global
        else:
            input_dict["box2global"] = ego2global @ lidar2ego  #lidar2global


        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            cam_intrinsic = []
            cam2coors = []
            sensor2egos = []
            camego2globals = []
            ego2sensors = []
            cam2lidars = []
            for cam_type, cam_info in info["cams"].items():
                image_paths.append(cam_info["data_path"])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                lidar2cam_t = (
                    cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                )
                cam2lidar = np.eye(4)
                cam2lidar[:3, :3] = cam_info["sensor2lidar_rotation"]
                cam2lidar[:3, 3] = cam_info["sensor2lidar_translation"]
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = copy.deepcopy(cam_info["cam_intrinsic"])
                cam_intrinsic.append(intrinsic)
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T
                lidar2img_rts.append(lidar2img_rt)
                cam2lidars.append(cam2lidar)
                w,x,y,z = cam_info["sensor2ego_rotation"]
                sensor2ego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sensor2ego_tran = torch.Tensor(
                    cam_info["sensor2ego_translation"])
                sensor2ego = sensor2ego_rot.new_zeros((4, 4))
                sensor2ego[:3, :3] = sensor2ego_rot
                sensor2ego[:3, -1] = sensor2ego_tran
                sensor2ego[3, 3] = 1
                # camego2global_rot = torch.Tensor(
                #     Quaternion(cam_info["ego2global_rotation"]).rotation_matrix)
                # camego2global_tran = torch.Tensor(
                #     cam_info["ego2global_translation"])
                # camego2global = camego2global_rot.new_zeros((4, 4))
                # camego2global[:3, :3] = camego2global_rot
                # camego2global[:3, -1] = camego2global_tran
                
                
                cam2coors.append(sensor2ego.clone())
                # sensor2egos.append(sensor2ego)
                # print("sensor2ego",sensor2ego)
                # print("viewpad",viewpad)
                # print("sensor2ego.shape",sensor2ego.shape)
                # print("viewpad.shape",viewpad.shape)
                ego2sensors.append(viewpad @ np.array(torch.inverse(sensor2ego)))
                # camego2globals.append(camego2global) 


                
            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsic,
                    ego2sensors=ego2sensors,
                    cam2coor=np.float32(np.stack(cam2coors)),
                    cam2lidar=np.float32(np.stack(cam2lidars)),
                    # sensor2ego=sensor2egos,
                    # camego2global=camego2globals,
                    
                )
            )


        # if self.modality["use_camera"]:
        #     image_paths = []
        #     lidar2img_rts = []
        #     cam_intrinsic = []
        #     img2lidar_rots = []
        #     img2lidar_trans = []
        #     img2lidars = []
        #     for cam_type, cam_info in info["cams"].items():     
        #         image_paths.append(cam_info["data_path"])
        #         # obtain lidar to image transformation matrix
        #         img2lidar_rots.append(copy.deepcopy(cam_info["sensor2lidar_rotation"]))
        #         img2lidar_trans.append(copy.deepcopy(cam_info["sensor2lidar_translation"]))
        #         lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
        #         lidar2cam_t = (
        #             cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
        #         )
        #         lidar2cam_rt = np.eye(4)
        #         lidar2cam_rt[:3, :3] = lidar2cam_r.T
        #         lidar2cam_rt[3, :3] = -lidar2cam_t
        #         intrinsic = copy.deepcopy(cam_info["cam_intrinsic"])
        #         cam_intrinsic.append(intrinsic)
        #         viewpad = np.eye(4)
        #         viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
        #         lidar2img_rt = viewpad @ lidar2cam_rt.T
        #         lidar2img_rts.append(lidar2img_rt)
        #         img2lidar = np.eye(4)
        #         img2lidar[:3, :3] = cam_info["sensor2lidar_rotation"]
        #         img2lidar[:3, 3] = cam_info["sensor2lidar_translation"]
        #         img2lidars.append(copy.deepcopy(img2lidar))
        #     input_dict.update(
        #         dict(
        #             img_filename=image_paths,
        #             lidar2img=lidar2img_rts,
        #             cam_intrinsic=cam_intrinsic,
        #             img2lidar_trans=np.float32(np.stack(img2lidar_trans)),
        #             img2lidar_rots=np.float32(np.stack(img2lidar_rots)),
        #             img2lidars=np.float32(np.stack(img2lidars)),
        #             # cams=info["cams"],
        #         )
        #     )

        # if not self.test_mode and "" not in self.pipeline :
        annos = self.get_ann_info(index)
        input_dict.update(annos)
        return input_dict
    
    
    def get_ann_info(self, index):
        info = self.data_infos[index]
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        if 'gt_boxes_ego' in info:
            gt_bboxes_3d = np.array(info["gt_boxes_ego"])
        else:
            gt_bboxes_3d = info["gt_boxes"][mask]
        gt_labels_3d = []
        if 'gt_lables_ego' in info:
            gt_labels_3d = np.array(info['gt_lables_ego'])
        else:
            gt_names_3d = info["gt_names"][mask]
            for cat in gt_names_3d:
                if cat in self.CLASSES:
                    gt_labels_3d.append(self.CLASSES.index(cat))
                else:
                    gt_labels_3d.append(-1)
            gt_labels_3d = np.array(gt_labels_3d)
        if 'gt_boxes_ego' not in info:
            if self.with_velocity:
                gt_velocity = info["gt_velocity"][mask]
                nan_mask = np.isnan(gt_velocity[:, 0])
                gt_velocity[nan_mask] = [0.0, 0.0]
                gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
        )
        if 'instance_inds_ego' in info:
            instance_inds = np.array(info["instance_inds_ego"], dtype=np.int)
            anns_results["instance_inds"] = instance_inds
        else:
            if "instance_inds" in info:
                instance_inds = np.array(info["instance_inds"], dtype=np.int)[mask]
                anns_results["instance_inds"] = instance_inds
                
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None, tracking=False):
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(
                det, threshold=self.tracking_threshold if tracking else None
            )
            sample_token = self.data_infos[sample_id]["token"]
            # boxes = lidar_nusc_box_to_global(
            #     self.data_infos[sample_id],
            #     boxes,
            #     mapped_class_names,
            #     self.det3d_eval_configs,
            #     self.det3d_eval_version,
            # )
            boxes = ego_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.det3d_eval_configs,
                self.det3d_eval_version,
            )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if tracking and name in [
                    "barrier",
                    "traffic_cone",
                    "construction_vehicle",
                ]:
                    continue
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = NuScenes3DDetTrackDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = NuScenes3DDetTrackDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                )
                if not tracking:
                    nusc_anno.update(
                        dict(
                            detection_name=name,
                            detection_score=box.score,
                            attribute_name=attr,
                        )
                    )
                else:
                    nusc_anno.update(
                        dict(
                            tracking_name=name,
                            tracking_score=box.score,
                            tracking_id=str(box.token),
                        )
                    )

                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(
        self, result_path, logger=None, result_name="img_bbox", tracking=False
    ):
        from nuscenes import NuScenes

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False
        )
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }
        if not tracking:
            from nuscenes.eval.detection.evaluate import NuScenesEval

            nusc_eval = NuScenesEval(
                nusc,
                config=self.det3d_eval_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
            )
            nusc_eval.main(render_curves=False)

            # record metrics
            metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"
            for name in self.CLASSES:
                for k, v in metrics["label_aps"][name].items():
                    val = float("{:.4f}".format(v))
                    detail[
                        "{}/{}_AP_dist_{}".format(metric_prefix, name, k)
                    ] = val
                for k, v in metrics["label_tp_errors"][name].items():
                    val = float("{:.4f}".format(v))
                    detail["{}/{}_{}".format(metric_prefix, name, k)] = val
                for k, v in metrics["tp_errors"].items():
                    val = float("{:.4f}".format(v))
                    detail[
                        "{}/{}".format(metric_prefix, self.ErrNameMapping[k])
                    ] = val

            detail["{}/NDS".format(metric_prefix)] = metrics["nd_score"]
            detail["{}/mAP".format(metric_prefix)] = metrics["mean_ap"]
        else:
            from nuscenes.eval.tracking.evaluate import TrackingEval

            nusc_eval = TrackingEval(
                config=self.track3d_eval_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
                nusc_version=self.version,
                nusc_dataroot=self.data_root,
            )
            metrics = nusc_eval.main()

            # record metrics
            metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            print(metrics)
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"
            keys = [
                "amota",
                "amotp",
                "recall",
                "motar",
                "gt",
                "mota",
                "motp",
                "mt",
                "ml",
                "faf",
                "tp",
                "fp",
                "fn",
                "ids",
                "frag",
                "tid",
                "lgd",
            ]
            for key in keys:
                detail["{}/{}".format(metric_prefix, key)] = metrics[key]

        return detail

    def format_results(self, results, jsonfile_prefix=None, tracking=False):
        assert isinstance(results, list), "results must be a list"

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        if not ("pts_bbox" in results[0] or "img_bbox" in results[0]):
            result_files = self._format_bbox(
                results, jsonfile_prefix, tracking=tracking
            )
        else:
            result_files = dict()
            for name in results[0]:
                print(f"\nFormating bboxes of {name}")
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {
                        name: self._format_bbox(
                            results_, tmp_file_, tracking=tracking
                        )
                    }
                )
        return result_files, tmp_dir

    def evaluate(
        self,
        results,
        metric=None,
        logger=None,
        jsonfile_prefix=None,
        result_names=["img_bbox"],
        show=False,
        show_dir=None,
        out_dir=None,
        pipeline=None,
    ):
        result_surround = None
        result_occ = None
        bbox_task = True
        gt_occ = []
        if len(results[0].keys()) != 0:
            if 'surround_results' in results[0]:
                result_surround = [result['surround_results'] for result in results] 
            if 'occ_results' in results[0]:
                result_occ = [result['occ_results'] for result in results] 
                if not self.valid_for_detection:
                    bbox_task = False
            if 'boxes' in results[0] and bbox_task:
                results =  [result['boxes'] for result in results] 
                # results = [item for sublist in results for item in sublist]
            else:
                bbox_task = False

        results_dict = dict()
        if result_occ is not None:
            if self.surround_occ:
                self.occ_eval_metrics = Metric_mIoU_surround(
                    num_classes=17,
                    use_lidar_mask=False,
                    use_image_mask=True)
            else:
                self.occ_eval_metrics = Metric_mIoU(
                    num_classes=18,
                    use_lidar_mask=False,
                    use_image_mask=True)
            save_occ = []
            print('\nStarting Evaluation...')
            for index, occ_pred in enumerate(tqdm(result_occ)):
                occ_dict = {}
                info = self.data_infos[index]
                if self.surround_occ:
                    occ_base_path = './data/nuscenes/nuscenes_occ/samples/'
                    lidar_path = info['lidar_path'].split('/')[-1] + '.npy'
                    occ_path = os.path.join(occ_base_path, lidar_path)
                    occ_gt = np.load(occ_path)
                    occ_gt = occ_gt.astype(np.float32)
                    # occ_gt -= 1
                    # occ_gt[occ_gt == -1] = 255
                    voxel = np.ones([200,200,16]) * 17
                    voxel[occ_gt[:, 0].astype(np.int), occ_gt[:, 1].astype(np.int), occ_gt[:, 2].astype(np.int)] = occ_gt[:, 3]
                    voxel-=1
                    mask_lidar = (voxel != -1).astype(bool)
                    mask_camera = (voxel != -1).astype(bool)
                    gt_semantics = voxel
                    gt_semantics[gt_semantics == -1] = 16

                    # occ_gt = occ_gt.astype(np.int32)
                else:    # occ_gt = occ_gt[occ_gt > 0]
                    occ_gt = np.load(os.path.join(info['occ_path'],'labels.npz'))
                    gt_semantics = occ_gt['semantics']
                    gt_occ.append(gt_semantics)
                    mask_lidar = occ_gt['mask_lidar'].astype(bool)
                    mask_camera = occ_gt['mask_camera'].astype(bool)
                # occ_pred = occ_pred
                self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)
                
                occ_dict['voxel_semantics_preds'] = occ_pred
                occ_dict['voxel_semantics'] = gt_semantics
                occ_dict['mask_lidar'] = mask_lidar
                occ_dict['mask_camera'] = mask_camera
                occ_dict['path'] = info['occ_path']
                save_occ.append(occ_dict)
                
                # if self.save_results == 'all' or  self.save_results == 'occ':
                #     path = info['occ_path'].split('/')
                #     path ='./pred/' + path[-2] + '/' + path[-1]
                #     if not os.path.exists(path):
                #         os.makedirs(path)
                #     np.save(os.path.join(path,'occ_pred.npy'), occ_pred)
                # if index%100==0 and show_dir is not None:
                #     gt_vis = self.vis_occ(gt_semantics)
                #     pred_vis = self.vis_occ(occ_pred)
                #     mmcv.imwrite(np.concatenate([gt_vis, pred_vis], axis=1),
                #                 os.path.join(show_dir + "%d.jpg"%index))
                    
            
            dict_iou = self.occ_eval_metrics.count_miou()
            results_dict.update(dict_iou)
            if self.save_results == 'all' or  self.save_results == 'occ':
                with open('occ_results.pkl', 'wb') as f:
                    pickle.dump(save_occ, f)
                
        if result_surround is not None:
            if self.use_semantic:
                class_names = {0: 'IoU'}
                class_num = len(self.occ_classes) + 1
                for i, name in enumerate(self.occ_classes):
                    class_names[i + 1] = self.occ_classes[i]
                results_occ = np.stack(result_surround, axis=0)
                results_occ = results_occ.mean(0)[0]
                mean_ious = []
                for i in range(class_num):
                    tp = results_occ[i, 0]
                    p = results_occ[i, 1]
                    g = results_occ[i, 2]
                    union = p + g - tp
                    mean_ious.append(tp / union)
                    print("class: {}, iou: {}".format(class_names[i], tp / union))
                for i in range(class_num):
                    results_dict[class_names[i]] = mean_ious[i]
                    
                results_dict['mIoU'] = np.mean(np.array(mean_ious)[1:])
                print("mIoU: {}".format(results_dict['mIoU']))
            else:
                result_surround = np.stack(result_surround, axis=0).mean(0)
                results_dict={'Acc':result_surround[0],
                            'Comp':result_surround[1],
                            'CD':result_surround[2],
                            'Prec':result_surround[3],
                            'Recall':result_surround[4],
                            'F-score':result_surround[5]}
        if bbox_task:
            for metric in ["detection", "tracking"]:
                tracking = metric == "tracking"
                if tracking and not self.tracking:
                    continue
                result_files, tmp_dir = self.format_results(
                    results, jsonfile_prefix, tracking=tracking
                )

                # if isinstance(result_files, dict):
                for name in result_names:
                    ret_dict = self._evaluate_single(
                        result_files, tracking=tracking
                    )
                results_dict.update(ret_dict)
                # elif isinstance(result_files, str):
                #     results_dict = self._evaluate_single(
                #         result_files, tracking=tracking
                #     )
                if tmp_dir is not None:
                    tmp_dir.cleanup()

        if self.save_results == 'all' or  self.save_results == 'det':
            show = True
            out_dir = "./origin_images"
            self.show(results, save_dir=out_dir, show=show, pipeline=pipeline)


        # data_loader = DataLoader(
        #     EgoPoseDataset(self.data_infos),
        #     batch_size=1,
        #     shuffle=False,
        #     num_workers=8
        # )
        # lidar_origins = [batch[1] for i,batch in enumerate(data_loader)]
        
        # results_dict.update(calc_rayiou(result_occ, gt_occ, lidar_origins))
            
        return results_dict

    def show(self, results, save_dir=None, show=False, pipeline=None):
        save_dir = "./" if save_dir is None else save_dir
        save_dir = os.path.join(save_dir, "visual")
        print_log(os.path.abspath(save_dir))
        pipeline = Compose(pipeline)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        videoWriter = None
        print("len(results): ",len(results))
        for i, result in enumerate(results):
            if "img_bbox" in result.keys():
                result = result["img_bbox"]
            get_data = self.get_data_info(i)
            data_info = pipeline(get_data)
            imgs = []
            raw_imgs = data_info["img"]
            # lidar_path = data_info['lidar_path']
            lidar2img = get_data['ego2sensors']
            pred_bboxes_3d = result["boxes_3d"][
                result["scores_3d"] > self.vis_score_threshold
            ]

            # if "instance_ids" in result and self.tracking:
            #     color = []
            #     for id in result["instance_ids"].cpu().numpy().tolist():
            #         color.append(
            #             self.ID_COLOR_MAP[int(id % len(self.ID_COLOR_MAP))]
            #         )
            if "labels_3d" in result:
                color = []
                for id in result["labels_3d"].cpu().numpy().tolist():
                    color.append(self.ID_COLOR_MAP[id])
            else:
                color = (255, 0, 0)

            # ===== draw boxes_3d to images =====
            for j, img_origin in enumerate(raw_imgs):
                img = img_origin.copy()
                # if len(pred_bboxes_3d) != 0:
                #     img = draw_lidar_bbox3d_on_img(
                #         pred_bboxes_3d,
                #         img,
                #         lidar2img[j],
                #         img_metas=None,
                #         color=color,
                #         thickness=3,
                #     )
                imgs.append(img)

            # ===== draw boxes_3d to BEV =====
            bev = draw_lidar_bbox3d_on_bev(
                pred_bboxes_3d,
                bev_size=img.shape[0] * 2,
                color=color,
            )

            # ===== put text and concat =====
            # for j, name in enumerate(
            #     [
            #         "front",
            #         "front right",
            #         "front left",
            #         "rear",
            #         "rear left",
            #         "rear right",
            #     ]
            # ):
                # imgs[j] = cv2.rectangle(
                #     imgs[j],
                #     (0, 0),
                #     (440, 80),
                #     color=(255, 255, 255),
                #     thickness=-1,
                # )
                # w, h = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                # text_x = int(220 - w / 2)
                # text_y = int(40 + h / 2)

                # imgs[j] = cv2.putText(
                #     imgs[j],
                #     name,
                #     (text_x, text_y),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     2,
                #     (0, 0, 0),
                #     2,
                #     cv2.LINE_AA,
                # )
            image = np.concatenate(
                [
                    np.concatenate([imgs[2], imgs[0], imgs[1]], axis=1),
                    np.concatenate([imgs[4], imgs[3], imgs[5]], axis=1),
                ],
                axis=0,
            )
            image = np.concatenate([image, bev], axis=1)

            # ===== save video =====
            # if videoWriter is None:
            #     videoWriter = cv2.VideoWriter(
            #         os.path.join(save_dir, "video.avi"),
            #         fourcc,
            #         7,
            #         image.shape[:2][::-1],
            #     )
            cv2.imwrite(os.path.join(save_dir, f"{i}.jpg"), image)
        #     videoWriter.write(image)
        # videoWriter.release()


def output_to_nusc_box(detection, threshold=None):
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()
    if "instance_ids" in detection:
        ids = detection["instance_ids"]  # .numpy()
    if threshold is not None:
        if "cls_scores" in detection:
            mask = detection["cls_scores"].numpy() >= threshold
        else:
            mask = scores >= threshold
        box3d = box3d[mask]
        scores = scores[mask]
        labels = labels[mask]
        ids = ids[mask]

    if hasattr(box3d, "gravity_center"):
        box_gravity_center = box3d.gravity_center.numpy()
        box_dims = box3d.dims.numpy()
        nus_box_dims = box_dims[:, [1, 0, 2]]
        box_yaw = box3d.yaw.numpy()
    else:
        box3d = box3d.numpy()
        box_gravity_center = box3d[..., :3].copy()
        box_dims = box3d[..., 3:6].copy()
        nus_box_dims = box_dims[..., [1, 0, 2]]
        box_yaw = box3d[..., 6].copy()

    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    # box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if hasattr(box3d, "gravity_center"):
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (*box3d[i, 7:9], 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        if "instance_ids" in detection:
            box.token = ids[i]
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(
    info,
    boxes,
    classes,
    eval_configs,
    eval_version="detection_cvpr_2019",
):
    box_list = []
    for i, box in enumerate(boxes):
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list


def ego_nusc_box_to_global(
    info,
    boxes,
    classes,
    eval_configs,
    eval_version="detection_cvpr_2019",
):
    box_list = []
    for i, box in enumerate(boxes):
        # Move box to ego vehicle coord system
        # box.rotate(pyquaternion.Quaternion(info["lidar2ego_rotation"]))
        # box.translate(np.array(info["lidar2ego_translation"]))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list