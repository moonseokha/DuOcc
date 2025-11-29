# Copyright (c) OpenMMLab. All rights reserved.
import pickle

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from nuscenes_converter import create_nuscenes_infos as create_nuscenes_infos
import os

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

def transform_matrix(translation, rotation):
    """Converts a translation and rotation into a 4x4 transformation matrix.

    Args:
        translation: <np.float: 3>. Translation in x, y, z.
        rotation: <np.float: 4>. Rotation as quaternion in w, x, y, z.

    Returns:
        <np.float: 4, 4>. A transformation matrix.
    """
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = Quaternion(rotation).rotation_matrix
    matrix[:3, 3] = translation
    return matrix

def add_ann_adj_info(extra_tag):
    nuscenes_version = 'v1.0-trainval'
    dataroot = './data/nuscenes/'
    nuscenes = NuScenes(nuscenes_version, dataroot)
    for set in ['train', 'val']:
        dataset = pickle.load(
            open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set), 'rb'))
        id=0
        while(id < len(dataset['infos'])):
        # for id in range(len(dataset['infos'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]
            # token = dataset['infos']['token']
            # get sweep adjacent frame info
            sd_rec = nuscenes.get("sample_data", nuscenes.get("sample",info['token'])["data"]["LIDAR_TOP"])
            pose_record = nuscenes.get("ego_pose", sd_rec["ego_pose_token"])
            #get yaw_pitch_roll from quaternion
            lidar2ego_translation = info['lidar2ego_translation']
            lidar2ego_rotation = info['lidar2ego_rotation']
            ego2global_translation = info['ego2global_translation']
            ego2global_rotation = info['ego2global_rotation']
            lidar2ego = transform_matrix(
                lidar2ego_translation, lidar2ego_rotation)
            ego2global = transform_matrix(
                ego2global_translation, ego2global_rotation)
            
            lidar2global = np.dot(ego2global, lidar2ego)
            global2lidar = np.linalg.inv(lidar2global)
            # get angle from global2lidar
            # angle = np.arctan2(global2lidar[1, 0], global2lidar[0, 0])
            angle = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            
            
            sample = nuscenes.get('sample', info['token'])
            anns = sample['anns']

            next_pos = list()
            gt_boxes = list()
            gt_labels = list()
            instance_list = list()
            trans = -np.array(ego2global_translation)
            rot = Quaternion(ego2global_rotation).inverse
            instance_list_ego = list()
            for ann in anns:
                ann_info = nuscenes.get('sample_annotation', ann)
                velocity = nuscenes.box_velocity(ann_info['token'])
                instance_list.append(ann_info['instance_token'])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                # Use ego coordinate.
                if (map_name_from_general_to_detection[ann_info['category_name']]
                        not in classes
                        or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
                    continue
                box = Box(
                    ann_info['translation'],
                    ann_info['size'],
                    Quaternion(ann_info['rotation']),
                    velocity=velocity,
                )
                box.translate(trans)
                box.rotate(rot)
                box_xyz = np.array(box.center)
                box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
                box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
                box_velo = np.array(box.velocity[:2])
                gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
                gt_boxes.append(gt_box)
                gt_labels.append(
                    classes.index(
                        map_name_from_general_to_detection[ann_info['category_name']]))
                instance_list_ego.append(nuscenes.getind("instance",ann_info['instance_token']))

            
            
            dataset['infos'][id]['gt_boxes_ego'] = gt_boxes
            dataset['infos'][id]['gt_lables_ego'] = gt_labels
            dataset['infos'][id]['instance_inds_ego'] = instance_list_ego
            if len(gt_boxes)==0:
                if set == 'train':
                    dataset['infos'].pop(id)
                    continue
            scene = nuscenes.get('scene',nuscenes.get("sample",dataset['infos'][id]['token'])['scene_token'])
            dataset['infos'][id]['occ_path'] = './data/nuscenes/gts/%s/%s'%(scene['name'],dataset['infos'][id]['token'])
            id +=1

        if not os.path.exists('./data/nuscenes_anno_pkls'):
            os.makedirs('./data/nuscenes_anno_pkls')
        with open('./data/nuscenes_anno_pkls/%s_infos_aug_%s.pkl' % (extra_tag, set),
                  'wb') as fid:
            pickle.dump(dataset, fid)
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="nuscenes converter")
    parser.add_argument("--root_path", type=str, default="./data/nuscenes/")
    parser.add_argument("--info_prefix", type=str, default="nuscenes_occ")
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument("--max_sweeps", type=int, default=10)
    args = parser.parse_args()

    versions = args.version.split(",")
    for version in versions:
        create_nuscenes_infos(
            args.root_path,
            args.info_prefix,
            version=version,
            max_sweeps=args.max_sweeps,
        )
    add_ann_adj_info(args.info_prefix)
