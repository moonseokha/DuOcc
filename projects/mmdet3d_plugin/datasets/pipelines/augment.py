import torch

import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from PIL import Image
import copy

@PIPELINES.register_module()
class ResizeCropFlipImage(object):
    def __call__(self, results):
        aug_config = results.get("aug_config")
        if aug_config is None:
            return results
        aug_config = aug_config["2d"]
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        post_rots = []
        post_trans = []
        results["intrinsic_mat"] = copy.deepcopy(results["cam_intrinsic"])
        for i in range(N):
            post_rot = torch.eye(3)
            post_tran = torch.zeros(3)
            img, mat = self._img_transform(
                np.uint8(imgs[i]), aug_config[i],
            )
            new_imgs.append(np.array(img).astype(np.float32))
            post_rot[:2, :2] = torch.tensor(copy.deepcopy(mat[:2, :2]))
            post_tran[:2] = torch.tensor(copy.deepcopy(mat[:2, 2]))
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            results["lidar2img"][i] = mat @ results["lidar2img"][i]
            results["ego2sensors"][i] = mat @ results["ego2sensors"][i]
            if "cam_intrinsic" in results:
                results["cam_intrinsic"][i][:3, :3] *= aug_config[i]["resize"]
                # results["cam_intrinsic"][i][:3, :3] = (
                #     mat[:3, :3] @ results["cam_intrinsic"][i][:3, :3]
                # )

        results["img"] = new_imgs
        results["post_rots"] = torch.stack(post_rots)
        results["post_trans"] = torch.stack(post_trans)
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        return results

    def _img_transform(self, img, aug_configs):
        H, W = img.shape[:2]
        resize = aug_configs.get("resize", 1)
        resize_dims = (int(W * resize), int(H * resize))
        crop = aug_configs.get("crop", [0, 0, *resize_dims])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)

        origin_dtype = img.dtype
        if origin_dtype != np.uint8:
            min_value = img.min()
            max_vaule = img.max()
            scale = 255 / (max_vaule - min_value)
            img = (img - min_value) * scale
            img = np.uint8(img)
        img = Image.fromarray(img)
        img = img.resize(resize_dims).crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        img = np.array(img).astype(np.float32)
        if origin_dtype != np.uint8:
            img = img.astype(np.float32)
            img = img / scale + min_value

        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] *= resize
        transform_matrix[:2, 2] -= np.array(crop[:2])
        if flip:
            flip_matrix = np.array(
                [[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]]
            )
            transform_matrix = flip_matrix @ transform_matrix
        rotate = rotate / 180 * np.pi
        rot_matrix = np.array(
            [
                [np.cos(rotate), np.sin(rotate), 0],
                [-np.sin(rotate), np.cos(rotate), 0],
                [0, 0, 1],
            ]
        )
        rot_center = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        rot_matrix[:2, 2] = -rot_matrix[:2, :2] @ rot_center + rot_center
        transform_matrix = rot_matrix @ transform_matrix
        extend_matrix = np.eye(4)
        extend_matrix[:3, :3] = transform_matrix
        return img, extend_matrix

@PIPELINES.register_module()
class BBoxRotation_DuOcc(object):
    def __init__(self,surround_occ=False):
        self.surround_occ = surround_occ

    def __call__(self, results):
        rotate_angle = results["aug_config"]["3d"]["rotate_bda"]
        scale_ratio = results["aug_config"]["3d"]["scale_bda"]
        flip_dx = results["aug_config"]["3d"]["flip_dx"]
        flip_dy = results["aug_config"]["3d"]["flip_dy"]
        test_mode = results["aug_config"]["3d"]["test_mode"]
        # rotate_angle, scale_ratio, flip_dx, flip_dy = self.sample_bda_augmentation()
        angle = rotate_angle / 180 * np.pi
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = np.array([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ np.array([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ np.array([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
            
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        mat = np.eye(4)
        mat[:3, :3] = rot_mat

        rot_mat_inv = np.linalg.inv(mat)
        num_view = len(results["ego2sensors"])
        if 'voxel_semantics' in results:
            # for view in range(num_view):
            #     # lidar2img -> ego2img
            #     results["lidar2img"][view] = results["lidar2img"][view] @ np.linalg.inv(results["lidar2ego"])
            
            if flip_dx:
                results['voxel_semantics'] = results['voxel_semantics'][::-1,...].copy()
                results['mask_lidar'] = results['mask_lidar'][::-1,...].copy()
                results['mask_camera'] = results['mask_camera'][::-1,...].copy()
            if flip_dy:
                results['voxel_semantics'] = results['voxel_semantics'][:,::-1,...].copy()
                results['mask_lidar'] = results['mask_lidar'][:,::-1,...].copy()
                results['mask_camera'] = results['mask_camera'][:,::-1,...].copy()
                
        for view in range(num_view):
            results["ego2sensors"][view] = (
                results["ego2sensors"][view] @ rot_mat_inv
            )
        
        results["box2img"] = results["lidar2img"]
        # results["img2coor"] = results["box2img"].copy().inv()
        if "box2global" in results:
            results["box2global"] = results["box2global"] @ rot_mat_inv

        if "lidar2ego" in results:
            if self.surround_occ:
                results["lidar2ego"] = mat @ results["lidar2ego"] @ rot_mat_inv
            else:
                results["lidar2ego"] = mat @ results["lidar2ego"]

        if "gt_bboxes_3d" in results and (test_mode==False):
            results["gt_bboxes_3d"] = self.box_rotate(
                results["gt_bboxes_3d"], rot_mat, angle , scale_ratio, flip_dx, flip_dy
            )
            
        # if "future_gt_bboxes_3d" in results:
        #     print("future_gt_bboxes_3d")
        #     results["future_gt_bboxes_3d"] = self.box_rotate(
        #         results["future_gt_bboxes_3d"], rot_mat, angle , scale_ratio, flip_dx, flip_dy, only_xyz=True,
        #     )
            
        results["bda"]=rot_mat[:3,:3]
        return results

    @staticmethod
    def box_rotate(bbox_3d, rot_mat, angle, scale_ratio, flip_dx, flip_dy,only_xyz=False):
        bbox_3d[:, :3] = (rot_mat @ np.expand_dims(bbox_3d[:, :3],axis=-1)).squeeze(-1)
        if only_xyz:
            bbox_3d[:, 3] += angle
            return bbox_3d
        bbox_3d[:, 3:6] *= scale_ratio
        bbox_3d[:, 6] += angle
        if flip_dx:
            bbox_3d[:,6] = 2 * np.arcsin(1.0) - bbox_3d[:,6]
        if flip_dy:
            bbox_3d[:, 6] = -bbox_3d[:, 6]
        bbox_3d[:, 7:] = (
                rot_mat[:2, :2] @ np.expand_dims(bbox_3d[:, 7:],axis=-1)).squeeze(-1)    
        return bbox_3d


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(
                    -self.brightness_delta, self.brightness_delta
                )
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str
