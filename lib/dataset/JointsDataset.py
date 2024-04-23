# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import copy
import logging

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform, get_scale

logger = logging.getLogger(__name__)


class JointsDataset(Dataset):

    def __init__(self, cfg, image_set, is_train, transform=None):
        self.cfg = cfg
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.maximum_person = cfg.MULTI_PERSON.MAX_PEOPLE_NUM

        self.is_train = is_train

        this_dir = os.path.dirname(__file__)
        dataset_root = os.path.join(this_dir, '../..', cfg.DATASET.ROOT)
        self.dataset_root = os.path.abspath(dataset_root)
        self.root_id = cfg.DATASET.ROOTIDX
        self.image_set = image_set
        self.dataset_name = cfg.DATASET.TEST_DATASET

        self.data_format = cfg.DATASET.DATA_FORMAT
        self.data_augmentation = cfg.DATASET.DATA_AUGMENTATION

        self.num_views = cfg.DATASET.CAMERA_NUM

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.NETWORK.TARGET_TYPE
        self.image_size = np.array(cfg.NETWORK.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.NETWORK.HEATMAP_SIZE)
        self.sigma = cfg.NETWORK.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

        self.space_size = np.array(cfg.MULTI_PERSON.SPACE_SIZE)
        self.space_center = np.array(cfg.MULTI_PERSON.SPACE_CENTER)
        self.initial_cube_size = np.array(cfg.MULTI_PERSON.INITIAL_CUBE_SIZE)

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx, is_ray):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if data_numpy is None:
            # logger.error('=> fail to read {}'.format(image_file))
            # raise ValueError('Fail to read {}'.format(image_file))
            return None, None, None, None, None, None

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)


        height, width, _ = data_numpy.shape
        c = np.array([width / 2.0, height / 2.0])
        s = get_scale((width, height), self.image_size)
        r = 0

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans, (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        denorm_img = torch.tensor(input, dtype=torch.float32) / 255.0
        if is_ray:
            ray_batch={
                'denorm_img': denorm_img,
                'camera': db_rec['camera'],
                'transform': trans
            }
            return ray_batch
        else:
            if self.transform:
                input = self.transform(input)

            meta = {
                'image': image_file,
                'camera': db_rec['camera'],
                'denorm_img': denorm_img,
                'transform': trans
            }

            return input, meta