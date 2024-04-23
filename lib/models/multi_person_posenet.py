# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import pose_resnet
from core.loss import NeRFLoss
from models.nerf_mlp import VanillaNeRFRadianceField
from models.nerf_method import NerfMethodNet
from models.nerf_test import NerfMethodNetTest

class MultiPersonPoseNet(nn.Module):
    def __init__(self, backbone, cfg):
        super(MultiPersonPoseNet, self).__init__()
        self.backbone = backbone

        self.model = VanillaNeRFRadianceField(
            net_depth=4,  # The depth of the MLP.
            net_width=256,  # The width of the MLP.
            skip_layer=3,  # The layer to add skip layers to.
            feature_dim=6 + cfg.NERF.DIM * 2,  # + RGB original img #self.num_joints
            net_depth_condition=1,  # The depth of the second part of MLP.
            net_width_condition=128
        )
        self.nerf_method = NerfMethodNet(cfg)
        self.nerf_test = NerfMethodNetTest(cfg)

        self.use_feat_level = cfg.NERF.USE_FEAT_LEVEL
        self.image_size = cfg.NETWORK.IMAGE_SIZE
        self.begin = cfg.NERF.FEAT_BEGIN_LAYER
        self.end = cfg.NERF.FEAT_END_LAYER


    def deal_with_feats2d(self, feats2d):
        lay = len(feats2d[0])
        assert self.end <= lay, 'end is larger than lay'
        w, h = self.image_size[0], self.image_size[1]
        feats = []
        for index in range(self.begin, self.end):
            feature_index = [f[index] for f in feats2d] # [f0index, f1index, f2index, f3index]
            fff = []
            for feat in feature_index: # [tensor(bs, 256, h,w )  *4]
                ff = F.adaptive_avg_pool2d(feat, (h, w))
                fff.append(ff)
            feats.append(fff)
        return feats


    def forward(self, views=None, meta=None, ray_batches=None):
        all_heatmaps, feats2d = [], []
        for view in views:
            heatmaps, feat2d = self.backbone(view, self.use_feat_level)
            all_heatmaps.append(heatmaps)
            feats2d.append(feat2d)

        # all_heatmaps = targets_2d
        device = all_heatmaps[0].device

        '''fs = self.deal_with_feats2d(feats2d)
        feats2d = fs[0]'''
        
        # for 98 's  15heatmap train
        w, h = self.image_size[0], self.image_size[1]
        feats2d = all_heatmaps  # [(bs ,15,h,w) *4]  [(bs ,256,h,w) *4] ->16
        feats2d = [F.adaptive_avg_pool2d(feat, (h, w)) for feat in feats2d]



        # calculate nerf loss
        criterion_nerf = NeRFLoss().cuda()

        # nerf train & loss
        if self.training:
            preds_rays = self.nerf_method(meta, self.model, ray_batches, device, feats2d)
        else:
            preds_rays = self.nerf_test(meta, self.model, ray_batches, device, feats2d)
        rgbs, gts, masks = [], [], []
        for ret in preds_rays:
            rgbs.append(ret['outputs_coarse']['rgb'])
            gts.append(ret['gt_rgb'])
            masks.append(ret['outputs_coarse']['mask'])
        loss_nerf = criterion_nerf(rgbs, gts, masks)
        return loss_nerf




def get_multi_person_pose_net(cfg, is_train=True):
    if cfg.BACKBONE_MODEL:
        backbone = eval(cfg.BACKBONE_MODEL + '.get_pose_net')(cfg, is_train=is_train)
    else:
        backbone = None
    model = MultiPersonPoseNet(backbone, cfg)
    return model
