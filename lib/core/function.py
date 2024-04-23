from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import copy

import torch
import numpy as np

logger = logging.getLogger(__name__)


def train_3d(config, model, optimizer, loader, epoch, output_dir, writer_dict, device=torch.device('cuda'), dtype=torch.float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_nerf = AverageMeter()

    model.train()

    if model.module.backbone is not None:
        model.module.backbone.eval()  # Comment out this line if you want to train 2D backbone jointly


    end = time.time()
    for i, (inputs, meta, ray_batches) in enumerate(loader):
        data_time.update(time.time() - end)

        if 'panoptic' in config.DATASET.TEST_DATASET:
            loss_nerf = model(views=inputs, meta=meta, ray_batches=ray_batches)
        elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
            pass


        loss_nerf = loss_nerf.mean()
        losses_nerf.update(loss_nerf.item())
        optimizer.zero_grad()
        loss_nerf.backward()
        optimizer.step()


        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss_nerf: {loss_nerf.val:.6f} ({loss_nerf.avg:.6f})\t' \
                  'Memory {memory:.1f}'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                    data_time=data_time, loss_nerf=losses_nerf, memory=gpu_memory_usage)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss_nerf', losses_nerf.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


def validate_3d(config, model, loader, output_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_nerf = AverageMeter()
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (inputs, meta, ray_batches) in enumerate(loader):
            data_time.update(time.time() - end)
            if 'panoptic' in config.DATASET.TEST_DATASET:
                loss_nerf = model(views=inputs, meta=meta, ray_batches=ray_batches)
            elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
                pass

            loss_nerf = loss_nerf.mean()
            losses_nerf.update(loss_nerf.item())

            batch_time.update(time.time() - end)
            end = time.time()
            if i % config.PRINT_FREQ == 0 or i == len(loader) - 1:
                gpu_memory_usage = torch.cuda.memory_allocated(0)
                msg = 'Test: [{0}/{1}]\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed: {speed:.1f} samples/s\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss_nerf: {loss_nerf.val:.6f} ({loss_nerf.avg:.6f})\t' \
                      'Memory {memory:.1f}'.format(
                        i, len(loader), batch_time=batch_time,
                        speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                        data_time=data_time, loss_nerf=losses_nerf, memory=gpu_memory_usage)
                logger.info(msg)


    return  losses_nerf.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
