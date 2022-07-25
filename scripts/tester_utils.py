from numpy.core.numeric import Inf
import torch
import numpy as np
from glob import glob
import medpy.metric as md

import sys
import os

sys.path.insert(0, os.path.dirname(__file__) + '/..')

from utils.util import _eval_dice, _eval_haus, _connectivity_region_analysis


@torch.no_grad()
def test_normal_unet(site_index, test_net, args, info=False):
    test_net.eval()
    if info:
        eval_funcs = [md.dc, md.hd95, md.jc, md.assd]
    else:
        eval_funcs = [md.jc]

    if args.dataset == 'fundus':
        test_data_list = glob(
            '/raid/wjc/data/SpecializedFedSeg/{}/Site{}/test/image/*.npy'.
            format(args.dataset, site_index + 1))
    elif args.dataset == 'pmr':
        test_data_list = glob(
            '/raid/wjc/data/SpecializedFedSeg/{}/Site{}/test/image/*'.format(
                args.dataset, site_index + 1))
    elif args.dataset == 'polyp':
        test_data_list = glob(
            '/raid/wjc/data/SpecializedFedSeg/{}/Site{}/test/image/*'.format(
                args.dataset, site_index + 1))
    dice_array = []

    score_values = np.zeros(
        (len(eval_funcs), args.num_classes, len(test_data_list)))

    for fid, filename in enumerate(test_data_list):
        if args.dataset == 'fundus':
            data = np.load(filename)
            image = np.expand_dims(data[..., :3].transpose(2, 0, 1),
                                   axis=0) / 255.
            image = torch.from_numpy(image).cuda().float()

            mask = np.load(filename.replace('image', 'mask'))
            pred = test_net(image).cpu().numpy()

            for c in range(args.num_classes):
                _pred_y = pred[0, c] > 0.5
                _mask = mask > c
                _pred_y = _connectivity_region_analysis(_pred_y)
                for e_i in range(len(eval_funcs)):
                    score_values[e_i, c, fid] = eval_funcs[e_i](_pred_y, _mask)

        elif args.dataset == 'pmr':
            image_names = os.listdir(filename)
            preds = []
            masks = []
            for image_name in image_names:
                image = np.load(os.path.join(filename, image_name))[np.newaxis,
                                                                    np.newaxis]
                mask = np.load(
                    os.path.join(filename.replace('image', 'mask'),
                                 image_name))
                image = torch.from_numpy(image).cuda().float()
                pred = test_net(image).cpu().numpy()[0, 0] > 0.5
                pred = _connectivity_region_analysis(pred)
                preds.append(pred)
                masks.append(mask)
            preds = np.array(preds)
            masks = np.array(masks)
            for e_i in range(len(eval_funcs)):
                score_values[e_i, 0, fid] = eval_funcs[e_i](preds, masks)
        elif args.dataset == 'polyp':
            data = np.load(filename)
            image = np.expand_dims(data[..., :3].transpose(2, 0, 1),
                                   axis=0) / 255.
            image = torch.from_numpy(image).cuda().float()

            mask = np.load(filename.replace('image', 'mask')) > 0.5
            pred = test_net(image).cpu().numpy()[0, 0] > 0.5
            if np.max(pred) == 0:
                pred[0, 0] = 1
            for e_i in range(len(eval_funcs)):
                score_values[e_i, 0, fid] = eval_funcs[e_i](pred, mask)

    if info:
        return score_values
    else:
        score_values = np.mean(score_values, axis=-1)
        return score_values


@torch.no_grad()
def test_lc_unet(site_index, test_net, seg_heads, args, info=False):
    test_net.eval()
    if info:
        eval_funcs = [md.dc, md.assd, md.jc, md.assd]
    else:
        eval_funcs = [md.jc]

    if args.dataset == 'fundus':
        test_data_list = glob(
            '/raid/wjc/data/SpecializedFedSeg/{}/Site{}/test/image/*.npy'.
            format(args.dataset, site_index + 1))
    elif args.dataset == 'pmr':
        test_data_list = glob(
            '/raid/wjc/data/SpecializedFedSeg/{}/Site{}/test/image/*'.format(
                args.dataset, site_index + 1))
    elif args.dataset == 'polyp':
        test_data_list = glob(
            '/raid/wjc/data/SpecializedFedSeg/{}/Site{}/test/image/*'.format(
                args.dataset, site_index + 1))

    score_values = np.zeros(
        (len(eval_funcs), args.num_classes, len(test_data_list)))

    for fid, filename in enumerate(test_data_list):
        if args.dataset == 'fundus':
            data = np.load(filename)
            image = np.expand_dims(data[..., :3].transpose(2, 0, 1),
                                   axis=0) / 255.
            image = torch.from_numpy(image).cuda().float()

            mask = np.load(filename.replace('image', 'mask'))
            pred = test_net(image, site_index, -1, seg_heads).cpu().numpy()

            for c in range(args.num_classes):
                _pred_y = pred[0, c] > 0.5
                _mask = mask > c
                _pred_y = _connectivity_region_analysis(_pred_y)
                for e_i in range(len(eval_funcs)):
                    score_values[e_i, c, fid] = eval_funcs[e_i](_pred_y, _mask)

        elif args.dataset == 'pmr':
            image_names = os.listdir(filename)
            preds = []
            masks = []
            for image_name in image_names:
                image = np.load(os.path.join(filename, image_name))[np.newaxis,
                                                                    np.newaxis]
                mask = np.load(
                    os.path.join(filename.replace('image', 'mask'),
                                 image_name))
                image = torch.from_numpy(image).cuda().float()
                pred = test_net(image, site_index, -1,
                                seg_heads).cpu().numpy()[0, 0] > 0.5
                pred = _connectivity_region_analysis(pred)
                preds.append(pred)
                masks.append(mask)
            preds = np.array(preds)
            masks = np.array(masks)
            for e_i in range(len(eval_funcs)):
                score_values[e_i, 0, fid] = eval_funcs[e_i](preds, masks)
        elif args.dataset == 'polyp':
            data = np.load(filename)
            image = np.expand_dims(data[..., :3].transpose(2, 0, 1),
                                   axis=0) / 255.
            image = torch.from_numpy(image).cuda().float()

            mask = np.load(filename.replace('image', 'mask'))
            pred = test_net(image, site_index, -1,
                            seg_heads).cpu().numpy()[0, 0] > 0.5

            for e_i in range(len(eval_funcs)):
                score_values[e_i, 0, fid] = eval_funcs[e_i](pred, mask)
    if info:
        return score_values
    else:
        score_values = np.mean(score_values, axis=-1)
        return score_values
