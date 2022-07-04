# -*- coding: utf-8 -*-
"""
2D Unet-like architecture code in Pytorch
"""
import math, sys, os

sys.path.insert(0, os.path.dirname(__file__) + '/../')
import numpy as np
from networks.layers import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

from networks.soft_attn import Soft
from networks.unet2d import ConvD, ConvU


class HeadCalibration(nn.Module):
    def __init__(self, n_classes, n_fea):
        super(HeadCalibration, self).__init__()
        self.n_classes = n_classes
        self.head = nn.Conv2d(n_fea * n_classes * 2, n_classes, 1)
        self.soft = Soft()

    def forward(self, uncertainty, preds, fea):
        fea_list = []
        att_maps = []
        for c in range(self.n_classes):
            fea2 = self.soft(uncertainty[:, c].unsqueeze(1), fea) + fea
            fea_list.append(fea2)
            fea3 = self.soft(preds[:, c].unsqueeze(1), fea) + fea
            fea_list.append(fea3)
        fea_list = torch.cat(fea_list, dim=1)
        o = self.head(fea_list)
        o = F.sigmoid(o)
        return o


class PersonalizedChannelSelection(nn.Module):
    def __init__(self, f_dim, emb_dim):
        super(PersonalizedChannelSelection, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Sequential(nn.Conv2d(emb_dim, f_dim, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(f_dim, f_dim, 1, bias=False))
        self.fc2 = nn.Sequential(
            nn.Conv2d(f_dim * 2, f_dim // 16, 1, bias=False), nn.ReLU(),
            nn.Conv2d(f_dim // 16, f_dim, 1, bias=False))

    def forward_emb(self, emb):
        emb = emb.unsqueeze(-1).unsqueeze(-1)
        emb = self.fc1(emb)
        return emb

    def forward(self, x, emb):
        b, c, w, h = x.size()

        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)

        # site embedding
        emb = self.forward_emb(emb)

        # avg
        avg_out = torch.cat([avg_out, emb], dim=1)
        avg_out = self.fc2(avg_out)

        # max
        max_out = torch.cat([max_out, emb], dim=1)
        max_out = self.fc2(max_out)

        out = avg_out + max_out
        hmap = self.sigmoid(out)

        x = x * hmap + x

        return x, hmap


class Unet2D(nn.Module):
    def __init__(self, c=3, n=16, norm='in', num_classes=2, emb_n=64, pcs=0):
        super(Unet2D, self).__init__()
        self.if_pcs = pcs

        self.convd1 = ConvD(c, n, norm, first=True)
        self.convd2 = ConvD(n, 2 * n, norm)
        self.convd3 = ConvD(2 * n, 4 * n, norm)
        self.convd4 = ConvD(4 * n, 8 * n, norm)
        self.convd5 = ConvD(8 * n, 16 * n, norm)

        self.convu4 = ConvU(16 * n, norm, first=True)
        self.convu3 = ConvU(8 * n, norm)
        self.convu2 = ConvU(4 * n, norm)

        self.convu1 = ConvU(2 * n, norm)

        self.seg1 = nn.Conv2d(2 * n, num_classes, 1)

        self.pcs = PersonalizedChannelSelection(16 * n, emb_n)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, emb):
        hmaps = []
        batch_size = x.size(0)
        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        # attention
        if self.if_pcs:
            x5, hmap5 = self.pcs(x5, emb)
        else:
            hmap5 = None

        y4 = self.convu4(x5, x4)
        y3 = self.convu3(y4, x3)
        y2 = self.convu2(y3, x2)
        y1 = self.convu1(y2, x1)

        y1_pred = self.seg1(y1)

        predictions = F.sigmoid(input=y1_pred)

        return y1, predictions, x5, hmap5


class LC_Fed(nn.Module):
    def __init__(self, args):
        super(LC_Fed, self).__init__()
        self.if_pcs = args.pcs
        self.if_hc = args.hc
        self.unet = Unet2D(c=args.c_in,
                           n=16,
                           norm=args.norm,
                           num_classes=args.num_classes,
                           emb_n=args.client_num,
                           pcs=self.if_pcs)
        if self.if_hc:
            self.hc = HeadCalibration(n_classes=args.num_classes, n_fea=32)
        self.site_num = args.client_num

    @torch.no_grad()
    def get_current_emb(self, site_index, k):
        emb = np.zeros((k, self.site_num))
        emb[:, site_index] = 1
        emb = torch.from_numpy(emb).float().cuda()
        return emb

    @torch.no_grad()
    def get_un_map_by_head(self, fea, seg_heads):
        pred_list = []
        for i in range(self.site_num):
            _pred = seg_heads[i](fea)
            _pred = torch.sigmoid(_pred)
            pred_list.append(_pred)
        pred_list = [_pred.unsqueeze(0) for _pred in pred_list]
        pred_list = torch.cat(pred_list, dim=0)
        uncertainty = torch.std(pred_list, dim=0, keepdim=False)

        return uncertainty

    @torch.no_grad()
    def get_un_map(self, x, bs):
        pred_list = []
        for i in range(self.site_num):
            _site_emb = self.get_current_emb(i, bs)
            _, _pred, _, _ = self.unet(x, _site_emb)
            pred_list.append(_pred)

        pred_list = [_pred.unsqueeze(0) for _pred in pred_list]
        pred_list = torch.cat(pred_list, dim=0)
        uncertainty = torch.std(pred_list, dim=0, keepdim=False)

        return uncertainty

    @torch.no_grad()
    def get_emb(self, site_index):
        _site_emb = self.get_current_emb(site_index, 1)
        _emb = self.unet.att5.forward_emb(_site_emb)
        return _emb

    def get_hmaps(self, x):
        bs = x.size(0)
        hmaps = []
        for i in range(self.site_num):
            _site_emb = self.get_current_emb(i, bs)
            _, _hmap = self.unet.pcs(x, _site_emb)
            hmaps.append(_hmap)
        return hmaps

    @torch.no_grad()
    def get_hmap(self, x, site_index):
        bs = x.size(0)
        _emb = self.get_current_emb(site_index, bs)
        _, _, _, hmap = self.unet(x, _emb)
        return hmap

    @torch.no_grad()
    def forward_for_eval(self, x, site_index, seg_heads):
        bs = x.size(0)
        _emb = self.get_current_emb(site_index, bs)
        fea, preds, _, _ = self.unet(x, _emb)
        if self.if_hc:
            uncertainty = self.get_un_map_by_head(fea, seg_heads)
            o = self.hc(uncertainty, preds, fea)
            return o
        else:
            return preds

    def forward_for_train(self, x, site_index, stage, seg_heads):
        bs = x.size(0)
        _emb = self.get_current_emb(site_index, bs)
        if stage == 0:
            assert seg_heads is None
            fea, pred, x5, hmap = self.unet(x, _emb)
            return pred, x5, hmap
        elif stage == 1:
            assert self.hc is not None
            assert seg_heads is not None
            fea, preds, _, _ = self.unet(x, _emb)
            uncertainty = self.get_un_map_by_head(fea, seg_heads)
            o = self.hc(uncertainty.detach(), preds.detach(), fea.detach())
            return o
        else:
            raise NotImplementedError

    def forward(self, x, site_index, stage, seg_heads):
        if self.training:
            o = self.forward_for_train(x, site_index, stage, seg_heads)
        else:
            o = self.forward_for_eval(x, site_index, seg_heads)
        return o


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    x = torch.rand((4, 3, 384, 384)).cuda()
    net = Unet2D().cuda()
    for (name, data) in net.named_parameters():
        print(name)
    y = net(x)