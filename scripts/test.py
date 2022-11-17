import os
import sys
from numpy.lib.npyio import load

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from scripts.tester_utils import test_normal_unet, test_lc_unet

import torch
import argparse
import random
import numpy as np
from glob import glob

from utils.summary import create_logger, DisablePrint, create_summary
from utils.util import _eval_dice, _eval_haus, _connectivity_region_analysis, parse_fn_haus
from utils.util import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='lcfed', help='model_name')
parser.add_argument('--arch', type=str, default='ours')
parser.add_argument('--dataset',
                    type=str,
                    default='polyp',
                    help='dataset name')

parser.add_argument('--norm', type=str, default='in', help='normalization')
parser.add_argument('--pcs', type=int, default=1, help='using pcs')
parser.add_argument('--pcs_n', type=int, default=1)
parser.add_argument('--hc', type=int, default=1, help='using hc')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
args = parser.parse_args()

txt_path = 'logs/{}/{}/txt/'.format(args.dataset, args.exp)
log_path = 'logs/{}/{}/log/'.format(args.dataset, args.exp)
model_path = 'logs/{}/{}/model/'.format(args.dataset, args.exp)
npy_path = 'logs/{}/{}/npy/'.format(args.dataset, args.exp)
import os

os.makedirs(npy_path, exist_ok=True)
logger = create_logger(0, save_dir=txt_path)
print = logger.info

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.dataset == 'fundus':
    from dataloaders.fundus_dataloader import Dataset, RandomNoise
    args.client_num = 4
    args.num_classes = 2
    args.c_in = 3
elif args.dataset == 'pmr':
    from dataloaders.pmr_dataloader import Dataset, RandomNoise
    args.client_num = 6
    args.num_classes = 1
    args.c_in = 1
elif args.dataset == 'polyp':
    from dataloaders.polyp_dataloader import Dataset, RandomNoise
    args.client_num = 4
    args.num_classes = 1
    args.c_in = 3
else:
    raise NotImplementedError
assert args.num_classes > 0 and args.client_num > 1
print(args)


def get_current_emb(site_index, k=1):
    emb = np.zeros((k, args.client_num))
    emb[:, site_index] = 1
    emb = torch.from_numpy(emb).float()
    return emb


if __name__ == "__main__":
    # define dataset, model, optimizer for each client
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    dataloader_clients = []
    net_clients = []
    optimizer_clients = []
    for client_idx in range(args.client_num):
        if args.arch == 'ours':
            from networks.multi_lcnet import LC_Fed
            net = LC_Fed(args)
            print('arch:' + args.arch)
        elif 'bn' in args.exp:
            from networks.unet2d import Unet2D
            net = Unet2D(c=args.c_in, num_classes=args.num_classes, norm='bn')
        else:
            from networks.unet2d import Unet2D
            net = Unet2D(c=args.c_in, num_classes=args.num_classes, norm='in')

        net = net.cuda()
        net = load_model(
            net, model_path + '/Site{}_best.pth'.format(client_idx + 1))

        net_clients.append(net)

    if args.arch == 'ours':
        import copy
        seg_heads = copy.deepcopy([_net.unet.seg1 for _net in net_clients])
    print('[INFO] Initialized success...')
    _iou = ''
    _dc = ''
    _hd = ''
    _assd = ''
    overall = np.zeros((4, ))
    ious = []
    for site_index in range(args.client_num):
        this_net = net_clients[site_index]
        this_net.eval()
        dice_list = []
        if args.arch == 'ours':
            dice, haus, iou, assd = test_lc_unet(site_index,
                                                 this_net,
                                                 seg_heads,
                                                 args,
                                                 info=True)
        else:
            dice, haus, iou, assd = test_normal_unet(site_index,
                                                     this_net,
                                                     args,
                                                     info=True)
        np.save(npy_path + f'/site_{site_index}.npy',
                np.concatenate([dice, haus, iou, assd], axis=0))
        _iou += '&{:.2f}'.format(np.mean(iou) * 100)
        _dc += '&{:.2f}'.format(np.mean(dice) * 100)
        _hd += '&{:.2f}'.format(np.mean(haus))
        _assd += '&{:.2f}'.format(np.mean(assd))
        overall += np.array(
            [np.mean(iou),
             np.mean(dice),
             np.mean(haus),
             np.mean(assd)])
        ious.append(np.mean(iou) * 100)
    overall /= args.client_num
    context = _iou + '&{:.2f}'.format(
        overall[0] * 100) + _dc + '&{:.2f}'.format(
            overall[1] * 100) + _hd + '&{:.2f}'.format(
                overall[2]) + _assd + '&{:.2f}'.format(overall[3])
    print(_iou + '&{:.2f}'.format(overall[0] * 100))
    print(np.std(ious))
    print(_assd + '&{:.2f}'.format(overall[3]))
    print(context)