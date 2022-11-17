import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from tqdm import tqdm
from tensorboardX import SummaryWriter

import argparse

import time
import random
import numpy as np

from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# network codes
from networks.lcrepnet import LC_Fed
#
#
#
import medpy.metric as md
import copy

from utils.losses import dice_loss
from utils.summary import create_logger, DisablePrint, create_summary
from utils.util import _eval_dice, _eval_haus, _connectivity_region_analysis, parse_fn_haus

from scripts.trainer_utils import set_global_grad, update_global_model_with_keys, update_global_model
from scripts.tester_utils import test_lc_unet

from dataloaders.fundus_dataloader import Dataset, RandomNoise

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='test', help='model_name')
parser.add_argument('--dataset',
                    type=str,
                    default='fundus',
                    help='dataset name')

parser.add_argument('--head_iter',
                    type=int,
                    default=10,
                    help='iter number of head in local update')

parser.add_argument('--max_epoch',
                    type=int,
                    default=300,
                    help='maximum epoch number to train')
parser.add_argument('--batch_size',
                    type=int,
                    default=6,
                    help='batch_size per gpu')
parser.add_argument('--base_lr',
                    type=float,
                    default=0.001,
                    help='basic learning rate of each site')
parser.add_argument('--load_weight',
                    type=int,
                    default=0,
                    help='load pre-trained weight from local site')

parser.add_argument('--alpha',
                    type=float,
                    default=0,
                    help='contrast loss weight')

parser.add_argument('--norm', type=str, default='in', help='normalization')

parser.add_argument('--pcs', type=int, default=1, help='using pcs')

parser.add_argument('--hc', type=int, default=1, help='using hc')

parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='4', help='GPU to use')
args = parser.parse_args()

args.exp = args.exp + '_pcs_{}_hc_{}_{}_alpha_{}'.format(
    args.pcs, args.hc, args.norm, args.alpha)

txt_path = 'logs/{}/{}/txt/'.format(args.dataset, args.exp)
log_path = 'logs/{}/{}/log/'.format(args.dataset, args.exp)
model_path = 'logs/{}/{}/model/'.format(args.dataset, args.exp)
os.makedirs(txt_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)
logger = create_logger(0, save_dir=txt_path)
print = logger.info
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
base_lr = args.base_lr
max_epoch = args.max_epoch

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

# ------------------  start training ------------------ #
# weight average
client_weight = np.ones((args.client_num, )) / args.client_num
print(client_weight)

if __name__ == "__main__":

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    dataloader_clients = []
    net_clients = []
    optimizer_clients = []
    for client_idx in range(args.client_num):
        net = LC_Fed(args)
        net = net.cuda()

        if args.load_weight:
            pass

        dataset = Dataset(client_idx=client_idx, split='train', transform=None)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=1,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn,
                                drop_last=True)
        # dataloader
        dataloader_clients.append(dataloader)

        # segmentation model
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=args.base_lr,
                                     betas=(0.9, 0.999))
        optimizer_clients.append(optimizer)
        net_clients.append(net)

    to_ignore = ['seg', 'hc']
    global_keys = []
    ignored_keys = []
    for k in net.state_dict().keys():
        ignore_tag = 0
        for ignore_key in to_ignore:
            if ignore_key in k:
                ignore_tag = 1
        if not ignore_tag:
            global_keys.append(k)
        else:
            ignored_keys.append(k)
    print(global_keys)
    print(ignored_keys)

    if args.load_weight:
        # update_global_model(net_clients, client_weight)
        update_global_model_with_keys(net_clients, client_weight, global_keys)

    print('[INFO] Initialized success...')

    # start federated learning
    best_score = 0
    writer = SummaryWriter(log_path)
    lr_ = base_lr

    c_loss_func = nn.MSELoss()
    alpha = 1
    for epoch_num in range(max_epoch):
        seg_heads = copy.deepcopy([_net.unet.seg1 for _net in net_clients])
        for client_idx in range(args.client_num):
            dataloader_current = dataloader_clients[client_idx]
            net_current = net_clients[client_idx]
            net_current.train()
            optimizer_current = optimizer_clients[client_idx]
            time1 = time.time()

            for i_batch, sampled_batch in enumerate(dataloader_current):
                time2 = time.time()

                # obtain training data
                volume_batch, label_batch = sampled_batch['image'].cuda(
                ), sampled_batch['label'].cuda()

                # train seg
                if i_batch < args.head_iter:
                    set_global_grad(net_current, global_keys, False)
                else:
                    set_global_grad(net_current, global_keys, True)
                pred, x5, hmap = net_current(volume_batch, client_idx, 0, None)
                s1_loss = dice_loss(pred, label_batch)

                hmaps = net_current.get_hmaps(x5.detach())
                c_loss = 0
                for other_client in range(args.client_num):
                    if not other_client == client_idx:
                        c_loss += c_loss_func(hmaps[client_idx],
                                              hmaps[other_client].detach())
                c_loss = -c_loss / (args.client_num - 1)

                s1_loss = s1_loss + c_loss * args.alpha
                optimizer_current.zero_grad()
                s1_loss.backward()
                optimizer_current.step()

                # train calibration net
                if args.hc:
                    pred, x5, hmap = net_current(volume_batch, client_idx, 1,
                                                 seg_heads)
                    s2_loss = dice_loss(pred, label_batch)
                    optimizer_current.zero_grad()
                    s2_loss.backward()
                    optimizer_current.step()

                iter_num = len(dataloader_current) * epoch_num + i_batch

                if iter_num % 10 == 0:
                    if args.hc:
                        writer.add_scalar('loss/site{}'.format(client_idx + 1),
                                          s2_loss, iter_num)
                    else:
                        writer.add_scalar('loss/site{}'.format(client_idx + 1),
                                          s1_loss, iter_num)
                    if args.hc:
                        print(
                            'Epoch: [%d] client [%d] iteration [%d / %d] : s1 loss : %f s2 loss : %f'
                            % (epoch_num, client_idx, i_batch,
                               len(dataloader_current), s1_loss.item(),
                               s2_loss.item()))
                    else:
                        print(
                            'Epoch: [%d] client [%d] iteration [%d / %d] : s1 loss : %f '
                            % (epoch_num, client_idx, i_batch,
                               len(dataloader_current), s1_loss.item()))

        ## model aggregation
        update_global_model_with_keys(net_clients, client_weight, global_keys)

        ## evaluation
        overall_score = 0
        for site_index in range(args.client_num):
            this_net = net_clients[site_index]
            dice_list = []
            print("[Test] epoch {} testing Site {}".format(
                epoch_num, site_index + 1))

            score_values = test_lc_unet(site_index, this_net, seg_heads, args)
            writer.add_scalar('Score/site{}'.format(site_index + 1),
                              np.mean(score_values[0]), epoch_num)
            overall_score += np.mean(score_values[0])
        overall_score /= args.client_num
        writer.add_scalar('Score_Overall', overall_score, epoch_num)

        if overall_score > best_score:
            best_score = overall_score
            ## save model
            save_mode_path = os.path.join(model_path, 'best.pth')
            torch.save(net_clients[0].state_dict(), save_mode_path)

            for site_index in range(args.client_num):
                save_mode_path = os.path.join(
                    model_path, 'Site{}_best.pth'.format(site_index + 1))
                torch.save(net_clients[site_index].state_dict(),
                           save_mode_path)
        print('[INFO] Dice Overall: {:.2f} Best Dice {:.2f}'.format(
            overall_score * 100, best_score * 100))
