import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
import random
from scipy import ndimage
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,\
    generate_binary_structure


class Dataset(Dataset):
    """ LA Dataset """
    def __init__(self, client_idx=None, split='train', transform=None):
        self.root_dir = '/raid/wjc/data/SpecializedFedSeg/fundus/'
        self.transform = transform
        self.split = split
        self.client_name = ['Site1', 'Site2', 'Site3', 'Site4']
        self.image_list = glob(
            self.root_dir +
            '/{}/{}/image/*'.format(self.client_name[client_idx], split))

        print("total {} slices".format(len(self.image_list)))

    def __len__(self):
        # return len(self.image_list)
        return 300

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.image_list) - 1)
        raw_file = self.image_list[idx]
        image = np.load(raw_file).transpose((2, 0, 1))
        mask = np.load(raw_file.replace('image', 'mask'))
        image = torch.from_numpy(image) / 255.

        _mask = []
        for _c in range(2):
            _mask.append((mask > _c).copy())
        mask = torch.from_numpy(np.array(_mask))
        sample = {"image": image, "label": mask}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant',
                           constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant',
                           constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1],
                      d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1],
                      d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.4):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = torch.clamp(
            torch.rand(image.size()) * self.sigma, -2 * self.sigma,
            2 * self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


if __name__ == '__main__':
    for _id in range(6):
        x = Dataset(_id, 'train')
        y = Dataset(_id, 'test')
        n = len(x.image_list) + len(y.image_list)
        print(_id, n)