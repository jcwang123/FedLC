import os
from re import L
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
    """ PMR Dataset """
    def __init__(self, client_idx=None, split='train', transform=None):
        self.root_dir = ''  # todo
        self.transform = transform
        self.split = split
        self.client_name = [
            'Site1', 'Site2', 'Site3', 'Site4', 'Site5', 'Site6'
        ]
        self.image_list = glob(
            self.root_dir +
            '/{}/{}/image/*/*'.format(self.client_name[client_idx], split))

        print("total {} slices".format(len(self.image_list)))

    def __len__(self):
        # return len(self.image_list)
        return 300

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.image_list) - 1)
        raw_file = self.image_list[idx]
        image = np.load(raw_file)[np.newaxis]
        mask = np.load(raw_file.replace('image', 'mask'))[np.newaxis]
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        sample = {"image": image, "label": mask}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.5):
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