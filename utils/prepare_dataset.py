import os
from typing import OrderedDict
import SimpleITK as sitk
# import nibabel as nib
import os
import numpy as np
from glob import glob
import time
import shutil
import matplotlib.pyplot as plt
import cv2, sys
import SimpleITK as sitk
import matplotlib.pyplot as plt


def save_npy(data, p):
    dir = os.path.dirname(p)
    os.makedirs(dir, exist_ok=True)
    np.save(p, data)


def mr_norm(x, r=0.99):
    # normalize mr image
    # x: w,h
    _x = x.flatten().tolist()
    _x.sort()
    vmax = _x[int(len(_x) * r)]
    vmin = _x[0]
    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / vmax
    return x


def prepare_pmr(save_dir):
    ori_dir = '/raid/wjc/data/PMR'
    site_names = os.listdir(ori_dir)
    site_names.sort()
    for i in range(len(site_names)):
        seg_paths = glob(ori_dir + '/' + site_names[i] + '/*mentation*')
        seg_paths.sort()
        print('[INFO]', site_names[i], len(seg_paths))
        img_paths = [p[:-20] + '.nii.gz' for p in seg_paths]
        for j in range(len(seg_paths)):
            itk_image = sitk.ReadImage(img_paths[j])
            itk_mask = sitk.ReadImage(seg_paths[j])
            image = sitk.GetArrayFromImage(itk_image)
            mask = sitk.GetArrayFromImage(itk_mask)

            train_or_test = 'train' if j < (int(len(seg_paths) *
                                                0.8)) else 'test'
            case_name = img_paths[j].split('/')[-1][:6]

            cnt = np.zeros(2, )
            for k in range(image.shape[0]):
                base_name = 'slice_{:03d}.npy'.format(k)
                slice_image = mr_norm(image[k])
                slice_mask = mask[k] > 0
                if slice_mask.max() > 0:
                    cnt[1] += 1
                else:
                    continue
                cnt[0] += 1
                save_npy(
                    slice_image,
                    save_dir + '/pmr/Site{}/{}/image/{}/{}'.format(
                        i + 1, train_or_test, case_name, base_name))
                save_npy(
                    slice_mask, save_dir + '/pmr/Site{}/{}/mask/{}/{}'.format(
                        i + 1, train_or_test, case_name, base_name))
                # print(slice_image.shape, slice_mask.shape, slice_image.max(),
                #   slice_image.min(), slice_mask.max())
            print(cnt)


def prepare_fundus(save_dir):
    #  download fundus data to $ori_dir
    ori_dir = '/raid/wjc/data/Fundus'
    for site_index in range(1, 5):
        image_paths = glob(ori_dir +
                           '/Site{}/*/ROIs/image/*.png'.format(site_index))
        for image_path in image_paths:
            mask_path = image_path.replace('image', 'mask')
            assert (os.path.exists(mask_path))
            img = cv2.imread(image_path)[:, :, ::-1]
            img = cv2.resize(img, (384, 384), cv2.INTER_CUBIC)

            mask = 2 - np.array(cv2.imread(mask_path, 0) / 127, dtype='uint8')
            mask = cv2.resize(mask, (384, 384), cv2.INTER_NEAREST)
            site_p = image_path.split('/')[-5]
            train_or_test = image_path.split('/')[-4]
            file_name = image_path.split('/')[-1][:-4]

            save_npy(
                img, save_dir + '/fundus/{}/{}/image/{}'.format(
                    site_p, train_or_test, file_name))
            save_npy(
                mask, save_dir + '/fundus/{}/{}/mask/{}'.format(
                    site_p, train_or_test, file_name))


def prepare_polyp(save_dir):
    #  download fundus data to $ori_dir
    ori_dir = '/raid/wjc/data/polyp'
    site_names = ['Kvasir', 'ETIS', 'CVC-ColonDB', 'CVC-ClinicDB']

    #   221:273
    for train_or_test in ['train', 'test']:
        # train
        image_paths = [[], [], [], []]
        if train_or_test == 'train':
            for image_path in glob(
                    '/raid/wjc/data/polyp/TrainDataset/images/*'):
                if image_path.split('/')[-1][0] == 'c':
                    image_paths[0].append(image_path)
                else:
                    image_paths[-1].append(image_path)
            image_paths[2] = [
                '/raid/wjc/data/polyp/TestDataset/CVC-ColonDB/images/{}.png'.
                format(i) for i in range(1, 221)
            ] + [
                '/raid/wjc/data/polyp/TestDataset/CVC-ColonDB/images/{}.png'.
                format(i) for i in range(273, 380)
            ]
            image_paths[1] = [
                '/raid/wjc/data/polyp/TestDataset/CVC-ColonDB/images/{}.png'.
                format(i) for i in range(1, 171)
            ]
        else:
            image_paths[0] = glob(
                '/raid/wjc/data/polyp/TestDataset/Kvasir/images/*')
            image_paths[-1] = glob(
                '/raid/wjc/data/polyp/TestDataset/CVC-ClinicDB/images/*')
            image_paths[2] = [
                '/raid/wjc/data/polyp/TestDataset/CVC-ColonDB/images/{}.png'.
                format(i) for i in range(221, 273)
            ]
            image_paths[1] = [
                '/raid/wjc/data/polyp/TestDataset/CVC-ColonDB/images/{}.png'.
                format(i) for i in range(171, 197)
            ]
        for i, site_name in enumerate(site_names):
            for image_path in image_paths[i]:
                mask_path = image_path.replace('images', 'masks')
                assert (os.path.exists(mask_path))
                img = cv2.imread(image_path)[:, :, ::-1]
                img = cv2.resize(img, (384, 384), cv2.INTER_CUBIC)

                mask = np.array((cv2.imread(mask_path, 0) > 0) * 1,
                                dtype='uint8')
                mask = cv2.resize(mask, (384, 384), cv2.INTER_NEAREST)

                site_p = 'Site{}'.format(i + 1)

                file_name = image_path.split('/')[-1][:-4]

                save_npy(
                    img, save_dir + '/polyp/{}/{}/image/{}'.format(
                        site_p, train_or_test, file_name))
                save_npy(
                    mask, save_dir + '/polyp/{}/{}/mask/{}'.format(
                        site_p, train_or_test, file_name))


if __name__ == '__main__':
    # prepare_fundus(save_dir='/raid/wjc/data/SpecializedFedSeg')
    # prepare_pmr(save_dir='/raid/wjc/data/SpecializedFedSeg')
    prepare_polyp(save_dir='/raid/wjc/data/SpecializedFedSeg')
