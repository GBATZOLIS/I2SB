# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import io

from PIL import Image
import lmdb

import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import Dataset

from ipdb import set_trace as debug
from tqdm import tqdm 
import numpy as np

def lmdb_loader(path, lmdb_data):
    # In-memory binary streams
    # print(path)
    with lmdb_data.begin(write=False, buffers=True) as txn:
        bytedata = txn.get(path.encode())
    img = Image.open(io.BytesIO(bytedata))
    return img.convert('RGB')

def _build_lmdb_dataset(
        root, log, transform=None, target_transform=None,
        loader=lmdb_loader):
    """
    You can create this dataloader using:
    train_data = _build_lmdb_dataset(traindir, transform=train_transform)
    valid_data = _build_lmdb_dataset(validdir, transform=val_transform)
    """

    print(f'root:{root}')
    root = str(root)
    if root.endswith('/'):
        root = root[:-1]
    pt_path = os.path.join(
        root + '_faster_imagefolder.lmdb.pt')
    lmdb_path = os.path.join(
        root + '_faster_imagefolder.lmdb')

    if os.path.isfile(pt_path) and os.path.isdir(lmdb_path):
        log.info('[Dataset] Loading pt {} and lmdb {}'.format(pt_path, lmdb_path))
        data_set = torch.load(pt_path)
    else:
        data_set = datasets.ImageFolder(
            root, None, None, None)
        torch.save(data_set, pt_path, pickle_protocol=4)
        log.info('[Dataset] Saving pt to {}'.format(pt_path))
        log.info('[Dataset] Building lmdb to {}'.format(lmdb_path))

        allocate_ram_memory_in_gbs = 200 #set it 200GB for the full ImageNet dataset.
        allocated_bytes = allocate_ram_memory_in_gbs * (1024 ** 3)
        env = lmdb.open(lmdb_path, map_size=allocated_bytes)
        with env.begin(write=True) as txn:
            for _path, class_index in tqdm(data_set.imgs):
                with open(_path, 'rb') as f:
                    data = f.read()
                
                #print(data)
                txn.put(_path.encode('ascii'), data)
    data_set.lmdb_data = lmdb.open(
        lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False,
        meminit=False)
    # reset transform and target_transform
    data_set.samples = data_set.imgs
    data_set.transform = transform
    data_set.target_transform = target_transform
    data_set.loader = lambda path: loader(path, data_set.lmdb_data)

    return data_set

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def build_DiT_train_transform(image_size):
    return transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

def build_DiT_test_transform(image_size):
    return transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

def build_train_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) # [0,1] --> [-1, 1]
    ])

def build_test_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) # [0,1] --> [-1, 1]
    ])

def build_lmdb_dataset(opt, log, train, transform=None):
    """ resize -> crop -> to_tensor -> norm(-1,1) """
    fn = opt.dataset_dir / ('train' if train else 'val')

    if transform is None:
        build_transform = build_train_transform if train else build_test_transform
        transform = build_transform(opt.image_size)
    elif transform == 'DiT':
        build_transform = build_DiT_train_transform if train else build_DiT_test_transform
        transform = build_transform(opt.image_size)

    dataset = _build_lmdb_dataset(fn, log, transform=transform)
    log.info(f"[Dataset] Built Imagenet dataset {fn=}, size={len(dataset)}!")
    return dataset

def readlines(fn):
    file = open(fn, "r").readlines()
    return [line.strip('\n\r') for line in file]

def build_lmdb_dataset_val10k(opt, log, transform=None):

    fn_10k = readlines(f"dataset/val_faster_imagefolder_10k_fn.txt")
    label_10k = readlines(f"dataset/val_faster_imagefolder_10k_label.txt")

    if transform is None: transform = build_test_transform(opt.image_size)
    dataset = _build_lmdb_dataset(opt.dataset_dir / 'val', log, transform=transform)
    dataset.samples = [(fn, int(label)) for fn, label in zip(fn_10k, label_10k)]

    assert len(dataset) == 10_000
    log.info(f"[Dataset] Built Imagenet val10k, size={len(dataset)}!")
    return dataset

class InpaintingVal10kSubset(Dataset):
    def __init__(self, opt, log, mask):
        super(InpaintingVal10kSubset, self).__init__()

        assert mask in ["center", "freeform1020", "freeform2030"]
        self.mask_type = mask
        self.dataset = build_lmdb_dataset_val10k(opt, log)

        from corruption.inpaint import get_center_mask, load_freeform_masks
        if self.mask_type == "center":
            self.mask = get_center_mask([opt.image_size, opt.image_size]) # [1,256,256]
        else:
            self.masks = load_freeform_masks(mask)[:,0,...] # [10000, 256, 256]
            assert len(self.dataset) == len(self.masks)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        mask = self.mask if self.mask_type == "center" else self.masks[[index]]
        return *self.dataset[index], mask
