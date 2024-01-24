# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import copy
import argparse
import random
from pathlib import Path
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu

from logger import Logger
import distributed_util as dist_util
from i2sb import Runner, download_ckpt
from corruption import build_corruption
from dataset import imagenet
from i2sb import ckpt_util

import colored_traceback.always
from ipdb import set_trace as debug

RESULT_DIR = Path("results")

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def build_subset_per_gpu(opt, dataset, log):
    n_data = len(dataset)
    n_gpu  = opt.global_size
    n_dump = (n_data % n_gpu > 0) * (n_gpu - n_data % n_gpu)

    # create index for each gpu
    total_idx = np.concatenate([np.arange(n_data), np.zeros(n_dump)]).astype(int)
    idx_per_gpu = total_idx.reshape(-1, n_gpu)[:, opt.global_rank]
    log.info(f"[Dataset] Add {n_dump} data to the end to be devided by {n_gpu=}. Total length={len(total_idx)}!")

    # build subset
    indices = idx_per_gpu.tolist()
    subset = Subset(dataset, indices)
    log.info(f"[Dataset] Built subset for gpu={opt.global_rank}! Now size={len(subset)}!")
    return subset

def collect_all_subset(sample, log):
    batch, *xdim = sample.shape
    # Ensure the tensor is contiguous
    contiguous_sample = sample.contiguous()
    gathered_samples = dist_util.all_gather(contiguous_sample, log)
    gathered_samples = [sample.cpu() for sample in gathered_samples]
    # [batch, n_gpu, *xdim] --> [batch*n_gpu, *xdim]
    return torch.stack(gathered_samples, dim=1).reshape(-1, *xdim)

def build_partition(opt, full_dataset, log):
    n_samples = len(full_dataset)

    part_idx, n_part = [int(s) for s in opt.partition.split("_")]
    assert part_idx < n_part and part_idx >= 0
    assert n_samples % n_part == 0

    n_samples_per_part = n_samples // n_part
    start_idx = part_idx * n_samples_per_part
    end_idx = (part_idx+1) * n_samples_per_part

    indices = [i for i in range(start_idx, end_idx)]
    subset = Subset(full_dataset, indices)
    log.info(f"[Dataset] Built partition={opt.partition}, {start_idx=}, {end_idx=}! Now size={len(subset)}!")
    return subset

def build_val_dataset(opt, log, corrupt_type):
    val_dataset = imagenet.build_lmdb_dataset(opt, log, train=False, transform=opt.dataset_transform) # full 50k val
    '''
    if "sr4x" in corrupt_type:
        val_dataset = imagenet.build_lmdb_dataset(opt, log, train=False) # full 50k val
    elif "inpaint" in corrupt_type:
        mask = corrupt_type.split("-")[1]
        val_dataset = imagenet.InpaintingVal10kSubset(opt, log, mask) # subset 10k val + mask
    elif corrupt_type == "mixture":
        from corruption.mixture import MixtureCorruptDatasetVal
        val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log)
        val_dataset = MixtureCorruptDatasetVal(opt, val_dataset) # subset 10k val + mixture
    else:
        val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log) # subset 10k val
    '''
    # build partition
    if opt.partition is not None:
        val_dataset = build_partition(opt, val_dataset, log)
    return val_dataset

def get_recon_imgs_fn(opt, nfe):
    sample_dir = RESULT_DIR / opt.ckpt / "samples_nfe{}{}".format(
        nfe, "_clip" if opt.clip_denoise else ""
    )
    os.makedirs(sample_dir, exist_ok=True)

    recon_imgs_fn = sample_dir / "recon{}.pt".format(
        "" if opt.partition is None else f"_{opt.partition}"
    )
    return recon_imgs_fn

def compute_batch(ckpt_opt, corrupt_type, corrupt_method, out):
    if "inpaint" in corrupt_type:
        clean_img, y, mask = out
        corrupt_img = clean_img * (1. - mask) + mask
        x1          = clean_img * (1. - mask) + mask * torch.randn_like(clean_img)
    elif corrupt_type == "mixture":
        clean_img, corrupt_img, y = out
        mask = None
    else:
        clean_img, y = out
        mask = None
        corrupt_img = corrupt_method(clean_img.to(opt.device))
        x1 = corrupt_img.to(opt.device)

    cond = x1.detach() if ckpt_opt.cond_x1 else None
    if ckpt_opt.add_x1_noise: # only for decolor
        x1 = x1 + torch.randn_like(x1)

    return corrupt_img, x1, mask, cond, y

def find_last_saved_checkpoint(sample_dir, opt):
    # List all files matching the pattern 'recon{}_*.pt'
    pattern = f"recon{'_' + opt.partition if opt.partition is not None else ''}_*.pt"
    files = list(sample_dir.glob(pattern))

    if not files:
        return None, 0  # No files found, start from the beginning

    # Assuming only one file is present
    last_file = files[0]
    last_num = int(last_file.stem.split('_')[-1])

    return last_file, last_num

def get_recon_imgs_fn(opt, nfe, num_images):
    sample_dir = RESULT_DIR / opt.ckpt / "samples_nfe{}{}".format(
        nfe, "_clip" if opt.clip_denoise else ""
    )
    os.makedirs(sample_dir, exist_ok=True)

    recon_imgs_fn = sample_dir / "recon{}_{}.pt".format(
        "" if opt.partition is None else f"_{opt.partition}", num_images
    )
    return recon_imgs_fn

@torch.no_grad()
def main(opt):
    log = Logger(opt.global_rank, ".log")

    # get (default) ckpt option
    ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / opt.ckpt)
    corrupt_type = ckpt_opt.corrupt
    nfe = opt.nfe or ckpt_opt.interval-1

    # build corruption method
    corrupt_method = build_corruption(opt, log, corrupt_type=corrupt_type)

    # build imagenet val dataset
    val_dataset = build_val_dataset(opt, log, corrupt_type)
    n_samples = len(val_dataset)

    # build dataset per gpu and loader
    subset_dataset = build_subset_per_gpu(opt, val_dataset, log)
    val_loader = DataLoader(subset_dataset,
        batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,
    )

    # build runner
    runner = Runner(ckpt_opt, log, save_opt=False)

    # handle use_fp16 for ema
    if opt.use_fp16:
        runner.ema.copy_to() # copy weight from ema to net
        runner.net.diffusion_model.convert_to_fp16()
        runner.ema = ExponentialMovingAverage(runner.net.parameters(), decay=0.99) # re-init ema with fp16 weight

    # ... [rest of the setup code remains unchanged] ...

    # Determine the starting point for processing
    sample_dir = RESULT_DIR / opt.ckpt / "samples_nfe{}{}".format(
        opt.nfe or ckpt_opt.interval-1, "_clip" if opt.clip_denoise else ""
    )
    
    last_saved_file, last_saved_num = find_last_saved_checkpoint(sample_dir, opt)

    recon_imgs = []
    original_imgs = []  # List to store original images
    ys = []
    num = last_saved_num

    for loader_itr, out in enumerate(val_loader):
        # Skip already processed images
        if loader_itr * opt.batch_size < last_saved_num:
            continue

        # ... [rest of the processing code remains unchanged] ...

        corrupt_img, x1, mask, cond, y = compute_batch(ckpt_opt, corrupt_type, corrupt_method, out)

        # Store original images
        original_imgs.append(corrupt_img.cpu())

        # ... [rest of the reconstruction and gathering code] ...

        num += len(gathered_recon_img)
        log.info(f"Collected {num} recon images!")

        # Check if 1000 images have been processed and save
        if num - last_saved_num >= 1000 or loader_itr == len(val_loader) - 1:
            arr = torch.cat(recon_imgs, axis=0)[:num]
            orig_arr = torch.cat(original_imgs, axis=0)[:num]  # Original images
            label_arr = torch.cat(ys, axis=0)[:num]
            
            recon_imgs_fn = get_recon_imgs_fn(opt, nfe, num)
            if opt.global_rank == 0:
                # Save both reconstructed and original images
                torch.save({"recon_arr": arr, "orig_arr": orig_arr, "label_arr": label_arr}, recon_imgs_fn)
                log.info(f"Saved at {recon_imgs_fn}")
                
                # Remove previous file
                if last_saved_file is not None:
                    os.remove(last_saved_file)
                    log.info(f"Removed {last_saved_file}")

            last_saved_num = num
            last_saved_file = recon_imgs_fn

        dist.barrier()

    del runner

    log.info(f"Sampling complete! Collect recon_imgs={arr.shape}, orig_imgs={orig_arr.shape}, ys={label_arr.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,  default=0)
    parser.add_argument("--n-gpu-per-node", type=int,  default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,  default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,  default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,  default=1,           help="The number of nodes in multi node env")

    # data
    parser.add_argument("--image-size",     type=int,  default=256)
    parser.add_argument("--dataset-dir",    type=Path, default="/dataset",  help="path to LMDB dataset")
    parser.add_argument("--vae-model-name", type=str,   default='stabilityai/sd-vae-ft-ema', help="Diffusers stability AI vae - ema weights")
    parser.add_argument("--dataset-transform",    type=str,  default=None,  help="dataset transformation sequence")
    parser.add_argument("--partition",      type=str,  default=None,        help="e.g., '0_4' means the first 25% of the dataset")

    # sample
    parser.add_argument("--batch-size",     type=int,  default=32)
    parser.add_argument("--ckpt",           type=str,  default=None,        help="the checkpoint name from which we wish to sample")
    parser.add_argument("--nfe",            type=int,  default=None,        help="sampling steps")
    parser.add_argument("--clip-denoise",   action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--use-fp16",       action="store_true",            help="use fp16 network weight for faster sampling")

    arg = parser.parse_args()

    opt = edict(
        distributed=(arg.n_gpu_per_node > 1),
        device="cuda",
    )
    opt.update(vars(arg))

    # one-time download: ADM checkpoint
    download_ckpt("data/")

    set_seed(opt.seed)

    if opt.distributed:
        size = opt.n_gpu_per_node

        processes = []
        for rank in range(size):
            opt = copy.deepcopy(opt)
            opt.local_rank = rank
            global_rank = rank + opt.node_rank * opt.n_gpu_per_node
            global_size = opt.num_proc_node * opt.n_gpu_per_node
            opt.global_rank = global_rank
            opt.global_size = global_size
            print('Node rank %d, local proc %d, global proc %d, global_size %d' % (opt.node_rank, rank, global_rank, global_size))
            p = Process(target=dist_util.init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        dist_util.init_processes(0, opt.n_gpu_per_node, main, opt)
