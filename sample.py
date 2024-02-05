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
import lpips
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

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

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

def get_image_save_path(opt, image_type, nfe):
    """
    Generates the file path for saving images.

    Args:
    opt (edict): Options dictionary containing configuration settings.
    image_type (str): Type of the image, either 'recon' for reconstructed or 'corrupt' for corrupted images.
    nfe (int): Number of function evaluations or steps used in the reconstruction process.

    Returns:
    Path: The file path for saving the images.
    """
    assert image_type in ['recon', 'corrupt'], "image_type must be either 'recon' or 'corrupt'"

    sample_dir = RESULT_DIR / opt.ckpt / "samples_nfe{}{}".format(
        nfe, "_clip" if opt.clip_denoise else ""
    )
    os.makedirs(sample_dir, exist_ok=True)

    image_save_path = sample_dir / f"{image_type}{'_' if opt.partition is not None else ''}{opt.partition}.pt"
    return image_save_path


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

@torch.no_grad()
def main(opt):
    log = Logger(opt.global_rank, ".log")

    # Get (default) checkpoint option
    ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / opt.ckpt)
    corrupt_type = ckpt_opt.corrupt
    nfe = opt.nfe or ckpt_opt.interval - 1

    # Build corruption method
    corrupt_method = build_corruption(opt, log, corrupt_type=corrupt_type)

    # Build ImageNet validation dataset
    val_dataset = build_val_dataset(opt, log, corrupt_type)
    n_samples = len(val_dataset)

    # Build dataset per GPU and loader
    subset_dataset = build_subset_per_gpu(opt, val_dataset, log)
    val_loader = DataLoader(subset_dataset,
                            batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False)

    # Build runner
    runner = Runner(ckpt_opt, log, save_opt=False)

    # Handle use_fp16 for EMA
    if opt.use_fp16:
        runner.ema.copy_to()  # Copy weight from EMA to net
        runner.net.diffusion_model.convert_to_fp16()
        runner.ema = ExponentialMovingAverage(runner.net.parameters(), decay=0.99)  # Re-init EMA with fp16 weight

    # Create save folder
    corrupt_imgs_fn = get_image_save_path(opt, 'corrupt', nfe)
    recon_imgs_fn = get_image_save_path(opt, 'recon', nfe)
    log.info(f"Recon images will be saved to {recon_imgs_fn}!")

    corrupt_imgs = []
    recon_imgs = []
    real_imgs = []  # Store real images
    ys = []
    num = 0

    # Initialize lists to collect LPIPS scores
    lpips_model = lpips.LPIPS(net='alex').to(opt.device)
    lpips_scores_corrupt = []
    lpips_scores_recon = []

    for loader_itr, out in enumerate(val_loader):
        if loader_itr > 5:
            break

        real, _ = out
        real = real.to(opt.device)
        corrupt_img, x1, mask, cond, y = compute_batch(ckpt_opt, corrupt_type, corrupt_method, out)

        xs, _ = runner.ddpm_sampling(
            ckpt_opt, x1, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, nfe=nfe, verbose=opt.n_gpu_per_node == 1
        )
        recon_img = xs[:, 0, ...].to(opt.device)

        assert recon_img.shape == corrupt_img.shape

        if loader_itr == 0 and opt.global_rank == 0:  # Debug
            os.makedirs(".debug", exist_ok=True)
            tu.save_image((corrupt_img + 1) / 2, ".debug/corrupt.png")
            tu.save_image((recon_img + 1) / 2, ".debug/recon.png")
            log.info("Saved debug images!")

        # Compute LPIPS scores for corrupted and reconstructed images
        lpips_score_corrupt = lpips_model(real, corrupt_img).mean()
        lpips_score_recon = lpips_model(real, recon_img).mean()
        
        # Aggregate scores locally (per GPU/process)
        lpips_scores_corrupt.append(lpips_score_corrupt.item())
        lpips_scores_recon.append(lpips_score_recon.item())

        gathered_recon_img = collect_all_subset(recon_img, log)
        recon_imgs.append(gathered_recon_img)

        y = y.to(opt.device)
        gathered_y = collect_all_subset(y, log)
        ys.append(gathered_y)

        gathered_real_img = collect_all_subset(real, log)  # Collect real images
        real_imgs.append(gathered_real_img)  # Append to real images list

        gathered_corrupt_img = collect_all_subset(corrupt_img, log)
        corrupt_imgs.append(gathered_corrupt_img)

        num += len(gathered_recon_img)
        log.info(f"Collected {num} recon images!")
        dist.barrier()

    del runner

    corrupt_arr = torch.cat(corrupt_imgs, axis=0)[:n_samples]
    arr = torch.cat(recon_imgs, axis=0)[:n_samples]
    real_arr = torch.cat(real_imgs, axis=0)[:n_samples]  # Concatenate real images
    label_arr = torch.cat(ys, axis=0)[:n_samples]

    # After the loop, reduce the scores across all GPUs
    # Convert lists to tensors for reduction
    lpips_scores_corrupt_tensor = torch.tensor(lpips_scores_corrupt, device=opt.device)
    lpips_scores_recon_tensor = torch.tensor(lpips_scores_recon, device=opt.device)

    # Sum scores across all GPUs
    dist.reduce(lpips_scores_corrupt_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(lpips_scores_recon_tensor, dst=0, op=dist.ReduceOp.SUM)

    # Handle global rank 0 specific operations
    if opt.global_rank == 0:
        N=5
        # Initialize a list to hold the images for the grid
        images_for_grid = []

        # Loop through the first N samples and add real, corrupt, and recon images to the list
        for i in range(N):
            images_for_grid.append(real_arr[i])
            images_for_grid.append(corrupt_arr[i])
            images_for_grid.append(arr[i])

        # Stack the list of images into a single tensor
        images_tensor = torch.stack(images_for_grid)

        # Create the grid with N rows, each row has 3 images
        # Since we're creating N rows with 3 images per row, set nrow=3
        image_grid = tu.make_grid(images_tensor, nrow=3, padding=2, normalize=True, value_range=(-1, 1))

        # Save the grid to a file
        save_path = ".debug/comparison.png"
        tu.save_image(image_grid, save_path)
        
        log.info(f"Saved comparison grid to {save_path}")
    
        # Only the master process computes the average
        avg_lpips_corrupt = lpips_scores_corrupt_tensor.sum().item() / (len(lpips_scores_corrupt) * opt.global_size)
        avg_lpips_recon = lpips_scores_recon_tensor.sum().item() / (len(lpips_scores_recon) * opt.global_size)
        
        # Log the averaged LPIPS scores
        log.info(f"Average LPIPS score for corruption: {avg_lpips_corrupt}")
        log.info(f"Average LPIPS score for reconstruction: {avg_lpips_recon}")

        torch.save({"arr": arr, "label_arr": label_arr}, recon_imgs_fn)
        log.info(f"Save at {recon_imgs_fn}")

        torch.save({"arr": corrupt_arr, "label_arr": label_arr}, corrupt_imgs_fn)
        log.info(f"Save corrupted images at {corrupt_imgs_fn}")

        # Print min and max values for all types of images
        recon_min, recon_max = torch.min(arr), torch.max(arr)
        corrupt_min, corrupt_max = torch.min(corrupt_arr), torch.max(corrupt_arr)
        real_min, real_max = torch.min(real_arr), torch.max(real_arr)

        log.info(f"Minimum and maximum values of recon_arr: {recon_min}, {recon_max}")
        log.info(f"Minimum and maximum values of corrupt_arr: {corrupt_min}, {corrupt_max}")
        log.info(f"Minimum and maximum values of real images: {real_min}, {real_max}")

    dist.barrier()  # Ensure synchronization before finishing
    log.info(f"Sampling complete! Collect recon_imgs={arr.shape}, real_imgs={real_arr.shape}, ys={label_arr.shape}")



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
