# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import pickle

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
import torchmetrics

import distributed_util as dist_util
from evaluation import build_resnet50

from . import util
from .network import Image256Net
from .diffusion import Diffusion

from ipdb import set_trace as debug
import lpips

def build_optimizer_sched(opt, net, log):

    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()

class Runner(object):
    def __init__(self, opt, log, save_opt=True):
        super(Runner,self).__init__()

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
        self.net = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)

        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")

        self.net.to(opt.device)
        self.ema.to(opt.device)

        self.log = log

        #evaluation
        self.lpips_model = lpips.LPIPS(net='vgg').to(opt.device)

    def compute_label(self, step, x0, xt):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def sample_batch(self, opt, loader, corrupt_method):
        if opt.corrupt == "mixture":
            clean_img, corrupt_img, y = next(loader)
            mask = None
        elif "inpaint" in opt.corrupt:
            clean_img, y = next(loader)
            with torch.no_grad():
                corrupt_img, mask = corrupt_method(clean_img.to(opt.device))
        else:
            clean_img, y = next(loader)
            with torch.no_grad():
                corrupt_img = corrupt_method(clean_img.to(opt.device))
            mask = None

        # os.makedirs(".debug", exist_ok=True)
        # tu.save_image((clean_img+1)/2, ".debug/clean.png", nrow=4)
        # tu.save_image((corrupt_img+1)/2, ".debug/corrupt.png", nrow=4)
        # debug()

        y  = y.detach().to(opt.device)
        x0 = clean_img.detach().to(opt.device)
        x1 = corrupt_img.detach().to(opt.device)
        if mask is not None:
            mask = mask.detach().to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)
        cond = x1.detach() if opt.cond_x1 else None

        if opt.add_x1_noise: # only for decolor
            x1 = x1 + torch.randn_like(x1)

        assert x0.shape == x1.shape

        return x0, x1, mask, y, cond

    def train(self, opt, train_dataset, val_dataset, corrupt_method):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader   = util.setup_loader(val_dataset,   opt.microbatch)

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=1000).to(opt.device)
        self.resnet = build_resnet50().to(opt.device)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                # ===== sample boundary pair =====
                x0, x1, mask, y, cond = self.sample_batch(opt, train_loader, corrupt_method)

                # ===== compute loss =====
                step = torch.randint(0, opt.interval, (x0.shape[0],))

                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                label = self.compute_label(step, x0, xt)

                pred = net(xt, step, cond=cond)
                assert xt.shape == label.shape == pred.shape

                if mask is not None:
                    pred = mask * pred
                    label = mask * label

                loss = F.mse_loss(pred, label)
                loss.backward()

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if (it+1) % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if (it+1) % 5000 == 0:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()

            if (it+1) == 500 or (it+1) % 10000 == 0: # 0, 0.5k, 3k, 6k 9k
                net.eval()
                self.evaluation(opt, it+1, val_loader, corrupt_method)
                net.train()
        self.writer.close()

    @torch.no_grad()
    def ddpm_sampling(self, opt, x1, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or opt.interval-1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe+1)

        # create log steps
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0
        self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)
        if mask is not None:
            mask = mask.to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):
                step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                out = self.net(xt, step, cond=cond)
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose,
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return xs, pred_x0


    @torch.no_grad()
    def evaluation(self, opt, it, val_loader, corrupt_method, num_batches=65):
        def log_image(tag, img, nrow=10):
            # Log image that is already on CPU
            self.writer.add_image(it, tag, tu.make_grid((img + 1) / 2, nrow=nrow))

        def calculate_accuracy(img):
            pred = self.resnet(img.to(opt.device)) # input range [-1,1]
            accu = self.accuracy(pred, y.to(opt.device)).item()
            return accu

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        corrupt_accu_scores = []
        recon_accu_scores = []
        corrupt_lpips_scores = []
        recon_lpips_scores = []

        for batch_idx in range(num_batches):
            img_clean, img_corrupt, mask, y, cond = self.sample_batch(opt, val_loader, corrupt_method)

            x1 = img_corrupt.to(opt.device)

            xs, pred_x0s = self.ddpm_sampling(
                opt, x1, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, nfe=10, verbose=opt.global_rank==0
            )

            log.info("Collecting tensors ...")
            img_clean   = all_cat_cpu(opt, log, img_clean)
            img_corrupt = all_cat_cpu(opt, log, img_corrupt)
            y           = all_cat_cpu(opt, log, y)
            xs          = all_cat_cpu(opt, log, xs)
            pred_x0s    = all_cat_cpu(opt, log, pred_x0s)
            

            batch, len_t, *xdim = xs.shape
            assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
            assert xs.shape == pred_x0s.shape
            assert y.shape == (batch,)

            xs = xs[:, 0, ...] #img_recon

            # Calculate accuracy (ensure tensors are on the correct device)
            corrupt_accuracy = calculate_accuracy(img_corrupt)
            corrupt_accu_scores.append(corrupt_accuracy)
            recon_accuracy = calculate_accuracy(xs)
            recon_accu_scores.append(recon_accuracy)

            # Convert images for LPIPS before moving them off the GPU
            recon_lpips = self.lpips_model((xs.to(opt.device) + 1) / 2, (img_clean.to(opt.device) + 1) / 2).mean().item()
            recon_lpips_scores.append(recon_lpips)
            corrupt_lpips = self.lpips_model((img_corrupt.to(opt.device) + 1) / 2, (img_clean.to(opt.device) + 1) / 2).mean().item()
            corrupt_lpips_scores.append(corrupt_lpips)

            # Move tensors to CPU after computations if needed for logging
            if batch_idx == 0:  # Log images only for the first batch
                img_clean_cpu = img_clean.detach().cpu()
                img_corrupt_cpu = img_corrupt.detach().cpu()
                img_recon = xs.detach().cpu()
                log_image("image/clean", img_clean_cpu)
                log_image("image/corrupt", img_corrupt_cpu)
                log_image("image/recon", img_recon)
                # Note: Logging of pred_x0s and xs is omitted for brevity

        # Calculate and log mean scores
        mean_corrupt_accu = sum(corrupt_accu_scores) / len(corrupt_accu_scores)
        mean_recon_accu = sum(recon_accu_scores) / len(recon_accu_scores)
        mean_corrupt_lpips = sum(corrupt_lpips_scores) / len(corrupt_lpips_scores)
        mean_recon_lpips = sum(recon_lpips_scores) / len(recon_lpips_scores)

        self.writer.add_scalar(it, "accuracy/corrupt", mean_corrupt_accu)
        self.writer.add_scalar(it, "accuracy/recon", mean_recon_accu)
        self.writer.add_scalar(it, "LPIPS/corrupt", mean_corrupt_lpips)
        self.writer.add_scalar(it, "LPIPS/recon", mean_recon_lpips)

        torch.cuda.empty_cache()

