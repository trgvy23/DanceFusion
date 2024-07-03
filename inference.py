from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from datetime import datetime
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader

from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from model.cfg_sampler import ClassifierFreeSampleModel

from vis import SMPLSkeleton
from data_loaders.d2m.quaternion import ax_from_6v

from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
from scipy import linalg

import numpy as np
import torch
import os

import pickle
from pathlib import Path

from evaluation.features.kinetic import extract_kinetic_features
from evaluation.features.manual_new import extract_manual_features

import torch

torch.multiprocessing.set_sharing_strategy('file_system')

from evaluation.metrics_new import quantized_metrics, calc_and_save_feats

def inference(args, eval_motion_loaders, origin_loader, out_dir, log_file, replication_times, diversity_times, mm_num_times, run_mm=False):
    device = "cuda:0" if args.cuda else "cpu"
    njoints = 24
    smpl = SMPLSkeleton(device=device)
        
    for batch in origin_loader:
        motion_0, motion_1_with_transition, filenames = batch["motion_feats_0"], batch["motion_feats_1"], batch["filename"]
        motion = torch.from_numpy(np.concatenate((motion_0, motion_1_with_transition), axis=1)).to(device)
        
        b, s, c = motion.shape
        
        sample_contact, motion = torch.split(
        motion, (4, motion.shape[2] - 4), dim=2)
        pos = motion[:, :, :3].to(motion.device)  # np.zeros((sample.shape[0], 3))
        q = motion[:, :, 3:].reshape(b, s, njoints, 6)
        # go 6d to ax
        q = ax_from_6v(q).to(motion.device)
        
        for q_, pos_, filename in zip(q, pos, filenames):
            if out_dir is not None:
                full_pose = (smpl.forward(q_.unsqueeze(0), pos_.unsqueeze(0)).squeeze(0).detach().cpu().numpy())
                outname = f'{args.inference_dir}/gt/{"".join(os.path.splitext(os.path.basename(filename))[0])}.pkl'
                out_path = os.path.join(out_dir, outname)
                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "wb") as file_pickle:
                    pickle.dump(
                        {
                            "smpl_poses": q_.squeeze(0).reshape((-1, njoints * 3)).cpu().numpy(),
                            "smpl_trans": pos_.squeeze(0).cpu().numpy(),
                            "full_pose": full_pose,
                        },
                        file_pickle,
                    )
    
    with open(log_file, 'a') as f:
        for replication in range(replication_times):
            motion_loaders = {}
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader

            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}', file=f, flush=True)
            
            gt_root = f'{args.inference_dir}/gt'
            pred_root = [f'{args.inference_dir}/inference']
            
            calc_and_save_feats(gt_root)
            
            for pred_root in pred_root:
                print(pred_root, file=f, flush=True)
                calc_and_save_feats(pred_root)
                print(quantized_metrics(pred_root, gt_root), file=f, flush=True)

        print(f'!!! DONE !!!')
        print(f'!!! DONE !!!', file=f, flush=True)

def evaluation(args, log_file, num_samples_limit, run_mm, mm_num_samples, mm_num_repeats, mm_num_times, diversity_times, replication_times):
    print("[eval_batch_size]: ", args.eval_batch_size)

    dist_util.setup_dist(args.device)
    logger.configure()

    logger.log("creating data loader...")
    origin_loader = get_dataset_loader(args, batch_size=args.eval_batch_size, train=False)

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, origin_loader)

    logger.log(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to("cuda:0" if args.cuda else "cpu")
    model.eval()  # disable random masking

    eval_motion_loaders = {
        'vald': lambda: get_mdm_loader(
            args, model, diffusion, args.eval_batch_size,
            origin_loader, mm_num_samples, mm_num_repeats, num_samples_limit, args.guidance_param
        )
    }

    inference(args, eval_motion_loaders, origin_loader, args.out_dir, log_file, replication_times, diversity_times, mm_num_times, run_mm=run_mm)

if __name__ == '__main__':
    args = evaluation_parser()
    fixseed(args.seed)
    
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    log_file = os.path.join(os.path.dirname(args.model_path), 'inference_{}_{}'.format(name, niter))
    
    if args.guidance_param != 1.:
        log_file += f'_gscale{args.guidance_param}'
    log_file += f'_inpaint{args.inpainting_frames}'
    log_file += '.log'
    
    num_samples_limit = None  # None means no limit (eval over all dataset)
    run_mm = False
    mm_num_samples = 0
    mm_num_repeats = 0
    mm_num_times = 0
    diversity_times = 300
    replication_times = 1  # about 3 Hrs
    
    evaluation(args, log_file, num_samples_limit, run_mm, mm_num_samples, mm_num_repeats, mm_num_times, diversity_times, replication_times)
    