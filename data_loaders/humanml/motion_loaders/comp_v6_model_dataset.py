import torch
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm
from utils import dist_util
from teach.data.tools import lengths_to_mask 
import numpy as np
import random

import pickle
import os

from data_loaders.d2m.quaternion import ax_from_6v, quat_slerp
from pytorch3d.transforms import (axis_angle_to_quaternion, quaternion_to_axis_angle)

from vis import SMPLSkeleton

class CompCCDGeneratedDataset(Dataset):

    def __init__(self, args, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, num_samples_limit, scale=1.):
        self.dataloader = dataloader
        # dataloader = self.dataloader
        self.dataset = self.dataloader.dataset
        assert mm_num_samples < len(self.dataloader.dataset)
        clip_denoised = False  # FIXME - hardcoded=

        real_num_batches = len(self.dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // self.dataloader.batch_size + 1

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // self.dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        
        self.device = "cuda:0" if args.cuda else "cpu"

        model.eval()
        
        nfeats = 151
        njoints = 24
        self.smpl = SMPLSkeleton(device=self.device)

        with torch.no_grad():
            # args.inpainting_frames = 0
            for i, batch in tqdm(enumerate(dataloader)):
                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                bs = len(batch['length_0'])

                model_kwargs_0 = {}
                model_kwargs_0['y'] = {}
                model_kwargs_0['y']['lengths'] = batch['length_0']
                model_kwargs_0['y']['music'] = batch['music_0'].to(self.device)
                model_kwargs_0['y']['mask'] = lengths_to_mask(model_kwargs_0['y']['lengths'], 
                                    dist_util.dev()).unsqueeze(1).unsqueeze(2)

                model_kwargs_1 = {}
                model_kwargs_1['y'] = {}

                model_kwargs_1['y']['lengths'] = batch['length_1']
                model_kwargs_1['y']['music'] = batch['music_1'].to(self.device)
                model_kwargs_1['y']['mask'] = lengths_to_mask(model_kwargs_1['y']['lengths'], 
                                    dist_util.dev()).unsqueeze(1).unsqueeze(2)
                # add CFG scale to batch
                if scale != 1.:
                    model_kwargs_0['y']['scale'] = torch.ones(len(model_kwargs_0['y']['lengths']),
                                                            device=self.device) * scale
                    model_kwargs_1['y']['scale'] = torch.ones(len(model_kwargs_1['y']['lengths']),
                                                            device=self.device) * scale

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = False
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                
                for t in range(repeat_times):
                    sample_0 = diffusion.p_sample_loop (
                        model,
                        (bs, nfeats, 1, model_kwargs_0['y']['mask'].shape[-1]),
                        noise=None,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs_0,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=None,
                        progress=False,
                        dump_steps=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )
                    
                    sample_1 = diffusion.p_sample_loop (
                        model,
                        (bs, nfeats, 1, model_kwargs_1['y']['mask'].shape[-1]),
                        noise=None,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs_1,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=None,
                        progress=False,
                        dump_steps=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )
                    
                    num_rows = 1
                    motion = torch.cat(( sample_0[:, :, :, -45 :], sample_1[:, :, :, : 45]), -1)
                    assert motion.shape == (bs, nfeats, 1, 90)
                    input_motions = motion.repeat((num_rows, 1, 1, 1))
                    
                    max_frames = input_motions.shape[-1]
                    assert max_frames == input_motions.shape[-1]
                    gt_frames_per_sample = {}
                    
                    model_kwargs_2 = {}
                    model_kwargs_2['y'] = {}

                    model_kwargs_2['y']['lengths'] = [90 for len in batch['length_0']]
                    model_kwargs_2['y']['music'] = torch.cat((model_kwargs_0['y']['music'][:, -45 * 4800:], model_kwargs_1['y']['music'][:, :45 * 4800]), dim=1).to(self.device)
                    model_kwargs_2['y']['mask'] = lengths_to_mask(model_kwargs_2['y']['lengths'], 
                                        dist_util.dev()).unsqueeze(1).unsqueeze(2)
                    # add CFG scale to batch
                    if scale != 1.:
                        model_kwargs_2['y']['scale'] = torch.ones(len(model_kwargs_2['y']['lengths']),
                                                                device=self.device) * scale
                    
                    if args.inpainting_frames > 0:
                        total_hist_frame = 45
                        condition_frame = 45 - args.inpainting_frames
                        hist_lst = [feats[:,:,:len] for feats, len in zip(sample_0, batch['length_0'])]
                        hframes = torch.stack([x[:,:,-total_hist_frame : -condition_frame] for x in hist_lst])
                        
                        fut_lst = [feats[:,:,:len] for feats, len in zip(sample_1, batch['length_1'])]
                        fut_frames = torch.stack([x[:,:,condition_frame:total_hist_frame] for x in fut_lst])

                        model_kwargs_2['y']['hframes'] = hframes
                        model_kwargs_2['y']['fut_frames'] = fut_frames
                        
                    model_kwargs_2['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.float,
                                                                device=input_motions.device)  # True means use gt motion
                    for i, length in enumerate(model_kwargs_2['y']['lengths']):
                        start_idx, end_idx = args.inpainting_frames, 90 - args.inpainting_frames
                        gt_frames_per_sample[i] = list(range(0, start_idx)) + list(range(end_idx, max_frames))
                        model_kwargs_2['y']['inpainting_mask'][i, :, :, start_idx: end_idx] = False  # do inpainting in those frames
                        mask_slope = args.inpainting_frames // 2
                        for f in range(mask_slope):
                            if start_idx-f < 0:
                                continue
                            model_kwargs_2['y']['inpainting_mask'][i, :, :, start_idx-f] = f/mask_slope
                            if end_idx+f >= length:
                                continue
                            model_kwargs_2['y']['inpainting_mask'][i, :, :, end_idx+f] = f/mask_slope
                    
                    sample_2 = diffusion.p_sample_loop (
                        model,
                        (bs, nfeats, 1, model_kwargs_2['y']['mask'].shape[-1]),
                        noise=None,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs_2,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=None,
                        progress=False,
                        dump_steps=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )
                    
                    assert sample_0.shape == sample_1.shape == sample_2.shape == (bs, nfeats, 1, 90)
                    
                    sample = []
                    
                    for idx in range(bs):
                        motion_0_result = sample_0[idx].squeeze().unsqueeze(dim=0).permute(0, 2, 1)
                        motion_1_result = sample_1[idx].squeeze().unsqueeze(dim=0).permute(0, 2, 1)
                        motion_2_result = sample_2[idx].squeeze().unsqueeze(dim=0).permute(0, 2, 1)
                        
                        assert motion_0_result.shape == motion_1_result.shape == motion_2_result.shape == (1, 90, nfeats)
                        
                        motion_result = torch.cat((motion_0_result[:, :45, :], motion_2_result, motion_1_result[:, -45:, :]), dim=1)
                        
                        assert motion_result.shape == (1, 180, nfeats)
                        
                        if motion_result.shape[2] == nfeats:
                            sample_contact, motion_result = torch.split(
                                motion_result, (4, motion_result.shape[2] - 4), dim=2
                            )
                        else:
                            sample_contact = None
                        # do the FK all at once
                        b, s, c = motion_result.shape
                        pos = motion_result[:, :, :3].to(self.device)  # np.zeros((sample.shape[0], 3))
                        q = motion_result[:, :, 3:].reshape(b, s, njoints, 6)
                        # go 6d to ax
                        q = ax_from_6v(q).to(self.device)

                        b, s, c1, c2 = q.shape
                        if b > 1:
                            # if long mode, stitch position using linear interp
                            assert s % 2 == 0
                            half = s // 2
                            assert half == 45

                            fade_out = torch.ones((1, s, 1)).to(pos.device)
                            fade_in = torch.ones((1, s, 1)).to(pos.device)
                            fade_out[:, half:, :] = torch.linspace(1, 0, half)[None, :, None].to(
                                pos.device
                            )
                            fade_in[:, :half, :] = torch.linspace(0, 1, half)[None, :, None].to(
                                pos.device
                            )

                            pos[:-1] *= fade_out
                            pos[1:] *= fade_in

                            full_pos = torch.zeros((s + half * (b - 1), 3)).to(pos.device)
                            id_ = 0
                            for pos_slice in pos:
                                full_pos[id_ : id_ + s] += pos_slice
                                id_ += half

                            # stitch joint angles with slerp
                            slerp_weight = torch.linspace(0, 1, half)[None, :, None].to(pos.device)

                            left, right = q[:-1, half:], q[1:, :half]
                            # convert to quat
                            left, right = (
                                axis_angle_to_quaternion(left),
                                axis_angle_to_quaternion(right),
                            )
                            merged = quat_slerp(left, right, slerp_weight)  # (b-1) x half x ...
                            # convert back
                            merged = quaternion_to_axis_angle(merged)

                            full_q = torch.zeros((s + half * (b - 1), c1, c2)).to(pos.device)
                            full_q[:half] += q[0, :half]
                            id_ = half
                            for q_slice in merged:
                                full_q[id_ : id_ + half] += q_slice
                                id_ += half
                            full_q[id_ : id_ + half] += q[-1, half:]
                            
                            full_pos = full_pos.unsqueeze(0)
                            full_q = full_q.unsqueeze(0)
                            
                            assert full_pos.shape == (1, 180, 3)
                            assert full_q.shape == (1, 180, njoints, 3)
                            
                            full_pose = (
                                self.smpl.forward(full_q, full_pos).squeeze(0).detach().cpu().numpy()
                            )  # b, s, 24, 3
                            
                            if njoints == 24:
                                assert full_pose.shape == (180, njoints, 3)
                            else:
                                assert full_pose.shape == (180, 55, 3)
                            
                            filename = batch['filename'][idx]
                            outname = f'{args.inference_dir}/inference/{"".join(os.path.splitext(os.path.basename(filename))[0])}.pkl'
                            out_path = os.path.join("./", outname)
                            # Create the directory if it doesn't exist
                            os.makedirs(os.path.dirname(out_path), exist_ok=True)
                            with open(out_path, "wb") as file_pickle:
                                pickle.dump(
                                    {
                                        "smpl_poses": full_q.squeeze(0).reshape((-1, njoints * 3)).cpu().numpy(),
                                        "smpl_trans": full_pos.squeeze(0).cpu().numpy(),
                                        "full_pose": full_pose.squeeze(),
                                    },
                                    file_pickle,
                                )
                                
                            sample.append(full_pose)
                        else:
                            full_pos = pos.squeeze().unsqueeze(0)
                            full_q = q.squeeze().unsqueeze(0)
                            
                            assert full_pos.shape == (1, 180, 3)
                            assert full_q.shape == (1, 180, njoints, 3)
                            
                            full_pose = (
                                self.smpl.forward(full_q, full_pos).squeeze(0).detach().cpu().numpy()
                            )  # b, s, 24, 3
                            
                            if njoints == 24:
                                assert full_pose.shape == (180, njoints, 3)
                            else:
                                assert full_pose.shape == (180, 55, 3)
                            
                            filename = batch['filename'][idx]
                            outname = f'{args.inference_dir}/inference/{"".join(os.path.splitext(os.path.basename(filename))[0])}.pkl'
                            out_path = os.path.join("./", outname)
                            # Create the directory if it doesn't exist
                            os.makedirs(os.path.dirname(out_path), exist_ok=True)
                            with open(out_path, "wb") as file_pickle:
                                pickle.dump(
                                    {
                                        "smpl_poses": full_q.squeeze(0).reshape((-1, njoints * 3)).cpu().numpy(),
                                        "smpl_trans": full_pos.squeeze(0).cpu().numpy(),
                                        "full_pose": full_pose.squeeze(),
                                    },
                                    file_pickle,
                                )
                                
                            sample.append(full_pose)
                        
                    
                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i],
                                    'length': 180,
                                    'music': torch.cat((model_kwargs_0['y']['music'][bs_i], model_kwargs_1['y']['music'][bs_i]), axis=0),
                                    'filename': batch["filename"][bs_i],
                                    } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts
                    
        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions


    def __len__(self):
        return len(self.generated_motion)

    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, length, music, filename = data['motion'], data['length'], data['music'], data['filename']
        
        return motion, length, music, filename