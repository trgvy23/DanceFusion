import glob
import os, sys
import re
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
sys.path.append(os.getcwd())

from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                  quaternion_multiply,
                                  quaternion_to_axis_angle)
from data_loaders.d2m.quaternion import ax_from_6v
from torch.utils.data import Dataset

from data_loaders.d2m.preprocess import Normalizer, vectorize_many
from data_loaders.d2m.quaternion import ax_to_6v
from vis import SMPLSkeleton
import pickle

modir = 'data_loaders\d2m\\aistpp_dataset\motions'
    
def process_dataset(root_pos, local_q):
    # FK skeleton
    
    # to Tensor
    root_pos = torch.Tensor(root_pos).unsqueeze(0)
    local_q = torch.Tensor(local_q).unsqueeze(0)
    # to ax
    bs, sq, c = local_q.shape
    local_q = local_q.reshape((bs, sq, -1, 3))

    # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
    root_q = local_q[:, :, :1, :]  # sequence x 1 x 3
    root_q_quat = axis_angle_to_quaternion(root_q)
    rotation = torch.Tensor(
        [0.7071068, 0.7071068, 0, 0]
    )  # 90 degrees about the x axis
    root_q_quat = quaternion_multiply(rotation, root_q_quat)
    root_q = quaternion_to_axis_angle(root_q_quat)
    local_q[:, :, :1, :] = root_q

    # don't forget to rotate the root position too 😩
    pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
    root_pos = pos_rotation.transform_points(
        root_pos
    )  # basically (y, z) -> (-z, y), expressed as a rotation for readability

    # do FK
    # positions = smpl.forward(local_q, root_pos)  # batch x sequence x 24 x 3
    # feet = positions[:, :, (7, 8, 10, 11)]
    # feetv = torch.zeros(feet.shape[:3])
    # feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
    # contacts = (feetv < 0.01).to(local_q)  # cast to right dtype

    # to 6d
    local_q = ax_to_6v(local_q)

    # now, flatten everything into: batch x sequence x [...]
    l = [root_pos, local_q]
    global_pose_vec_input = vectorize_many(l).float().detach()

    assert not torch.isnan(global_pose_vec_input).any()

    print(f"Dataset Motion Features Dim: {global_pose_vec_input.shape}")

    return global_pose_vec_input[0]

def load_data(data_path):
    # open data path
    motion_path = os.path.join(data_path)
    # sort motions and sounds
    motions = sorted(glob.glob(os.path.join(motion_path, "*.pkl")))
    
    all_pos = []
    all_q = []
    all_names = []
    
    for motion in motions:
        data = pickle.load(open(motion, "rb"))
        pos = data["smpl_trans"]
        q = data["smpl_poses"]
        scale = data["smpl_scaling"][0]
        pos /= scale
        
        all_pos.append(pos)
        all_q.append(q)
        all_names.append(motion)
    
    # all_pos = np.array(all_pos)  # N x seq x 3
    # all_q = np.array(all_q)
    
    # all_pos = all_pos[:, :: 2, :]
    # all_q = all_q[:, :: 2, :]

    data = {"pos": all_pos, "q": all_q, "filenames": all_names}
    return data

def create_noramlizer():
    data_li = []

    data = load_data(modir)

    # print(len(data["pos"]))
    for pos, q in zip(data["pos"], data["q"]):
        all_pos = np.array(pos)  # N x seq x 3
        all_q = np.array(q)
        
        all_pos = all_pos[:: 2, :]
        all_q = all_q[:: 2, :]
        
        dataset = process_dataset(all_pos, all_q)
        data_li.append(dataset.unsqueeze(0))

    data_li = torch.cat(data_li, dim=1)
    data_li_ori = data_li.clone()
    Normalizer_ = Normalizer(data_li)
    torch.save(Normalizer_, 'data_loaders/d2m/aistpp_dataset/AIST_Normalizer.pth')


    reNorm = torch.load('data_loaders/d2m/aistpp_dataset/AIST_Normalizer.pth')
    data_newnormed = reNorm.normalize(data_li)
    data_newunnormed = reNorm.unnormalize(data_newnormed)
    print(data_newnormed[0,:20])
    print(data_newunnormed[0,:20])
    print(data_li_ori[0,:20])
    
def unnomarlize(pred_dir):
    out_dir = os.path.join(pred_dir)
    reNorm = torch.load('data_loaders/d2m/aistpp_dataset/AIST_Normalizer.pth')
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    smpl = SMPLSkeleton(device=device)
    
    motions = sorted(glob.glob(os.path.join(pred_dir, "*.pkl")))
    for motion in motions:
        with open(motion, 'rb') as f:
            data = pickle.load(f)

        # poses = torch.Tensor(data['full_pose']).unsqueeze(dim=0)
        q = torch.Tensor(data['smpl_poses']).reshape((-1, 24, 3)).unsqueeze(dim=0)
        pos = torch.Tensor(data['smpl_trans']).unsqueeze(dim=0)
        
        b, s, njoints, _ = q.shape
        
        q = ax_to_6v(q)

        # now, flatten everything into: batch x sequence x [...]
        l = [pos, q]
        global_pose_vec_input = vectorize_many(l).float().detach()

        assert not torch.isnan(global_pose_vec_input).any()
        
        data_newunnormed = reNorm.normalize(global_pose_vec_input)
        
        pos = data_newunnormed[:, :, :3].to(device)  # np.zeros((sample.shape[0], 3))
        q = data_newunnormed[:, :, 3:].reshape(b, s, njoints, 6)
        # go 6d to ax
        q = ax_from_6v(q).to(device)
        
        full_pose = (
            smpl.forward(q, pos).squeeze(0).detach().cpu().numpy()
        )  # b, s, 24, 3
        
        outname = f'{out_dir}/{"".join(os.path.splitext(os.path.basename(motion))[0])}.pkl'
        
        out_path = os.path.join(outname)
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        with open(outname, "wb") as file_pickle:
            pickle.dump(
                {
                    "smpl_poses": q.squeeze(0).reshape((-1, njoints * 3)).cpu().numpy(),
                    "smpl_trans": pos.squeeze(0).cpu().numpy(),
                    "full_pose": full_pose
                },
                file_pickle,
            )
        
if __name__ == '__main__':
    
    pred_dir = "evaluate_result\dhms-guidance\\val\ckpt16-guidance-2.5\inference"
    out_dir = "evaluate_result\dhms-guidance\\val\ckpt16-guidance-2.5\inference_normed"
    
    unnomarlize(pred_dir)