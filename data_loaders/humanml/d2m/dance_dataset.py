import glob
import os
import pickle
import random
from functools import cmp_to_key
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                  quaternion_multiply,
                                  quaternion_to_axis_angle)
from torch.utils.data import Dataset

from data_loaders.d2m.preprocess import vectorize_many
from data_loaders.d2m.quaternion import ax_to_6v
from vis import SMPLSkeleton

floor_height = 0

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class AISTPPDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        train: bool,
        feature_type: str = "baseline",
        data_len: int = -1,
    ):
        self.data_path = data_path
        self.raw_fps = 60
        self.data_fps = 30
        assert self.data_fps <= self.raw_fps
        self.data_stride = self.raw_fps // self.data_fps

        self.train = train
        self.name = "Train" if self.train else "Test"
        self.feature_type = feature_type

        self.data_len = data_len
        data = self.load_aistpp()  # Call this last

        # process data, convert to 6dof etc
        pose_input_0 = self.process_dataset(data["pos_0"], data["q_0"])
        pose_input_1 = self.process_dataset(data["pos_1"], data["q_1"])
       
        self.data = {
            "pose_0": pose_input_0,
            "pose_1": pose_input_1,
            "length_0": data['length_0'],
            "length_1": data['length_1'],
            "filenames": data["filenames"],
        }
        assert len(pose_input_0) == len(data["filenames"])
        self.length = len(pose_input_0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        feature = torch.from_numpy(np.load(filename_))
        feature_0 = feature[: self.data['length_0']]
        feature_1 = feature[self.data['length_0'] :]
        seq_0, d_0 = feature_0.shape
        seq_1, d_1 = feature_1.shape
        return {
            "pose_0": self.data['pose_0'][idx],
            "pose_1": self.data['pose_1'][idx],
            "length_0": self.data['length_0'],
            "length_1": self.data['length_1'],
            "music_0": feature_0.reshape(seq_0 * d_0),
            "music_1": feature_1.reshape(seq_1 * d_1),
            "filename": filename_
        }

    def load_aistpp(self):
        # open data path
        split_data_path = os.path.join(
            self.data_path, "train" if self.train else "test"
        )

        # Structure:
        # data
        #   |- train
        #   |    |- motion_sliced
        #   |    |- wav_sliced
        #   |    |- baseline_features
        #   |    |- jukebox_features
        #   |    |- motions
        #   |    |- wavs

        motion_path = os.path.join(split_data_path, "motions_sliced")
        sound_path = os.path.join(split_data_path, f"jukebox_feats_sliced")
        # sort motions and sounds
        motions = sorted(glob.glob(os.path.join(motion_path, "*.pkl")))
        features = sorted(glob.glob(os.path.join(sound_path, "*.npy")))

        # stack the motions and features together
        all_pos = []
        all_q = []
        all_pos1 = []
        all_q1 = []
        all_names = []
        assert len(motions) == len(features)
        for motion, feature in zip(motions, features):
            # make sure name is matching
            m_name = os.path.splitext(os.path.basename(motion))[0]
            f_name = os.path.splitext(os.path.basename(feature))[0]
            assert m_name == f_name, str((motion, feature))
            # load motion
            
            data = pickle.load(open(motion, "rb"))
            pos_0 = data["pos_0"]
            q_0 = data["q_0"]
            pos1 = data["pos_1"]
            q1 = data["q_1"]
            all_pos.append(pos_0)
            all_q.append(q_0)
            all_pos1.append(pos1)
            all_q1.append(q1)
            all_names.append(feature)

        all_pos = np.array(all_pos)  # N x seq x 3
        all_q = np.array(all_q)  # N x seq x (joint * 3)
        all_pos1 = np.array(all_pos1)  # N x seq x 3
        all_q1 = np.array(all_q1)  # N x seq x (joint * 3)
        # downsample the motions to the data fps
        all_pos = all_pos[:, :: self.data_stride, :]
        all_q = all_q[:, :: self.data_stride, :]
        all_pos1 = all_pos1[:, :: self.data_stride, :]
        all_q1 = all_q1[:, :: self.data_stride, :]
        data = {"pos_0": all_pos, "q_0": all_q, 
                "pos_1": all_pos1, "q_1": all_q1,
                "length_0": data['length_0'],
                "length_1": data['length_1'],
                "filenames": all_names}
        return data

    def process_dataset(self, root_pos, local_q):
        # FK skeleton
        smpl = SMPLSkeleton()
        # to Tensor
        root_pos = torch.Tensor(root_pos)
        local_q = torch.Tensor(local_q)
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
        positions = smpl.forward(local_q, root_pos)  # batch x sequence x 24 x 3
        feet = positions[:, :, (7, 8, 10, 11)]
        feetv = torch.zeros(feet.shape[:3])
        feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
        contacts = (feetv < 0.01).to(local_q)  # cast to right dtype

        # to 6d
        local_q = ax_to_6v(local_q)

        # now, flatten everything into: batch x sequence x [...]
        l = [contacts, root_pos, local_q]
        global_pose_vec_input = vectorize_many(l).float().detach()

        assert not torch.isnan(global_pose_vec_input).any()
        data_name = "Train" if self.train else "Test"

        # cut the dataset
        if self.data_len > 0:
            global_pose_vec_input = global_pose_vec_input[: self.data_len]

        global_pose_vec_input = global_pose_vec_input

        # print(f"{data_name} Dataset Motion Features Dim: {global_pose_vec_input.shape}")

        return global_pose_vec_input