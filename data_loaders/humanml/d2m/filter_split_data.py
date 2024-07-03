import glob
import os
import pickle
import shutil
from pathlib import Path
import numpy as np
import torch
import sys
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append(os.getcwd()) 

def fileToList(f):
    out = open(f, "r").readlines()
    out = [x.strip() for x in out]
    out = [x for x in out if len(x)]
    return out

def get_train_test_list(dataset_path):
    filter_list = set(fileToList(f"{dataset_path}/splits/ignore_list.txt"))
    train_list = set(fileToList(f"{dataset_path}/splits/crossmodal_train.txt"))
    test_list = set(fileToList(f"{dataset_path}/splits/crossmodal_test.txt"))
    val_list = set(fileToList(f"{dataset_path}/splits/crossmodal_val.txt"))
    test_list.update(val_list)
    
    return filter_list, train_list, test_list

def split_data(dataset_path):
    filter_list, train_list, test_list = get_train_test_list(dataset_path)
    # train - test split
    for split_list, split_name in zip([train_list, test_list], ["train", "test"]):
        Path(f"{dataset_path}/{split_name}/motions").mkdir(parents=True, exist_ok=True)
        Path(f"{dataset_path}/{split_name}/wavs").mkdir(parents=True, exist_ok=True)
        for sequence in split_list:
            if sequence in filter_list:
                continue
            motion = f"{dataset_path}/motions/{sequence}.pkl"
            wav = f"{dataset_path}/wavs/{sequence}.wav"
            assert os.path.isfile(motion)
            assert os.path.isfile(wav)
            motion_data = pickle.load(open(motion, "rb"))
            trans = motion_data["smpl_trans"]
            pose = motion_data["smpl_poses"]
            scale = motion_data["smpl_scaling"]
            out_data = {"pos": trans, "q": pose, "scale": scale}
            pickle.dump(out_data, open(f"{dataset_path}/{split_name}/motions/{sequence}.pkl", "wb"))
            shutil.copyfile(wav, f"{dataset_path}/{split_name}/wavs/{sequence}.wav")