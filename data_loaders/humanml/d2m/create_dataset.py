import argparse
import os
from pathlib import Path

from audio_extraction.baseline_features import \
    extract_folder as baseline_extract
from audio_extraction.jukebox_features import extract_folder as jukebox_extract
from filter_split_data import *
from slice import *


def create_dataset(opt):
    path_folder = opt.datapath
    # split the data according to the splits files
    print("Creating train / test split")
    split_data(path_folder)
    motion_len = opt.motion_len
    
    # process dataset to extract audio features
    if opt.extract_baseline:
        print("Extracting baseline features")
        baseline_extract(f"{path_folder}/train/wavs", f"{path_folder}/train/music_npy")
        baseline_extract(f"{path_folder}/test/wavs", f"{path_folder}/test/music_npy")
    if opt.extract_jukebox:
        print("Extracting jukebox features")
        jukebox_extract(f"{path_folder}/train/wavs_sliced", f"{path_folder}/train/jukebox_feats")
        jukebox_extract(f"{path_folder}/test/wavs_sliced", f"{path_folder}/test/jukebox_feats")
        
    # slice motions/music into sliding windows to create training dataset
    print("Slicing train data")
    slice_aistpp(f"{path_folder}/train/motions", f"{path_folder}/train/music_npy", f"{path_folder}/train/wavs", 0.5, motion_len)
    print("Slicing test data")
    slice_aistpp(f"{path_folder}/test/motions", f"{path_folder}/test/music_npy", f"{path_folder}/test/wavs", 0.5, motion_len)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        type=str,
        default="./aistpp_dataset/",
        help="path to folder containing motions and music",
    )
    parser.add_argument(
        "--motion_len",
        type=float,
        default=6.0,
        help="each slice's length in second",
    )
    parser.add_argument("--extract-baseline", action="store_true")
    parser.add_argument("--extract-jukebox", action="store_true")
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    create_dataset(opt)