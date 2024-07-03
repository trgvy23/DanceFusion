import glob
import os
import pickle

import librosa as lr
import numpy as np
import soundfile as sf
from tqdm import tqdm
import math
import numpy as np

def slice_audio(audio_file, stride, length, out_dir, num_slices):
    # stride, length in seconds
    FPS = 30
    audio = np.load(audio_file)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * FPS)
    stride_step = int(stride * FPS)
    while start_idx <= len(audio) - window and idx < num_slices:
        audio_slice = audio[start_idx : start_idx + window]
        np.save(f"{out_dir}/{file_name}_slice{idx}.npy", audio_slice)
        start_idx += stride_step
        idx += 1
    return idx

def slice_motion(motion_file, stride, length, out_dir, num_slices, segment_len):
    motion = pickle.load(open(motion_file, "rb"))
    pos, q = motion["pos"], motion["q"]
    scale = motion["scale"][0]

    file_name = os.path.splitext(os.path.basename(motion_file))[0]
    # normalize root position
    pos /= scale
    start_idx = 0
    window = int(length * 60)
    stride_step = int(stride * 60)
    segment_len_ = int(segment_len * 60)
    slice_count = 0
    # slice until done or until matching audio slices
    while start_idx <= len(pos) - window and slice_count < num_slices:
        
        pos_0, q_0 = (
            pos[start_idx : start_idx + segment_len_],
            q[start_idx : start_idx + segment_len_],
        ) # lenght_0
        pos_1, q_1 = (
            pos[start_idx + segment_len_ : start_idx + window],
            q[start_idx + segment_len_ : start_idx + window],
        ) # lenght_0
                    
        out = {"pos_0": pos_0, 
                "q_0":q_0,
                "pos_1": pos_1,
                "q_1": q_1,
                "length_0": int(segment_len_ / 2),
                "length_1": int(segment_len_ / 2),
                }
        
        # out = {"pos": pos_slice, "q": q_slice, "pos1": pos_slice_1, "q1": q_slice_1}
        pickle.dump(out, open(f"{out_dir}/{file_name}_slice{slice_count}.pkl", "wb"))
        start_idx += stride_step
        slice_count += 1
    return slice_count

def slice_aistpp(motion_dir, wav_feature_dir, wav_dir, stride=0.5, length=6):
    wavs_feature = sorted(glob.glob(f"{wav_feature_dir}/*.npy"))
    motions = sorted(glob.glob(f"{motion_dir}/*.pkl"))
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    
    wav_feature_out = wav_feature_dir + "_sliced"
    motion_out = motion_dir + "_sliced"
    wav_out = wav_dir + "_sliced"
    
    os.makedirs(wav_feature_out, exist_ok=True)
    os.makedirs(motion_out, exist_ok=True)
    os.makedirs(wav_out, exist_ok=True)
    
    assert len(wavs_feature) == len(motions) == len(wavs)
    
    for wav_feature, motion, wav in tqdm(zip(wavs_feature, motions, wavs)):
        # make sure name is matching
        m_name = os.path.splitext(os.path.basename(motion))[0]
        w_name = os.path.splitext(os.path.basename(wav_feature))[0]
        wav_name = os.path.splitext(os.path.basename(wav))[0]
        
        assert m_name == w_name == wav_name, str((motion, wav_feature, wav))
        
        wav_slices = slice_audio_wav(wav, stride, length, wav_out)
        motion_slices = slice_motion(motion, stride, length, motion_out, wav_slices, length // 2)
        audio_slices = slice_audio(wav_feature, stride, length, wav_feature_out, wav_slices)
        # make sure the slices line up
        assert audio_slices == motion_slices == wav_slices, str(
            (wav_feature, motion, wav, audio_slices, motion_slices, wav_slices)
        )

def slice_audio_wav(audio_file, stride, length, out_dir):
    # stride, length in seconds
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx : start_idx + window]
        sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
        start_idx += stride_step
        idx += 1
    return idx