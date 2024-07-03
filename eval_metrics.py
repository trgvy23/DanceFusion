from evaluation.metrics_new import *
from evaluation.beat_align import *
from evaluation.metrics_new import quantized_metrics, calc_and_save_feats
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        type=str,
        default="output\\result.log",
        help="path to result log file",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="output\\gt",
        help="path to ground truth dir",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default="output\\inference",
        help="path to prediction dir",
    )
    parser.add_argument(
        "--music_dir",
        type=str,
        default="custom_input",
        help="path to music dir",
    )
    opt = parser.parse_args()
    return opt
        
if __name__ == '__main__':
    
    opt = parse_opt()
    log_file = opt.log

    gt_root = opt.gt_dir
    calc_and_save_feats(gt_root)
    
    pred_root = opt.pred_dir
    music_dir = opt.music_dir
    
    with open(log_file, 'a') as f:
        print(pred_root, file=f, flush=True)
        calc_and_save_feats(pred_root)
        print("FID Metrics", file=f, flush=True)
        print(quantized_metrics(pred_root, gt_root), file=f, flush=True)
        print("Beat Accuracy", file=f, flush=True)
        print(calc_ba_score(pred_root, music_dir), file=f, flush=True)