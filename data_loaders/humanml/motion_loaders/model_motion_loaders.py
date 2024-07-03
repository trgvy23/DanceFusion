from torch.utils.data import DataLoader
from data_loaders.humanml.motion_loaders.comp_v6_model_dataset import CompCCDGeneratedDataset
import numpy as np
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

# our loader
def get_mdm_loader(args, model, diffusion, batch_size, ground_truth_loader, mm_num_samples, mm_num_repeats, num_samples_limit, scale):
    opt = {
        'name': 'test',  # FIXME
    }
    dataset = CompCCDGeneratedDataset(args, model, diffusion, ground_truth_loader, mm_num_samples, mm_num_repeats, num_samples_limit, scale)

    motion_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True, num_workers=4)

    return motion_loader