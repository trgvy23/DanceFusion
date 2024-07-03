from model.mdm import MDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
import torch.nn.functional as F


def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model_and_diffusion(args, data):
    model = MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args, data):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = "music"

    # SMPL defaults
    data_rep = 'rot6d'

    njoints = 151
    nfeats = 1
    
    feature_dim = 4800
    cond_drop_prob = args.cond_drop_prob
    
    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats,
            'translation': True, 'pose_rep': data_rep, 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': F.gelu, 'data_rep': data_rep, 'cond_mode': cond_mode,
            'action_emb': action_emb,
            'inpainting_frames': args.inpainting_frames, 'motion_mask': args.motion_mask, 'music_dim': feature_dim,
            'cond_drop_prob': cond_drop_prob,
            #! here
            # # FlowMDM
            # 'diffusion_steps': args.diffusion_steps,
            # 'max_seq_att': args.max_seq_att, 
            # 'bpe_denoising_step': args.bpe_denoising_step,
            # 'bpe_training_ratio': args.bpe_training_ratio,
            # 'rpe_horizon': args.rpe_horizon,
            # 'use_chunked_att': args.use_chunked_att,
            }

def create_gaussian_diffusion(args):
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 1000
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_mse=args.lambda_mse,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
    )