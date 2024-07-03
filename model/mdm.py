import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
from teach.data.tools import lengths_to_mask
from vis import SMPLSkeleton
from einops import rearrange, reduce, repeat

def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift

class MDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', #clip_dim=512,
                 **kargs):
        super().__init__()

        #self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.action_emb = kargs.get('action_emb', None)
        
        self.music_dim = 4800 * 90 #baseline feats
        # pos_dim = 3
        # rot_dim = self.njoints * self.nfeats  # 24 joints, 6dof
        # self.input_feats = pos_dim + rot_dim + 4
    
        # output_feats = self.input_feats

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_drop_prob = kargs.get('cond_drop_prob', 0.)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = self.cond_drop_prob
        
        self.max_seq_att = kargs.get('max_seq_att', 1024)
        self.input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim)
        self.process_cond_input = [nn.Linear(2*self.latent_dim, self.latent_dim) for _ in range(self.num_layers)]
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        self.motion_mask = kargs['motion_mask']
        self.inpainting_frames = kargs['inpainting_frames']
        
        if self.inpainting_frames > 0:
            # self.hist_frames = 5
            self.seperation_token = nn.Parameter(torch.randn(latent_dim))
            self.skel_embedding = nn.Linear(self.njoints, self.latent_dim)
        
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                    num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'music' in self.cond_mode:
                self.embed_music = nn.Linear(self.music_dim, self.latent_dim)

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            masked_cond = cond * (1. - mask)
            return masked_cond
        else:
            return cond

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        
        time_emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        
        music_emb = self.embed_music(y['music'])
        emb = time_emb + self.mask_cond(music_emb, force_mask=force_mask)

        x = self.input_process(x)
        
        mask = lengths_to_mask(y['lengths'], x.device)
        if y.get('hframes', None) == None:
            token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.motion_mask:
                output = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
            else:
                output = self.seqTransEncoder(xseq)[1:]
        else:
            token_mask = torch.ones((bs, 2 + 2*self.inpainting_frames), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
            sep_token = torch.tile(self.seperation_token, (bs,)).reshape(bs, -1).unsqueeze(0)
            hframes = y['hframes'].squeeze(2).permute(2, 0, 1) #TODO find out the diff 
            hframes_emb = self.skel_embedding(hframes)
            fut_frames = y['fut_frames'].squeeze(2).permute(2, 0, 1)
            fut_frames_emb = self.skel_embedding(fut_frames)
            # hframes_emb = hframes_emb.permute(1, 0, 2) # [5 b dim]
            xseq = torch.cat((emb, hframes_emb, fut_frames_emb, sep_token, x), axis=0)
            # TODO add attention mask
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)[2 + 2*self.inpainting_frames:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        self.trajectEmbedding_1 = nn.Linear(3, self.latent_dim)
        self.trajectEmbedding_2 = nn.Linear(3, self.latent_dim)
        

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        # print("INPUT PROCESS: ", bs, njoints, nfeats, nframes)
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        #! Global Trajectory
        x_root = x[:, :, 4:7]
        
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        
        x_lambda = self.trajectEmbedding_1(x_root)
        x_beta = self.trajectEmbedding_2(x_root)
        
        x = featurewise_affine(x, (x_lambda, x_beta))
        
        return x


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output