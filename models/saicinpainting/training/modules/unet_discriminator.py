# -*- coding: utf-8 -*-
"""UNetLatentDiscriminator — faithful port of InvSR's UNet2DConditionDiscriminator.

Source: InvSR (CVPR 2025, arXiv 2412.09013), file
src/diffusers/models/unets/unet_2d_condition_discriminator.py + config
configs/sd-turbo-sr-ldis.yaml (discriminator section), adapted to this
repo's diffusers 0.27.0.dev0 building blocks. Verified against the original:

  * projected-input conditioning: the RAW latent goes through `feature_in`
    (Conv 4->32ch) and is re-injected (bicubic-resized, channel-concat) into
    EVERY down block — InvSR forward L1223-1249; down-block in_channels =
    block_out_channels[i-1] + norm_num_groups — InvSR __init__ L341-344;
  * multi-scale heads GroupNorm/SiLU/Conv(c->c)/GroupNorm/SiLU/Conv(c->1)
    on the mid output AND on every up-block output — InvSR __init__
    L398-415, L477-493 (4 logit maps for a 64x64 input);
  * timestep conditioning via standard sinusoidal Timesteps->TimestepEmbedding
    (positional, flip_sin_to_cos=True, freq_shift=0, dim 128*4=512) — the
    anti-shortcut design: D always knows the noise level, cannot exploit it;
  * G loss = -mean over heads of mean(logits) (trainer L950-959);
    D loss = hinge per head, averaged (trainer L1612+).

Documented deviations (information-free in OUR setting):
  1. addition_embed_type='text' aug path OMITTED — their prompt varies per
     image, ours is the CONSTANT empty-prompt embed; mixing a constant into
     the time embedding adds no information. Cross-attn conditioning KEPT.
  2. GLIGEN/FreeU/PEFT/T2I-Adapter branches unused by their config — omitted.

Hyperparameters = InvSR config verbatim (block_out (128,256,512), layers
(1,2,2), transformer_layers 1, head dims (8,16,16) -> heads (16,16,32),
groups 32, eps 1e-5, silu, downsample_padding 1, dropout 0), with
cross_attention_dim 1024 -> 768 (SD1.5 empty-prompt embeds).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.models.unets.unet_2d_blocks import (
    get_down_block, get_mid_block, get_up_block)


class UNetLatentDiscriminator(nn.Module):
    """Multi-scale UNet discriminator over 4-ch latents, timestep-aware."""

    def __init__(self, in_channels=4, cross_attention_dim=768):
        super().__init__()
        # ---- InvSR sd-turbo-sr-ldis.yaml discriminator section (verbatim) ----
        block_out_channels = (128, 256, 512)
        layers_per_block = (1, 2, 2)
        down_block_types = ('DownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D')
        mid_block_type = 'UNetMidBlock2DCrossAttn'
        up_block_types = ('CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'UpBlock2D')
        attention_head_dim = (8, 16, 16)          # -> num heads = out_ch // head_dim
        transformer_layers_per_block = 1
        norm_num_groups = 32
        norm_eps = 1e-5
        act_fn = 'silu'
        downsample_padding = 1
        dropout = 0.0
        temb_dim = block_out_channels[0] * 4      # 512 (InvSR _set_time_proj)

        heads = tuple(c // d for c, d in zip(block_out_channels, attention_head_dim))

        # input (InvSR __init__ L246-252)
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)
        self.feature_in = nn.Conv2d(in_channels, norm_num_groups, 3, padding=1)
        # time (standard UNet path)
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos=True,
                                   downscale_freq_shift=0.0)
        self.time_embedding = TimestepEmbedding(block_out_channels[0], temb_dim)

        def _head(c):
            return nn.Sequential(                      # InvSR __init__ L398-415
                nn.GroupNorm(min(norm_num_groups, c // 4), c, eps=norm_eps),
                nn.SiLU(),
                nn.Conv2d(c, c, 3, padding=1),
                nn.GroupNorm(min(norm_num_groups, c // 4), c, eps=norm_eps),
                nn.SiLU(),
                nn.Conv2d(c, 1, 3, padding=1),
            )
        # down (InvSR __init__ L341-373: in_ch = prev_out + feature channels)
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, kind in enumerate(down_block_types):
            input_channel = output_channel + norm_num_groups
            output_channel = block_out_channels[i]
            is_final = i == len(block_out_channels) - 1
            self.down_blocks.append(get_down_block(
                kind, num_layers=layers_per_block[i],
                in_channels=input_channel, out_channels=output_channel,
                temb_channels=temb_dim, add_downsample=not is_final,
                resnet_eps=norm_eps, resnet_act_fn=act_fn,
                transformer_layers_per_block=transformer_layers_per_block,
                num_attention_heads=heads[i], resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim if 'CrossAttn' in kind else None,
                downsample_padding=downsample_padding, dropout=dropout))

        # mid
        self.mid_block = get_mid_block(
            mid_block_type, temb_channels=temb_dim,
            in_channels=block_out_channels[-1],
            resnet_eps=norm_eps, resnet_act_fn=act_fn, resnet_groups=norm_num_groups,
            output_scale_factor=1,
            transformer_layers_per_block=transformer_layers_per_block,
            num_attention_heads=heads[-1],
            cross_attention_dim=cross_attention_dim, dropout=dropout)

        # up + one head per up block (InvSR __init__ L420-493)
        self.up_blocks = nn.ModuleList([])
        self.out_blocks = nn.ModuleList([_head(block_out_channels[-1])])
        rev_out = list(reversed(block_out_channels))
        rev_layers = list(reversed(layers_per_block))
        rev_heads = list(reversed(heads))
        output_channel = rev_out[0]
        for i, kind in enumerate(up_block_types):
            is_final = i == len(block_out_channels) - 1
            prev_output_channel = output_channel
            output_channel = rev_out[i]
            input_channel = rev_out[min(i + 1, len(block_out_channels) - 1)]
            self.up_blocks.append(get_up_block(
                kind, num_layers=rev_layers[i] + 1,
                in_channels=input_channel, out_channels=output_channel,
                prev_output_channel=prev_output_channel, temb_channels=temb_dim,
                add_upsample=not is_final,
                resnet_eps=norm_eps, resnet_act_fn=act_fn,
                resolution_idx=i,
                transformer_layers_per_block=transformer_layers_per_block,
                num_attention_heads=rev_heads[i], resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim if 'CrossAttn' in kind else None,
                dropout=dropout))
            self.out_blocks.append(_head(output_channel))
    def forward(self, sample, timestep, encoder_hidden_states, return_features=False):
        """InvSR forward (L1150-1325), adapter/controlnet branches dropped.

        Returns the list of multi-scale logit maps; optionally also the
        feature maps entering each head (for feature-matching, our stand-in
        for InvSR's finetuned latent-LPIPS which is not available offline).
        """
        t_emb = self.time_proj(timestep.to(sample.device))
        emb = self.time_embedding(t_emb)

        inputs_feature = self.feature_in(sample)          # projected input
        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        feats = []
        for downsample_block in self.down_blocks:
            inputs_down = F.interpolate(inputs_feature, size=sample.shape[-2:],
                                        mode='bicubic')   # InvSR L1225
            if getattr(downsample_block, 'has_cross_attention', False):
                sample, res = downsample_block(
                    hidden_states=torch.cat([sample, inputs_down], dim=1),
                    temb=emb, encoder_hidden_states=encoder_hidden_states)
            else:
                sample, res = downsample_block(
                    hidden_states=torch.cat([sample, inputs_down], dim=1), temb=emb)
            down_block_res_samples += res

        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
        feats.append(sample)
        out = [self.out_blocks[0](sample)]                # deepest head

        for i, upsample_block in enumerate(self.up_blocks):
            res = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            if getattr(upsample_block, 'has_cross_attention', False):
                sample = upsample_block(hidden_states=sample, temb=emb,
                                        res_hidden_states_tuple=res,
                                        encoder_hidden_states=encoder_hidden_states)
            else:
                sample = upsample_block(hidden_states=sample, temb=emb,
                                        res_hidden_states_tuple=res)
            feats.append(sample)
            out.append(self.out_blocks[i + 1](sample))    # one head per up block

        if return_features:
            return out, feats
        return out


if __name__ == '__main__':
    torch.manual_seed(0)
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = UNetLatentDiscriminator(in_channels=4, cross_attention_dim=768).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'params: {n_params:,}')
    x = torch.randn(2, 4, 64, 64, device=dev)
    t = torch.tensor([100, 250], device=dev)
    prompt = torch.randn(2, 77, 768, device=dev)
    logits, feats = net(x, t, prompt, return_features=True)
    print('logits:', [tuple(l.shape) for l in logits])
    print('feats :', [tuple(f.shape) for f in feats])
    g_loss = sum(-l.mean() for l in logits) / len(logits)
    g_loss.backward()
    gn = sum(p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None)
    print(f'G-style loss {float(g_loss):.4f}, grad|sum| {gn:.3e} — OK')
