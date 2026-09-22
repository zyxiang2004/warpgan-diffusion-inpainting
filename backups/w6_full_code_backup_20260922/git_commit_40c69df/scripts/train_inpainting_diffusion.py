"""Training entrypoint for the WarpGAN diffusion inpainting port (Step 3).

Hydra entry (run from the repo root):

    python scripts/train_inpainting_diffusion.py [--cli overrides]
        config : configs/train_inpainting_diffusion.yaml (+ CLI deltas)
        resume : checkpoint_path=.../iteration_N.pt -> exp_dir is DERIVED from
                 the ckpt path (two levels up); config saved as
                 config_resume.yaml. Live recipe overrides (v16) are passed on
                 the command line — see scripts/v16_resume.sh for the exact set.
        fresh  : exp_dir gets a [YYYYMMDD-HHMMSS] prefix and must NOT exist
                 (WARP_GAN_OVERWRITE=1 reuses it).

Then: Coach(opts) assembles the full stack (module map in
training/coach_inpainting_diffusion.py docstring) and .train() runs the
real/synth 1:1 alternation loop. Full file-level map: docs/ARCHITECTURE_DIFFUSION.md.
"""
import os
import random

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

import sys
sys.path.append('.')
sys.path.append('..')

from training.coach_inpainting_diffusion import Coach
from utils.common import addtime2path

SEED = 2107  # same as orig scripts/train_inpainting.py
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


@hydra.main(config_path='../configs', config_name='train_inpainting_diffusion')
def main(opts: DictConfig):
    if opts.checkpoint_path is not None:
        checkpoint_path = os.path.abspath(opts.checkpoint_path)
        # .../<experiment>/checkpoints/iteration_N.pt -> <experiment>
        opts.exp_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        os.makedirs(opts.exp_dir, exist_ok=True)
        print(f'Resuming in existing experiment directory: {opts.exp_dir}')
    else:
        opts.exp_dir = addtime2path(opts.exp_dir)
        if os.path.exists(opts.exp_dir):
            if os.environ.get('WARP_GAN_OVERWRITE', '0') == '1':
                print(f'WARP_GAN_OVERWRITE=1 -> reusing {opts.exp_dir}')
            else:
                raise Exception(f'Oops... {opts.exp_dir} already exists '
                                f'(set WARP_GAN_OVERWRITE=1 to reuse)')
        os.makedirs(opts.exp_dir, exist_ok=True)

    print(OmegaConf.to_yaml(opts))
    config_name = 'config.yaml' if opts.checkpoint_path is None else 'config_resume.yaml'
    OmegaConf.save(opts, os.path.join(opts.exp_dir, config_name))

    coach = Coach(opts)
    coach.train()


if __name__ == '__main__':
    main()
