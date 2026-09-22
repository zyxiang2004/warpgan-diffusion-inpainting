"""v14 GAN/FM tests (2026-08-31): the ORIG third loss layer (distribution
anchor). No big models loaded — NLayerDiscriminator + R1 loss + FM are small
enough to build directly on CPU, so the DATA FLOW itself is tested here.
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_v13_recipe import _Stub, check  # reuse harness (it stubs CUDA ops)

from models.saicinpainting.training.modules import make_discriminator
from models.saicinpainting.training.losses.adversarial import make_discrim_loss
from models.saicinpainting.training.losses.feature_matching import feature_matching_loss


def test_discriminator_interface():
    d = make_discriminator('pix2pixhd_nlayer', input_nc=3, ndf=64, n_layers=4)
    x = torch.randn(2, 3, 256, 256)
    pred, feats = d(x)
    check('D forward returns (pred, feats)', pred.dim() == 4 and isinstance(feats, list))
    check('D pred is a 1-channel patch map', pred.dim() == 4 and pred.shape[1] == 1,
          f'{tuple(pred.shape)}')
    check('D feats count = n_layers+1', len(feats) == 5)
    # features are a function of input (FM gradient path exists) — DIFFERENT
    # inputs (clone() would make FM==0 and prove nothing)
    x2 = torch.randn_like(x).requires_grad_(True)
    _, f2 = d(x2)
    feature_matching_loss(f2, feats).backward()
    check('FM gradient reaches the fake input',
          float(x2.grad.abs().sum()) > 0)


def test_r1_discriminator_loss():
    loss = make_discrim_loss('r1', gp_coef=0.001, weight=10,
                             mask_as_fake_target=False, allow_scale_mask=True)
    real = torch.randn(1, 3, 256, 256).requires_grad_(True)
    fake = torch.randn(1, 3, 256, 256)
    d = make_discriminator('pix2pixhd_nlayer', input_nc=3, ndf=64, n_layers=4)
    dr, _ = d(real)
    df, _ = d(fake)
    dl, m = loss.discriminator_loss(real_batch=real, fake_batch=fake,
                                    discr_real_pred=dr, discr_fake_pred=df,
                                    mask=None)
    check('R1 D-loss finite and >0', torch.isfinite(dl) and float(dl) > 0,
          f'{float(dl):.4f}')
    check('R1 gp present in metrics', float(m['discr_real_gp']) >= 0,
          f"gp={float(m['discr_real_gp']):.6f}")
    gl, _ = loss.generator_loss(real_batch=real, fake_batch=fake,
                                discr_real_pred=dr, discr_fake_pred=df, mask=None)
    check('G-side NS loss finite', torch.isfinite(gl), f'{float(gl):.4f}')


def test_fm_full_frame_orig_semantics():
    """orig coach L1189-1194: mask_for_fm=None -> plain per-layer MSE mean."""
    fa = [torch.randn(1, 8, 32, 32), torch.randn(1, 16, 16, 16)]
    fb = [torch.randn(1, 8, 32, 32), torch.randn(1, 16, 16, 16)]
    v = feature_matching_loss(fa, fb, mask=None)
    ref = torch.stack([torch.nn.functional.mse_loss(a, b)
                       for a, b in zip(fa, fb)]).mean()
    check('FM(None) == per-layer MSE mean', torch.allclose(v, ref, rtol=1e-6))
    check('FM identical features == 0',
          float(feature_matching_loss(fa, [f.clone() for f in fa], None)) == 0.0)


def test_config_defaults():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'train_inpainting_diffusion.yaml'))
    check('config: fm weight 10 (orig 100, gradual)', float(cfg.losses.feature_matching.weight) == 10)
    check('config: g-adv weight 0 (FM-first)', float(cfg.losses.adversarial.weight) == 0)
    check('config: r1 gp_coef 0.001 (orig)', float(cfg.losses.adversarial.gp_coef) == 0.001)
    check('config: D = pix2pixhd_nlayer 3/64/4 (orig)',
          cfg.losses.discriminator.kind == 'pix2pixhd_nlayer'
          and int(cfg.losses.discriminator.ndf) == 64
          and int(cfg.losses.discriminator.n_layers) == 4)
    check('config: D lr 1e-4 (orig)',
          float(cfg.losses.optimizers.discriminator.lr) == 1e-4)


def test_gan_fm_g_loss_wiring():
    """_gan_fm_g_loss on a stubbed coach: correct queueing + zero when off."""
    from training.coach_inpainting_diffusion import Coach
    s = _Stub()
    s.use_gan_fm = False
    r = Coach._gan_fm_g_loss(s, {'pred_x0_img_g': torch.rand(1, 3, 8, 8)},
                             torch.rand(1, 3, 512, 512), 'p1')
    check('off state returns None', r is None)
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'training', 'coach_inpainting_diffusion.py')).read()
    check('coach: pass1 FM wired', "self._gan_fm_g_loss(out1, x, 'p1')" in src)
    check('coach: pass2 FM wired', "self._gan_fm_g_loss(out2, x, 'p2')" in src)
    check('coach: D step after real update', 'self._discriminator_step()' in src)
    check('coach: synth forward has NO gan_fm call',
          src.count('_gan_fm_g_loss(') == 3)  # def + p1 call + p2 call


if __name__ == '__main__':
    test_discriminator_interface()
    test_r1_discriminator_loss()
    test_fm_full_frame_orig_semantics()
    test_config_defaults()
    test_gan_fm_g_loss_wiring()
    import tests.test_v13_recipe as _h
    print('=' * 60)
    print(f'v14 GAN/FM tests: {_h.PASS} PASS / {_h.FAIL} FAIL')
    sys.exit(1 if _h.FAIL else 0)
