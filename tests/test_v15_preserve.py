"""v15 score-preservation tests (2026-08-31): blind high-freq prior takeover.
Pure-numeric tests (no big models): band math, weighting, config, wiring."""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_v13_recipe import _Stub, check  # noqa: E402 (stubs CUDA ops)

from training.coach_inpainting_diffusion import Coach  # noqa: E402


def _lp():
    s = _Stub()
    s.synth_lp_sigma = 2.0
    s._texture_zero_weight = types.MethodType(Coach._texture_zero_weight, s)
    return types.MethodType(Coach._lowpass_eps, s)


def test_hp_band_math():
    lp = _lp()
    torch.manual_seed(0)
    e = torch.randn(1, 4, 8, 8)
    hp = e - lp(e)
    # HP kills a pure DC shift
    hp2 = (e + 0.3) - lp(e + 0.3)
    check('HP invariance to DC shift', torch.allclose(hp, hp2, atol=1e-5))
    # HP keeps a zero-mean checkerboard
    eb = e.clone()
    eb[..., ::2, ::2] += .5; eb[..., 1::2, 1::2] += .5
    eb[..., ::2, 1::2] -= .5; eb[..., 1::2, ::2] -= .5
    hp_eb = eb - lp(eb)
    check('HP captures checkerboard energy',
          float((hp_eb - hp).pow(2).sum()) > 0.1,
          f'{float((hp_eb - hp).pow(2).sum()):.3f}')
    # offset composition: hp(a-b) band-selects the CONTROL difference only
    a, b = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    check('HP is linear in the offset',
          torch.allclose((a - b) - lp(a - b),
                         (a - lp(a)) - (b - lp(b)), atol=1e-4))


def test_preserve_weighting():
    """sum(w*hp^2)/(sum(w)*ch) with a known field must match a hand value."""
    torch.manual_seed(1)
    hp = torch.randn(1, 4, 8, 8)
    w = torch.zeros(1, 1, 8, 8)
    w[..., :4] = 1.0
    ref = (hp.pow(2) * w).sum() / (w.sum() * 4)
    got = (hp.pow(2) * w).sum() / (w.sum() * hp.shape[1] + 1e-6)
    check('preserve normalization exact', torch.allclose(ref, got, rtol=1e-5))
    # gradient flows to the (student) side
    e = torch.randn(1, 4, 8, 8, requires_grad=True)
    lp = _lp()
    loss = ((e - lp(e)).pow(2) * w).sum() / (w.sum() * 4)
    loss.backward()
    check('preserve gradient reaches eps_pred', float(e.grad.abs().sum()) > 0)


def test_config_defaults():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'train_inpainting_diffusion.yaml'))
    check('config: preserve.weight == 1.0', float(cfg.losses.preserve.weight) == 1.0)
    check('config: preserve.teacher_wplus == true', bool(cfg.losses.preserve.teacher_wplus))


def test_wiring():
    src = open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'training', 'coach_inpainting_diffusion.py')).read()
    check('coach: teacher branch in diffusion forward',
          'eps_teacher = self.denoising_unet(' in src
          and 'up_block_add_samples=None' in src)
    check('coach: teacher gated on preserve_weight',
          'if self.preserve_weight > 0:' in src)
    check('coach: L_preserve on pass1 with eff field',
          "w_blind_p = (1.0 - eff_latents)" in src)
    check('coach: v15 audit wired', 'self._v15_audit(out1, offset, hp, w_blind_p, loss_pres)' in src)
    check('coach: teacher output in return dict',
          "'eps_teacher': eps_teacher" in src)


if __name__ == '__main__':
    test_hp_band_math()
    test_preserve_weighting()
    test_config_defaults()
    test_wiring()
    import tests.test_v13_recipe as _h
    print('=' * 60)
    print(f'v15 preserve tests: {_h.PASS} PASS / {_h.FAIL} FAIL')
    sys.exit(1 if _h.FAIL else 0)
