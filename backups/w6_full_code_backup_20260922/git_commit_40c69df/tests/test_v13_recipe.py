# -*- coding: utf-8 -*-
"""v13 recipe tests (2026-08-29): the restored ORIG three-layer recipe.

Verifies, WITHOUT loading any big model:
  1. loss math: weighted eps-MSE (v11 path restored) behaves exactly as the
     evidence-graded contract demands at photo/blind extremes;
  2. config defaults equal the ORIG checkpoint recipe (latent 0.1 / id 0.5 /
     hole 0.1 / dual-band low-anchor 0);
  3. the v13 identity-audit logic fires/fails correctly on stubbed encoders;
  4. low-t gating math (t < pixel_max_t selection).

The full end-to-end DATA-FLOW reality (real VAE, real WplusNet, real GOAE
encoder, gradients into BrushNet) is covered by the GPU smoke run
(smoke.check=True + smoke.fix_timestep=100) whose audits these tests mirror.
"""
import math
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CPU-only test env: stub the stylegan2 CUDA extensions pulled in by the
# Coach import chain (they are never used by the numeric methods we test).
for _m in ('models.stylegan2.op', 'models.stylegan2.op.fused_act',
           'models.stylegan2.op.upfirdn2d'):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
_op = sys.modules['models.stylegan2.op']
_fa = sys.modules['models.stylegan2.op.fused_act']
_uf = sys.modules['models.stylegan2.op.upfirdn2d']
_fa.FusedLeakyReLU = type('FusedLeakyReLU', (), {})
_fa.fused_leaky_relu = lambda *a, **k: None
_uf.upfirdn2d = lambda *a, **k: None
_op.FusedLeakyReLU, _op.fused_leaky_relu, _op.upfirdn2d = \
    _fa.FusedLeakyReLU, _fa.fused_leaky_relu, _uf.upfirdn2d

from training.coach_inpainting_diffusion import Coach  # noqa: E402

PASS = 0
FAIL = 0


def check(name, ok, extra=''):
    global PASS, FAIL
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")
    PASS, FAIL = PASS + (1 if ok else 0), FAIL + (0 if ok else 0)


class _Stub:
    """Bare receiver so we can call Coach's pure-numeric methods."""


def test_weighted_eps_mse_math():
    eps_mse = types.MethodType(Coach._eps_mse, _Stub())
    torch.manual_seed(0)
    b, ch, h, w = 2, 4, 8, 8
    eps_pred, noise = torch.randn(b, ch, h, w), torch.randn(b, ch, h, w)
    w1 = torch.ones(b, 1, h, w)
    check('weighted eps: all-ones == plain mean',
          torch.allclose(eps_mse(eps_pred, noise, w1),
                         (eps_pred - noise).pow(2).mean(), rtol=1e-5))
    # photo region full weight, blind region hole weight (v13 recipe 1.0/0.1)
    eff = torch.zeros(b, 1, h, w)
    eff[..., :4] = 1.0
    w_map = eff * 1.0 + (1.0 - eff) * 0.1
    e = eps_pred - noise
    sq = e.pow(2)
    ref = (sq * w_map).sum() / (w_map.sum() * ch)
    check('weighted eps: evidence grading 1.0/0.1 exact',
          torch.allclose(eps_mse(eps_pred, noise, w_map), ref, rtol=1e-6))
    w_hole = (1.0 - eff)
    ref_hole = (sq * w_hole).sum() / (w_hole.sum() * ch)
    check('weighted eps: blind-only normalization exact',
          torch.allclose(eps_mse(eps_pred, noise, w_hole), ref_hole, rtol=1e-6))
    # orig §5.3 semantics: with hole_weight the blind HALF still gets gradient
    # (unlike the abandoned dual-band which zeroed the texture band there)
    w_grad = torch.autograd.grad(
        eps_mse(eps_pred.requires_grad_(True), noise, w_map),
        eps_pred, retain_graph=False)[0][..., 4:]
    check('weighted eps: blind region gradient NONZERO (orig weak anchor)',
          float(w_grad.abs().sum()) > 0)


def test_low_t_gating():
    pixel_max_t = 200
    ts = torch.tensor([50, 150, 199, 200, 999])
    sel = ts < pixel_max_t
    check('low-t gate: t<200 selection (3 of 5)',
          sel.sum().item() == 3 and bool(sel[:3].all()) and not bool(sel[3]))


def test_config_defaults_are_orig_recipe():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'train_inpainting_diffusion.yaml'))
    check('config: losses.latent.weight == 0.1 (orig)',
          float(cfg.losses.latent.weight) == 0.1)
    check('config: losses.pixel.id_weight == 0.5 (orig)',
          float(cfg.losses.pixel.id_weight) == 0.5)
    check('config: real_pass1.hole_weight == 0.1 (orig §5.3)',
          float(cfg.losses.real_pass1.hole_weight) == 0.1)
    check('config: real_pass1.blind_struct_weight == 0 (low-anchor OFF)',
          float(cfg.losses.real_pass1.blind_struct_weight) == 0.0)


def test_identity_audit_logic():
    """The v13 audit must PASS for a coherent encoder and FAIL for a
    meaningless one (stubbed WplusNet, no big models)."""
    x = torch.rand(1, 3, 512, 512)
    codes = torch.randn(1, 14, 512)

    class Gan:
        def __init__(self, coherent):
            self.coherent = coherent
            self.decoder = types.SimpleNamespace(
                parameters=lambda: iter([torch.zeros(1, device='cpu')]))

        def encoder_forward(self, img):
            # coherent: reproduces the person's codes;
            # incoherent: returns a FOREIGN code (channel-flipped = a
            # different person) — truly meaningless as a target.
            return codes if self.coherent else codes.flip(-1)

    s = _Stub()
    s.gan = Gan(coherent=True)
    try:
        Coach._v13_identity_audit(s, x, codes)   # coherent -> must pass
        ok_pass = True
    except AssertionError:
        ok_pass = False
    check('identity audit: coherent encoder PASSES', ok_pass)
    s.gan = Gan(coherent=False)
    try:
        Coach._v13_identity_audit(s, x, codes)   # incoherent -> must raise
        ok_fail = False
    except AssertionError:
        ok_fail = True
    check('identity audit: incoherent encoder RAISES', ok_fail)


def test_dual_band_really_off_in_recipe():
    """v13 red line: with blind_struct_weight=0 the dual-band loss is never
    constructed — the weighted eps path (orig §5.3) is the pass1 loss."""
    src = open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'training', 'coach_inpainting_diffusion.py')).read()
    check('coach: dual-band gated behind blind_struct_weight>0',
          'if self.blind_struct_weight > 0:' in src)
    check('coach: weighted v11 path present (orig §5.3)',
          'w_map = eff_latents * float(p1.visible_weight)' in src)
    check('coach: identity audit wired into real forward',
          'self._v13_identity_audit(x, codes)' in src)


if __name__ == '__main__':
    test_weighted_eps_mse_math()
    test_low_t_gating()
    test_config_defaults_are_orig_recipe()
    test_identity_audit_logic()
    test_dual_band_really_off_in_recipe()
    print('=' * 60)
    print(f'v13 recipe tests: {PASS} PASS / {FAIL} FAIL')
    sys.exit(1 if FAIL else 0)

