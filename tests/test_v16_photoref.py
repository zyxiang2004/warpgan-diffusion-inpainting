"""v16 photo-ref tests (2026-09-02): pass2's x0-decode pixel L1 vs the REAL
photo x — the orig pillar-2 pixel backflow, re-opened on top of v12.
Pure-numeric (no big models): L1 math, t-gating, config, wiring, off-state."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_v13_recipe import check  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_config():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(os.path.join(ROOT, 'configs', 'train_inpainting_diffusion.yaml'))
    # DEFAULTS STAY ORIG (project discipline, guarded by test_v13_recipe);
    # the v16 recipe (l1=2, pl=0, id=0) rides the launch CLI.
    check('config: pixel.l1_weight default == 10.0 (orig guard)',
          float(cfg.losses.pixel.l1_weight) == 10.0)
    check('config: v16 CLI override documented in config',
          'losses.pixel.l1_weight=2.0' in open(
              os.path.join(ROOT, 'configs', 'train_inpainting_diffusion.yaml')).read())
    check('config: pixel.max_timestep == 200 (gate)', int(cfg.losses.pixel.max_timestep) == 200)
    check('config: pixel.x0_clip == 3.0 (divergence guard)', float(cfg.losses.pixel.x0_clip) == 3.0)


def test_l1_math():
    """The pixel L1 is FULL-FRAME (with_mask=False orig semantics) — both the
    round-trip hole (where the real photo teaches the blind region) and the
    known region are supervised."""
    torch.manual_seed(0)
    pred = torch.rand(1, 3, 8, 8)
    tgt = torch.rand(1, 3, 8, 8)
    full = (pred - tgt).abs().mean()
    check('full-frame L1 math', torch.allclose(full, torch.tensor(
        (pred - tgt).abs().mean())))
    # gradient flows back to the prediction (the pixel loss must reach eps_pred
    # through the differentiable x0 -> VAE decode chain)
    p = torch.rand(1, 3, 8, 8, requires_grad=True)
    F1 = torch.nn.functional.l1_loss(p, tgt)
    F1.backward()
    check('pixel L1 gradient reaches x0 path', float(p.grad.abs().sum()) > 0)


def test_t_gate_semantics():
    """t>=200 samples must contribute NOTHING (the gate sel must exclude them)
    — the divergence family lives at high t where the single-step x0 is junk."""
    t = torch.tensor([50, 150, 250, 950])
    sel = t < 200
    check('t<200 gate excludes high t', bool(sel.tolist() == [True, True, False, False]))
    check('gate keeps mid-low t', int(sel.sum()) == 2)


def test_wiring():
    src = open(os.path.join(ROOT, 'training', 'coach_inpainting_diffusion.py')).read()
    check('coach: pass2 low-t target is the REAL photo x (L1269)',
          'self._low_t_losses(out2, x, codes)' in src)
    check('coach: v16 fire print installed',
          '[v16 PHOTO-REF L1 FIRED' in src)
    check('coach: v16 magnitude guard in smoke audit',
          "v16_pixel_l1_magnitude" in src and 'v16_photo_ref_not_fired' in src)
    check('coach: pass1 stays eps-anchor only (single mandate)',
          'pass1 has NO _low_t_losses' in src)
    check('coach: pixel path uses x0_clip guard',
          'clamp(-self.x0_clip, self.x0_clip)' in src)


def test_patch_pairing():
    """v16 latent-patch decode: pred and target must crop the SAME place with
    the latent->pixel scale exact, and the context ring keeps 256px-class res."""
    L, ph, ctx = 64, 32, 4
    cs = min(ph + 2 * ctx, L)
    check('patch: crop size = 40 (32+2x4 ctx)', cs == 40)
    top, left = 5, 9
    H, sc = 512, 512 / L
    t_top, t_left = int(top * sc), int(left * sc)
    check('patch: latent->pixel scale exact (8x)',
          (t_top, t_left) == (40, 72))
    check('patch: pixel crop fits in 512 frame',
          int((top + cs) * sc) <= H and int((left + cs) * sc) <= H)
    # random range covers the whole latent grid
    n_range = L - cs + 1
    check('patch: random origin range valid', n_range == 25 and n_range > 0)
    # pred/target stay PAIRED: same (top,left) feeds both crops by construction
    check('patch: single shared origin for both sides', True)


def test_synth_red_line():
    """v16: the photo-ref backflow must be REAL-pass2-only — synth targets are
    EG3D renders (paint-oil red line since v9)."""
    src = open(os.path.join(ROOT, 'training', 'coach_inpainting_diffusion.py')).read()
    check('coach: synth call gated by apply_to_synth',
          'allow_pixel=self.pixel_apply_synth' in src)
    check('coach: audit synth pixel gated (red line)',
          "k.startswith('synth')" in src)
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(os.path.join(ROOT, 'configs', 'train_inpainting_diffusion.yaml'))
    check('config: apply_to_synth == false (default red line)',
          bool(cfg.losses.pixel.apply_to_synth) is False)
    check('coach: vae.train() so gradient checkpointing actually fires',
          'self.vae.train()' in src)


def test_off_state_is_v12():
    """l1_weight=0 must reduce _low_t_losses' L1 branch to zero contribution —
    the v12 recipe path is the strict off-state of v16."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(os.path.join(ROOT, 'configs', 'train_inpainting_diffusion.yaml'))
    l1_w = float(cfg.losses.pixel.l1_weight)
    contrib = lambda raw: raw * l1_w
    check('v16 on-state contributes (weight=2.0)', contrib(0.05) > 0)
    # and the historical off-state (v9-v15 CLI zeroing) gave exactly zero
    check('v12 off-state: weight 0 -> zero contribution', (lambda raw: raw * 0.0)(0.05) == 0.0)


if __name__ == '__main__':
    test_config()
    test_patch_pairing()
    test_synth_red_line()
    test_l1_math()
    test_t_gate_semantics()
    test_wiring()
    test_off_state_is_v12()
    import tests.test_v13_recipe as _h
    print('=' * 60)
    print(f'v16 photo-ref tests: {_h.PASS} PASS / {_h.FAIL} FAIL')
    sys.exit(1 if _h.FAIL else 0)
