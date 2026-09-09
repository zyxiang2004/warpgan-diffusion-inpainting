import torch
import torch.nn as nn
import torch.nn.functional as F


class ReferenceAttentionProcessor(nn.Module):
    """Frozen SD attention plus gated W+ RCA / ReferenceNet residual branches.

    The W+ branch owns Q, K, V and output projections, which is the ordinary RCA
    fallback described by the project design. ReferenceNet is also residual-gated;
    replacing the stock processor therefore preserves the pretrained UNet exactly
    when both gates are zero.
    """

    def __init__(self, hidden_size, cross_attention_dim=768):
        super().__init__()
        # Define the complete processor interface for both self- and cross-attn.
        # This avoids branch-dependent attributes and keeps checkpoint inspection
        # and logging structurally well-defined.
        self.reference_scale = None
        self.reference_local_scale = None
        self.to_k_reference = None
        self.to_v_reference = None
        self.to_out_reference = None
        if cross_attention_dim is not None:
            self.to_q_wplus = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_k_wplus = nn.Linear(cross_attention_dim, hidden_size, bias=False)
            self.to_v_wplus = nn.Linear(cross_attention_dim, hidden_size, bias=False)
            self.to_out_wplus = nn.Linear(hidden_size, hidden_size, bias=True)
            self.wplus_scale = nn.Parameter(torch.zeros(1))
            self.wplus_output_mode = "independent"
        else:
            self.to_q_wplus = None
            self.to_k_wplus = None
            self.to_v_wplus = None
            self.to_out_wplus = None
            self.wplus_scale = None
            # ReferenceNet features live in the same hidden space as self-attention,
            # but reusing the frozen SD K/V/O projections leaves only one scalar
            # gate trainable. Independent K/V/O adapters let real-photo supervision
            # learn *how* to use reference texture while the zero gate preserves the
            # pretrained SD function exactly at initialization.
            self.to_k_reference = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_v_reference = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_out_reference = nn.Linear(hidden_size, hidden_size, bias=True)
            self.reference_scale = nn.Parameter(torch.zeros(1))
            # ---- v7a: SOURCE-SEPARATING K/V for the mirror evidence ----
            # v7 post-mortem: with a SHARED K/V the softmax share stayed at the
            # token-count baseline (0.489->0.486 over 16K steps) — nothing in
            # the frozen stack can learn to ENCODE the two sources differently,
            # so the attention cannot discriminate evidence quality. Fix: give
            # the mirror its own trainable K/V (SD warm-start, lr = brushnet's,
            # per the v2 low-lr lesson). Softmax competition and the shared
            # output projection are UNCHANGED — the model gains the ABILITY to
            # separate sources, not a hand-set strength.
            self.to_k_reference_mirror = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_v_reference_mirror = nn.Linear(hidden_size, hidden_size, bias=False)
            self.mirror_attn_share = None  # read-only gauge (see v7)
            # Target-aligned local reference path. It uses the same pretrained
            # V/O initialization but bypasses global token matching at visible
            # positions, preserving eyelashes, eyelids and hair strands.
            self.reference_local_scale = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def initialize_from_attention(self, attn):
        """Copy the pretrained SD projections into the independent RCA branch."""
        if self.to_q_wplus is None:
            self.to_k_reference.weight.copy_(attn.to_k.weight)
            self.to_v_reference.weight.copy_(attn.to_v.weight)
            # v7a: mirror K/V warm-start from the same SD projections — the
            # initial K-space is IDENTICAL for both sources (share starts at the
            # token-count baseline, exactly like v7), then only training can
            # differentiate them. Zero-perturbation start preserved.
            self.to_k_reference_mirror.weight.copy_(attn.to_k.weight)
            self.to_v_reference_mirror.weight.copy_(attn.to_v.weight)
            self.to_out_reference.weight.copy_(attn.to_out[0].weight)
            if attn.to_out[0].bias is not None:
                self.to_out_reference.bias.copy_(attn.to_out[0].bias)
            else:
                self.to_out_reference.bias.zero_()
            return
        self.to_q_wplus.weight.copy_(attn.to_q.weight)
        self.to_k_wplus.weight.copy_(attn.to_k.weight)
        self.to_v_wplus.weight.copy_(attn.to_v.weight)
        self.to_out_wplus.weight.copy_(attn.to_out[0].weight)
        if attn.to_out[0].bias is not None:
            self.to_out_wplus.bias.copy_(attn.to_out[0].bias)
        else:
            self.to_out_wplus.bias.zero_()

    @staticmethod
    def _to_heads(tensor, heads):
        head_dim = tensor.shape[-1] // heads
        return tensor.view(tensor.shape[0], -1, heads, head_dim).transpose(1, 2)

    @staticmethod
    def _from_heads(tensor):
        return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], -1)

    def _reference_tokens(self, feat, query_tokens, dtype):
        """Resolve a (possibly multi-scale) reference feature into a token seq.

        Single-image behavior is bit-identical to the upstream processor
        (best-matching scale, bilinear resize to the query side). Multiple
        reference images are token-concatenated by the caller (teacher ④:
        x_mirror joins the ReferenceNet branch in PARALLEL with the primary
        reference — global attention needs no spatial alignment).
        """
        if isinstance(feat, (list, tuple)):
            feat = min(
                feat,
                key=lambda item: abs(item.shape[-2] * item.shape[-1] - query_tokens),
            )
        if feat.shape[-2] * feat.shape[-1] != query_tokens:
            side = int(query_tokens ** 0.5)
            if side * side == query_tokens:
                feat = F.interpolate(feat, size=(side, side), mode='bilinear', align_corners=False)
        return feat.flatten(2).transpose(1, 2).to(dtype)

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None,
        attention_mask=None, temb=None, reference_features=None,
        reference_features_extra=None, reference_validity=None,
        wplus_features=None, scale=1.0,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        branch_input = hidden_states
        query = self._to_heads(attn.to_q(branch_input), attn.heads)

        if encoder_hidden_states is not None:
            context = encoder_hidden_states
            if attn.norm_cross:
                context = attn.norm_encoder_hidden_states(context)
            prepared_mask = None
            if attention_mask is not None:
                prepared_mask = attn.prepare_attention_mask(attention_mask, context.shape[1], batch_size)
                prepared_mask = prepared_mask.view(batch_size, attn.heads, -1, prepared_mask.shape[-1])
            key = self._to_heads(attn.to_k(context), attn.heads)
            value = self._to_heads(attn.to_v(context), attn.heads)
        else:
            prepared_mask = None
            key = self._to_heads(attn.to_k(branch_input), attn.heads)
            value = self._to_heads(attn.to_v(branch_input), attn.heads)

        base_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=prepared_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = self._from_heads(base_output).to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None and wplus_features is not None:
            if getattr(self, 'wplus_backbone_fusion', False):
                # Teacher variant: W+ tokens share the backbone cross-attention.
                # They are concatenated with the text tokens so to_k/to_v/to_q of
                # the backbone itself process them, exactly like the original
                # w-plus-adapter trains its adapters on the main path.
                wplus_seq = wplus_features.to(branch_input.dtype)
                context = torch.cat([context, wplus_seq], dim=1)
                key = self._to_heads(attn.to_k(context), attn.heads)
                value = self._to_heads(attn.to_v(context), attn.heads)
                base_output = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=prepared_mask, dropout_p=0.0, is_causal=False
                )
                hidden_states = self._from_heads(base_output).to(query.dtype)
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
            else:
                query_w = self._to_heads(self.to_q_wplus(branch_input), attn.heads)
                key_w = self._to_heads(self.to_k_wplus(wplus_features), attn.heads)
                value_w = self._to_heads(self.to_v_wplus(wplus_features), attn.heads)
                w_output = F.scaled_dot_product_attention(
                    query_w, key_w, value_w, attn_mask=None, dropout_p=0.0, is_causal=False
                )
                w_output = self.to_out_wplus(self._from_heads(w_output).to(query.dtype))
                hidden_states = hidden_states + w_output * torch.tanh(self.wplus_scale)

        # v7 soft retrieval: mirror tokens JOIN the primary reference tokens on
        # the same K/V axis (one softmax competition — the mutual-self-attention
        # paradigm). Where the mirror genuinely matches (real skin on the far
        # cheek) it wins attention mass; where it does not (asymmetric hair) the
        # mass stays with the inversion tokens. No hand-set strength anywhere.
        if encoder_hidden_states is None and reference_features is not None:
            query_tokens = branch_input.shape[1]
            feat = reference_features.get(branch_input.shape[-1]) if isinstance(reference_features, dict) else None
            feat_extra = (
                reference_features_extra.get(branch_input.shape[-1])
                if isinstance(reference_features_extra, dict) else None
            )
            if feat is not None:
                ref_seq = self._reference_tokens(feat, query_tokens, branch_input.dtype)
                n_primary = ref_seq.shape[1]
                key_prim = self.to_k_reference(ref_seq)
                val_prim = self.to_v_reference(ref_seq)
                if feat_extra is not None:
                    # v7a: mirror tokens pass through their OWN K/V — the only
                    # trainable path able to encode "evidence source" into the
                    # key space; softmax competition then decides usage.
                    ref_seq_extra = self._reference_tokens(feat_extra, query_tokens, branch_input.dtype)
                    key_prim = torch.cat([key_prim, self.to_k_reference_mirror(ref_seq_extra)], dim=1)
                    val_prim = torch.cat([val_prim, self.to_v_reference_mirror(ref_seq_extra)], dim=1)
                ref_key = self._to_heads(key_prim, attn.heads)
                ref_value = self._to_heads(val_prim, attn.heads)
                ref_output = F.scaled_dot_product_attention(
                    query, ref_key, ref_value, attn_mask=None, dropout_p=0.0, is_causal=False
                )
                # read-only gauge: attention mass on the mirror tokens
                if feat_extra is not None and not torch.is_grad_enabled():
                    with torch.no_grad():
                        qn = F.normalize(query.float(), dim=-1)
                        kn = F.normalize(ref_key.float(), dim=-1)
                        aff = qn @ kn.transpose(-1, -2) * (query.shape[-1] ** 0.5)
                        probs = aff.softmax(dim=-1)
                        self.mirror_attn_share = float(
                            probs[..., n_primary:].sum() / (probs.sum() + 1e-8))
                ref_output = self._from_heads(ref_output).to(query.dtype)
                ref_output = self.to_out_reference(ref_output)
                hidden_states = hidden_states + ref_output * torch.tanh(self.reference_scale)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor