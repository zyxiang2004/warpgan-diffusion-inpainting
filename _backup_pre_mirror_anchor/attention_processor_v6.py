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
            # ---- mirror branch (teacher ④, Step-2 retry 2026-08-24) ----
            # Fully INDEPENDENT side branch for reference_features_extra
            # (x_mirror): own K/V adapters (same warm-start), own zero-init
            # gate. The previous token-concat + shared-gate design injected an
            # UN-LEARNABLE constant perturbation and destroyed the trained
            # BrushNet (see STEP0 report §8.13). Zero gate => step-0 output is
            # bit-identical to the no-mirror baseline; the gradient then
            # decides whether mirror information is used at all.
            self.to_k_reference_extra = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_v_reference_extra = nn.Linear(hidden_size, hidden_size, bias=False)
            self.reference_extra_scale = nn.Parameter(torch.zeros(1))
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
            self.to_k_reference_extra.weight.copy_(attn.to_k.weight)
            self.to_v_reference_extra.weight.copy_(attn.to_v.weight)
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

        if encoder_hidden_states is None and reference_features is not None:
            query_tokens = branch_input.shape[1]
            feat = reference_features.get(branch_input.shape[-1]) if isinstance(reference_features, dict) else None
            feat_extra = (
                reference_features_extra.get(branch_input.shape[-1])
                if isinstance(reference_features_extra, dict) else None
            )
            if feat is not None:
                ref_seq = self._reference_tokens(feat, query_tokens, branch_input.dtype)
                ref_key = self._to_heads(self.to_k_reference(ref_seq), attn.heads)
                ref_value = self._to_heads(self.to_v_reference(ref_seq), attn.heads)
                ref_output = F.scaled_dot_product_attention(
                    query, ref_key, ref_value, attn_mask=None, dropout_p=0.0, is_causal=False
                )
                ref_output = self._from_heads(ref_output).to(query.dtype)
                ref_output = self.to_out_reference(ref_output)
                hidden_states = hidden_states + ref_output * torch.tanh(self.reference_scale)

            # teacher ④, retry: INDEPENDENT mirror side branch — separate K/V,
            # separate attention, separate LEARNED gate starting at tanh(0)=0.
            # No token concat into the main reference attention (the v1 design
            # that injected an unlearnable constant perturbation).
            if feat_extra is not None:
                extra_seq = self._reference_tokens(feat_extra, query_tokens, branch_input.dtype)
                extra_key = self._to_heads(self.to_k_reference_extra(extra_seq), attn.heads)
                extra_value = self._to_heads(self.to_v_reference_extra(extra_seq), attn.heads)
                extra_output = F.scaled_dot_product_attention(
                    query, extra_key, extra_value, attn_mask=None, dropout_p=0.0, is_causal=False
                )
                extra_output = self._from_heads(extra_output).to(query.dtype)
                extra_output = self.to_out_reference(extra_output)  # shared output proj (same space)
                hidden_states = hidden_states + extra_output * torch.tanh(self.reference_extra_scale)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor