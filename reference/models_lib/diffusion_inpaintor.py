import torch
from torch import nn
from diffusers import AutoencoderKL, BrushNetModel, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from models.mapper.w_proj import WProjModel
from models.referencenet.attention_processor import ReferenceAttentionProcessor
from models.referencenet.unet_2d_condition import UNet2DConditionModel as ReferenceNet
from models.saicinpainting.utils import set_requires_grad
from utils.diffusion_inpainting import sample_brushnet_inpainting
from utils.warp.Splatting import Warper


class DiffusionInpaintor(nn.Module):
    """Production loader for the WarpGAN diffusion inpainting checkpoint."""

    def __init__(
        self,
        checkpoint_path,
        device="cuda:0",
        sd_path=None,
        brushnet_path=None,
    ):
        super().__init__()
        import os
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if sd_path is None:
            sd_path = os.path.join(_project_root, "pretrained_models", "AI-ModelScope", "stable-diffusion-v1-5")
        if brushnet_path is None:
            brushnet_path = os.path.join(_project_root, "pretrained_models", "brushnet")
        self.device = torch.device(device)
        self.vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(self.device).eval()
        self.scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")
        self.unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet").to(self.device).eval()

        processors = {}
        for name in self.unet.attn_processors.keys():
            cross_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            else:
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            processor = ReferenceAttentionProcessor(hidden_size, cross_dim).to(self.device)
            processor.initialize_from_attention(
                self.unet.get_submodule(name.removesuffix(".processor"))
            )
            processors[name] = processor
        self.unet.set_attn_processor(processors)

        self.reference_net = ReferenceNet.from_pretrained(sd_path, subfolder="unet").to(self.device).eval()
        self.brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float32).to(self.device).eval()
        self.w_mapper = WProjModel(cross_attention_dim=768, embeddings_dim=512).to(self.device).eval()
        self.warper = Warper()

        tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(self.device).eval()
        tokens = tokenizer(
            "", padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            self.empty_prompt_embeds = text_encoder(tokens.input_ids.to(self.device))[0]
        del tokenizer, text_encoder

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        architecture_version = int(checkpoint.get("architecture_version", -1))
        if architecture_version not in (2, 3, 4, 5) or checkpoint.get("supervision") != "novel_view_target":
            raise RuntimeError(
                "Expected a WarpGAN diffusion v2/v3/v4/v5 checkpoint trained with novel-view supervision"
            )
        self.architecture_version = architecture_version
        geometry_contract = checkpoint.get("geometry_condition") or {}
        self.geometry_condition_enabled = bool(geometry_contract.get("enable", False))
        self.geometry_mode = str(geometry_contract.get("mode", "raw_warp"))
        self.geometry_lowpass_kernel = int(geometry_contract.get("lowpass_kernel", 31))
        if "w_mapper_state_dict" not in checkpoint or "rca_state_dict" not in checkpoint:
            raise RuntimeError("Checkpoint is missing W+ mapper or RCA processor weights")
        self.w_mapper.load_state_dict(checkpoint["w_mapper_state_dict"], strict=True)
        missing_processors = [
            name for name in self.unet.attn_processors
            if name not in checkpoint["rca_state_dict"]
        ]
        if missing_processors:
            raise RuntimeError(f"Checkpoint is missing RCA processors: {missing_processors[:3]}")
        for name, processor in self.unet.attn_processors.items():
            legacy_state = checkpoint["rca_state_dict"][name]
            incompatible = processor.load_state_dict(
                legacy_state, strict=architecture_version == 5
            )
            if architecture_version < 5 and incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Unexpected legacy RCA keys for {name}: {incompatible.unexpected_keys}"
                )
        self.checkpoint_step = int(checkpoint.get("global_step", -1))

        for module in (self.vae, self.unet, self.reference_net, self.brushnet, self.w_mapper):
            set_requires_grad(module, False)
            module.eval()

    @torch.inference_mode()
    def extract_reference_features(self, source_image):
        source_latents = (
            self.vae.encode(source_image * 2.0 - 1.0).latent_dist.mode()
            * self.vae.config.scaling_factor
        )
        batch_size = source_latents.shape[0]
        features = self.reference_net.extract_features(
            sample=source_latents,
            timestep=torch.zeros(batch_size, device=self.device, dtype=torch.long),
            encoder_hidden_states=self.empty_prompt_embeds.expand(batch_size, -1, -1),
        )
        if self.architecture_version == 2:
            # Reproduce the legacy v2 behavior: one (last/deepest) feature per
            # channel width. V3 checkpoints use the full multiscale lists.
            return {
                channels: values[-1] if isinstance(values, (list, tuple)) else values
                for channels, values in features.items()
            }
        return features

    def _align_reference_features(self, features, source_depth, source_camera, target_camera):
        aligned, validities = {}, {}
        for channels, values in features.items():
            values = values if isinstance(values, (list, tuple)) else [values]
            aligned[channels], validities[channels] = [], []
            for feature in values:
                warped, validity, _ = self.warper.forward_warp(
                    img1=feature,
                    depth1=source_depth,
                    c1=source_camera,
                    c2=target_camera,
                )
                aligned[channels].append(warped)
                validities[channels].append(validity)
        return aligned, validities

    @torch.inference_mode()
    def forward(
        self, warp_image, mask, wplus_codes, source_image, steps=50, seed=42,
        source_depth=None, source_camera=None, target_camera=None,
        geometry_image=None,
    ):
        warp_image = warp_image.to(self.device).float().clamp(0, 1)
        mask = mask.to(self.device).float().clamp(0, 1)
        source_image = source_image.to(self.device).float().clamp(0, 1)
        wplus_codes = wplus_codes.to(self.device).float()
        if geometry_image is not None:
            geometry_image = geometry_image.to(self.device).float().clamp(0, 1)
        if (
            self.geometry_condition_enabled
            and self.geometry_mode not in ("none", "raw_warp")
            and geometry_image is None
        ):
            raise ValueError(
                "This checkpoint requires a target-view geometry_image for "
                f"geometry mode {self.geometry_mode!r}. Production inference must "
                "use the same condition contract as training."
            )
        reference_features = self.extract_reference_features(source_image)
        aligned_features, reference_validity = None, None
        if source_depth is not None and source_camera is not None and target_camera is not None:
            aligned_features, reference_validity = self._align_reference_features(
                reference_features,
                source_depth.to(self.device).float(),
                source_camera.to(self.device).float(),
                target_camera.to(self.device).float(),
            )
        return sample_brushnet_inpainting(
            vae=self.vae,
            denoising_unet=self.unet,
            brushnet=self.brushnet,
            scheduler_config=self.scheduler.config,
            empty_prompt_embeds=self.empty_prompt_embeds,
            warp_image=warp_image,
            mask=mask,
            geometry_image=geometry_image if self.geometry_condition_enabled else None,
            geometry_lowpass_kernel=self.geometry_lowpass_kernel,
            geometry_mode=self.geometry_mode,
            wplus_features=self.w_mapper(wplus_codes),
            reference_features=reference_features,
            aligned_reference_features=aligned_features,
            reference_validity=reference_validity,
            num_inference_steps=steps,
            seed=seed,
        )