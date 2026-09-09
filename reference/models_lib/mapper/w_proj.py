import torch
import torch.nn as nn

class WProjModel(nn.Module):
    """
    用于将 14x512 的 W+ 潜码映射为 Stable Diffusion 的 768 维 Token 序列
    【修复版】：保留 Token 维度的独立性，避免语义坍塌，并移除了无效的 zero_scale
    """
    def __init__(self, cross_attention_dim=768, embeddings_dim=512, clip_extra_context_tokens=4, num_tokens=14):
        super().__init__()
        
        self.proj_index = [4, 8, 12, num_tokens]
        self.cross_attention_dim = cross_attention_dim
        self.num_extra_tokens = clip_extra_context_tokens 
        
        # 【修复点 1】：将输入维度统一设为 embeddings_dim (512)
        # 使得 Linear 层在 Sequence 维度上逐 Token 映射，保留 W+ 固有的层级空间表达
        self.proj_0 = nn.Linear(embeddings_dim, cross_attention_dim, bias=False)
        self.proj_1 = nn.Linear(embeddings_dim, cross_attention_dim, bias=False)
        self.proj_2 = nn.Linear(embeddings_dim, cross_attention_dim, bias=False)
        self.proj_3 = nn.Linear(embeddings_dim, cross_attention_dim, bias=False)
        
        self.clip_proj = nn.Linear(embeddings_dim, cross_attention_dim, bias=False)
        
        # 保留 LayerNorm，用于规范化 3D 潜码映射后的分布，使其对齐 CLIP 文本域
        self.norm = nn.LayerNorm(cross_attention_dim)

        # 【修复点 2】：已移除 self.zero_scale，避免与 Attention Processor 的零门控产生梯度互锁

    def forward(self, image_embeds):
        # image_embeds shape: [Batch, 14, 512]
        
        # 【修复点 3】：移除 reshape(batch_size, -1)
        # 输入直接为 [Batch, N, 512]，Linear 层会自动在最后一个维度映射，输出 [Batch, N, 768]
        embeds_0 = self.proj_0(image_embeds[:, :self.proj_index[0], ...])
        embeds_1 = self.proj_1(image_embeds[:, self.proj_index[0]:self.proj_index[1], ...])
        embeds_2 = self.proj_2(image_embeds[:, self.proj_index[1]:self.proj_index[2], ...])
        embeds_3 = self.proj_3(image_embeds[:, self.proj_index[2]:self.proj_index[3], ...])
        
        # 步骤 A: 投影生成 base tensor [Batch, 768]
        clip_tokens = self.clip_proj(image_embeds.mean(dim=1))
        
        # 步骤 B: 增加维度 [Batch, 1, 768]，然后重复指定的次数
        clip_tokens = clip_tokens.unsqueeze(1).repeat(1, self.num_extra_tokens, 1)

        # 步骤 C: 维度完全对齐的最终拼接
        # 此时 embeds_0 ~ embeds_3 已经是 3D 张量，不再需要额外 unsqueeze(1)
        wplus_tokens = torch.cat([
            embeds_0,
            embeds_1,
            embeds_2,
            embeds_3,
            clip_tokens
        ], dim=1)
        
        # 直接返回规范化后的结果，零初始化保护由下游的 Cross Attention Processor 负责
        return self.norm(wplus_tokens)