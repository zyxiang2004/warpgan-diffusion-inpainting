你这个问题抓得很准，而且我查了一圈后发现：**最近 diffusion restoration / personalization / distillation 里，已经有一条相当清晰的技术路线能把 GAN 时代的“输出级外观监督”接回 ε-prediction diffusion。**

先给结论：对你们这个 **SD1.5 + ε prediction + 3D 投影破损图 + reference + 人脸新视角补全**，我不建议继续让“3D GAN 渲染的空洞 GT”承担高频纹理监督。更合理的是把监督拆成：

$$3D render负责几何/低频结构+预训练SD负责自然图像prior+reference负责身份/个体纹理+real face discriminator/perceptual prior负责照片质感$$

这比“想办法让 ε-MSE 更好地拟合渲染图”更符合你真正想解决的问题。

## 一、核心问题：Diffusion 训练时其实可以构造“输出图像”
你说：

diffusion 训练时没有“输出图像”，只有噪声预测，所以 GAN 的 output loss 没地方加。

这句话只对了一半。

对于你们当前的 ε-prediction：

$$z_{t}=\sqrt{\bar{\alpha }_{t}}z_{0}+\sqrt{1-\bar{\alpha }_{t}}\epsilon$$

UNet 输出：

$$\hat{\epsilon }_{\theta}\left(z_{t},\ t,\ c\right)$$

那么**在任意训练 timestep** $t$，都可以直接得到当前网络对应的 clean latent estimate：

$$\hat{z}_{0}=\frac{z_{t}-\sqrt{1-\bar{\alpha }_{t}}\hat{\epsilon }_{\theta}}{\sqrt{\bar{\alpha }_{t}}}$$

然后：

$$\hat{I}_{0}=D_{VAE}\left(\hat{z}_{0}\right)$$

这里的 $\hat{I}_{0}$就是一个**可微分的“当前预测最终图像”**。

它不是完整运行 50-step DDIM 后的最终结果，但是它和 diffusion 当前 denoising field 完全相连，因此：

$$L_{LPIPS}\left(\hat{I}_{0},\ I\right)L_{GAN}\left(D\left(\hat{I}_{0}\right)\right)L_{ID}\left(\hat{I}_{0},\ I_{ref}\right)$$

都可以直接反向传播回 UNet。

而且**不需要把 50 个 DDIM step 全部展开反传。**

这个思路现在已经不只是理论推断。CVPR 2025 的 **Arbitrary-steps Image Super-resolution via Diffusion Inversion** 就直接把 L2、LPIPS 和 GAN loss 加到了 diffusion 的预测结果上；它甚至专门使用 latent-space LPIPS / discriminator 来降低内存消耗。(OpenAccess)

另外 Diffusion2GAN 也讨论了一个很实际的问题：SD 的 latent 如果先 VAE decode 到 512×512 再算 LPIPS，非常吃显存，因此他们提出了 **LatentLPIPS**，直接在 Stable Diffusion latent 上做 perceptual distance。([Minguk Kang](https://mingukkang.github.io/Diffusion2GAN/static/paper/diffusion2gan_arxiv_v2.pdf?utm_source=chatgpt.com))

所以你们现在最大的观念变化应该是：

**ε-MSE 不必是唯一损失。ε prediction 只是参数化方式，而不是限制你不能在** $\hat{x}_{0}$**上做 perceptual / adversarial / identity supervision。**

## 二、我认为和你们问题最相关的一篇：CVPR 2025 InvSR
**Arbitrary-steps Image Super-resolution via Diffusion Inversion**

Yue, Liao, Loy, CVPR 2025

这篇我建议你们**优先精读**。

论文使用预训练 diffusion prior 做图像恢复，而且训练中组合：

$$L=L_{2}+\lambda _{LPIPS}L_{LPIPS}+\lambda _{GAN}L_{GAN}$$

它最重要的意义不是 SR 本身，而是直接证明：

**pretrained diffusion + image/perceptual loss + adversarial loss 是可以共存的。**

而且它的 GAN 用 hinge loss，discriminator 作用于 diffusion 产生的重建结果。(OpenAccess)

这和你们的矛盾几乎一模一样：

你们：

$$\epsilon _{\theta}\rightarrow \hat{z}_{0}\rightarrow VAE\rightarrow \hat{I}_{0}\rightarrow D$$

他们本质也是：

$$diffusion prediction\rightarrow \hat{x}_{0}\rightarrow L_{perceptual}+L_{GAN}.$$

所以如果你问我：

有没有一篇比较“硬”的近期论文可以支持“给 diffusion 加 GAN/LPIPS 输出损失”？

**InvSR 是我目前最推荐你引用的一篇。** (OpenAccess)

## 三、另一条非常值得看的路线：Adversarial Diffusion

### 1. Adversarial Diffusion Distillation, 2023
Sauer et al. 的 **Adversarial Diffusion Distillation (ADD)** 把 diffusion teacher signal 和 adversarial loss 结合，让 student 在极少采样步下仍保持高图像保真度。(arXiv)

它的出发点很有启发性：

### Diffusion loss
负责：

mode coverage、semantic distribution、整体生成能力。

### Adversarial loss
负责：

high-frequency realism、锐利细节、真实图像 manifold。

这正好对应你现在的问题：

你的 ε-MSE 学到的是：

“这个区域大概长成什么”。

但是你缺的是：

“这个皮肤应该看起来像真实摄影皮肤，而不是 3D render skin。”

GAN discriminator 对这个问题反而极其擅长。

### 2. Adversarial Distribution Matching / DMDX, ICCV 2025
更近的是 ICCV 2025 的：

**Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis**

它甚至同时用了：

- latent-space discriminator
- pixel-space discriminator
并明确写出了对生成的 $\tilde{x}_{0}$做 adversarial supervision。(OpenAccess)

这篇对你们的启示非常直接：

discriminator 不一定非得判别最终 50-step DDIM output。

完全可以判：

$$\hat{x}_{0}\left(z_{t}\right)$$

或者：

$$\hat{z}_{0}$$

甚至两个都判。

## 四、但是你们这里有一个比“怎么加 GAN”更根本的问题
你现在描述的 target：

可见区域 = real photograph
空洞区域 = 3D GAN render

这在 diffusion finetuning 里其实非常危险。

因为标准 ε-MSE：

$$L_{\epsilon}=\| \epsilon -\epsilon _{\theta}\left(z_{t},\ t,\ c\right)\| ^{2}$$

本质上要求模型学习训练 target 的完整 data distribution。

所以对于 hole：

$$p_{train}\left(hole\right)\approx p_{3D-render}$$

你却希望 inference：

$$p_{output}\left(hole\right)\approx p_{photograph}.$$

目标本身就是矛盾的。

## 五、为什么 Diffusion Inpainting 没 GAN，也能生成“照片级纹理”？
这是你第二个问题的核心。

答案其实很重要：

**纹理通常不是从当前 inpainting GT 学来的，而是从大规模 pretrained diffusion prior 里来的。**

比如经典 **RePaint** 完全没有重新训练一个 GAN 来学习 hole texture。

它直接使用一个已经在真实图像上训练好的 DDPM：

$$p_{\theta}\left(x\right)$$

然后 inference 时只不断把已知区域重新注入 reverse diffusion。

所以：

### mask 外
来自 observed pixels。

### mask 内
来自：

$$pretrained natural-image diffusion prior$$

而不是：

$$paired masked-image GT texture$$

RePaint甚至指出，传统 pixel/perceptual reconstruction supervision 容易形成简单的纹理延伸，而 diffusion 的生成 prior 可以给极端 mask 生成更合理的内容。(OpenAccess)

这件事对你们非常关键。

## 六、你们现在可能恰恰把 SD1.5 最宝贵的东西教坏了
SD1.5 原来已经知道：

- 真实毛孔长什么样；
- 胡须的 stochastic high-frequency structure；
- 发丝如何交叉；
- skin highlight；
- subsurface-like skin appearance；
- 相机噪声；
- 真实皮肤的空间频率分布。
然后你用：

3D GAN render hole

作为 GT fine-tune。

模型于是慢慢学：

$$p_{photo}\rightarrow p_{render}$$

于是最后出现：

**塑料皮肤 / 油画感。**

从 diffusion 的角度看这并不奇怪。

它只是很好地完成了你的监督目标。

## 七、这个问题在文献里叫：fine-tuning destroys / shifts pretrained prior
最经典的来源之一就是：

### DreamBooth，CVPR 2023
DreamBooth明确观察到：

对少量 subject 数据持续 fine-tune，会逐渐破坏 pretrained class prior，降低原模型生成同一类别其他实例的能力。

因此提出：

$$L=L_{subject}+\lambda _{prior}L_{prior-preservation}.$$

其中 prior 数据直接由 frozen pretrained model 生成。(OpenAccess)

虽然 DreamBooth解决的是 personalization，但是你们的问题实际上更严重：

不是“小数据过拟合照片”，而是**错误域——render domain——在把 photorealistic prior 往 render domain 拉。**

## 八、2024 Diff-Tuning：更直接研究 diffusion fine-tuning 的“遗忘”
**Diffusion Tuning: Transferring Diffusion Models via Chain of Forgetting**

专门研究 pretrained diffusion 在 downstream fine-tuning 中的 transfer / forgetting，并发现沿 denoising trajectory 存在系统性的 forgetting behaviour；它提出专门的正则方式保持预训练知识。(arXiv)

这篇可以支撑你们论文里的一个核心论点：

我们不能简单 full-finetune SD1.5 来拟合含有 synthetic rendering artifacts 的 target，否则会破坏其 photographic prior。

## 九、所以我更建议你们采用一种“解耦监督”
我会把你们现在：

$$L=L_{\epsilon}$$

改成类似：

$$L=L_{diffusion}+\lambda _{structure}L_{structure}+\lambda _{photo}L_{photo}+\lambda _{ref}L_{ref}+\lambda _{prior}L_{prior}$$

它们各自承担完全不同的职责。

### ① ε-MSE：保留 diffusion training
继续：

$$L_{\epsilon}=\| \epsilon -\epsilon _{\theta}\| _{2}^{2}.$$

但是：

### real visible 区
正常监督。

### render hole
**不要再让它强监督 high-frequency appearance。**

可以：

$$L_{\epsilon}=L_{\epsilon}^{visible}+\lambda _{h}L_{\epsilon}^{hole}$$

其中：

$$\lambda _{h}≪1$$

甚至逐渐降到 0。

## 十、3D render 只负责结构，不应该负责 texture GT
这个设计我觉得尤其适合你的系统。

在 hole：

不要：

$$\hat{I}_{hole}\approx I_{GAN-render}$$

pixel-to-pixel。

改成约束：

$$LowPass\left(\hat{I}_{hole}\right)\approx LowPass\left(I_{render}\right)$$

或者约束：

- face parsing
- edge
- normal
- depth
- landmark
- low-frequency RGB
- coarse latent feature
即：

$$3Drender\rightarrow geometry/pose/structureteacher$$

而不是：

$$3Drender\rightarrow appearanceteacher.$$

这一点我认为是你们系统设计上最值得改的地方。

## 十一、照片级质感：在 X上加 masked discriminator
比如：

$$\hat{z}_{0}=\frac{z_{t}-\sqrt{1-\bar{\alpha }_{t}}\epsilon _{\theta}}{\sqrt{\bar{\alpha }_{t}}}$$

decode：

$$\hat{I}_{0}=VAE_{D}\left(\hat{z}_{0}\right).$$

然后：

$$I_{h}=M_{h}\odot  \hat{I}_{0}.$$

判别器不需要知道你这个 hole 对应的真实 GT 是什么。

### Real
来自真正的人脸照片：

$$I_{real}\sim p_{face-photo}.$$


### Fake
来自：

$$M_{h}\odot  \hat{I}_{0}.$$

于是：

$$D_{patch}:\left(\begin{matrix}I_{real}\rightarrow real \\ \hat{I}_{h}\rightarrow fake\end{matrix}\right)$$

这样它监督的是：

$$p\left(\hat{I}_{hole}\right)\approx p\left(I_{realface}\right)$$

而不是：

$$\hat{I}_{hole}=I_{render}.$$

**这就把你原来 GAN 系统最成功的东西重新引入 diffusion 了。**

## 十二、我甚至建议做 Local Patch discriminator，而不是整图 D
因为你们不是在解决：

“这是不是一张人脸？”

SD1.5早就会。

你真正要判的是：

“这块 64×64 皮肤像不像照片？”

“这个 hair patch 是发丝还是油画？”

“这个 cheek texture 是否有真实摄影统计？”

因此建议：

$$D_{skin}$$

和/或：

$$D_{patch}$$

输入随机 crop：

$$32\times 32, 64\times 64, 128\times 128.$$

Real crops：

真实照片上的：

- forehead
- cheek
- chin
- hair
Fake crops：

生成 hole 相同语义区域。

这个会比 full-face discriminator 更直接针对你们的问题。

## 十三、一个非常关键的训练细节：不要所有 t 都算 LPIPS/GAN
因为：

$$\hat{z}_{0}=\frac{z_{t}-\sqrt{1-\bar{\alpha }_{t}}\hat{\epsilon }}{\sqrt{\bar{\alpha }_{t}}}$$

当 $t$很大时：

$$\bar{\alpha }_{t}\rightarrow 0.$$

所以：

$$\frac{1}{\sqrt{\bar{\alpha }_{t}}}$$

会把 ε prediction error 极度放大。

这时的 $\hat{x}_{0}$本来就很不稳定。

如果你：

t = 900
decode $\hat{x}_{0}$
LPIPS + GAN

梯度很可能很脏。

因此我建议：

### ε-MSE
仍然：

$$t\sim U\left(0,\ T\right)$$

全范围训练。

### image-space loss
只在：

$$t<t_{perceptual}$$

时启用。

比如先实验：

$$t<200, 300, 500.$$

或者按 SNR：

$$SNR\left(t\right)>\tau .$$


## 十四、参考图纹理迁移：你真正应该重点看的文献
这个是你的第 4 个问题，而且可能比 GAN loss 更重要。

### 1. Paint by Example — CVPR 2023
这是 diffusion **reference/exemplar-guided inpainting** 的代表论文。

它不是只给 text condition，而是：

$$I_{reference}\rightarrow imageencoder\rightarrow condition\rightarrow diffusioninpainting.$$

并通过信息瓶颈和强 augmentation 防止模型退化成简单 copy-paste。(OpenAccess)

它非常适合回答：

“怎么把 reference appearance 输入 missing region？”

## 十五、IP-Adapter：我认为你们工程上最值得借鉴

### IP-Adapter
核心是：

$$F_{ref}=E_{image}\left(I_{ref}\right)$$

然后通过独立 image cross-attention：

$$Attention\left(Q,\ K_{image},\ V_{image}\right)$$

注入 SD UNet。

而原来的 text cross-attention 保留。

关键优势是：

**base diffusion 可以 freeze。**

IP-Adapter仅约 22M 参数，并通过 decoupled cross-attention 接入 image prompt。(arXiv)

这直接解决你第三个问题：

### “如何防止 fine-tuning 教坏 pretrained prior？”
答案之一就是：

$$Don’t fine-tune the photographic prior aggressively.$$

可以：

$$freeze SD1.5UNet$$

主要训练：

- condition encoder
- projection layer
- reference attention
- ControlNet branch
- LoRA

## 十六、AnyDoor — CVPR 2024：与你的“纹理迁移”非常接近
这篇我强烈推荐。

它明确区分：

### identity feature
“这个对象是谁”

和

### detail feature
“这个对象具体长什么纹理”。

论文指出，仅 identity feature 不足以保持 appearance details，所以专门增加了 detail features，用于保存：

- texture
- local appearance
- fine detail
同时允许：

- pose
- orientation
- lighting
变化。(OpenAccess)

这恰好就是你的问题：

参考图：

正脸真实 skin/hair texture

目标：

新视角

你不希望：

$$copy pixels$$

而希望：

$$preserve texture identity+allow viewpoint-dependent transformation.$$

AnyDoor 这个思想非常值得移植。

## 十七、对于“人脸 identity”，PVA 比普通 IP-Adapter 更贴近你
**Personalized Face Inpainting with Diffusion Models by Parallel Visual Attention**

WACV 2024

它专门解决：

$$reference face+masked face\rightarrow identity-preserving inpainting.$$

方法是在每个 cross-attention block 里增加：

$$Parallel Visual Attention$$

让 denoising features attend 到 reference identity features。(OpenAccess)

相比“reference embedding concat 到 condition”这种简单方法，它更加直接。

而且任务与你高度一致：

### face + reference + hole completion + diffusion
所以这是你的核心 related work 之一。

## 十八、InstantID — 很值得拿来解决 identity，而不是 texture
InstantID把：

- Face ID embedding
- facial landmarks
- text
一起作为 condition 注入 diffusion。

它主打的是**单张 reference、zero-shot identity preservation**，并兼容 SD1.5 / SDXL。(arXiv)

但需要注意：

$$identity\ne texture$$

ArcFace / FaceID 类 feature 很擅长保持：

- 五官
- face shape
- identity
却不一定会保存：

- 毛孔
- 局部皮肤纹理
- 发丝 pattern
- 痣/雀斑
- 局部光泽
所以我不建议只加 FaceID loss。

## 十九、最新一些：ConsistentID，TPAMI 2026
这篇已经正式到 TPAMI 2026：

**ConsistentID: Portrait Generation With Multimodal Fine-Grained Identity Preserving**

它明确指出，仅用 global ID embedding 不能很好保留**fine-grained facial details**，所以增加：

- localized facial features
- facial descriptions
- global facial context
- facial attention localization
来保存细粒度 identity。([PubMed](https://pubmed.ncbi.nlm.nih.gov/41525591/?utm_source=chatgpt.com))

这个和你的“毛孔/皮肤/局部 facial attributes”问题非常接近。

## 二十、2026 ReSem-Face：大遮挡尤其值得看
刚出的：

**When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions**

专门考虑：

large face occlusion / large missing region 下 reference identity 信息丢失。

它使用多 reference 提取 identity-conditioned semantic prior，再通过多 stream conditioning 注入 diffusion。(arXiv)

你的 3D projection hole 如果比较大，这篇值得重点跟。

## 二十一、Reference-Guided Face Restoration 2025
还有：

### Reference-Guided Identity Preserving Face Restoration
提出：

Composite Context

同时融合 reference 的：

$$high-level+low-level$$

信息，而不是只用一个 global embedding。

还设计了 Hard Example Identity Loss。(arXiv)

这篇其实也支持我前面说的：

**reference 要拆成 identity + appearance/detail 两类信息。**

## 二十二、不要直接对“新视角 reference”做 pixel LPIPS
这里有一个坑。

假设 reference 是正脸：

$$I_{r}$$

你生成侧脸：

$$\hat{I}_{t}.$$

直接：

$$LPIPS\left(\hat{I}_{t},\ I_{r}\right)$$

是不合理的。

因为：

- geometry不同；
- eye位置不同；
- hair投影不同；
- shading不同。
于是 LPIPS 会把：

$$geometry difference$$

错误当成：

$$appearance error.$$


## 二十三、Reference texture 更好的做法
我更建议分 3 层。

### Level 1：identity
ArcFace / AdaFace 类：

$$L_{ID}=1-cos\left(E_{face},\ \left(\hat{I}\right)E_{face}\left(I_{ref}\right)\right).$$

负责：

是不是同一个人。

### Level 2：局部 semantic correspondence
利用：

- face parsing
- UV
- 3D correspondence
- DINO patch feature
建立：

$$cheek_{ref}\leftrightarrow cheek_{target}forehead_{ref}\leftrightarrow forehead_{target}.$$

再比较 feature。

这对你们特别有优势：

**你们本来就有 3D。**

所以你们其实比普通 reference-guided diffusion 多一个非常强的信息：

$$3D correspondence$$

可以从 reference view 把 visible texture 投到目标视角。

## 二十四、这个地方甚至可能成为你们自己的创新点
因为一般的 IP-Adapter：

$$I_{ref}\rightarrow global/referencetokens\rightarrow UNet.$$

它不知道：

reference 左脸上的这个 patch，对应目标视角的哪块脸。

而你们有 3D geometry。

完全可以得到：

$$reference pixel3Dtarget surface$$

于是做：

### Geometry-aware Reference Attention
query：

$$Q\left(x,\ y\right)$$

只 attend 到 3D correspondence 附近的：

$$K_{ref}\left(u,\ v\right),V_{ref}\left(u,\ v\right).$$

这会比普通 IP-Adapter 强得多。

## 二十五、皮肤纹理和头发纹理最好也别完全放一起
因为二者性质不同。

### Skin
更接近：

- stationary/local texture
- skin pores
- high-frequency stochastic statistics
很适合：

$$PatchGAN+LPIPS+localfeaturestatistics.$$


### Hair
则高度：

- orientation-sensitive
- structural
- long-range correlated
不能只靠一个 32×32 PatchGAN。

需要：

- reference feature attention
- segmentation
- orientation/edge structure
- diffusion prior
一起约束。

所以：

$$L_{skin}\ne L_{hair}$$

其实可能是值得尝试的。

## 二十六、我给你一个最推荐的整体方案
如果现在让我直接改你们系统，我会从：

$$SD1.5 base尽量冻结$$

开始。

训练：

$$Control branch+Reference Adapter+LoRA / selected UNet layers$$

而不是 full fine-tune。

网络：

$$I_{proj},M\rightarrow E_{geometry}I_{ref}\rightarrow E_{ID},E_{detail}$$

共同送：

$$UNet\left(z_{t},\ t,\ F_{geometry},\ F_{ID},\ F_{detail}\right).$$


### Loss 我会设计成：

$$L=L_{\epsilon}+\lambda _{s}L_{structure}+\lambda _{p}L_{LPIPS}+\lambda _{a}L_{adv}+\lambda _{i}L_{ID}+\lambda _{r}L_{ref-detail}+\lambda _{prior}L_{preserve}$$

其中：

$$L_{\epsilon}$$

保证 diffusion trajectory。

$$L_{structure}$$

只让 3D GAN render 教：

- geometry
- coarse color
- illumination
- silhouette。

$$L_{LPIPS}$$

在：

$$\hat{x}_{0}$$

上算，并主要限制低/中 timestep。

$$L_{adv}$$

真实照片 patches vs hole generated patches。

$$L_{ID}$$

reference identity。

$$L_{ref-detail}$$

reference local feature / texture consistency。

$$L_{preserve}$$

保持原始 SD1.5 photographic prior。

## 二十七、一个很重要的 ablation 顺序
不要一下把 GAN、LPIPS、ArcFace、DINO、reference attention 全塞进去。

我建议按这个顺序：

1. **Baseline**：当前 ε-MSE。
2. **降低/移除 hole render ε supervision**，看看 plastic texture 是否明显改善。
3. **Freeze 大部分 SD / LoRA-only**，验证 pretrained photo prior 是否被保护。
4. 增加 $\hat{x}_{0}$-space **masked LPIPS**。
5. 增加 **local adversarial loss**。
6. 增加 **reference ID attention**。
7. 增加 **reference detail/3D correspondence attention**。
这个实验顺序会非常有解释力。

## 二十八、我给你筛出的文献优先级
如果时间有限，我建议先读下面这些：

| **优先级** | **文献** | **对你最有用的点** |
|---|---|---|
| ★★★★★ | **Arbitrary-steps Image Super-resolution via Diffusion Inversion, CVPR 2025** | diffusion 上直接加 LPIPS + GAN |
| ★★★★★ | **Personalized Face Inpainting with Diffusion Models by Parallel Visual Attention, WACV 2024** | reference face → diffusion inpainting |
| ★★★★★ | **AnyDoor, CVPR 2024** | identity feature + detail texture feature |
| ★★★★★ | **DreamBooth, CVPR 2023** | prior preservation / 防止微调破坏 prior |
| ★★★★☆ | **Adversarial Diffusion Distillation, 2023** | adversarial + diffusion prior |
| ★★★★☆ | **Adversarial Distribution Matching, ICCV 2025** | latent/pixel discriminator on diffusion output |
| ★★★★☆ | **Paint by Example, CVPR 2023** | exemplar/reference-guided inpainting |
| ★★★★☆ | **IP-Adapter, 2023** | freeze base diffusion + reference attention |
| ★★★★☆ | **InstantID, 2024** | face ID + spatial facial condition |
| ★★★★☆ | **ConsistentID, TPAMI 2026** | fine-grained facial identity/detail |
| ★★★★☆ | **ReSem-Face, 2026** | large-hole reference face inpainting |
| ★★★☆☆ | **RePaint, CVPR 2022** | 纹理来自 pretrained generative prior 而非 paired hole GT |
| ★★★☆☆ | **Diff-Tuning, 2024** | diffusion fine-tuning forgetting |

## 二十九、最重要的三篇，我建议你现在就下载精读
**① InvSR — CVPR 2025**
它直接回答你的第一个问题：

diffusion 怎样挂 LPIPS / GAN output loss。 (OpenAccess)

**② PVA Face Inpainting — WACV 2024**
直接回答：

reference face 怎样引导 diffusion 补全。 (OpenAccess)

**③ AnyDoor — CVPR 2024**
直接回答：

reference 的“identity”和“texture detail”为什么必须分开建模。 (OpenAccess)

然后用 **DreamBooth / Diff-Tuning** 来解决：

为什么不能把 SD1.5 photographic prior 微调坏。(OpenAccess)

### 最后，我对你们当前问题的判断
你们遇到的“油画/塑料”可能**并不主要是 diffusion 生成能力不足**。

更可能是：

真正应该做的是：

$$render告诉模型“这里是什么、几何应该怎样”$$

但让：

$$SD prior + real-photo distribution告诉模型“真实皮肤/头发应该长什么样”$$

再让：

$$reference image告诉模型“这个人的皮肤/头发具体应该是什么样”.$$

这三个职责一旦拆开，你这个问题就会从“怎样让 ε-MSE 学会照片纹理”，变成一个更合理的 **geometry-conditioned + reference-conditioned + photo-prior-preserving diffusion inpainting** 问题。我认为这条路线比继续调纯 ε-MSE 的权重、mask 权重或者 timestep sampling 要有潜力得多。
