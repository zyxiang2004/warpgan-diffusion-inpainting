[License: arXiv.org perpetual non-exclusive license](https://info.arxiv.org/help/license/index.html#licenses-available)

arXiv:2608.04820v1 \[cs.CV\] 05 Aug 2026

# When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions

Feng Ding    Shuhuai Xie    Yue Zhou    Yulan Zhang    Guopu Zhu    Mengyao Xiao

###### Abstract

Face inpainting with diffusion models has recently achieved impressive visual quality, yet preserving identity fidelity under significant occlusion and conflicting text guidance remains a major challenge. To address this issue, we present Reference Semantic Inpainting for Face (ReSem-Face), a cascaded diffusion framework that introduces an explicit identity-conditioned semantic prior for multi-reference face inpainting. Our approach distills representative identity features from multiple references to reconstruct missing semantic regions, which then guide the diffusion process through a multi-stream conditioning architecture. This design provides strong semantic constraints when pixels are absent and stabilizes identity reconstruction while remaining compatible with prompt-driven edits. Experiments on CelebAHQ-IDI-5 and VGGFace2 demonstrate that ReSem-Face yields more reliable identity-preserving completion under severe semantic masks and improves text-controlled editing quality compared with representative baselines.

Nanchang University, Shenzhen University, Huizhong University, Harbin Institute of Technology

## Introduction

Face inpainting ([Ju et al. 2024](#bib.bib11); [Zhuang et al. 2024](#bib.bib12); [Xiao et al. 2025](#bib.bib49)) refers to the task of completing missing or occluded facial regions while preserving the identity of a specific person and maintaining natural consistency with the visible context (e.g., expression, and background). Unlike generic inpainting, the goal is not only to produce a plausible face but to reconstruct the same face: subtle geometry and fine-grained textures that make a person recognizable must be retained even when key identity cues are severely removed by the mask.

Early works typically build on convolutional or GAN-based inpainting pipelines([Cai et al. 2019](#bib.bib14); [Li et al. 2017](#bib.bib13); [Goodfellow et al. 2014](#bib.bib45)), sometimes augmented with facial priors such as landmarks, parsing maps, or multi-scale discriminators to enhance realism and structural plausibility. These methods achieve visually coherent completions under mild to moderate occlusions and often produce sharp textures. However, when the missing region is large or covers identity-critical areas (e.g., eyes, nose, and mouth), they tend to hallucinate person-agnostic details, leading to identity drift; moreover, their reliance on local context makes them brittle under expression variations and hard to extend to controllable edits.

It is worth noting that diffusion models([Zhang et al. 2023](#bib.bib17); [Wu et al. 2023](#bib.bib48); [Zhao et al. 2023](#bib.bib47); [Zhu et al. 2024](#bib.bib46); [Wasserman et al. 2025](#bib.bib18); [Xie et al. 2025](#bib.bib20)) have recently advanced image synthesis and editing by progressively denoising from noise to data, offering strong generative priors and impressive fidelity in texture and global consistency. Motivated by this, diffusion-based inpainting frameworks ([Avrahami et al. 2022](#bib.bib15); [Avrahami et al. 2023](#bib.bib16); [Zhang et al. 2023](#bib.bib17); [Kim et al. 2025](#bib.bib19); [Huang et al. 2025](#bib.bib21)) have emerged as a promising direction for face restoration, and reference-guided variants further attempt to inject identity information from one or more reference images ([Luo et al. 2023](#bib.bib8); [Xu et al. 2024](#bib.bib9))so that the denoising process can reconstruct the correct subject rather than an average face.

Despite the success of reference-guided diffusion and personalized generation techniques, robust identity preservation under heavy occlusion remains challenging. In practice, reference images are often misaligned with the target, as the masked target offers little direct evidence for identity. More importantly, many existing designs inject identity cues implicitly during denoising and expect them to simultaneously determine who the subject is and what content should fill the missing region([Ruiz et al. 2023](#bib.bib43); [Li et al. 2024](#bib.bib44); [Gal et al. 2023](#bib.bib55); [Kumari et al. 2023](#bib.bib10)); under high uncertainty and noise, this coupling can weaken identity signals, and the problem becomes even harder when text-guided editing ([Zhang et al. 2020b](#bib.bib25); [Nichol et al. 2021](#bib.bib22); [Zhang et al. 2020a](#bib.bib24); [Xie et al. 2023](#bib.bib23); [Chen et al. 2024](#bib.bib26)) is introduced, where semantic instructions may inadvertently override or distort identity-specific details.

Motivated by the aforementioned limitations, we propose Reference Semantic Inpainting for Face (ReSem-Face), a diffusion-based ([Rombach et al. 2022](#bib.bib28)) personalized face inpainting framework that introduces an explicit identity-conditioned semantic prior to stabilize identity reconstruction under severe occlusion. Specifically, we design an Identity-Aware Semantic Pre-Inpainting module that aggregates cues from multiple references in a clean feature space and predicts semantic tokens describing the missing region’s identity-relevant structure and appearance; these tokens are then injected into the diffusion U-Net via dedicated semantic attention, operating alongside text guidance and an introduced Reference Identity Attention pathway to form complementary constraints. By decoupling identity-related semantics from the noisy denoising trajectory, ReSem-Face enables more faithful identity preservation and more reliable controllable editing, especially in challenging large-mask scenarios.

In summary, our contributions are as follows:

- •
  We propose a novel Identity-Aware Semantic Pre-Inpainting Module that predicts identity-aware geometry and texture semantics for missing facial regions, forming a high-level semantic prior for diffusion-based inpainting.
- •
  We integrate this semantic prior into the diffusion U-Net through a Reference Semantic Attention pathway, which works together with text cross-attention and Reference Identity Attention to enable tri-conditioning with text, identity, and semantic guidance.
- •
  Extensive experiments demonstrate that ReSem-Face achieves state-of-the-art identity preservation and strong controllability in both standalone face inpainting and text-guided editing scenarios.

## Related Work

### Face Inpainting

The past five years have witnessed rapid progress in face inpainting, driven by the demand to realistically recover occluded facial regions while keeping identity and fine-grained attributes ([Wang et al. 2021](#bib.bib27); [Xu et al. 2024](#bib.bib9); [Suvorov et al. 2022](#bib.bib5)). Early works focus on identity-guided completion under heterogeneous domains, typically combining staged fitting and refinement to better align the synthesized face with the surrounding context ([Li et al. 2021](#bib.bib1); [Luo et al. 2023](#bib.bib8)). To this end, Mask-aware transformer (MAT) ([Li et al. 2022](#bib.bib2); [Vaswani et al. 2017](#bib.bib42)) utilizes transformer-style long-range modeling to fill large holes with coherent global structure at high resolution. Due to the development of deep generative models, diffusion-based priors have been adopted to improve semantic plausibility and output diversity without retraining for specific mask types ([Lugmayr et al. 2022](#bib.bib3); [Chen et al. 2024](#bib.bib26); [Luo et al. 2023](#bib.bib8); [Rombach et al. 2022](#bib.bib28)). Some works ([Dong et al. 2022](#bib.bib4); [Luo et al. 2023](#bib.bib8)) propose to explicitly restore structural cues (e.g., sketches and edges) and inject them into texture completion, strengthening geometry–texture consistency. Furthermore, ([Suvorov et al. 2022](#bib.bib5)) proposes to use Fourier-based global receptive fields to better handle large irregular masks and high-resolution images. However, despite these advances, person-specific fidelity can still drift—subtle identity cues may change when the missing region is large—motivating more identity-aware face inpainting designs ([Motamed et al. 2023](#bib.bib6)).

### Reference-guided Face Inpainting

A widely adopted and highly effective solution to improve identity fidelity in face inpainting is to condition the model on one or multiple reference images of the same subject([Zhou et al. 2021](#bib.bib29); [Luo et al. 2023](#bib.bib8); [Motamed et al. 2023](#bib.bib6); [Varanka et al. 2024](#bib.bib30)), which provide identity-critical cues that are absent in heavily masked targets. In early representative efforts, reference-attention–based designs ([Yu et al. 2022](#bib.bib7)) explicitly align and fuse reference features with the corrupted input, encouraging the completion to resemble the reference identity more faithfully. Following that, dual-control formulations further disentangle the reference signal into high-level identity and low-level texture ([Luo et al. 2023](#bib.bib8)), enabling more controllable and higher-quality completion under large-scale missing regions. PVA (Xu et al. 2024) integrates Parallel Visual Attention into a pretrained diffusion inpainting model, injecting reference-image features into the denoising network to achieve identity-preserving and language-controllable face inpainting with lightweight per-identity tuning. Recent works (Yang et al. 2023; Chen et al. 2024a; Motamed et al. 2023) also conduct reference guidance via personalization-by-tuning and diffusion-based exemplar conditioning for stronger reference-driven generation and editing; meanwhile, reference-guided methods have also been extended to directional and diverse face inpainting to produce multiple plausible yet reference-consistent outcomes. Nevertheless, most existing reference-guided methods mainly introduce identity cues as denoising-time conditions, while the semantic content of heavily occluded regions is still inferred implicitly during diffusion. In contrast, our ReSem-Face explicitly predicts identity-aware semantic priors before diffusion denoising, providing additional geometry and texture constraints for large-mask identity reconstruction.

## Methods

We begin by noting that, in multi-reference identity-preserving face inpainting, a plausible synthesis needs to be generated for a large missing facial region while remaining faithful to the identity of a specific person provided by a set of reference images. This setting becomes even more challenging when the completion is further guided by a text prompt for attribute editing, since identity consistency and prompt compliance can be conflicting under severe occlusions. In this section, we will delve into the details of our proposed ReSem-Face framework, and the pipeline of ReSem-Face is shown in Fig. [1](#Sx3.F1 "Figure 1 ‣ Multi-Reference Identity Semantic Aggregation ‣ Identity-Aware Semantic Pre-Inpainting Module ‣ Methods ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"). We first provide a brief review of latent diffusion inpainting, and then present the identity-aware semantic pre-inpainting module, the Reference Identity Attention pathway, and the ReSemAttn-based injection scheme for semantic priors, followed by the training objectives and implementation details.

### Preliminaries

Diffusion-based generative models view image synthesis as learning to reverse a gradual noising process. Starting from a clean sample $x_{0}$, the forward process constructs a sequence $\{x_{t}\}_{t=1}^{T}$ by repeatedly injecting Gaussian noise with a variance schedule $\{\beta_{t}\}_{t=1}^{T}$:

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
x_{t+1}=\sqrt{1-\beta_{t}}\,x_{t}+\sqrt{\beta_{t}}\,\epsilon,\qquad\epsilon\sim\mathcal{N}(0,I),
``` |  | (1) |

where $0<\beta_{t}<1$. After sufficiently many steps, $x_{t}$ becomes nearly Gaussian and the original data is largely destroyed. By composing the forward transitions, one can directly relate $x_{t}$ to $x_{0}$ in closed form:

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
x_{t}=\sqrt{\bar{\alpha}_{t}}\,x_{0}+\sqrt{1-\bar{\alpha}_{t}}\,\epsilon_{t},
``` |  | (2) |

with $\bar{\alpha}_{t}=\prod_{i=1}^{t}(1-\beta_{i})$ denoting the cumulative signal decay. The generative model is then defined by a denoising network $\epsilon_{\theta}(x_{t},t)$ that predicts the noise added at each step. It is trained using a denoising score-matching objective

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
\mathcal{L}_{\mathrm{DSM}}=\mathbb{E}_{x_{0},t,\epsilon\sim\mathcal{N}(0,I)}\big[\|\epsilon-\epsilon_{\theta}(x_{t},t)\|_{2}^{2}\big],
``` |  | (3) |

where $t$ is sampled uniformly from $\{1,\dots,T\}$ and $x_{t}$ is obtained from $x_{0}$ via the forward process. At inference time, the model approximately inverts the diffusion chain, starting from Gaussian noise and iteratively denoising it. In practice, we adopt a DDIM-style sampler for efficient generation([Song et al. 2020](#bib.bib32)).

Running this procedure directly in pixel space is computationally demanding. Latent Diffusion Models (LDMs)([Rombach et al. 2022](#bib.bib28)) address this limitation by applying the same diffusion formulation in a compressed latent representation. Let $E_{V}(\cdot)$ and $D_{V}(\cdot)$ denote the encoder and decoder of a Variational Auto-Encoder (VAE) ([Kingma and Welling 2013](#bib.bib33)), and $z_{t}=E_{V}(x_{t})$, $x_{t}=D_{V}(z_{t})$ be the encoding and decoding operations. The diffusion model is then defined over latent variables $z_{t}$ instead of raw images. Moreover, LDMs are typically conditioned on text through cross-attention layers that attend to language features $y_{i}=E_{T}(T_{i})$ extracted by a pretrained text encoder (e.g., CLIP([Radford et al. 2021](#bib.bib34))). For inpainting, the latent model also receives the occluded image and the corresponding binary mask as additional inputs. Concretely, we form a mask-aware latent input

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
\tilde{z}_{t}=z_{t}\,\|\,u_{\downarrow}(m)\,\|\,E_{V}(m\odot x_{0}),
``` |  | (4) |

where $m$ is the mask, $u_{\downarrow}(m)$ denotes downsampling $m$ to the latent resolution, $\odot$ is element-wise multiplication, and $\|\,$ indicates channel-wise concatenation. The inpainting LDM is then trained with a noise-prediction loss analogous to the DDPM objective:

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
\mathcal{L}_{\mathrm{LDM}}=\mathbb{E}_{z_{0},y,m,t,\epsilon}\big[\|\epsilon-\epsilon_{\theta}(\tilde{z}_{t},y,t)\|_{2}^{2}\big],
``` |  | (5) |

where $z_{0}=E_{V}(x_{0})$ and $y$ denotes the text conditioning. Our method builds on top of such a latent diffusion inpainting backbone, which jointly exploits the text prompt, the masked image, and the mask to guide the denoising process.

### Identity-Aware Semantic Pre-Inpainting Module

To provide the diffusion backbone with high-level priors describing the latent structure of the occluded facial region, we introduce an Identity-Aware Semantic Pre-Inpainting Module. This module forms an additional semantic pathway complementary to Reference Identity Attention, producing identity-conditioned semantic tokens that encode both geometry and fine-grained appearance. The overall design follows a cascaded ([Vaswani et al. 2017](#bib.bib42)) architecture reminiscent of CAT-Diffusion but differs fundamentally in its purpose, semantic targets, and integration strategy. The full framework of ReSem-Face is shown in Fig. [1](#Sx3.F1 "Figure 1 ‣ Multi-Reference Identity Semantic Aggregation ‣ Identity-Aware Semantic Pre-Inpainting Module ‣ Methods ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions")

#### Multi-Reference Identity Semantic Aggregation

Given a set of reference images $\mathcal{R}_{p}=\{x_{i}^{r}\}_{i=1}^{N_{p}}$ from the same identity, we first extract per-reference identity-aware tokens using a frozen identity encoder $E_{\mathrm{id}}$, where the FaceNet branch is implemented as a ResNet-50 model trained with ArcFace ([Deng et al. 2019](#bib.bib36)). To suppress intra-identity variations, we perform cross-image token interaction via multi-head attention over all reference tokens, aggregating them into a unified identity semantic bank:

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
\mathbf{Z}^{r}=\mathrm{Norm}\!\left(\frac{1}{N_{p}}\sum_{i=1}^{N_{p}}\mathrm{MSA}\!\big(\mathbf{Z}^{r}_{i},\mathbf{Z}^{r}_{1:N_{p}},\mathbf{Z}^{r}_{1:N_{p}}\big)\right),
``` |  | (6) |

where $\mathbf{Z}^{r}_{i}=E_{\mathrm{id}}(x_{i}^{r})$, and $\mathbf{Z}^{r}$ serves as a consistent identity memory queried by the masked target in subsequent stages.

![Refer to caption](./5e0b5d265169888a00b7598170f9b26d5e9e0ff1.png)

Figure 1: The pipeline consists of three main components: (1) Identity Semantic Bank Construction, which aggregates identity features from multiple reference images; (2) Semantic Pre-Inpainting, where a semantic inpainter predicts semantic tokens for the masked region in the clean feature space; and (3) Tri-Conditioned Diffusion, where the predicted tokens are injected into the denoising backbone via the Reference Semantic Attention (ReSemAttn) pathway, which operates in parallel with text cross-attention and Reference Identity Attention to guide the generation.

#### Identity-conditioned semantic pre-inpainting

Given a masked target image $x^{t}$ and its binary mask $m$, we encode the visible context with a CLIP-based mask-aware visual encoder $E_{\mathrm{vis}}$ to obtain target tokens $\mathbf{Z}^{t}$. We then fuse $\mathbf{Z}^{t}$ with the identity bank $\mathbf{Z}^{r}$ through multi-head cross-attention, so that identity cues can be propagated into partially observed or fully missing regions. The fused representation is decoded by a lightweight transformer decoder with two heads: (i) a geometry head that predicts structure-oriented semantics (e.g., component layout and shape consistency), and (ii) a texture head that predicts identity-dependent appearance traits. Their outputs are finally projected into a fixed-length sequence of identity-conditioned semantic tokens $\mathbf{S}_{p}$ for the occluded region.

#### Identity-aware semantic supervision

To ensure $\mathbf{S}_{p}$ captures both semantic correctness and identity consistency, we supervise the geometry/textural predictions and enforce identity alignment using features extracted from the ground-truth region $x^{\mathrm{gt}}$. In addition, we adopt a frozen teacher model to provide stronger semantic targets and distill high-level knowledge into $\mathbf{S}_{p}$, which stabilizes semantic prediction under large occlusions and improves prompt alignment:

|  |  |  |  |  |
|----|----|----|----|----|
|  | $\displaystyle\mathcal{L}_{\mathrm{pre}}$ | $\displaystyle=\lambda_{\mathrm{id}}\!\left(1-\cos\!\left(\phi_{\mathrm{id}}(x^{\mathrm{gt}}),\,\phi_{\mathrm{id}}(\mathbf{S}_{p})\right)\right)+\lambda_{\mathrm{geo}}\left\|\phi_{\mathrm{geo}}(x^{\mathrm{gt}})-\mathbf{G}\right\|_{1}$ |  | (7) |
|  |  | $\displaystyle+\lambda_{\mathrm{tex}}\left\|\phi_{\mathrm{tex}}(x^{\mathrm{gt}})-\mathbf{T}\right\|_{1}+\lambda_{\mathrm{t}}\left\|\psi_{\mathrm{t}}(x^{\mathrm{gt}})-\psi_{\mathrm{t}}(\mathbf{S}_{p})\right\|_{1}.$ |  |  |

where $\phi_{\mathrm{id}},\phi_{\mathrm{geo}},\phi_{\mathrm{tex}}$ extract identity/geometry/texture descriptors, $\mathbf{G}$ and $\mathbf{T}$ denote the geometry/texture head outputs, and $\psi_{\mathrm{t}}(\cdot)$ denotes teacher features used for distillation([Navaneet et al. 2022](#bib.bib31)).

#### Reference Identity Attention

To complement the semantic prior with direct identity evidence, we introduce a Reference Identity Attention (RIA) pathway for identity-conditioned denoising. Given the aggregated reference identity tokens $\mathbf{H}_{p}$, RIA lets the intermediate U-Net features selectively retrieve identity-specific cues from the reference set and injects them into the denoising process as an identity guidance branch. Different from the semantic pathway, which predicts explicit geometry and texture priors for the missing region, RIA focuses on preserving subject-level identity consistency during generation. In this way, identity retrieval and semantic prior injection are modeled as two complementary conditions within the tri-conditioning framework.

#### Reference Semantic Attention for Diffusion Conditioning

We inject the predicted semantic tokens $\mathbf{S}_{p}$ into the diffusion U-Net via a dedicated Reference Semantic Attention (ReSemAttn) pathway. At each transformer block, ReSemAttn operates in parallel with text cross-attention and Reference Identity Attention (RIA), providing complementary constraints to guide denoising:

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
\begin{split}\mathbf{Z}_{\ell+1}&=\mathrm{SelfAttn}(\mathbf{Z}_{\ell})+\mathrm{CrossAttn}_{\mathrm{text}}(\mathbf{Z}_{\ell},\mathbf{Y})\\
&\quad+\mathrm{RIA}(\mathbf{Z}_{\ell},\mathbf{H}_{p})+\mathrm{ReSemAttn}(\mathbf{Z}_{\ell},\mathbf{S}_{p}),\end{split}
``` |  | (8) |

where $\mathbf{Z}_{\ell}$ is the hidden state at block $\ell$, $\mathbf{Y}$ is the text condition, and $\mathbf{H}_{p}$ denotes the reference identity tokens. This tri-pathway conditioning encourages identity-faithful and semantically coherent completion, especially when visible evidence is sparse.

### Training

Following the standard latent diffusion inpainting formulation, ReSem-Face optimizes a diffusion-based inpainting objective augmented with reference identity conditioning and identity-aware semantic supervision.

#### Diffusion Loss

We adopt the standard latent diffusion objective, where the model predicts noise added at timestep $t$. Given masked latent $\tilde{\mathbf{z}}_{t}$, text prompt $\mathbf{y}$, reference identity tokens $\mathbf{H}_{p}$, and semantic tokens $\mathbf{S}$, the denoising loss is:

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
\mathcal{L}_{\mathrm{diff}}=\mathbb{E}_{\mathbf{z}_{0},t,\epsilon}\left[\left\|\epsilon-\epsilon_{\theta}(\tilde{\mathbf{z}}_{t},t,\mathbf{y},\mathbf{H}_{p},\mathbf{S})\right\|_{2}^{2}\right].
``` |  | (9) |

This objective aligns all three conditioning streams (text, identity, semantic) with the diffusion trajectory.

#### Identity Preservation Loss

To ensure that personalized details are preserved in the generated face, we introduce an *identity-consistency loss* between the reconstructed image $\hat{x}$ and the ground-truth $x^{\mathrm{gt}}$. Using a pretrained identity extractor $\phi_{\mathrm{id}}$, we compute:

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
\mathcal{L}_{\mathrm{id}}=1-\cos\!\left(\phi_{\mathrm{id}}(\hat{x}),\phi_{\mathrm{id}}(x^{\mathrm{gt}})\right).
``` |  | (10) |

This constraint guides the diffusion model toward identity-faithful synthesis, especially in large-missing-region cases.

#### Reference Semantic Loss

Our Identity-Aware Semantic Pre-Inpainting Module predicts geometry and texture semantic tokens $(G,T)$ for the occluded region. To align predicted semantics with the ground truth region’s structure and appearance, we propose a *Reference Semantic Loss*:

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
\mathcal{L}_{\mathrm{sem}}=\lambda_{1}\left\|\phi_{\mathrm{geo}}(x^{\mathrm{gt}})-G\right\|_{1}+\lambda_{2}\left\|\phi_{\mathrm{tex}}(x^{\mathrm{gt}})-T\right\|_{1}.
``` |  | (11) |

These terms supervise the semantic branch to produce accurate high-level priors before diffusion denoising.

## Experiments

### Experiment Settings

#### Dataset and Pre-processing

We conduct experiments on CelebAHQ-IDI-5 ([Xu et al. 2024](#bib.bib9)), a tailored benchmark for multi-reference face inpainting containing 1,963 identities, and VGGFace2 ([Cao et al. 2018](#bib.bib52)), a large-scale dataset comprising over 3.3 million images across 9,131 identities. Unlike the aligned CelebA-HQ, VGGFace2 serving as a challenging testbed to evaluate our model’s generalization capability in unconstrained "in-the-wild" scenarios. Following standard protocols, we utilize 5 reference images per identity and evaluate on unseen identities. To mimic real-world occlusions, we apply semantic masks including lower-face, eye&brow, whole-face, and random regions. The input consists of a masked target image, the corresponding binary mask, and the reference set. All images are kept with original alignment and resized to the diffusion backbone’s input resolution. During training, we randomly sample masks to enhance robustness, while evaluation is performed on specific mask categories to assess performance under diverse occlusion semantics.

#### Implementation Details

ReSem-Face follows a two-stage training scheme. Stage I optimizes the semantic pre-inpainter via $\mathcal{L}_{\mathrm{pre}}$ using Adam with a learning rate of $1\times 10^{-5}$ for 20K iterations on 4 A40 GPUs. Stage II finetunes the diffusion backbone. We freeze the original U-Net layers and the Identity Encoder, and train only the introduced modules, including ReSemAttn, and Reference Identity Attention, using the joint objective of $\mathcal{L}_{\mathrm{diff}}$, $\mathcal{L}_{\mathrm{id}}$, and $\mathcal{L}_{\mathrm{sem}}$. This stage runs for 200K iterations using AdamW with a learning rate of $1.6\times 10^{-5}$ and a weight decay of $10^{-2}$, employing classifier-free guidance by dropping conditioning with a probability of 0.1. At inference, we perform a lightweight personalization of 40 steps for each target identity.

#### Evaluation Metrics

We compare ReSem-Face against eight baselines including LDI ([Rombach et al. 2022](#bib.bib28)), Custom Diffusion ([Kumari et al. 2023](#bib.bib10)), Textual Inversion ([Gal et al. 2023](#bib.bib55)), ReF-LDM ([Hsiao et al. 2024](#bib.bib54)), PVA ([Xu et al. 2024](#bib.bib9)), TransRef ([Liu et al. 2025](#bib.bib53)), OmniGen ([Xiao et al. 2025](#bib.bib49)), and HiFi-Inpaint ([Liu et al. 2026](#bib.bib56)). To evaluate identity preservation, we report Identity Similarity computed via CosFace ([Wang et al. 2018](#bib.bib35)) alongside perceptual metrics such as FID ([Heusel et al. 2017](#bib.bib37)), KID ([Bińkowski et al. 2018](#bib.bib38); [Karras et al. 2020](#bib.bib39)), PSNR, SSIM, and LPIPS ([Zhang et al. 2018](#bib.bib50)). For text controllability assessed on whole-face masks using 15 edit prompts ([Xu et al. 2024](#bib.bib9)), we measure CLIPScore ([Hessel et al. 2021](#bib.bib40)), ImageReward ([Xu et al. 2023](#bib.bib51)), and Facial Attribute Accuracy(Attr-Acc) ([Liu et al. 2015](#bib.bib41)). Note that TransRef is excluded from text-based evaluations due to its lack of text conditioning.

### Identity-Preserving Face Inpainting

#### Qualitative Results

Fig. [2](#Sx4.F2 "Figure 2 ‣ Qualitative Results ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions") compares ReSem-Face with eight baselines on CelebAHQ-IDI-5. LDI often suffers from severe identity drift, while Custom Diffusion and PVA improve identity consistency but frequently yield over-smoothed textures or structural inconsistencies under large occlusions. Although OmniGen and TransRef enhance reference alignment, they still exhibit subtle identity mismatches or boundary artifacts. In contrast, ReSem-Face generates the most faithful completions across all mask types. By leveraging the identity-conditioned semantic prior, our method accurately reconstructs identity-defining structures and sharp textures, significantly outperforming baselines in maintaining identity coherence when visual evidence is sparse.

![Refer to caption](./5854bc8220c82008573906057fdc7b961456c1e2.png)

Figure 2: Inpainting results of ReSem-Face and baselines on the test set of CelebAHQ-IDI-5 dataset.

#### Quantitative Results

Tab. [1](#Sx4.T1 "Table 1 ‣ Quantitative Results ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions") summarizes the quantitative comparisons on CelebAHQ-IDI-5. First, reference-guided methods substantially outperform generic inpainting and lightweight personalization in identity similarity. Second, while TransRef achieves high PSNR and SSIM due to its pixel-aligned prompting, ReSem-Face attains the best FID and LPIPS. This indicates that our method generates more perceptually natural textures compared to pixel-wise optimization. Most importantly, ReSem-Face achieves the highest ID score of 0.766, surpassing the strongest baseline by a clear margin. These results confirm that the proposed semantic prior effectively enhances both identity fidelity and perceptual realism under large occlusions.

| Method | Venue | FT.Time | ID $\uparrow$ | FID $\downarrow$ | KID $\downarrow$ | PSNR $\uparrow$ | SSIM $\uparrow$ | LPIPS $\downarrow$ |
|----|----|----|----|----|----|----|----|----|
| LDI ([Rombach et al. 2022](#bib.bib28)) | CVPR’22 | $-$ | 0.326 | 8.79 | 2.92 | 24.52 | 0.815 | 0.138 |
| CD ([Kumari et al. 2023](#bib.bib10)) | CVPR’23 | $\sim$3min | 0.688 | 15.86 | 7.87 | 26.85 | 0.862 | 0.132 |
| TI ([Gal et al. 2023](#bib.bib55)) | ICLR’23 | $\sim$6min | 0.615 | 17.50 | 9.12 | 25.40 | 0.830 | 0.138 |
| ReF-LDM ([Hsiao et al. 2024](#bib.bib54)) | NeurIPS’24 | $-$ | 0.725 | 8.35 | 4.55 | 27.50 | 0.885 | 0.122 |
| PVA ([Xu et al. 2024](#bib.bib9)) | WACV’24 | $\sim$1min | 0.736 | 8.16 | 4.27 | 27.92 | 0.895 | 0.118 |
| TransRef ([Liu et al. 2025](#bib.bib53)) | Neurocomputing’25 | $-$ | 0.713 | 8.56 | 4.79 | 28.45 | 0.910 | 0.128 |
| OmniGen ([Xiao et al. 2025](#bib.bib49)) | CVPR’25 | $-$ | 0.704 | 9.45 | 4.82 | 26.14 | 0.848 | 0.126 |
| HiFi-Inpaint ([Liu et al. 2026](#bib.bib56)) | CVPR’26 | $-$ | 0.722 | 9.79 | 4.91 | 26.34 | 0.896 | 0.125 |
| ReSem-Face (Ours) | $-$ | $\sim$1min | 0.766 | 7.90 | 3.75 | 28.31 | 0.904 | 0.116 |

Table 1: Quantitative results on the test set of CelebAHQ-IDI-5. We quantify the per-identity tuning overhead in the "FT.Time" column using a single RTX A40, where "ID" denotes the identity similarity score. Fine-tuning is limited to 40 steps for all applicable models. KID is scaled by $10^{-3}$. The best and second-best results are highlighted in bold and underline, respectively.

| Method | Venue | FT.Time | ID $\uparrow$ | FID $\downarrow$ | KID $\downarrow$ | PSNR $\uparrow$ | SSIM $\uparrow$ | LPIPS $\downarrow$ |
|----|----|----|----|----|----|----|----|----|
| LDI ([Rombach et al. 2022](#bib.bib28)) | CVPR’22 | $-$ | 0.312 | 17.50 | 6.85 | 21.45 | 0.715 | 0.172 |
| CD ([Kumari et al. 2023](#bib.bib10)) | CVPR’23 | $\sim$3min | 0.585 | 21.30 | 12.40 | 22.10 | 0.742 | 0.165 |
| TI ([Gal et al. 2023](#bib.bib55)) | ICLR’23 | $\sim$6min | 0.525 | 22.80 | 13.50 | 21.50 | 0.725 | 0.175 |
| ReF-LDM ([Hsiao et al. 2024](#bib.bib54)) | NeurIPS’24 | $-$ | 0.645 | 17.90 | 8.85 | 23.80 | 0.775 | 0.148 |
| PVA ([Xu et al. 2024](#bib.bib9)) | WACV’24 | $\sim$1min | 0.638 | 18.10 | 8.95 | 24.15 | 0.782 | 0.145 |
| TransRef ([Liu et al. 2025](#bib.bib53)) | Neurocomputing’25 | $-$ | 0.612 | 18.85 | 9.55 | 24.12 | 0.795 | 0.152 |
| OmniGen ([Xiao et al. 2025](#bib.bib49)) | CVPR’25 | $-$ | 0.605 | 19.42 | 10.12 | 23.05 | 0.768 | 0.158 |
| HiFi-Inpaint ([Liu et al. 2026](#bib.bib56)) | CVPR’26 | $-$ | 0.612 | 17.82 | 8.52 | 23.94 | 0.788 | 0.151 |
| ReSem-Face (Ours) | $-$ | $\sim$1min | 0.672 | 16.85 | 7.62 | 24.56 | 0.792 | 0.142 |

Table 2: Cross-dataset evaluation on the test set of VGGFace2. The metric definitions and fine-tuning settings are identical to those in Tab. [1](#Sx4.T1 "Table 1 ‣ Quantitative Results ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").

#### Cross-Dataset Evaluation on VGGFace2

![Refer to caption](./48595610acdd3b192961f6fc83ca684cc1d766e7.png)

Figure 3: Inpainting results of ReSem-Face and baselines on the test set of VGGFace2 dataset.

To evaluate generalization capabilities on in-the-wild data, we conduct experiments on the VGGFace2 test set ([Cao et al. 2018](#bib.bib52)), with qualitative comparisons shown in Fig. [3](#Sx4.F3 "Figure 3 ‣ Cross-Dataset Evaluation on VGGFace2 ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"). We utilize the official test split with faces aligned and resized to $512\times 512$, noting that the resulting ground-truth images often exhibit blurriness which affects reference-based metrics. Quantitative results in Tab. [2](#Sx4.T2 "Table 2 ‣ Quantitative Results ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions") show that ReSem-Face leads in identity preservation with an ID score of 0.672. Although all methods suffer from domain gaps and resolution mismatches compared to CelebA-HQ, ReSem-Face consistently achieves the lowest FID and LPIPS. This confirms that our framework generalizes well to unseen identities and remains robust even when reference images are of lower quality.

#### User Study

Next, we carry out a user study to examine whether the inpainted images conform to human preferences. In the experiments, we randomly sample 1K images from the test set of CelebAHQ-IDI-5. We invite 10 evaluators (5 males and 5 females) with diverse education backgrounds: art design (4), psychology (2), computer science (2), and business (2). We present the evaluators with the masked inputs, reference images, text prompts, and the inpainted results from different methods in a randomized order. We ask them to assign scores (1$\sim$5) from three aspects: 1) Identity Fidelity: whether the completed face maintains the identity characteristics of the reference images; 2) Text Alignment: whether the generated content aligns with the semantic attributes described in the text prompt; 3) Visual Realism: whether the inpainted region is visually coherent with the surrounding context and structurally natural. Table [3](#Sx4.T3 "Table 3 ‣ User Study ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions") summarizes the averaged results of different approaches. As indicated by the results, ReSem-Face leads the competition by a clear margin against the other baselines in terms of both identity preservation and text-driven editability, aligning best with human perception.

| Method            | LDI  | CD   | OmniGen | PVA  | ReSem-Face |
|-------------------|------|------|---------|------|------------|
| Identity Fidelity | 1.45 | 2.82 | 3.15    | 3.72 | 4.35       |
| Text Alignment    | 2.15 | 2.50 | 3.88    | 3.25 | 4.18       |
| Visual Realism    | 2.90 | 3.20 | 3.65    | 3.58 | 4.40       |

Table 3: User study results on 1K randomly sampled test images from CelebAHQ-IDI-5. We report the average user preference score (1-5), where higher is better.

### Text Controllability

#### Qualitative Results

Fig. [4](#Sx4.F4 "Figure 4 ‣ Qualitative Results ‣ Text Controllability ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions") presents qualitative comparisons for text-controlled inpainting. Custom Diffusion responds to prompts but compromises identity under strong edits. PVA maintains stability yet over-smoothes complex attributes. OmniGen captures global structure but struggles to balance fine-grained attributes with identity constraints. In contrast, ReSem-Face delivers the most compelling results across prompts, faithfully executing target attributes while preserving identity-defining structures and natural textures, demonstrating that our semantic prior effectively bridges language control and identity preservation.

| Method | ID $\uparrow$ | CLIPScore $\uparrow$ | ImageReward $\uparrow$ | Attr-Acc (%) $\uparrow$ |
|----|----|----|----|----|
| CD | 0.617 | 0.256 | 0.52 | 81.5 |
| OmniGen | 0.602 | 0.289 | 0.68 | 89.2 |
| PVA | 0.632 | 0.309 | 0.61 | 91.8 |
| ReSem-Face | 0.656 | 0.318 | 0.65 | 92.5 |

Table 4: Quantitative results on CelebAHQ-IDI-5 for text-controlled face inpainting. We additionally report ImageReward to evaluate human preference and Attr-Acc for attribute classification accuracy. The best and second-best results are highlighted in bold and underline, respectively.

|  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|
| Components |  |  | Identity-Preserving |  |  | Text-Controlled |  |
| \# | SemPrior | ReSemAttn | ID $\uparrow$ | FID $\downarrow$ | KID $\times 10^{-3}$ $\downarrow$ | ID $\uparrow$ | CLIPScore $\uparrow$ |
| 1 |  |  | 0.702 | 9.61 | 5.12 | 0.592 | 0.311 |
| 2 | ✓ |  | 0.747 | 9.62 | 5.11 | 0.613 | 0.309 |
| 3 | ✓ | ✓ | 0.766 | 7.90 | 3.75 | 0.656 | 0.318 |

Table 5: Component ablation analysis on CelebAHQ-IDI-5. We evaluate the impact of the semantic prior and its injection strategy on both identity-preserving inpainting and text-controlled editing.

![Refer to caption](./b796dfcbad7cec59c02934faf3d9a22ccdb99ba0.png)

Figure 4: Qualitative comparisons of identity-preserving text-controlled inpainting. Prompts for editing are shown at the bottom of each row. The column tabs, “CD, OmniGen, PVA” denote Custom Diffusion ([Kumari et al. 2023](#bib.bib10)), OmniGen ([Xiao et al. 2025](#bib.bib49)), PVA ([Xu et al. 2024](#bib.bib9)), respectively.

#### Quantitative results

Tab. [4](#Sx4.T4 "Table 4 ‣ Qualitative Results ‣ Text Controllability ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions") reports quantitative results, including CLIPScore, ImageReward, and Attr-Acc. Baseline methods often sacrifice identity for prompt alignment. In contrast, ReSem-Face achieves the best overall trade-off, obtaining the highest ID similarity, CLIPScore, and Attr-Acc while maintaining competitive ImageReward. These results confirm that our semantic prior enables precise editing without compromising identity consistency.

### Ablation Study

We validate the effectiveness of the proposed semantic prior and its injection strategy through component-wise ablation (Tab. [5](#Sx4.T5 "Table 5 ‣ Qualitative Results ‣ Text Controllability ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions")) and hyperparameter analysis (Fig. [5](#Sx4.F5 "Figure 5 ‣ Component Analysis ‣ Ablation Study ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions")).

#### Component Analysis

We compare three variants: (1) no semantic prior, (2) semantic prior without injection using auxiliary supervision only, and (3) the full model with ReSemAttn injection. Results in Tab. [5](#Sx4.T5 "Table 5 ‣ Qualitative Results ‣ Text Controllability ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions") show that removing the semantic prior entirely leads to a significant performance drop, where the ID score decreases from 0.766 to 0.702, confirming that high-level semantics are essential when visual evidence is missing. Furthermore, Var. 2 yields only marginal gains, verifying that explicit injection via ReSemAttn is crucial for stabilizing identity reconstruction. Notably, our full model (Var. 3) utilizes a decoupled semantic architecture consisting of $\mathcal{H}_{geo}$ and $\mathcal{H}_{tex}$ to mitigate “semantic blurring” where identity-rich features interfere with rigid geometric parsing. This architectural choice ensures stable semantic injection and structural consistency without compromising the fidelity of identity texture, leading to the superior performance observed in ReSem-Face.

![](./7c31fc7e777c7bcc108f663988c2705d7923399f.svg)

(a) Impact of Reference Number

![](./ae2f93b39a4b4b086c9e2a5160641d98e0b9205e.svg)

(b) ID vs. CLIP Trade-off

Figure 5: Ablation studies on the number of reference images and free guidance scale. (a) Increasing the number of reference images steadily improves identity consistency across different free guidance scales. (b) Increasing guidance scale improves text alignment (CLIP) but leads to identity degradation (ID); we choose $s=4.0$ as the optimal trade-off point.

#### Hyperparameter Analysis

We further investigate the impact of key hyperparameters in Fig. [5](#Sx4.F5 "Figure 5 ‣ Component Analysis ‣ Ablation Study ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"). First, regarding the number of reference images (Fig. [5](#Sx4.F5 "Figure 5 ‣ Component Analysis ‣ Ablation Study ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions")(a)), we observe that increasing the references from 1 to 6 consistently boosts identity similarity. Notably, using multi-reference guidance maintains high identity fidelity even under strong guidance scales. Second, regarding the conflict between text and identity (Fig. [5](#Sx4.F5 "Figure 5 ‣ Component Analysis ‣ Ablation Study ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions")(b)), we analyze the trade-off under varying guidance scales ($s$). While higher scales improve CLIPScore, they often degrade identity preservation. Our analysis identifies $s=4.0$ as the optimal sweet spot, achieving precise attribute editing with minimal loss in identity fidelity.

## Conclusion

In this paper, we presented ReSem-Face, a semantic-enhanced diffusion framework for identity-preserving and text-controllable face inpainting under large occlusions. By combining identity-aware semantic pre-inpainting, Reference Identity Attention, and Reference Semantic Attention, ReSem-Face provides complementary identity and semantic constraints during denoising. Experiments on CelebAHQ-IDI-5 and VGGFace2 demonstrate its superiority in identity fidelity, visual realism, and text controllability over representative baselines.

## References

- Avrahami et al. (2023) O. Avrahami, O. Fried, and D. Lischinski Blended latent diffusion. ACM transactions on graphics (TOG) 42 (4), pp. 1–11. Cited by: [Introduction](#Sx1.p3.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Avrahami et al. (2022) O. Avrahami, D. Lischinski, and O. Fried Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 18208–18218. Cited by: [Introduction](#Sx1.p3.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Bińkowski et al. (2018) M. Bińkowski, D. J. Sutherland, M. Arbel, and A. Gretton Demystifying mmd gans. arXiv preprint arXiv:1801.01401. Cited by: [Evaluation Metrics](#Sx4.SSx1.SSSx3.p1.1 "Evaluation Metrics ‣ Experiment Settings ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Cai et al. (2019) J. Cai, H. Han, S. Shan, and X. Chen FCSR-gan: joint face completion and super-resolution via multi-task learning. IEEE Transactions on Biometrics, Behavior, and Identity Science 2 (2), pp. 109–121. Cited by: [Introduction](#Sx1.p2.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Cao et al. (2018) Q. Cao, L. Shen, W. Xie, O. M. Parkhi, and A. Zisserman Vggface2: a dataset for recognising faces across pose and age. In 2018 13th IEEE international conference on automatic face & gesture recognition (FG 2018), pp. 67–74. Cited by: [Dataset and Pre-processing](#Sx4.SSx1.SSSx1.p1.1 "Dataset and Pre-processing ‣ Experiment Settings ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Cross-Dataset Evaluation on VGGFace2](#Sx4.SSx2.SSSx3.p1.1 "Cross-Dataset Evaluation on VGGFace2 ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Chen et al. (2024) Y. Chen, J. Chen, Y. Pan, Y. Li, T. Yao, Z. Chen, and T. Mei Improving text-guided object inpainting with semantic pre-inpainting. In European conference on computer vision, pp. 110–126. Cited by: [Introduction](#Sx1.p4.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Face Inpainting](#Sx2.SSx1.p1.1 "Face Inpainting ‣ Related Work ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Deng et al. (2019) J. Deng, J. Guo, N. Xue, and S. Zafeiriou Arcface: additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4690–4699. Cited by: [Multi-Reference Identity Semantic Aggregation](#Sx3.SSx2.SSSx1.p1.1 "Multi-Reference Identity Semantic Aggregation ‣ Identity-Aware Semantic Pre-Inpainting Module ‣ Methods ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Dong et al. (2022) Q. Dong, C. Cao, and Y. Fu Incremental transformer structure enhanced image inpainting with masking positional encoding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11358–11368. Cited by: [Face Inpainting](#Sx2.SSx1.p1.1 "Face Inpainting ‣ Related Work ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Gal et al. (2023) R. Gal, Y. Alaluf, Y. Atzmon, O. Patashnik, A. H. Bermano, G. Chechik, and D. Cohen-Or An image is worth one word: personalizing text-to-image generation using textual inversion. In The Eleventh International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=NAQvF08TcyG) Cited by: [Introduction](#Sx1.p4.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Evaluation Metrics](#Sx4.SSx1.SSSx3.p1.1 "Evaluation Metrics ‣ Experiment Settings ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Table 1](#Sx4.T1.1.1.4.1 "In Quantitative Results ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Table 2](#Sx4.T2.1.1.4.1 "In Quantitative Results ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Goodfellow et al. (2014) I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio Generative adversarial nets. Advances in neural information processing systems 27. Cited by: [Introduction](#Sx1.p2.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Hessel et al. (2021) J. Hessel, A. Holtzman, M. Forbes, R. Le Bras, and Y. Choi Clipscore: a reference-free evaluation metric for image captioning. In Proceedings of the 2021 conference on empirical methods in natural language processing, pp. 7514–7528. Cited by: [Evaluation Metrics](#Sx4.SSx1.SSSx3.p1.1 "Evaluation Metrics ‣ Experiment Settings ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Heusel et al. (2017) M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30. Cited by: [Evaluation Metrics](#Sx4.SSx1.SSSx3.p1.1 "Evaluation Metrics ‣ Experiment Settings ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Hsiao et al. (2024) C. Hsiao, Y. Liu, C. Yang, S. Kuo, K. Jou, and C. Chen Ref-ldm: a latent diffusion model for reference-based face image restoration. Advances in Neural Information Processing Systems 37, pp. 74840–74867. Cited by: [Evaluation Metrics](#Sx4.SSx1.SSSx3.p1.1 "Evaluation Metrics ‣ Experiment Settings ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Table 1](#Sx4.T1.1.1.5.1 "In Quantitative Results ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Table 2](#Sx4.T2.1.1.5.1 "In Quantitative Results ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Huang et al. (2025) J. Huang, T. Liu, Y. Wu, X. Qu, L. Liu, and X. Hu MTADiffusion: mask text alignment diffusion model for object inpainting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 18325–18334. Cited by: [Introduction](#Sx1.p3.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Ju et al. (2024) X. Ju, X. Liu, X. Wang, Y. Bian, Y. Shan, and Q. Xu Brushnet: a plug-and-play image inpainting model with decomposed dual-branch diffusion. In European Conference on Computer Vision, pp. 150–168. Cited by: [Introduction](#Sx1.p1.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Karras et al. (2020) T. Karras, M. Aittala, J. Hellsten, S. Laine, J. Lehtinen, and T. Aila Training generative adversarial networks with limited data. Advances in neural information processing systems 33, pp. 12104–12114. Cited by: [Evaluation Metrics](#Sx4.SSx1.SSSx3.p1.1 "Evaluation Metrics ‣ Experiment Settings ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Kim et al. (2025) S. Kim, S. Suh, and M. Lee Rad: region-aware diffusion models for image inpainting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 2439–2448. Cited by: [Introduction](#Sx1.p3.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Kingma and Welling (2013) D. P. Kingma and M. Welling Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114. Cited by: [Preliminaries](#Sx3.SSx1.p2.1 "Preliminaries ‣ Methods ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Kumari et al. (2023) N. Kumari, B. Zhang, R. Zhang, E. Shechtman, and J. Zhu Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1931–1941. Cited by: [Introduction](#Sx1.p4.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Figure 4](#Sx4.F4 "In Qualitative Results ‣ Text Controllability ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Evaluation Metrics](#Sx4.SSx1.SSSx3.p1.1 "Evaluation Metrics ‣ Experiment Settings ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Table 1](#Sx4.T1.1.1.3.1 "In Quantitative Results ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Table 2](#Sx4.T2.1.1.3.1 "In Quantitative Results ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Li et al. (2021) J. Li, Z. Li, J. Cao, X. Song, and R. He Faceinpainter: high fidelity face adaptation to heterogeneous domains. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5089–5098. Cited by: [Face Inpainting](#Sx2.SSx1.p1.1 "Face Inpainting ‣ Related Work ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Li et al. (2022) W. Li, Z. Lin, K. Zhou, L. Qi, Y. Wang, and J. Jia Mat: mask-aware transformer for large hole image inpainting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10758–10768. Cited by: [Face Inpainting](#Sx2.SSx1.p1.1 "Face Inpainting ‣ Related Work ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Li et al. (2017) Y. Li, S. Liu, J. Yang, and M. Yang Generative face completion. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3911–3919. Cited by: [Introduction](#Sx1.p2.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Li et al. (2024) Z. Li, M. Cao, X. Wang, Z. Qi, M. Cheng, and Y. Shan Photomaker: customizing realistic human photos via stacked id embedding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8640–8650. Cited by: [Introduction](#Sx1.p4.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Liu et al. (2025) T. Liu, L. Liao, D. Chen, J. Xiao, Z. Wang, C. Lin, and S. Satoh TransRef: multi-scale reference embedding transformer for reference-guided image inpainting. Neurocomputing 632, pp. 129749. Cited by: [Evaluation Metrics](#Sx4.SSx1.SSSx3.p1.1 "Evaluation Metrics ‣ Experiment Settings ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Table 1](#Sx4.T1.1.1.7.1 "In Quantitative Results ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Table 2](#Sx4.T2.1.1.7.1 "In Quantitative Results ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Liu et al. (2026) Y. Liu, D. Zhou, J. Wang, X. Gao, G. Liu, J. Li, Q. Zhang, Q. Lyu, L. Guo, S. Wen, et al. Hifi-inpaint: towards high-fidelity reference-based inpainting for generating detail-preserving human-product images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1994–2004. Cited by: [Evaluation Metrics](#Sx4.SSx1.SSSx3.p1.1 "Evaluation Metrics ‣ Experiment Settings ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Table 1](#Sx4.T1.1.1.9.1 "In Quantitative Results ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Table 2](#Sx4.T2.1.1.9.1 "In Quantitative Results ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Liu et al. (2015) Z. Liu, P. Luo, X. Wang, and X. Tang Deep learning face attributes in the wild. In Proceedings of the IEEE international conference on computer vision, pp. 3730–3738. Cited by: [Evaluation Metrics](#Sx4.SSx1.SSSx3.p1.1 "Evaluation Metrics ‣ Experiment Settings ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Lugmayr et al. (2022) A. Lugmayr, M. Danelljan, A. Romero, F. Yu, R. Timofte, and L. Van Gool Repaint: inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11461–11471. Cited by: [Face Inpainting](#Sx2.SSx1.p1.1 "Face Inpainting ‣ Related Work ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Luo et al. (2023) W. Luo, S. Yang, and W. Zhang Reference-guided large-scale face inpainting with identity and texture control. IEEE Transactions on Circuits and Systems for Video Technology 33 (10), pp. 5498–5509. Cited by: [Introduction](#Sx1.p3.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Face Inpainting](#Sx2.SSx1.p1.1 "Face Inpainting ‣ Related Work ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Reference-guided Face Inpainting](#Sx2.SSx2.p1.1 "Reference-guided Face Inpainting ‣ Related Work ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Motamed et al. (2023) S. Motamed, J. Xu, C. H. Wu, C. Häne, J. Bazin, and F. De la Torre Patmat: person aware tuning of mask-aware transformer for face inpainting. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 22778–22787. Cited by: [Face Inpainting](#Sx2.SSx1.p1.1 "Face Inpainting ‣ Related Work ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Reference-guided Face Inpainting](#Sx2.SSx2.p1.1 "Reference-guided Face Inpainting ‣ Related Work ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Navaneet et al. (2022) K. Navaneet, S. A. Koohpayegani, A. Tejankar, and H. Pirsiavash Simreg: regression as a simple yet effective tool for self-supervised knowledge distillation. arXiv preprint arXiv:2201.05131. Cited by: [Identity-aware semantic supervision](#Sx3.SSx2.SSSx3.p2.1 "Identity-aware semantic supervision ‣ Identity-Aware Semantic Pre-Inpainting Module ‣ Methods ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Nichol et al. (2021) A. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. McGrew, I. Sutskever, and M. Chen Glide: towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741. Cited by: [Introduction](#Sx1.p4.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Radford et al. (2021) A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. Cited by: [Preliminaries](#Sx3.SSx1.p2.1 "Preliminaries ‣ Methods ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Rombach et al. (2022) R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695. Cited by: [Introduction](#Sx1.p5.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Face Inpainting](#Sx2.SSx1.p1.1 "Face Inpainting ‣ Related Work ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Preliminaries](#Sx3.SSx1.p2.1 "Preliminaries ‣ Methods ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Evaluation Metrics](#Sx4.SSx1.SSSx3.p1.1 "Evaluation Metrics ‣ Experiment Settings ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Table 1](#Sx4.T1.1.1.2.1 "In Quantitative Results ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Table 2](#Sx4.T2.1.1.2.1 "In Quantitative Results ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Ruiz et al. (2023) N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman Dreambooth: fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22500–22510. Cited by: [Introduction](#Sx1.p4.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Song et al. (2020) J. Song, C. Meng, and S. Ermon Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502. Cited by: [Preliminaries](#Sx3.SSx1.p1.4 "Preliminaries ‣ Methods ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Suvorov et al. (2022) R. Suvorov, E. Logacheva, A. Mashikhin, A. Remizova, A. Ashukha, A. Silvestrov, N. Kong, H. Goka, K. Park, and V. Lempitsky Resolution-robust large mask inpainting with fourier convolutions. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 2149–2159. Cited by: [Face Inpainting](#Sx2.SSx1.p1.1 "Face Inpainting ‣ Related Work ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Varanka et al. (2024) T. Varanka, T. Toivonen, S. Tripathy, G. Zhao, and E. Acar Pfstorer: personalized face restoration and super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2372–2381. Cited by: [Reference-guided Face Inpainting](#Sx2.SSx2.p1.1 "Reference-guided Face Inpainting ‣ Related Work ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Vaswani et al. (2017) A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin Attention is all you need. Advances in neural information processing systems 30. Cited by: [Face Inpainting](#Sx2.SSx1.p1.1 "Face Inpainting ‣ Related Work ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Identity-Aware Semantic Pre-Inpainting Module](#Sx3.SSx2.p1.1 "Identity-Aware Semantic Pre-Inpainting Module ‣ Methods ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Wang et al. (2018) H. Wang, Y. Wang, Z. Zhou, X. Ji, D. Gong, J. Zhou, Z. Li, and W. Liu Cosface: large margin cosine loss for deep face recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5265–5274. Cited by: [Evaluation Metrics](#Sx4.SSx1.SSSx3.p1.1 "Evaluation Metrics ‣ Experiment Settings ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Wang et al. (2021) X. Wang, Y. Li, H. Zhang, and Y. Shan Towards real-world blind face restoration with generative facial prior. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9168–9178. Cited by: [Face Inpainting](#Sx2.SSx1.p1.1 "Face Inpainting ‣ Related Work ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Wasserman et al. (2025) N. Wasserman, N. Rotstein, R. Ganz, and R. Kimmel Paint by inpaint: learning to add image objects by removing them first. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 18313–18324. Cited by: [Introduction](#Sx1.p3.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Wu et al. (2023) J. Z. Wu, Y. Ge, X. Wang, S. W. Lei, Y. Gu, Y. Shi, W. Hsu, Y. Shan, X. Qie, and M. Z. Shou Tune-a-video: one-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 7623–7633. Cited by: [Introduction](#Sx1.p3.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Xiao et al. (2025) S. Xiao, Y. Wang, J. Zhou, H. Yuan, X. Xing, R. Yan, S. Wang, T. Huang, and Z. Liu OmniGen: unified image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: [Introduction](#Sx1.p1.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Figure 4](#Sx4.F4 "In Qualitative Results ‣ Text Controllability ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Evaluation Metrics](#Sx4.SSx1.SSSx3.p1.1 "Evaluation Metrics ‣ Experiment Settings ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Table 1](#Sx4.T1.1.1.8.1 "In Quantitative Results ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Table 2](#Sx4.T2.1.1.8.1 "In Quantitative Results ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Xie et al. (2025) L. Xie, D. Pakhomov, Z. Wang, Z. Wu, Z. Chen, Y. Zhou, H. Zheng, Z. Zhang, Z. Lin, J. Zhou, et al. TurboFill: adapting few-step text-to-image model for fast image inpainting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 7613–7622. Cited by: [Introduction](#Sx1.p3.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Xie et al. (2023) S. Xie, Z. Zhang, Z. Lin, T. Hinz, and K. Zhang Smartbrush: text and shape guided object inpainting with diffusion model. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22428–22437. Cited by: [Introduction](#Sx1.p4.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Xu et al. (2024) J. Xu, S. Motamed, P. Vaddamanu, C. H. Wu, C. Haene, J. Bazin, and F. De la Torre Personalized face inpainting with diffusion models by parallel visual attention. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 5432–5442. Cited by: [Introduction](#Sx1.p3.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Face Inpainting](#Sx2.SSx1.p1.1 "Face Inpainting ‣ Related Work ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Figure 4](#Sx4.F4 "In Qualitative Results ‣ Text Controllability ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Dataset and Pre-processing](#Sx4.SSx1.SSSx1.p1.1 "Dataset and Pre-processing ‣ Experiment Settings ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Evaluation Metrics](#Sx4.SSx1.SSSx3.p1.1 "Evaluation Metrics ‣ Experiment Settings ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Table 1](#Sx4.T1.1.1.6.1 "In Quantitative Results ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions"), [Table 2](#Sx4.T2.1.1.6.1 "In Quantitative Results ‣ Identity-Preserving Face Inpainting ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Xu et al. (2023) J. Xu, X. Liu, Y. Wu, Y. Tong, Q. Li, M. Ding, J. Tang, and Y. Dong Imagereward: learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems 36, pp. 15903–15935. Cited by: [Evaluation Metrics](#Sx4.SSx1.SSSx3.p1.1 "Evaluation Metrics ‣ Experiment Settings ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Yu et al. (2022) J. Yu, K. Li, and J. Peng Reference-guided face inpainting with reference attention network. Neural Computing and Applications 34 (12), pp. 9717–9731. Cited by: [Reference-guided Face Inpainting](#Sx2.SSx2.p1.1 "Reference-guided Face Inpainting ‣ Related Work ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Zhang et al. (2020a) L. Zhang, Q. Chen, B. Hu, and S. Jiang Text-guided neural image inpainting. In Proceedings of the 28th ACM international conference on multimedia, pp. 1302–1310. Cited by: [Introduction](#Sx1.p4.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Zhang et al. (2023) L. Zhang, A. Rao, and M. Agrawala Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 3836–3847. Cited by: [Introduction](#Sx1.p3.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Zhang et al. (2018) R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586–595. Cited by: [Evaluation Metrics](#Sx4.SSx1.SSSx3.p1.1 "Evaluation Metrics ‣ Experiment Settings ‣ Experiments ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Zhang et al. (2020b) Z. Zhang, Z. Zhao, Z. Zhang, B. Huai, and J. Yuan Text-guided image inpainting. In Proceedings of the 28th ACM international conference on multimedia, pp. 4079–4087. Cited by: [Introduction](#Sx1.p4.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Zhao et al. (2023) S. Zhao, D. Chen, Y. Chen, J. Bao, S. Hao, L. Yuan, and K. K. Wong Uni-controlnet: all-in-one control to text-to-image diffusion models. Advances in Neural Information Processing Systems 36, pp. 11127–11150. Cited by: [Introduction](#Sx1.p3.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Zhou et al. (2021) Y. Zhou, C. Barnes, E. Shechtman, and S. Amirghodsi Transfill: reference-guided image inpainting by merging multiple color and spatial transformations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2266–2276. Cited by: [Reference-guided Face Inpainting](#Sx2.SSx2.p1.1 "Reference-guided Face Inpainting ‣ Related Work ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Zhu et al. (2024) R. Zhu, Y. Pan, Y. Li, T. Yao, Z. Sun, T. Mei, and C. W. Chen Sd-dit: unleashing the power of self-supervised discrimination in diffusion transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8435–8445. Cited by: [Introduction](#Sx1.p3.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
- Zhuang et al. (2024) J. Zhuang, Y. Zeng, W. Liu, C. Yuan, and K. Chen A task is worth one word: learning with task prompts for high-quality versatile image inpainting. In European Conference on Computer Vision, pp. 195–211. Cited by: [Introduction](#Sx1.p1.1 "Introduction ‣ When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions").
