# luce-render: getting close to real time — literature study 2024–2026

Status: research note, 2026-10-07. Input for luce-render's speed work after v1 ([DESIGN.md](../DESIGN.md)).
It follows [RENDERER-STUDY.md](RENDERER-STUDY.md). That study covers Cycles X, pbrt-v4, OpenPBR and OpenPGL in source;
this one covers what was **published** from SIGGRAPH 2024 to SIGGRAPH 2026.

Scope:
- **Venues:** SIGGRAPH 2024/2025/2026 technical papers (ACM TOG 43(4), 44(4), 45(4) and the conference track),
  SIGGRAPH Asia 2024/2025 (TOG 43(6), 44(6)), EGSR 2024/2025/2026, HPG 2024/2025/2026, and Eurographics 2025/2026 /
  I3D 2026 where the paper clearly matters.
- **Verification:** every paper below was found on an official program, on the kesen.realtimerendering.com lists,
  on a DOI, on arXiv or on a project page.
  - Papers marked **(background, pre-2024)** are older. They are cited because the newer work builds on them.
  - Papers marked **(title only)** are confirmed on an official program, but no abstract was found.
- **Numbers:** speed and quality figures are the authors' own, taken from abstracts and project pages. None of them
  were reproduced here.
- **Not copied:** as with the first study, nothing here is code to copy. Algorithms are described in our own words.

Scratch copies of the paper lists are in `/Users/sedov/Dev/luce_dev/.donors/realtime/` (kesen pages, SIGGRAPH 2026
fast-forward order PDF). They are reference only.

Our baseline (from DESIGN.md):
- **Integrator:** GPU-only wavefront (camera → intersect_closest → shade_surface/shade_miss → intersect_shadow →
  shade_shadow), queues with indirect dispatch, 2^20 paths in flight, 1 spp per pass.
- **Rays:** hardware ray queries, BVH2 fallback.
- **Spectral:** 4 hero wavelengths; scalar pdfs.
- **Materials and lights:** OpenPBR v1 lobes; light tree with MIS.
- **Speed:** ~9 ms per 1080p spectral spp at 12 bounces on an M4 Max, so ~110 spp/s.
- **No ML for now:** there is no cooperative-matrix / tensor-core API in luce-gpu, and luce-nn is paused.

---

## 1. Executive summary

### 1.1 The arithmetic of "converged in about a second"

At 9 ms/spp, one second gives about 110 spp at 1080p.

- **Product shots:** for a product shot (a few objects, area lights, a studio dome) 100 spp plus a decent denoiser
  is already "clean". The work there is mostly **denoising + adaptive sampling + transparent shadows/caustics
  through glass**.
- **Interiors:** for an interior lit through windows, 100 spp of plain path tracing is far from converged. The
  variance sits in indirect light that finds the light through a small opening. Here the payoff comes from
  **path guiding** and, in the viewport, a **radiance cache**.
- **Many-light scenes:** for many emissive meshes or dozens of rect lights, the light tree is the right base.
  **ReSTIR DI** in the viewport lifts it to roughly "one good shadow ray per pixel".
- **Raw speed:** making each spp cheaper (coherence, shorter viewport paths, faster shadow rays) multiplies every
  one of the above. It is the first thing to measure.

### 1.2 Ranked roadmap

| # | What | Why first / expected payoff | Effort | Needs ML? |
|---|---|---|---|---|
| 0 | **Measure** (per-kernel timestamps, occupancy per bounce, reference scenes, time-to-error curves) | Every later choice is decided by numbers. See §4. | S (days) | no |
| 1 | **Adaptive sampling + non-ML denoiser** (Cycles-style convergence test; variance-guided à-trous / SVGF-style filter on demodulated albedo; blend toward the noisy estimate as spp grows) | The largest perceived speedup for the viewport. 1 s goal is reachable for product scenes. | M (2–3 wk) | no |
| 2 | **Per-sample cost** (subgroup queue appends, material-sorted shading, shorter viewport paths, shadow-ray any-hit with transmittance) | 1.3–2× on trace/shade is typical. Transparent shadows through glass are a feature gap too. | M | no |
| 3 | **World-space hash grid** (one structure) holding (a) a **radiance cache** for viewport path termination and (b) **vMF guiding lobes** | Interiors: guiding gives large equal-time gains in indirect-dominated scenes. The cache shortens viewport paths to 2–4 bounces. | L (4–6 wk) | no |
| 4 | **ReSTIR DI (viewport)** on top of the light tree, with modern fixes (reservoir splatting, stratification, compatibility-guided neighbours) | Many-light scenes and small bright lights in the viewport. Final frames keep the light tree. | M–L | no |
| 5 | **Caustics:** MNEE for flagged glass casters (shadow caustics), then photon-based or ReSTIR-based caustics for final frames | Glass is a product-render staple; plain PT never converges SDS paths. | L | no |
| 6 | **Spectral/color noise:** ratio control variates across the 4 hero lanes | Cheap; directly attacks the color noise that hero sampling adds. | S–M | no |
| 7 | **ML denoiser** (OIDN-class U-Net on compute), later **neural radiance cache / NIRC** | Best final quality at low spp. Needs luce-nn and ideally cooperative matrices. | L | yes |
| 8 | **ReSTIR PT (Enhanced)** for the viewport's GI | The 2024–26 state of the art for real-time GI, but heavy machinery (shifts, Jacobians, spectral lanes). | XL | no |

Effort: S under a week, M 2–3 weeks, L a month or more, XL several months.

**Short version:**
1. Measure.
2. Then denoise and sample adaptively.
3. Then make each sample cheap.
4. Then build one hash grid that both guides and caches.
5. Then ReSTIR DI for the viewport.
6. Then caustics.

Everything neural waits for luce-nn. Everything below up to #6 runs on plain compute with int/float atomics and subgroups.

---

## 2. Literature by topic

Entry format: **Title** — authors. *Venue, year.* Link.
Each entry then says what the paper does, what it reports, its limits, and how it fits luce-render.

### 2.1 Path guiding

The 2024–26 work splits into two groups:
- **real-time, world-space and GPU-friendly** (VXPG, ReSTIR PG, RCPG);
- **offline-quality refinements** (MI reweighting, illumination-aware subdivision, MARS).

Neural guiding keeps improving, but every neural method needs online training.

**Real-Time Path Guiding Using Bounding Voxel Sampling (VXPG)** — Haolin Lu, Wesley Chang, Trevor Hedstrom, Tzu-Mao Li.
*SIGGRAPH 2024 (TOG 43(4)).* https://doi.org/10.1145/3658203 · https://suikasibyl.github.io/vxpg
- **How it works:** each frame it fills a spatial voxel grid with irradiance and geometry. A shading point picks a
  voxel with high expected contribution, using many-light-style clustering of (shading point, voxel) pairs to
  account for visibility. It then samples unbiasedly inside the chosen voxel's geometry.
- **Reported:** much lower perceptual error at equal time than earlier real-time guiding and VPL methods.
- **Limits:** voxel resolution bounds the quality. It is a "where to go next" (position) distribution, not a
  directional one.
- **Fit:** it needs no temporal history, so it combines with ReSTIR. It is GPU-native.
- **For us:** an interesting alternative to directional lobes, especially for indirect light from small bright
  regions. It is more machinery than a vMF grid.

**ReSTIR PG: Path Guiding with Spatiotemporally Resampled Paths** — Zheng Zeng, Markus Kettunen, Chris Wyman,
Lifan Wu, Ravi Ramamoorthi, Ling-Qi Yan, Daqi Lin. *SIGGRAPH Asia 2025.* https://doi.org/10.1145/3757377.3763813 ·
https://research.nvidia.com/labs/rtr/publication/zeng2025restirpg
- **Key observation:** the paths that ReSTIR PT accepts are already distributed roughly like the ideal local guiding
  density (incident radiance × BSDF × cosine).
- **How it works:** each frame it splats their bounce directions into a **world-space hash grid**. It fits
  **4-lobe vMF mixtures per cell with EM**. The next frame's initial candidates are guided by them.
- **Reported:**
  - lower variance, faster reaction to scene change and fewer correlation artifacts than guiding trained on raw
    path samples, at real-time rates;
  - overhead split: about 25% collect/splat, 25% fit, 50% sampling;
  - about 152 B per cell; 5–76 MB of grid per scene.
- **Limits:** the full loop needs ReSTIR PT.
- **For us:** the data structure (hash grid + 4 vMF lobes + per-frame fit) is exactly the GPU-friendly guiding we
  want. It also works when trained from plain path-traced samples, as in MCMM / Dittebrandt-style training. The
  ReSTIR feedback loop is an upgrade for later.

**RCPG: Real-Time Product Path Guiding Using Radiance Cascades** — J. Ikkala, P. Jääskeläinen, M. Mäkitalo.
*ACM TOG 2026.* https://doi.org/10.1145/3840291 · https://webpages.tuni.fi/vga/publications/RCPG.html
- **How it works:** it uses world-space radiance cascades as a hierarchical guiding structure. It derives bounds on
  radiance intervals so that the cascades do not misrepresent incoming directions. This gives **product
  sampling** (radiance × BSDF) in one hierarchical sampling process.
- **Properties:** the pdf can be evaluated in any direction, so it works with MIS and unbiased ReSTIR. Because it is
  world space, off-screen vertices are guided too.
- **Reported:** about **2× more paths per pixel** in a real-time budget than prior multi-sample-MIS guiding.
- **Limits:** no memory or ms breakdown is public. Radiance cascades are a sizeable new structure.
- **For us:** a strong v3 candidate if vMF lobes underperform on glossy surfaces.

**Multiple Importance Reweighting for Path Guiding** — Zhimin Fan, Yiming Wang, Chenxi Zhou, Ling-Qi Yan, Yanwen Guo,
Jie Guo. *SIGGRAPH 2025 (TOG 44(4)).* https://doi.org/10.1145/3731144 · https://zhiminfan.work/mi_reweight.html
- **Problem:** guided rendering trains in iterations, and earlier iterations' samples are noisy or wasted.
- **How it works:** it reweights **paths** (not just whole images, as in PPG's inverse-variance image combination)
  across iterations, AMIS-style. A storage budget splats low-value samples early.
- **Properties:** agnostic to the guiding method. Reports variance reduction with negligible bias at equal sample
  count and time.
- **Limits:** stores samples; memory/variance trade-off.
- **For us:** final-frame mode only. It lets the first, poorly trained passes still count.

**Illumination-Aware Spatial Subdivision for Path Guiding** — Fengshi Zheng, Christoph Peters, Sebastian Herholz,
Marco Manzi, Elmar Eisemann. *EGSR 2026.* https://doi.org/10.2312/sr.20261000 · https://spatial-subdiv.ewi.tudelft.nl
- **How it works:** it replaces the sample-count split rule of PPG/Open PGL-style trees. Cells keep "signatures"
  (mean radiance, radiance-weighted mean direction), and a statistical test splits only where the lighting
  changes. It applies on top of existing guiding. Code is public.
- **For us:** the same idea tells a hash grid **where to use finer cells**, for example by choosing the level per
  query from local signature variance.

**MARS: Multi-sample Allocation through Russian roulette and Splitting** — Joshua Meyer, Alexander Rath, Ömercan Yazici,
Philipp Slusallek. *SIGGRAPH Asia 2024.* https://arxiv.org/abs/2410.20429 · https://doi.org/10.1145/3680528.3687636
- **How it works:** it learns, per location and **per technique** (BSDF, guide, NEE…), how many samples to take
  (RR or splitting). It runs a fixed-point optimisation of image efficiency, stored in a light spatial structure.
- **Reported:** "consistent and substantial speedups" over EARS-style allocation with path guiding and BDPT.
- **Limits:** CPU research renderer. Splitting is awkward in a fixed-size path pool.
- **For us:** we can use its **RR half** (efficiency-aware survival probability per cell) in final frames.

**Efficient Neural Path Guiding with 4D Modeling** — Honghao Dong, Rui Su, Guoping Wang, Sheng Li. *SIGGRAPH Asia 2024.*
https://doi.org/10.1145/3680528.3687687 — neural guiding with time or **wavelength** as an extra dimension; online MLP
training. Our luminance-trained guide avoids the need; later, with ML.

**Neural Path Guiding with Distribution Factorization** — Pedro Figueiredo, Qihao He, Nima Khademi Kalantari.
*EGSR 2025.* https://arxiv.org/abs/2506.00839
- **What it does:** factors the directional pdf into two 1D pdfs predicted by a network. An auxiliary radiance
  network supplies normalisation.
- **Reported:** beats prior neural and non-neural guiding in hard scenes.
- **For us:** online neural training is needed, so it waits for luce-nn.

**Volume Scattering Probability Guiding** — Kehan Xu, Sebastian Herholz, Marco Manzi, Marios Papas, Markus Gross.
*SIGGRAPH Asia 2024.* https://kehanxuuu.github.io/vspg-website/ — guides the scattering probability in media; for us
once volumes arrive (OpenPBR v2/v3).

**Neural Resampling with Optimized Candidate Allocation** — Alexander Rath et al. (Disney Research). *EGSR 2025.*
https://diglib.eg.org/items/b52f338b-36d1-4565-a275-c3f11a49bfea/full — learned 5D incident radiance sampled in
product with the BSDF by RIS; strong equal-time gains; ML, later.

Background (pre-2024) that the above builds on:
- **Practical Path Guiding** (SD-tree) — Thomas Müller, Markus Gross, Jan Novák. *CGF 36(4), EGSR 2017.*
  https://doi.org/10.1111/cgf.13227
- **Markov Chain Mixture Models for Real-Time Direct Illumination** — Addis Dittebrandt, Vincent Schüßler,
  Johannes Hanika, Sebastian Herholz, Carsten Dachsbacher. *CGF 42(4), EGSR 2023.* https://cg.ivd.kit.edu/mcmm.php
  - Screen-space vMF mixtures driven by a per-pixel Markov chain; one ray per sample.
  - It shows that vMF lobes can be updated online, without an EM pass.

### 2.2 Radiance caching

No direct successor to the 2021 neural radiance cache appeared at SIGGRAPH or SIGGRAPH Asia in 2024–25. The useful
2024–26 results are three:
- the **two-level (MLMC) estimator**, which removes a cache's bias;
- **variance-aware termination**;
- **on-surface/world-space caches** at HPG.

**Neural Two-Level Monte Carlo Real-Time Rendering** — Mikhail Dereviannykh, Dmitrii Klepikov, Johannes Hanika,
Carsten Dachsbacher. *Eurographics 2025 (CGF 44(2)), Best Paper Honorable Mention.* https://arxiv.org/abs/2412.04634 ·
https://cg.ivd.kit.edu/nirc.php
- **How it works:** it splits the shading integral into a **cache integral**, estimated with many cheap samples of a
  Neural Incident Radiance Cache (NIRC), and a **residual** (path-traced radiance minus cache prediction) estimated
  with few samples. The sum is **unbiased**.
- **Reported:** a cache query costs 2–25× less than a path-traced sample. Big noise reduction, and it works in
  dynamic scenes.
- **Limits:** online neural training every frame.
- **Key idea for us:** the two-level split works with **any** cache, including a non-neural hash grid. It turns a
  biased viewport trick into an unbiased variance reducer.

**Optimizing Path Termination for Radiance Caching Through Explicit Variance Trading** — Lukas Kandlbinder,
Addis Dittebrandt, Alexander Schipek, Carsten Dachsbacher. *HPG 2024.* https://cg.ivd.kit.edu/variance-trading.php
- **How it works:** it decides **when** to terminate a path into a cache. Path variance becomes a locally computable
  quantity, using auxiliary statistics stored in the cache, and a global variance bound is met while bias is
  minimised.
- **Reported:** lower bias than ray-differential (spread-angle) heuristics at the same variance.
- **Fit:** works with any cache.
- **For us:** this is the termination rule we want for the viewport cache. NRC used a simple spread-angle heuristic
  instead.

**Radiance Caching with On-Surface Caches for Real-Time Global Illumination** — Wolfgang Tatzgern, Alexander Weinrauch,
Pascal Stadlbauer, Joerg H. Mueller, Martin Winter, Markus Steinberger. *HPG 2024.*
https://highperformancegraphics.org/slides24/hpg24_oscgi.pdf
- **What it does:** a two-level, on-surface, multi-resolution, hash-indexed world-space radiance cache for diffuse
  and glossy multi-bounce GI, shareable between viewers.
- **For us:** confirms the hash-grid design for a non-neural cache.

**Adaptive Multi-view Radiance Caching for Heterogeneous Participating Media** — Pascal Stadlbauer et al. *EG 2025.*
https://doi.org/10.1111/cgf.70051 — SH probe grid for multiple scattering in media; later, with volumes.

**Locality-aware Training for Online Radiance Caching in Path Tracing on Mobile Platforms** — Hyeonseung Yu et al.
*EGSR 2026 (title only).* An NRC-style cache for mobile GPUs; read when luce-nn resumes.

**Real-time Rendering with a Neural Irradiance Volume** — Arno Coomans et al. *EG 2026.* https://arxiv.org/abs/2602.12949
— precomputed neural irradiance probes for static scenes (~1 ms at 1080p); baked GI, out of scope.

**Dynamic Neural Radiosity with Multi-grid Decomposition** (Su et al., *SIGGRAPH Asia 2024*,
https://doi.org/10.1145/3680528.3687685) and **Vertex Features for Neural Global Illumination** (Su et al.,
*SIGGRAPH Asia 2025*, https://arxiv.org/abs/2508.07852): per-scene trained neural radiosity; not a fit without ML.

Background:
- **Real-time Neural Radiance Caching for Path Tracing** — Thomas Müller, Fabrice Rousselle, Jan Novák,
  Alexander Keller. *SIGGRAPH 2021.* https://arxiv.org/abs/2106.12372
  - Online-trained fused MLP queried at path termination. About 2.6 ms overhead at 1080p on NVIDIA tensor hardware.
  - Needs fused MLP training: tensor cores / cooperative matrix.

### 2.3 ReSTIR and resampling, 2024–2026

NVIDIA's group and its collaborators dominate this area. Many papers address the **failure modes** that show up when
ReSTIR is used for real:
- disocclusion;
- correlation and the choice of neighbours;
- color noise;
- defocus;
- level-of-detail changes;
- caustics.

**ReSTIR PT Enhanced: Algorithmic Advances for Faster and More Robust ReSTIR Path Tracing** — Daqi Lin, Markus Kettunen,
Chris Wyman. *I3D 2026 (PACMCGIT), Best Paper.* https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/
- **Reported:** 2–3× faster than ReSTIR PT, with lower error.
- **How:**
  - **Reciprocal neighbour selection** halves the cost of spatial reuse.
  - **Footprint-based reconnection criteria** make shift mappings more robust.
  - **Duplication maps** reduce correlation.
  - DI and GI share **one set of reservoirs**.
- **For us:** if we ever do ReSTIR PT, start from this paper, not from the 2022 one.

**Compatibility-Guided Neighbor Selection for ReSTIR** — Orion Junkins, Markus Kettunen, Daqi Lin, Ravi Ramamoorthi,
Chris Wyman. *HPG 2026, Best Paper.* https://doi.org/10.1145/3820024
- **What it does:** picks spatial-reuse neighbours by path compatibility.
- **Reported:** SMAPE −6–29%, temporal covariance −22–49%, for 2–5% extra cost.
- **For us:** cheap. It applies to ReSTIR DI too.

**Stochastic Pairwise MIS for Unbiased Large-Kernel Reuse in Real Time** — Trevor Hedstrom, Markus Kettunen, Daqi Lin,
Chris Wyman, Tzu-Mao Li. *Eurographics 2026 (CGF).* https://doi.org/10.1111/cgf.70391
- **What it does:** unbiased reuse from **many** neighbours, focused on those with contributing samples.
- **Reported:** large gains in disocclusions and under motion.

**Reservoir Splatting for Temporal Path Resampling and Motion Blur** — Jeffrey Liu, Daqi Lin, Markus Kettunen,
Chris Wyman, Ravi Ramamoorthi. *SIGGRAPH 2025.* https://doi.org/10.1145/3721238.3730646
- **How it works:** **forward-projects** the previous frame's primary hits into the pixels they land in, instead of
  back-projecting each pixel into the previous frame. This makes temporal reuse succeed more often under camera
  motion and subpixel jitter.
- **Reported:** motion blur and DOF without special shifts; up to 10% cheaper than Area ReSTIR.
- **For us:** the right temporal-reuse design for a progressive viewport with thin-lens DOF.

**Multi-Layer Reservoir Splatting for Temporal Reuse under Disocclusion** — Pengpei Hong, Song Zhang, Daqi Lin,
Markus Kettunen, Chris Wyman, Cem Yuksel. *SIGGRAPH 2026.* https://research.nvidia.com/labs/rtr/publication/hong2026multilayer/
— several screen-space layers so previously occluded samples are reused on disocclusion; small extra cost. Later.

**Real-Time Level-of-Detail Rendering with ReSTIR** — Yu-Chen Wang, Markus Kettunen, Daqi Lin, Chris Wyman, Lifan Wu,
Shuang Zhao. *SIGGRAPH 2026.* https://research.nvidia.com/labs/rtr/publication/wang2026levelofdetail/ — surface-point
mapping so reuse survives LoD/topology switches; relevant only once tessellation changes between frames.

**Area ReSTIR: Resampling for Real-Time Defocus and Antialiasing** — Song Zhang, Daqi Lin, Markus Kettunen, Cem Yuksel,
Chris Wyman. *SIGGRAPH 2024.* https://doi.org/10.1145/3658210 — reservoirs over film area and lens; superseded in cost
by reservoir splatting.

**Spatio-Temporal Control Variates with ReSTIR for Real-Time Rendering (ReSTCV)** — Zhong Shi, Cunhao Wu, Lifan Wu,
Kun Xu. *SIGGRAPH 2026, Honorable Mention.* https://doi.org/10.1145/3799902.3811113 · https://hercier.github.io/restcv/
- **Problem:** ReSTIR keeps **one** sample per pixel. With colorful lighting, the per-channel variance stays high.
- **How it works:** it stores accumulated color estimates with the reservoirs and uses neighbours' colors as
  **control variates**, corrected by differences.
- **Reported:** modest overhead, clearly cleaner color.
- **For us:** relevant for spectral. Our target function is scalar, so color noise is exactly the residual.

**Histogram Stratification for Spatio-Temporal Reservoir Sampling** — Corentin Salaün, Martin Bálint, Laurent Belcour,
Eric Heitz, Gurprit Singh, Karol Myszkowski. *SIGGRAPH 2025.* https://doi.org/10.1145/3721238.3730723
- **How it works:** it puts the candidates into local histograms and picks from them with QMC/antithetic patterns,
  instead of independent random selection.
- **Reported:** lower error for minimal overhead.
- **For us:** a drop-in for the reservoir selection step.

**Gradient-Domain ReSTIR Path Tracing** — Yu-Chen Wang, Markus Kettunen, Daqi Lin, Chris Wyman, Lifan Wu, Shuang Zhao.
*EG 2026.* https://research.nvidia.com/labs/rtr/publication/wang2026gradient — first real-time gradient-domain PT.

**ReSTIR BDPT: Bidirectional ReSTIR Path Tracing with Caustics** — Trevor Hedstrom, Markus Kettunen, Daqi Lin,
Chris Wyman, Tzu-Mao Li. *ACM TOG 2025, presented at SIGGRAPH 2026.* https://doi.org/10.1145/3744898 ·
https://research.nvidia.com/labs/rtr/publication/hedstrom2025restir
- **How it works:** GRIS over a technique-aware extended path space, with a bidirectional hybrid shift, plus
  **caustics reservoirs** that accumulate light-traced caustics across frames.
- **Reported:** about 50 ms/frame on their test GPU. MAPE 0.312 in 70 ms against 1.368 in 71 ms for ReSTIR PT
  (1080p, 1M light subpaths).
- **Limits:** a full BDPT plus ReSTIR stack.
- **For us:** the reference for "real-time caustics through reuse". XL effort.

**ReSTIR FG: Real-Time Reservoir Resampled Photon Final Gathering** — René Kern, Felix Brüll, Thorsten Grosch.
*EGSR 2024.* https://diglib.eg.org/handle/10.2312/sr20241155
- **What it does:** photon final gathering combined with ReSTIR. Real-time multi-bounce indirect light **and
  caustics**, including objects inside glass, which ReSTIR GI/PT handle poorly.
- **Follow-up:** **Guided ReSTIR FG+** (*EGSR 2026, title only*) targets large scenes and many lights.
- **For us:** the most direct "photons in a GPU viewport" design. See §3.6.

**ReSTIR Subsurface Scattering for Real-Time Path Tracing** — Mirco Werner, Vincent Schüßler, Carsten Dachsbacher.
*HPG 2024.* https://cg.ivd.kit.edu/restir-sss.php — for OpenPBR v3.

**Efficient Image-Space Shape Splatting for Monte Carlo Rendering** — Xiaochun Tong, Toshiya Hachisuka.
*SIGGRAPH Asia 2024 (TOG 43(6)).* https://cs.uwaterloo.ca/~thachisu/mcsplat.pdf
- **What it does:** reuses one path across a 2D shape of pixels at sublinear cost ("60 pixels with 3 shift
  mappings"), with telescoping debiasing.
- **Fit:** works over PT, PSSMLT or ReSTIR PT, and reduces ReSTIR's color noise.

Background:
- **ReSTIR DI** — Bitterli et al. *SIGGRAPH 2020.* https://doi.org/10.1145/3386569.3392481
- **GRIS / ReSTIR PT** — Lin et al. *SIGGRAPH 2022.* https://research.nvidia.com/labs/rtr/publication/lin2022generalized
- **Conditional ReSTIR** (final-gather reuse one bounce later, to cut blotches) — Kettunen, Lin, Ramamoorthi,
  Bashford-Rogers, Wyman. *SIGGRAPH Asia 2023.* https://research.nvidia.com/labs/rtr/publication/kettunen2023conditional/
- **Decorrelating ReSTIR Samplers via MCMC Mutations** — Sawhney, Lin, Kettunen, Bitterli, Ramamoorthi, Wyman, Pharr.
  *ACM TOG 2024.* https://arxiv.org/abs/2211.00166

### 2.4 Many-light sampling

**Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting** — Yusuke Tokuyoshi, Sho Ikeda,
Paritosh Kulkarni, Takahiro Harada. *SIGGRAPH Asia 2024.* https://doi.org/10.1145/3680528.3687647 ·
https://gpuopen.com/download/publications/Hierarchical_Light_Sampling_with_Accurate_Spherical_Gaussian_Lighting.pdf
- **How it works:** a light tree in which each node is a spherical-Gaussian light. Node importance is an analytic
  SG × BRDF product integral (NDF-filtered for anisotropic GGX; a cheaper form for diffuse).
- **Reported:** Toyshop scene (167k emissive triangles, 4K, equal time) gives RMSPE 27.2% against 31.0% for
  Conty–Kulla-style importance (our current pbrt-v4 light bounds are in this family).
- **Limits:** one isotropic lobe per cluster.
- **For us:** a direct upgrade of our tree's **importance function**; the tree build stays. Code is public
  (VSGL, for reading only). Effort M.

**Neural Importance Sampling of Many Lights** — Pedro Figueiredo, Qihao He, Steve Bako, Nima Khademi Kalantari.
*SIGGRAPH 2025.* https://arxiv.org/abs/2505.11729
- **How it works:** a network predicts **cluster-level** light-selection distributions, trained online by KL
  divergence. It learns a residual over the light hierarchy's own distribution, and the tree samples within clusters.
- **Fit:** the "residual on top of the light tree" structure is the right shape for us later.
- **Needs:** online training.

**Neural Visibility Cache for Real-Time Light Sampling** — Jakub Bokšanský, Daniel Meister. *JCGT 14(2), 2025.*
https://arxiv.org/abs/2506.05930 — learns light visibility so the tree stops picking occluded lights; ML.

**Adaptive Multiple Control Variates for Many-Light Rendering** — Xiaofeng Xu, Lu Wang. *EGSR 2025.*
https://doi.org/10.2312/sr.20251184 — modest gains, direct light only; low priority.

**Wavelet Representation and Sampling of Complex Luminaires** — Asen Atanasov, Vladimir Koylazov. *EGSR 2025, Best Paper.*
https://www.chaos.com/papers/wavelet-representation-and-sampling-of-complex-luminaires — turns a luminaire with internal
optics into a compact, importance-sampleable light field; relevant to product shots of lamps, later.

**One-more-vertex Next-Event Estimation with Hierarchical Geometry Sampling** — Jorge Garcia Pueyo, Nestor Monzon,
Adrian Jarabo, Adolfo Muñoz. *EGSR 2026.* https://doi.org/10.1111/cgf.70539 — a stochastic BVH walk samples an
intermediate vertex so NEE reaches indirect light from sparse geometry; orders-of-magnitude gains in its target cases.

**Many-Light Rendering Using ReSTIR-Sampled Shadow Maps** — Song Zhang, Daqi Lin, Chris Wyman, Cem Yuksel. *EG 2025.*
https://doi.org/10.1111/cgf.70059 — a rasterisation hybrid; not for us.

### 2.5 Adaptive sampling and sample allocation

**Forget Superresolution, Sample Adaptively (when Path Tracing)** — Martin Bálint, Corentin Salaün, Hans-Peter Seidel,
Karol Myszkowski. *SIGGRAPH 2026 (TOG 45(4)).* https://arxiv.org/abs/2602.08642
- **How it works:** an end-to-end learned sampler plus denoiser for **below 1 spp**. Stochastic sample placement
  gives usable gradients, and training is tonemapping- and perception-aware. It includes a gather-based pyramidal
  denoiser and a learnable albedo demodulation.
- **Reported:** better than uniform sparse sampling, especially on highlights and shadow edges.
- **For us:**
  - **Non-ML lesson:** below 1 spp, spending samples by perceptual importance beats super-resolution.
  - The full method needs ML.

**Neural Quadrature Rule and Autoregressive Adaptive Sampling** — Haolin Lu, Liwen Wu, Zimo Wang, Tzu-Mao Li,
Ravi Ramamoorthi. *SIGGRAPH 2026.* https://cseweb.ucsd.edu/~ravir/haolinsig26.pdf — neural quadrature for direct light; ML.

**Practical Error Estimation for Denoised Monte Carlo Image Synthesis** — Arthur Firmino, Ravi Ramamoorthi,
Jeppe Revall Frisvad, Henrik Wann Jensen. *SIGGRAPH 2024.* https://cseweb.ucsd.edu/~ravir/arthursig.pdf
- **How it works:** it estimates the per-pixel bias and variance of a **denoised** image. Squared error then follows a
  noncentral χ² distribution, which gives a **stopping criterion** at a user error threshold.
- **For us:** the final-frame "done" test. Render until the denoised error is under target, not to a fixed spp.

**Converging Algorithm-Agnostic Denoising for Monte Carlo Rendering** — Elena Denisova, Leonardo Bocchi. *HPG 2024.*
https://doi.org/10.1145/3675384
- **How it works:** it blends the denoised and noisy images per pixel, with weights from error estimates, and
  **provably converges to the reference** as spp grows. It works with any denoiser and also acts as a-posteriori
  adaptive sampling.
- **For us:** exactly the progressive-viewport behaviour we need. Early frames look denoised; the final image is
  unbiased.

**Statistical Error Reduction for Monte Carlo Rendering (StatER)** — Hiroyuki Sakai, Christian Freude, Michael Wimmer,
David Hahn. *SIGGRAPH Asia 2025.* https://users.cg.tuwien.ac.at/~hiroyuki/StatER/
- **What it does:** denoises both radiance **and the variance estimate**. The cleaner variance drives better
  variance-aware adaptive sampling.
- **Properties:** no training. Strongest at a few hundred spp.

**MARS** (§2.1) is per-technique allocation and belongs here too.

Background:
- Cycles' adaptive sampling: per-pixel error from two half-buffers; studied in RENDERER-STUDY §1.6.

### 2.6 Denoising usable with our path tracer

Non-ML options (usable now):

**A Statistical Approach to Monte Carlo Denoising (StatMC)** — Hiroyuki Sakai, Christian Freude, Thomas Auzinger,
David Hahn, Michael Wimmer. *SIGGRAPH Asia 2024.* https://users.cg.tuwien.ac.at/~hiroyuki/StatMC/
- **How it works:** per-pixel statistical tests (Welch-style) between neighbouring pixels' sample distributions
  decide what to average.
- **Reported:** quality comparable to neural denoisers, with no training and no hallucination. 28 ms at 256 spp in
  their CUDA timing table (OIDN 19.5 ms).
- **Needs:** per-pixel mean, variance and moment buffers, which our film can keep.
- **For us:** the best-documented non-ML **final-frame** denoiser of the period.

**Filtering-Based Reconstruction for Gradient-Domain Rendering** — Difei Yan, Shaokun Zheng, Ling-Qi Yan, Kun Xu.
*SIGGRAPH Asia 2024.* https://cg.cs.tsinghua.edu.cn/people/~kun/2024/GradFiltering.pdf — only if we render gradients.

**Imperfect Image-Space Control Variates for Monte Carlo Rendering** — Chanu Yang, Bochang Moon. *SIGGRAPH Asia 2025.*
https://doi.org/10.1145/3763335 — neighbouring pixels as control variates; needs a second, common-random-numbers render.

**DSCombiner: Double Shrinkage for Combining Biased and Unbiased Monte Carlo Renderings** — Chenxi Zhou et al.
*SIGGRAPH Asia 2025 (TOG 44(6)).* https://njucg.github.io/DSCombiner-Page/
- **What it does:** a statistical combination of a biased image (denoised, or cache-terminated) with an unbiased one.
  The neural refinement is optional.
- **For us:** the formal version of "blend denoised toward unbiased".

**A compact stochastic representation for Monte Carlo Path Traced images** — Matthias Sebastian Treder et al.
*SIGGRAPH Asia 2025.* https://doi.org/10.1145/3757377.3763846 — per-pixel Gaussian mixture of the samples; robust
firefly rejection without a fixed clamp.

ML options (later):

**Optimizing Vulkan Dispatch Schedules for Real-Time U-Net Denoising** — Karl Sassie, Johannes Hanika, Lucas Alber,
Reiner Dolp, Carsten Dachsbacher. *HPG 2026.* https://doi.org/10.1145/3820016
- **How it works:** ONNX is converted to Vulkan compute, and dispatch schedules (block sizes, layouts, fusion) are
  searched.
- **Reported:** the **OIDN U-Net runs fully on GPU compute, faster than TensorRT**.
- **For us:** the most directly applicable ML paper. It shows that an OIDN-class denoiser on plain Vulkan compute is
  practical. Inference only: no training and no tensor-core API needed, although cooperative matrices would help.

**Online Neural Denoising with Cross-Regression for Interactive Rendering** — Hajin Choi, Seokpyo Hong, Inwoo Ha,
Nahyup Kang, Bochang Moon. *SIGGRAPH Asia 2024.* https://cglab.gist.ac.kr/siga24crossdenoiser/ — temporal denoiser
trained on the fly, no dataset; needs online training.

**Neural Kernel Regression for Consistent Monte Carlo Denoising** — Qi Wang et al. *SIGGRAPH Asia 2024.*
https://doi.org/10.1145/3687949 — kernel-predicting denoiser that converges as spp grows.

**Ragged Neighborhood Attention for Spatiotemporal Neural Denoising of Deep Monte Carlo Renderings** — Xianyao Zhang,
Gerhard Röthlin, Tunç Ozan Aydin, Farnood Salehi, Marios Papas. *SIGGRAPH 2026.*
https://studios.disneyresearch.com/2026/07/16/ragged-neighborhood-attention-for-spatiotemporal-neural-denoising-of-deep-monte-carlo-renderings/
— deep-image (deep-Z) denoising; only for a deep-compositing output.

**Nonlinear Noise2Noise for Efficient Monte Carlo Denoiser Training** — Andrew Tinits, Stephen Mann.
*SIGGRAPH Asia 2025.* https://arxiv.org/abs/2512.24794 — trains HDR denoisers from noisy targets only, so no reference
render farm is needed if we train our own.

**Deep Residual Combiner** — Euan Hughes, Weijie Zhou, Toshiya Hachisuka. *EG 2026.*
https://diglib.eg.org/handle/10.1111/cgf70358 — learned fusion of denoised and unbiased estimates.

**Temporally Stable Metropolis Light Transport Denoising using Recurrent Transformer Blocks** — Chuhao Chen, Yuze He,
Tzu-Mao Li. *SIGGRAPH 2024.* https://czzzzh.github.io/MLTD/index.html — MLT-specific.

Background:
- **SVGF** — Schied et al. *HPG 2017.* https://doi.org/10.1145/3105762.3105770
- **A-SVGF** (temporal gradients) — Schied, Peters, Dachsbacher. *2018.* https://doi.org/10.1145/3233301
- These two remain the standard non-ML real-time filters: albedo demodulation, temporal accumulation, variance-guided
  à-trous.

### 2.7 Caustics and specular-diffuse-specular paths

**Sample Space Partitioning and Spatiotemporal Resampling for Specular Manifold Sampling** — Pengpei Hong, Meng Duan,
Beibei Wang, Cem Yuksel, Tizian Zeltner, Daqi Lin. *SIGGRAPH Asia 2025.* https://doi.org/10.1145/3757377.3763927 ·
https://graphics.cs.utah.edu/research/projects/psms-restir/
- **How it works:**
  - Tile-based sample-space partitioning (e.g. 32×32 tiles) keeps SMS Newton solves near good seeds.
  - A per-frame prior concentrates the initial guesses.
  - ReSTIR spatiotemporal reuse multiplies the effective samples.
- **Reported:** their Plane scene goes from 118 ms (SMS) to 35 ms.
- **For us:** the closest thing to **interactive glass caustics** that stays unbiased.

**Specular Polynomials** — Zhimin Fan, Jie Guo, Yiming Wang, Tianyu Xiao, Hao Zhang, Chenxi Zhou, Zhenyu Chen,
Pengpei Hong, Yanwen Guo, Ling-Qi Yan. *SIGGRAPH 2024.* https://arxiv.org/abs/2405.13409
- **How it works:** specular constraints are reformulated as polynomial systems, reduced to univariate root finding
  (resultants; bisection for more bounces). This finds **all** admissible specular paths between two points, with
  no Newton iterations and no seeds.
- **Properties:** deterministic, GPU-friendly, exact for one bounce.
- **Limits:** cost grows with the triangle tuples searched.

**Bernstein Bounds for Caustics** — Zhimin Fan, Chen Wang, Yiming Wang, Boxuan Li, Yuxuan Guo, Ling-Qi Yan, Yanwen Guo,
Jie Guo. *SIGGRAPH 2025 (TOG 44(4)).* https://doi.org/10.1145/3731145 · https://zhiminfan.work/bound_caustics.html
- **How it works:** bounds on vertex position and irradiance per triangle tuple (rational Bernstein bounds) let it
  **stochastically pick** high-contribution tuples for a specular-polynomial-style solve. It stays unbiased.
- **Limits:** at most 2 specular vertices; ignores Fresnel and visibility in the bounds; needs subdivision parameters.
- **For us:** the follow-up that makes specular polynomials scale. Research-grade.

**Computing Manifold Next-Event Estimation without Derivatives using the Nelder-Mead Method** — Ana Granizo-Hidalgo,
Nicolas Holzschuch. *EGSR 2024.* https://hal.science/hal-04702018
- **How it works:** solves the MNEE constraint with derivative-free Nelder–Mead instead of Newton steps.
- **Fit:** works with shading normals and displacement that have no analytic derivatives.
- **For us:** simplifies MNEE on interpolated-normal meshes.

**Photon-Driven Manifold Sampling** — Fei Lee, Jia-Wun Jhang, Chun-Fa Chang. *HPG 2024.*
https://scholar.lib.ntnu.edu.tw/en/publications/photon-driven-manifold-sampling/ — cached caustic photons seed unbiased
SMS walks; large variance reduction for multi-bounce caustics.

**Neural Progressive Photon Mapping** — Justin Benoist, Joey Litalien, Adrien Gruson. *EG 2026.*
https://profs.etsmtl.ca/agruson/publication/2026_NPPM/ — a small network predicts PPM kernel parameters and keeps it
consistent; state-of-the-art equal-time caustics; small ML.

**Segment-based Light Transport Simulation** — Wenyou Wang, Rex West, Toshiya Hachisuka. *SIGGRAPH 2025.*
https://cs.uwaterloo.ca/~thachisu/seglt.pdf — segments, not vertices, as the unit of transport; unifies photon density
estimation and path filtering under marginal MIS. Theory for a later photon/VCM mode.

**Less Can Be More: A Footprint-Driven Heuristic to Skip Wasted Connections and Merges in Bidirectional Rendering** —
Ömercan Yazici, Pascal Grittmann, Philipp Slusallek. *EGSR 2025.* https://doi.org/10.2312/sr.20251180 — prunes VCM
cost if we add VCM.

ReSTIR-based caustics: **ReSTIR BDPT** and **ReSTIR FG** (§2.3).

Background:
- **Manifold Next Event Estimation** — Johannes Hanika, Marc Droske, Luca Fascione. *CGF 34(4), EGSR 2015.*
  https://doi.org/10.1111/cgf.12681
  - Cycles uses it for flagged "shadow caustics" (studied in source in RENDERER-STUDY).
- **Specular Manifold Sampling** — Tizian Zeltner, Iliyan Georgiev, Wenzel Jakob. *SIGGRAPH 2020.*
  https://doi.org/10.1145/3386569.3392430
- **Efficient Caustics Rendering via Spatial and Temporal Path Reuse** — Xiaofeng Xu, Lu Wang, Beibei Wang.
  *CGF 42(7), 2023.* https://diglib.eg.org/handle/10.1111/cgf14975

### 2.8 GPU architecture, coherence, BVH

No 2024–26 paper in the surveyed venues targets wavefront ray sorting in particular. The useful results are these.

**GPU Coroutines for Flexible Splitting and Scheduling of Rendering Tasks** — Shaokun Zheng, Xin Chen, Zhong Shi,
Ling-Qi Yan, Kun Xu. *SIGGRAPH Asia 2024 (TOG 43(6)).* https://doi.org/10.1145/3687766
- **How it works:** you write a megakernel with suspension points, and the compiler splits it into stages that
  wavefront or persistent schedulers run.
- **For us:** confirms that wavefront staging is the right axis to tune. We already hand-split; the paper's scheduler
  comparisons are the useful part to read.

**On Ray Reordering Techniques for Faster GPU Ray Tracing** — Daniel Meister, Jakub Bokšanský, Michael Guthe,
Jiří Bittner. *arXiv 2025.* https://arxiv.org/abs/2506.11273
- **What it does:** evaluates sort keys (origin/direction, estimated termination point) in a **wavefront** path
  tracer with RTX trace kernels.
- **Reported:** **1.3–2.0× faster trace**, but the reordering overhead is hard to recover when tracing is
  hardware-accelerated.
- **For us:** sort for **shading** coherence (material id) first. Sort rays for tracing only on the software-BVH path.

**BVH build and traversal (for the software fallback and later GPU rebuilds):**
- **H-PLOC** — Carsten Benthin, Daniel Meister, Joshua Barczak, Rohan Mehalwal, John Tsakok, Andrew Kensler. *HPG 2024.*
  https://gpuopen.com/download/publications/HPLOC.pdf — single-kernel GPU build at PLOC++ quality, 1.1–3.6× faster.
- **Fused Collapsing for Wide BVH Construction** — Wilhem Barbier, Mathias Paulin. *HPG 2025.*
  https://diglib.eg.org/handle/10.1111/cgf70213 — wide builds 1.4–1.6× faster.
- **DOBB-BVH** — Michael A. Kern et al. *HPG 2025.* https://arxiv.org/abs/2506.22849 — OBB trees from a wide BVH;
  secondary rays +32% on average.
- **SOBB** (*EG 2025*, https://dcgi.fel.cvut.cz/projects/sobb/) and **uBVH** (*HPG 2025*,
  https://dcgi.fel.cvut.cz/projects/ubvh/) — Martin Káčerik, Jiří Bittner; uBVH reports incoherent rays 1.2–11.8×
  faster than an AABB BVH.
- **Memory-Efficient BVHs with Merged Nodes** — Jacob Haydel, Andrew Kensler, Cem Yuksel, Erik Brunvand. *HPG 2026.*
  https://doi.org/10.1145/3820018
- **Axis-Normalized Ray-Box Intersection** — Fabian Friederichs et al. *EG 2025.* https://doi.org/10.1111/cgf.70041 —
  about 11–14% faster slab tests; a free win in our BVH2 fallback.
- Scene-scale systems, context for later instancing: **Real-time Path Tracing of Massive Dynamic Foliage**
  (van Antwerpen et al., *HPG 2026*, https://doi.org/10.1145/3820021) and **Ray Tracing Massive Amounts of Animated
  Geometry** (Gruen et al., *HPG 2026*, https://doi.org/10.1145/3820014).

On M4 (hardware RT) the BVH layout is Apple's business; these matter for the fallback.

**Optimized and Aligned Anisotropic Monte Carlo Sampling Patterns** — Mirco Werner, Johannes Hanika,
Carsten Dachsbacher. *EGSR 2026.* https://cg.ivd.kit.edu/aligned-anisotropic-sampling.php
- **What it does:** per-pixel sampling patterns aligned to discontinuities.

**SZ Sequences: Binary-Constructed (0, 2^q)-Sequences** — Abdalla G. M. Ahmed, Matt Pharr, Victor Ostromoukhov,
Hui Huang. *SIGGRAPH Asia 2025 (TOG 44(6)).* https://arxiv.org/abs/2505.20434
- **What it does:** a drop-in Sobol replacement built with bit operations.
- **Reported:** up to **1.93× lower MRSE** in rendering tests.
- **For us:** cheap to try against our Owen-scrambled Sobol–Burley.

### 2.9 Spectral rendering efficiency and MIS

**Vector-Valued Monte Carlo Integration Using Ratio Control Variates** — Haolin Lu, Delio Vicini, Wesley Chang, Tzu-Mao Li.
*SIGGRAPH 2025 (TOG 44(4)), Best Paper.* https://doi.org/10.1145/3731175 · https://suikasibyl.github.io/vvmc/
- **What it does:** reduces the variance of **all channels** of a vector-valued integrand at once (RGB, spectra,
  derivatives) by reweighting samples with a ratio estimator. Importance sampling can only target one scalar.
- **Reported:** "almost negligible" cost, applicable nearly everywhere in a renderer.
- **For us:** this is the 2025 answer to hero-wavelength color noise. Our pdfs are luminance-driven, and the four
  lanes (→ XYZ) carry the residual chromatic variance. High-value research item.

**A Fluorescent Material Model for Non-Spectral Editing & Rendering** — Laurent Belcour, Alban Fichet, Pascal Barla.
*SIGGRAPH 2025.* https://arxiv.org/abs/2505.19672 — fluorescence in an RGB basis; a feature, not a speedup.

**Controlled Spectral Uplifting for Indirect-Light-Metamerism** — Mark van de Ruit, Elmar Eisemann.
*SIGGRAPH Asia 2024.* https://doi.org/10.1145/3680528.3687698 — compact spectral textures; memory, not speed.

**Correct your balance heuristic: Optimizing balance-style multiple importance sampling weights** — Qingqin Hua,
Pascal Grittmann, Philipp Slusallek. *SIGGRAPH 2025 (TOG 44(4)).* https://doi.org/10.1145/3730819
- **What it does:** learned correction factors multiplied into any MIS heuristic.
- **Reported:** better equal-time results for BDPT and RIS-based direct light.
- **For us:** a final-frame refinement once guiding adds a third technique (BSDF, guide, NEE).

Gap: no 2024–26 paper in these venues targets hero-wavelength sampling efficiency itself. Ratio control variates and
ReSTCV are the nearest.

### 2.10 MCMC (for completeness)

Not first-line for a GPU viewport: **Jump Restore Light Transport** (Sascha Holl, Gurprit Singh, Hans-Peter Seidel;
*SIGGRAPH Asia 2025*; https://arxiv.org/abs/2409.07148) parallelises any MCMC light transport, and **Rao-Blackwellized
Markov Chain Monte Carlo Light Transport** (same authors; *SIGGRAPH 2026*; https://arxiv.org/abs/2605.09117) cuts its variance.

---

## 3. Recommended designs for luce-render

### 3.0 Shared infrastructure: one world-space hash grid

Guiding, the radiance cache and (later) ReSTIR PG all want the same thing: a **world-space hash grid** keyed by
quantised position and a normal bin.

| Part | Design |
|---|---|
| **Key** | `hash(level, floor(p / cell(level)), octahedral normal bin (e.g. 8 bins))` |
| **Level** | Chosen per query from the camera footprint at that vertex, so cells stay about N pixels wide on screen (SHARC/NRC-style). For final frames, step down levels where the illumination signature varies (the idea from Illumination-Aware Spatial Subdivision, EGSR 2026). |
| **Table** | Open addressing, fixed capacity (e.g. 2^22 cells), 32-bit check key, linear probing up to 8 slots. Insertion by `atomicCompSwap` on the key. Eviction by age: stamp the last frame used and reclaim stale cells in a sweep kernel. |
| **Payload** | See §3.1 and §3.2. Stored as separate SoA arrays, so the cache and the guide can be enabled independently. |
| **Writes** | Path vertices append compact **training records** to a buffer: cell, direction, scalar value, pdf, and, for the cache, Spec radiance. A separate `grid_update` kernel folds them in. Float atomics may accumulate sufficient statistics directly. Keep one writer per cell per update step where it is cheap, for determinism in tests. |

Cost model: one hash probe per vertex (a few uncached loads), plus one record append per training vertex.

### 3.1 Path guiding, first version

**Goal:** reduce indirect variance in interiors and glossy interreflections, for both the viewport and final frames.

1. **Distribution.** Each cell holds **K = 4 vMF lobes**: weight, mean direction (oct-encoded), κ, plus sufficient
   statistics for an online EM step.
   - This is the representation ReSTIR PG uses (4 lobes per hash cell) and MCMM shows works online.
   - Directions are world space. Parallax is ignored in v1; cells are small.
2. **Training data.** Use **completed path segments**, not ReSTIR, in v1. When a path finishes or a vertex receives
   its contribution, the incident radiance along the sampled direction ω_i at vertex x is known.
   - Train on the **scalar luminance Y** of the 4-lane contribution × the |cos| and BSDF value at x, divided by the
     sampling pdf. That is, fit the product (cosine-weighted incident radiance), as ReSTIR PG and PPG-product
     variants do.
   - Store the luminance; never store a spectrum. This keeps pdfs scalar and wavelength-free, the rule from
     RENDERER-STUDY §2.5.
   - **Wavefront cost:** a path must remember its last few vertices (cell id, ω_i, pdf, throughput at that vertex),
     so the contribution can be credited back when radiance arrives. Keep a ring of 3–4 vertices in path state, about
     48–64 B. The radiance arriving at vertex k is the path's later radiance divided by
     the throughput accumulated up to k; store that per-vertex throughput **in luminance** to keep it cheap.
3. **Fitting.**
   - **Final frames:** a `guide_fit` kernel runs one weighted EM step per touched cell, once per pass. Use an
     exponential moving average of sufficient statistics (decay ~0.9 per pass while training, then freeze or slow down).
   - **Viewport:** the same step every frame with stronger decay, so lights moving or scene edits re-converge.
4. **Sampling.** One-sample MIS between BSDF and guide.
   - The probability α of picking the guide is set per cell: 0 for untrained cells (count < threshold), and capped
     at 0.5–0.7. Specular and near-specular lobes (roughness < ~0.1) never use the guide.
   - The guide pdf is product-ish: sample the vMF mixture, then multiply by the BSDF in MIS. Do not try to
     product-sample in v1.
   - **MIS weights:** combined pdf `α·p_guide + (1−α)·p_bsdf` for the direction, and the power heuristic against NEE
     as now.
   - **RR uses the unguided throughput**, as Cycles does: guiding must not change survival probabilities, or RR
     fights the guide.
5. **Spectral.** Nothing changes per lane. Guiding pdfs are scalars and the BSDF is evaluated per lane as now.
   Dispersion paths, where secondary lanes are terminated, still train on luminance.
6. **Later upgrades, in order:**
   - **MI reweighting** (Fan et al. 2025) to reuse the training passes in final frames;
   - **illumination-aware cell refinement** (Zheng et al. 2026);
   - **RCPG-style product sampling** if glossy surfaces stay noisy;
   - **ReSTIR PG** feedback once ReSTIR PT exists.
   VXPG stays an alternative if guiding toward small bright regions turns out to matter most.

**Expected payoff:** literature results for PPG/vMF-class guiding in indirect-dominated interiors are typically 2–10×
equal-time MSE reduction. Product shots gain little. Measure on the interior scenes (§4).

### 3.2 Radiance cache for the interactive viewport

**Goal:** short viewport paths (2–4 bounces) that still show multi-bounce light, without the bias being visible.

1. **Payload per cell:**
   - `Spec` outgoing radiance toward the hemisphere: v1 is **irradiance-like plus a diffuse/rough-glossy assumption**,
     stored as 4 lanes plus a luminance second moment;
   - a sample count and a frame stamp.
   - Spectral subtlety: radiance in the cache must be stored **per wavelength**, but each path carries different hero
     wavelengths. Store the cache in a **fixed spectral basis**, for example 8–16 bins over 360–830 nm or RGB
     sigmoid-style coefficients, accumulated by splatting each lane into its bin. Each lookup interpolates at the
     path's 4 wavelengths. The RGB build stores RGB.
   - This is the main spectral-specific design cost of caching.
2. **Update (SHARC-style, non-neural).** Every path vertex appends `(cell, L_lane[4], λ[4])` for the radiance it
   eventually observed (the same back-propagation ring as §3.1). `grid_update` blends it into the cell with a
   moving average whose weight depends on the sample count.
   - Multi-bounce comes for free: a path that terminates into the cache feeds cache values back into training,
     which is NRC's self-training trick in non-neural form.
3. **Termination rule.**
   - **v1:** a path terminates into the cache at bounce ≥ 2, once the path spread (sum of sqrt(pdf)-based spread
     angles × distance) exceeds about 2× the cell size. This is the NRC heuristic.
   - **v2:** variance-trading termination (Kandlbinder et al., HPG 2024), using the cell's stored second moment.
     It keeps total variance under a bound and minimises bias.
   - **Never** terminate after specular, near-specular or transmissive bounces. Caustic paths keep tracing.
4. **Bias control: the two-level estimator.** Following Dereviannykh et al. (EG 2025), but with our non-neural cache:
   - at each terminated vertex, a fraction of paths (e.g. 1 in 8, a viewport slider) keeps tracing and adds
     `(L_traced − L_cache)/q`;
   - this makes the estimator **unbiased** in expectation, and the progressive viewport converges to the right answer
     instead of to the cache's bias;
   - this matters for a product that also renders final frames: the viewport and the final frame must agree.
5. **Expected payoff:** viewport cost per spp drops roughly in proportion to the shortened mean path length. For 12 →
   ~3 mean bounces, expect 2–3× more spp/s in interiors (trace and shade dominate per-bounce cost), with low-frequency
   GI visible from the first frames.
6. **With ML later:** replace the hash-grid payload with an NRC/NIRC-style tiny MLP on a hash-grid encoding. This
   needs fused training, so cooperative matrices in luce-gpu or simdgroup matrices on Metal. Keep the two-level
   residual.

### 3.3 Many lights and ReSTIR DI

1. **Final frames:** keep the light tree.
   - Upgrade its importance from pbrt-v4 light bounds to **SG lighting importance** (Tokuyoshi et al., SIGGRAPH Asia
     2024). The reported gain is ~12% RMSPE at equal time on 167k emitters; more for glossy receivers.
   - Add the Cycles-style **min/max importance** and **wider leaves** already planned in DESIGN.md.
2. **Viewport: ReSTIR DI.**
   - **Candidates:** M = 8–32 light-tree samples per pixel plus 1 BSDF sample. The target function is the
     **luminance** of the unshadowed contribution at the pixel's 4 wavelengths (scalar, as all our pdfs).
   - **Temporal reuse via reservoir splatting** (Liu et al. 2025), not back-projection. It handles thin-lens DOF and
     motion and is cheaper than Area ReSTIR.
   - **Spatial reuse:** 2–3 neighbours chosen by **compatibility** (Junkins et al., HPG 2026; +2–5% cost) with
     pairwise MIS, and **histogram-stratified** candidate selection (Salaün et al. 2025).
   - **Color:** add **ReSTCV** (SIGGRAPH 2026) control variates to remove the per-channel noise of single-sample
     reuse. For spectral this is where hero-wavelength color noise shows.
   - **Spectral reuse rule:** a neighbour's light sample is re-evaluated at **this** pixel's wavelengths. Light
     samples are positions on emitters, so the shift is trivial (reconnection to the same light point). Spectral DI
     reuse is therefore easy; spectral **path** reuse is not.
   - **Bias:** use the unbiased (pairwise MIS) form. The final-frame mode does not use ReSTIR, so viewport and final
     agree up to noise.
3. **Wavefront fit:** ReSTIR DI adds 3 kernels:
   - `di_candidates`, which merges into shade_surface at the first vertex;
   - `di_temporal`;
   - `di_spatial_and_shadow`.
   Reservoirs cost ~32 B per pixel ×2 (current and previous frame). Only the **primary** vertex uses ReSTIR;
   secondary vertices keep light-tree NEE.

### 3.4 Adaptive sampling

1. **v1 (Cycles):** a two-half-buffer error estimate per pixel, a convergence threshold, and dilation of the active
   mask; already planned as `adaptive_*`.
2. **Driven by the denoiser, not raw noise.**
   - **Viewport:** after the first ~8–16 spp, allocate samples where the **denoised** image is uncertain. Use the
     denoiser's per-pixel variance after filtering, which is StatER's idea (variance itself denoised).
   - **Final frames:** stop when the Firmino et al. (SIGGRAPH 2024) error estimate of the denoised image falls below
     the target. "Render until good" replaces "render N spp".
3. **Per-technique allocation:** later, MARS-style per-cell RR/splitting factors for guide vs BSDF vs NEE, stored in
   the same hash grid. Final frames only.
4. **Below 1 spp:** when the viewport moves, render a **perceptual-importance-weighted** subset of pixels instead of
   dropping resolution. This is the non-ML reading of "Forget Superresolution" (SIGGRAPH 2026): edges, highlights and
   shadow boundaries first, from the previous frame's gradient and luminance. The camera kernel already skips pixels
   by mask, so this is a mask policy.

### 3.5 Denoiser

**No ML, viewport.** An SVGF-style filter:
- demodulate by first-hit albedo (we already write albedo and normal at the first diffuse vertex);
- temporal accumulation, with history clamped by reprojection validity;
- 4–5 à-trous passes guided by luminance variance, normal and depth.

The **converging blend** makes it progressive-safe:
- after accumulation, blend `denoised` and `noisy` per pixel with weights from the estimated error, following
  Denisova & Bocchi (HPG 2024), so that at high spp the image is the unbiased accumulation;
- this avoids the "viewport converges to a blurred image" trap;
- all of it is plain compute: subgroup reductions, shared-memory tiles.

**No ML, final frames:** a **StatMC-style statistical denoiser** (SIGGRAPH Asia 2024).
- It needs per-pixel mean, variance and a third moment, which the film can accumulate.
- It is comparable to neural denoisers at a few hundred spp, with no training and no hallucination.
- Optionally use **DSCombiner**-style shrinkage to fuse the cache-biased viewport estimate with unbiased samples.

**With ML later:** an **OIDN-class U-Net on compute**.
- Sassie et al. (HPG 2026) show that the OIDN U-Net runs on plain Vulkan compute, faster than TensorRT, with tuned
  dispatch schedules.
- That is inference only. It works without cooperative matrices, and it is a natural first workload for luce-nn when
  it resumes.
- Keep the converging blend around it. Consider **consistent** kernel-predicting designs (Neural Kernel Regression,
  SIGGRAPH Asia 2024) if we train our own, with Noise2Noise-style training (Tinits & Mann 2025) to avoid needing
  reference renders.

### 3.6 Caustics and transparent shadows through glass

Our current v1 treats glass as opaque to shadow rays. In order:

1. **Transparent shadows for cut-outs and thin glass** (`intersect_shadow` any-hit loop):
   - accumulate per-lane transmittance through cut-outs (exact) and **thin-walled** dielectrics (exact for thin
     sheets, as in Cycles).
   - For solid refracting glass, offer a **"shadow transparency" approximation** in the viewport only (a straight-line
     Fresnel transmittance product, biased and labelled as such). Product renders need it to look right before
     caustics are converged.
2. **MNEE for flagged caster/receiver pairs (final frames and viewport).**
   - Objects flagged "caustic caster" (glass), and receivers flagged "caustic receiver".
   - At a diffuse receiver vertex, NEE solves a 1–2 interface refractive chain to a sampled light point with Newton
     (MNEE).
   - Use the **derivative-free Nelder–Mead** variant (EGSR 2024) where shading-normal derivatives are a pain.
   - It is unbiased only with the SMS-style probability estimate. Cycles accepts MNEE's bias for speed; we should
     decide per mode: biased in the viewport, SMS-unbiased in final frames.
   - Effort L. It covers the classic "glass on table" product shot.
3. **Interactive caustics through resampling.**
   - **Tile-partitioned SMS + ReSTIR** (Hong et al., SIGGRAPH Asia 2025) for the viewport, once ReSTIR infrastructure
     exists. Reported 118 → 35 ms in their test.
   - **Photon-based** alternative: ReSTIR FG (EGSR 2024) for multi-bounce caustics and glass-enclosed objects.
4. **Final-frame general caustics** (water surfaces, multiple glass layers, focused lights):
   - **progressive photon mapping restricted to caustic paths** (photons that hit at least one specular surface), with
     MIS against PT. The literature path is the Segment-based (2025) and VCM-pruning (EGSR 2025) papers.
   - Photon-driven manifold sampling (HPG 2024) uses the same photons as SMS seeds.
   - Neural PPM (EG 2026) when ML exists.
   - ReSTIR BDPT (2025) is the XL option if real-time caustics become a product requirement.
5. **Guiding helps glossy caustics.** The vMF guide learns directions toward glossy reflectors, but not through
   delta interfaces. Keep the "never guide after specular" rule; caustics through delta glass need steps 2–4.

### 3.7 Per-sample cost (do alongside #1)

- **Subgroup-aggregated queue appends:** one atomic per subgroup. This is already planned. Expected to remove
  atomic contention in every queue push.
- **Material-sorted shade_surface:** a counting sort of the surface queue by (material class: diffuse / metal /
  glass / coat) before shading.
  - The literature (and Cycles) report the large gains for shading divergence, not for tracing.
  - The ray-reordering study above finds that trace-sorting overhead is hard to recover with hardware RT.
- **Viewport path length:** with the cache (§3.2), 3–4 bounces. Without it, cap the viewport at 6 with clamping
  stronger than in final frames.
- **Shadow rays:** opaque any-hit with early termination is already cheap. Keep the transparent loop off the opaque
  fast path, with a per-material flag.
- **BVH2 fallback:** axis-normalized ray-box tests (EG 2025); wide BVH (CWBVH, fused collapsing) when the fallback
  matters.
- **Sampler:** A/B test **SZ sequences** against Sobol–Burley (up to 1.93× lower MRSE reported).

### 3.8 Not now

- **ReSTIR PT** (Enhanced, 2026), ReSTIR BDPT, gradient-domain ReSTIR: XL. Revisit after #1–#6, for the viewport only.
- **All neural guiding, caching and light selection:** wait for luce-nn plus cooperative matrices.
- **MCMC integrators:** not GPU-viewport friendly.

---

## 4. What to measure

### 4.1 Instruments

1. **Per-kernel GPU timestamps** for camera, intersect_closest, shade_surface, shade_miss, intersect_shadow,
   shade_shadow, schedule, grid_update and denoise, per bounce. This needs luce-gpu timestamp queries.
2. **Occupancy per bounce:** live path count and queue lengths per kind (surface, miss, shadow), plus the fraction of
   the 2^20 pool active. This shows when the tail of long paths dominates.
3. **Rays per second,** split into closest and shadow rays, against the M4 Max hardware ceiling from a synthetic test
   (primary rays only).
4. **Shading divergence proxy:** distinct material classes per subgroup in shade_surface, sampled with a debug
   counter.
5. **Memory:** path state, hash grid, reservoirs and film buffers, in MB at 1080p and 4K.

### 4.2 Image quality and convergence

1. **Reference renders:** 64k spp per scene (overnight), stored as EXR.
2. **Time-to-error curves:**
   - relMSE and **FLIP** against the reference, plotted against wall-clock time (not spp), from 0.1 s to 60 s;
   - each feature reports **equal-time** improvement, matching the papers' methodology.
3. **Viewport metric:** time until FLIP < 0.05 (denoised) and < 0.02 (undenoised). The "converges in ~1 s" goal
   becomes a number.
4. **Temporal stability:** in a camera-orbit sequence, measure frame-to-frame FLIP of the denoised output against the
   same for the reference sequence. Flicker is the main ReSTIR/denoiser failure.
5. **Bias checks:**
   - viewport (cache + two-level) at high spp against the final-frame reference: the difference must go to zero;
   - with the two-level estimator off, measure the cache bias directly.
6. **Spectral checks:** chroma error (ΔE2000 of the mean color in regions) at equal time between spectral and RGB
   builds. This measures how much ratio control variates and ReSTCV help.

### 4.3 Scene set

| Scene | Stresses |
|---|---|
| Product: glass bottle + metal cap on a studio sweep, dome + 2 rect lights | transparent shadows, caustics, coat |
| Product: car-paint / coated object, HDRI dome | glossy noise, light tree vs dome |
| Interior: room lit only through a window (sun + sky) | guiding, cache, long paths |
| Interior at night: 50–200 small emitters (lamps, LED strips as emissive meshes) | many lights, ReSTIR DI |
| Glass-on-table caustic + a water surface pool | MNEE, SMS, photons |
| Cornell box (with and without spheres) | correctness and white furnace, existing |

### 4.4 Report per feature

For each feature report:
- equal-time relMSE/FLIP change on each scene;
- the per-spp ms change;
- memory;
- whether viewport and final agree.

Features land only if equal-time error drops on the scenes they target and does not regress elsewhere by more than
noise. This follows the "commit only on green" practice used for the rest of luce.

