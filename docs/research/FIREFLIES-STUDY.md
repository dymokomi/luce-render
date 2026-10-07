# luce-render: fireflies without clamping

Status: research note, 2026-10-07. It follows [REALTIME-STUDY.md](REALTIME-STUDY.md), whose §2.7 and §3.6 cover the
manifold methods for caustics (MNEE, SMS, specular polynomials). This note starts from a measurement of our own
Cornell box and asks what removes its fireflies while the image stays **unbiased**: its expected value is the true
image at every sample count, as the owner requires for physically accurate renders. Clamping is ruled out.

Every paper cited was found on a DOI, a publisher page or a project page. Numbers marked "reported" are the authors'
own. Nothing here is code to copy.

---

## 1. What makes the fireflies

### 1.1 Measurement

Setup:
- **Scene:** tests/cornell, 512 × 512, spectral, 12 bounces, no clamp, M4 Max.
- **Images:** each test image has 512 samples (seed 2). Its reference is the same scene at 16384 samples (seed 1).
- **Variants:** each replaces one material with the white diffuse wall material. This was a scratch edit of the test
  and is not committed.
- **Error:** per pixel, summed over RGB.
- **Top 0.1 %:** the share of all squared error that sits in the worst 262 pixels.
- **Outlier:** a pixel whose luminance is above 3 × its reference + 0.05.

| Scene                  | relMSE  | MSE      | error in top 0.1 % of pixels | outliers |
|------------------------|---------|----------|------------------------------|----------|
| full                   | 0.0416  | 0.00531  | 64 %                         | 162      |
| glass ball → diffuse   | 0.0057  | 0.00031  | 15 %                         | 0        |
| gold ball → diffuse    | 0.0390  | 0.00521  | 64 %                         | 153      |
| both → diffuse         | 0.0048  | 0.00027  | 17 %                         | 0        |

**The glass ball makes the fireflies.**
- It multiplies the error by 17 and puts two thirds of it into a few hundred pixels.
- The rough gold ball (roughness 0.35) barely matters. NEE works at its surface, so light reflected off it is found
  by ordinary light samples.
- Most outliers sit on the ceiling (the median outlier is in row 63 of 512). The rest sit on the floor around the ball.

### 1.2 The paths

The notation is E (eye), D (diffuse or rough), S (smooth glass) and L (light).

| Path        | Where                                       | How our path tracer finds it |
|-------------|---------------------------------------------|------------------------------|
| E D S S L   | the caustic on the floor under the ball     | only by a BSDF sample from the floor that passes through the ball and hits the 0.5 × 0.4 light |
| E D D S S L | ceiling and walls lit by that caustic       | the same lucky hit, one diffuse bounce later |
| E S S D S S L | the caustic seen through the ball         | practically never |

Next-event estimation cannot help with any of them:
- A shadow ray is straight, so it can't follow the bend through the glass. The ball blocks it.
- The renderer therefore finds this light only when a cosine-distributed direction happens to land on the small
  image of the light through the ball.
- That is rare. Each path that does land there carries the focused light of the whole caustic, and so becomes one
  bright pixel.

This is correct Monte Carlo behaviour: the variance is real, and the fireflies are not a bug. Reaching the error level
of the diffuse-glass scene would take about 7× more samples by relMSE (≈ 3,700). Single fireflies stay visible long
after that.

### 1.3 Checked and ruled out

| Suspect | Finding |
|---------|---------|
| **Light hits after smooth glass weighed wrongly** | Smooth glass is GGX at α = 1e-4, so its sampled pdf is huge. The power-heuristic weight of a light hit after it is ≈ 1, which is right because the light sample at the glass vertex contributes ≈ 0. |
| **Light-tree pdf differs between light samples and BSDF hits** | Both call `tree_pdf` from the same point (SV_VERTEX) and the same normal. The closed-form light tests in tests/gpu agree. |
| **Area sampling of the rect light near its edge** | Sampling by area makes the estimate unbounded near the light, but the diffuse-glass variant has no outliers at all, so it is not this scene's problem. It is still worth fixing (§3.5). |
| **Russian roulette** | Survival is min(0.95, max throughput), so a surviving path gains at most ~1/0.95. The fireflies arise before roulette, from the 1/pdf of the lucky sample. |
| **Hero-wavelength termination** | Our dielectrics have a constant IOR (no dispersion), so the secondary wavelengths are never terminated. |

### 1.4 What clamping costs

The cost was measured on 16384-sample references, clamp 10 against no clamp:

| Region                         | Light removed by clamp 10 |
|--------------------------------|---------------------------|
| whole image                    | 3.0 %                     |
| ceiling                        | 1.1 %                     |
| back wall                      | 2.1 %                     |
| floor strip                    | 18.9 %                    |
| caustic under the glass ball   | 59 %                      |

A clamp removes most of the caustic and a few percent of everything else. It never converges to the right image,
however many samples are taken. The no-clamp rule is right for final frames. The Render node's default clamp of 10 is
a preview setting and should be labelled that way (§4).

---

## 2. The literature, by what it fixes

### 2.1 Light tracing combined with path tracing (unbiased)

**Light tracing** follows paths from the lights. At every non-specular vertex it connects to the camera with one
shadow ray, finds the pixel, and adds the contribution to a splat buffer with atomics. This is the t = 1 strategy of
bidirectional path tracing.
- **Our dominant paths:**
  - L S S D: the light leaves the lamp, refracts twice through the ball, lands on the floor and connects to the camera.
  - L S S D D: the same path continues to the ceiling and connects to the camera from there.
  - **Both dominant classes in §1.2 are easy for light tracing.**
- **Combining with path tracing:** each path's weight is the power heuristic over the techniques that could have made
  it: a BSDF hit on the light, a light sample, and the camera connection. The weights need the reverse pdfs along the
  path, carried as running sums: the recursive MIS quantities of VCM (Georgiev et al., SIGGRAPH Asia 2012,
  https://doi.org/10.1145/2366145.2366211), restricted to three techniques.
- **Precedents:**
  - Mitsuba 3's `ptracer` traces from the lights and connects to the sensor at each bounce, in wavefront mode;
    its documentation says it does notably well with caustics
    (https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html).
  - LuxCoreRender has a "hybrid back/forward" mode for OpenCL GPUs. It assigns path classes to one technique each,
    instead of MIS-weighting them.
- **GPU design:** Davidovič, Křivánek, Hašan, Slusallek, *Progressive Light Transport Simulation on the GPU: Survey and
  Improvements*, ACM TOG 33(3) 2014, https://doi.org/10.1145/2602144. It compares PT, LT, BPT, PPM and VCM in one GPU
  framework, and splats light-tracing contributions with framebuffer atomics.
- **Unbiased:** yes.
- **Fit for us:** good.
  - Light paths are one more wavefront loop that reuses intersect_closest, intersect_shadow and the OpenPBR
    evaluation.
  - The splat is a float atomic add, which luce-gpu already has. Lanes are converted to the film's color before the
    add, as `finish` does now.
- **Does not fix:** E S S D S S L, the caustic seen through the ball, since the camera cannot connect through
  smooth glass.

### 2.2 Full BDPT: the light vertex cache (unbiased)

LVC-BPT comes from Davidovič et al. 2014 (§2.1):
- **How:** all light subpaths are traced first, and every vertex goes into one global cache. Each camera vertex then
  connects to a few cache vertices chosen uniformly. MIS uses the recursive weights, so nothing is kept per subpath.
  The cache is sized from probe paths.
- **Reported:** 30–60 % faster than the earlier GPU BPT implementations.
- **What it adds over §2.1:** connections from diffuse vertex to diffuse vertex. These matter for light that reaches
  the room indirectly (a lamp behind a shade, a sunlit floor seen around a corner).
- **Here:** light tracing already covers both dominant classes, so full BDPT is a later step.

Efficiency-aware MIS for bidirectional algorithms (Grittmann, Yazici, Georgiev, Slusallek, SIGGRAPH 2022,
https://doi.org/10.1145/3528223.3530126) picks, per pixel, how much each technique gets. It keeps a bidirectional
mode no slower than plain path tracing on scenes that don't need it.

### 2.3 Path guiding that learns caustics (unbiased)

Our guiding v1 trains only from camera paths, which is circular:
- At the floor, it learns that light arrives through the ball only from the rare paths that already went through
  the ball.
- Those paths are the fireflies themselves.

The fix is to train from the light side:

- **Vorba, Karlík, Šik, Ritschel, Křivánek**, *On-line Learning of Parametric Mixture Models for Light Transport
  Simulation*, SIGGRAPH 2014, https://doi.org/10.1145/2601097.2601203.
  - It fits directional mixtures by online EM, training the camera-side guide from photons traced from the lights.
  - Photons that pass through the ball teach the floor "light comes from the ball" before any camera path has found
    it.
  - With §2.1 this costs almost nothing extra: the light pass already makes those vertices.
- **Rath et al.**, *Variance-Aware Path Guiding*, SIGGRAPH 2020, https://doi.org/10.1145/3386569.3392441.
  - It trains toward the second moment instead of the radiance, and accounts for the other techniques (NEE, the
    defensive BSDF share).
  - That sends samples where the variance is, which here is toward the ball. It is a small change to our
    training target.
- **Ruppert, Herholz, Lensch**, *Robust Fitting of Parallax-Aware Mixtures for Path Guiding*, SIGGRAPH 2020,
  https://doi.org/10.1145/3386569.3392421.
  - It reprojects lobes to the shading point from an estimated source distance. Ours are not reprojected.
  - The ball and the light are close to the floor, so this matters here. Open PGL builds on this method.
- **Karlík et al.**, *MIS Compensation*, SIGGRAPH Asia 2019. It subtracts from the guide what BSDF sampling already
  covers. Reported 2.75× lower error in its headline example. Trivial once a guide pdf exists.
- **Rath, Yazici, Slusallek**, *Focal Path Guiding*, SIGGRAPH 2023, https://doi.org/10.1145/3588432.3591543.
  - It samples toward the points that many paths pass through, such as the focus behind a lens. A glass ball is a
    lens.
  - It is research-grade, and how much it helps from the receiver side here is untested.
- **Real-time guiding:** VXPG (Lu, Chang, Hedstrom, Li, SIGGRAPH 2024) guides toward lit voxels. It would aim the
  ceiling at the bright floor (E D D S S L), but not the floor at the ball.

Production state:
- Cycles runs guiding on the CPU only, and its manual says guiding "is not designed to be a caustic solver".
- Karma XPU users report refractive caustics needing more than 15000 samples. This is a forum report, not
  documentation.

Guided sampling stays unbiased as long as its pdf is exact and the BSDF keeps a defensive share.

### 2.4 Russian roulette and splitting (unbiased)

- **ADRRS:** Vorba, Křivánek, SIGGRAPH 2016, https://doi.org/10.1145/2897824.2925912.
  - It weighs a path's throughput by the cached incident radiance ahead of it.
  - Paths are killed below a window and **split** above it.
  - Splitting at the floor vertex sends several BSDF samples into the hard part of path space, without bias.
- **EARS:** Rath et al., SIGGRAPH 2022, https://doi.org/10.1145/3528223.3530168. It chooses the roulette and
  splitting factors that maximise 1 / (variance × time), using learnt per-region statistics.
- **Cycles:** survival is `min(sqrt(max throughput), 1)`. Dim paths live longer, and survival is capped at 1, not
  0.95.
- **For us:** splitting needs path states spawned mid-pass (a queue append), which the refill experiment already
  explored. It helps every estimator but does not, by itself, fix the caustic.

### 2.5 Manifold methods (unbiased with the SMS estimate)

These are covered in REALTIME-STUDY §2.7: MNEE, SMS, specular polynomials, and PSMS with ReSTIR (SIGGRAPH Asia 2025).
- They are the only unbiased route to E S S D S S L, the caustic seen through the ball.
- At a diffuse point they also give a direct "light sample through the glass", which fixes E D S S L from the camera
  side.
- They need flagged caster objects, Newton (or Nelder–Mead) solves and the SMS probability estimate to stay
  unbiased. Effort L.

### 2.6 Light sampling near the light (unbiased)

- **Ureña, Fajardo, King**, *An Area-Preserving Parametrization for Spherical Rectangles*, EGSR 2013,
  https://doi.org/10.1111/cgf.12151.
  - It samples a rect light uniformly in solid angle, which bounds the estimate near the light. Arnold and Cycles
    use it.
- **Peters**, *BRDF Importance Sampling for Polygonal Lights*, SIGGRAPH 2021. It adds the receiver's cosine and is
  exact for an unoccluded Lambertian receiver.
- **Fit:** both are a few hours of work, plus the matching pdf in the light-hit MIS.

### 2.7 Consistent and biased methods, for the record

| Method | Bias | Converges to the true image? |
|--------|------|------------------------------|
| Clamping (Cycles clamp direct/indirect) | loses energy (§1.4) | **no** |
| Filter glossy, caustics toggles | changes materials or drops paths | **no** |
| Path-space regularization (Kaplanyan, Dachsbacher, EG 2013) | blurs smooth vertices so NEE works through glass | only if the blur shrinks with samples |
| Progressive photon mapping / VCM merging; Octane and Redshift caustics | density estimation | only if the radius shrinks with samples |
| Median of means (Buisine et al., EGSR 2021); firefly reweighting (Zirr, Hanika, Dachsbacher, EG 2018) | biased at any finite count | consistent; Zirr keeps the unbiased sum recoverable |

Two uses fit the owner's rule, because the stored image stays the unbiased mean:
- a preview display that applies median of means or reweighting on top;
- a denoiser later.

Nothing in this table belongs in the accumulation of a final frame.

---

## 3. Plan for luce-render

### 3.1 Measure (small, first)

- Write the film as raw floats next to the PPM, and keep a second-moment plane. This gives a per-pixel variance
  image, i.e. a firefly map.
- Report relMSE, the top-0.1 % error share and relMSE × time in `LUCE_BENCH`, against a long reference.
- Keep the §1.1 variants available behind a test switch, so each fix is credited to the paths it targets.

### 3.2 Light tracing with MIS against path tracing (medium-large; the main fix)

Design:
- **Light paths.** Each pass also starts a share of its paths at a light. The light is chosen from the light tree by
  power, and a point and direction are chosen on it.
- **Shared kernels.** Light paths reuse intersect_closest and the OpenPBR evaluation.
- **Camera connection.** At each non-specular vertex: project the vertex to a pixel, trace a shadow ray to the
  camera, weigh by the camera's importance, and atomically add to a splat plane. The splat is converted to film color
  per lane, as `finish` does.
- **MIS.** Both kinds of path carry the VCM-style running pdf sums for three techniques: a BSDF hit on the light, a
  light sample, and the camera connection. Every path gets its power-heuristic weight, and the weights sum to 1.
- **Spectral.** A light path samples its own hero wavelengths. Each lane is splatted with that lane's pdf, so camera
  and light paths need not share wavelengths.
- **Budget.** Light paths are a setting: a share of each pass, 0 by default for preview and on for final frames.
  §2.2's efficiency-aware MIS can choose the share later.

Expected effect, which is a prediction to be verified by §3.1:
- Floor and ceiling fireflies (E D S S L, E D D S S L) go away.
- The caustic under the ball converges at the rate of the diffuse scene.
- A light path costs about one camera path.

Tests:
- A closed-form test with an unoccluded light, where light tracing alone must equal the reference.
- A check that the MIS weights sum to 1 on random paths.
- A light-tracing-only render of the Cornell box that agrees with the path-traced reference within noise.

### 3.3 Guiding trained from the light pass (small once §3.2 exists)

- Light-path vertices feed the guide grid at diffuse cells (Vorba 2014).
- Switch the training target to variance-aware (Rath 2020), and add parallax (Ruppert 2020) and MIS compensation
  (Karlík 2019).
- Guiding then helps the camera paths that light tracing does not reach: glossy receivers, and diffuse-to-diffuse
  paths away from the camera's view.

### 3.4 Manifold sampling for caustics seen through glass (large)

Use the MNEE/SMS plan in REALTIME-STUDY §3.6, with the SMS probability estimate so that final frames stay unbiased.
It covers E S S D S S L, which nothing above does.

### 3.5 Small, independent

- Sample rect lights uniformly in solid angle (Ureña 2013).
- Make survival `min(1, sqrt(max throughput))`, as Cycles does.
- Split at the first diffuse vertex once queue appends exist (ADRRS-lite).

### 3.6 Settings and naming

- The Render node keeps "Clamp indirect", but its default becomes 0 (off) for final renders.
- The inspector labels the clamp as biased.
- The Cornell benchmark stays unclamped, so speed comparisons always measure the unbiased image.

---

## 4. Results (2026-10-07)

§3.2 is built (docs/DESIGN.md, "Light tracing"); light paths are on by default
at 0.5 a pixel.

| Cornell box, 512 × 512, 512 samples | relMSE | ms a sample | relMSE x time |
| --- | --- | --- | --- |
| path tracing | 0.0392 | 3.14 | 0.123 |
| + light paths, 0.5 a pixel | 0.0041 | 4.79 | 0.020 |

- **Bias:** none measurable. The all-diffuse Cornell box agrees with path
  tracing to 0.01%; with the glass, light paths sit inside path tracing's own
  seed-to-seed spread (0.5% on the caustic floor).
- **Two bugs that looked like physics, and taught the rules:**
  - flat lights must stop light paths the mirror way they stop camera rays;
  - a camera connection must stop at a visible light the camera would hit.

  Each broke the reciprocity the weights assume.
- **Guiding trained by light paths** (§3.3): tried; it doubled the error on the
  gold ball and the clear coat (guided samples on glossy lobes) and did not help
  the glass. Not kept.
- **What remains:** caustics seen through glass or in a clear coat
  (E S S D S S L). §3.4's manifold sampling is next.

