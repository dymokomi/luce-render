# luce-render design

luce-render is a spectral path tracer that runs only on the GPU. The host, in
Luce Base, compiles a luce-geocore `GeometrySet` into buffers and records
compute passes on luce-gpu. All tracing happens in GLSL compute kernels: Metal
on macOS, Vulkan on Windows and Linux. There is no CPU renderer, and none is
planned.

The decisions below come from a source study of Cycles X, pbrt-v4, OpenPBR and
OpenPGL. [research/RENDERER-STUDY.md](research/RENDERER-STUDY.md) has the
citations and the alternatives. The references were studied, not copied: all
code here is our own.

## Goals

- **Fast:** a wavefront integrator, hardware ray queries where the GPU has them,
  a light tree, adaptive sampling, and later path guiding.
- **Correct color:** spectral by default, with a RGB build of the same code. The
  film is in XYZ, converted to ACEScg. Tone mapping happens downstream (luce-color,
  ACES 2.0).
- **One material model for now:** OpenPBR Surface. Shader assignment comes later.
- **One of several engines:** luced-3d's Render node picks the engine. luce-render
  is the first, behind the same scene description (USD-shaped cameras and lights in
  the GeometrySet).

## Scene

`compile_scene` produces the following (`src/render/scene.lucb`):

- **Triangles:** world-space points, three indices per triangle, and each
  triangle's polygon. Visible instances are realized. Analytic CAD is skipped until a
  Tessellate node; `skipped` counts it.
- **Cameras and lights:** flattened to world space from the scene component.
  Cameras follow UsdGeomCamera; lights follow UsdLux rect, disk, sphere and distant.
- **The BVH:** binned SAH, binary, with a 32-byte node (`src/render/bvh.lucb`). This
  is the software path. Ray queries (luce-gpu Tier B) replace it where the GPU
  supports them, with a two-level hierarchy of one BLAS per prototype and a TLAS of
  instances.

Planned layout changes for the integrator:

- **Triangle vertices:** pre-gathered as 3×vec4 per triangle, so intersection reads
  one place.
- **Per-vertex attributes:** octahedral normals and UVs.
- **Per-triangle material id.**
- **No `vec3[]` arrays:** std430 pads each element to 16 bytes.

## Gaussian splats

Splat clouds (luce-geocore's conventions: `orient`, `scale`, `opacity`,
linear `Cd`, the `sh` array, optional `restorient`) are ray traced as 3D
Gaussian Ray Tracing does (Moenne-Loccoz et al. 2024). The code is our own.

- **The model.** A ray meets each Gaussian once, at its point of maximum
  response. There it covers α = min(0.99, opacity · exp(-½ d²)), with d the
  ray's Mahalanobis distance from the center. Its color is the SH evaluated
  toward the ray's direction, added to the DC in the cloud's encoding,
  clamped and decoded, exactly as the viewport's `display_color`. Splats under
  1/255 or past 3σ do not count, as in the viewport.
- **The scene** (`splats.lucb`): every cloud and visible instance placed
  exactly (`placed_point_store`), 64 bytes a splat plus its SH halves, and a
  box out to where α falls to the cull. Hardware ray queries trace a Blas of
  those boxes (luce-gpu `Blas.create_boxes`), instanced beside the triangles'
  Blas with mask 2 (triangles have mask 1). The software path has a BVH of
  its own over the boxes, reached by address.
- **Traversal** (`shaders/splats.glsl`): a k-buffer. Each round gathers the 32
  nearest splats past the last round's farthest; once full, a generated
  intersection at the 32nd culls everything beyond it. The round is
  composited front to back, and rounds go on until a round is not full.
  Shadow rays and light paths need only the product of (1 - α), in any order.
  The k-buffer is distances and splats only, inserted by a fully unrolled
  pass so it stays in registers; α is evaluated again when a round is
  composited. Box candidates come once each (the Blas forbids duplicate
  any-hit candidates).
- **In the integrator** splats are a non-scattering layer that emits and
  absorbs:
  - A camera path's segment adds Σ Tᵢ αᵢ cᵢ times the path's throughput, then
    keeps T = Π (1 - αᵢ) of its throughput for what lies behind.
  - Shadow rays, light paths and light paths' camera connections keep T.
  - No technique samples splats as emitters, so their emission has one
    technique and weight 1. Transmittance is deterministic and the same for
    every technique that carries light along a segment, so the MIS weights
    are untouched and the estimate stays unbiased.
  - Captured light is emitted as captured. Relighting (Relight GSplats) bakes
    new colors into `Cd` and the SH before rendering.
  - Below T = 0.01 a segment goes on by Russian roulette (survives with
    probability T / 0.01 at T = 0.01), so expected emission and transmittance
    are exact.
- **Checked** (`tests/gpu/splats.lucb`): every pixel of overlapping, rotated,
  anisotropic splats with SH over an emissive quad, against a CPU composite in
  f64 (both builds, both traversals); a splat's shadow on a floor against
  1 - α; the software BVH against ray queries on 3000 random splats.

### Splat tracing speed

The bonsai capture (1.24M splats) at 1920 × 1080, splats alone, ms a sample
(the least of several runs, A and B alternated):

| | Metal (M4 Max, GPU shared) | RADV (Radeon 890M) | NVIDIA (RTX A5500 Laptop) |
| --- | --- | --- | --- |
| 16-hit k-buffer indexed at run time, α kept | 3046 | 673 | 150 |
| unrolled, α kept | | 597 | 99 |
| unrolled, α evaluated again | | 539 | 94 |
| 32 hits | | 433 | 75 |
| one ray query object a kernel | 1936 | 357 | 76 |

Another program held the Mac's GPU at 100% throughout, so its numbers are
several times what an idle M4 Max gives (the 16-hit kernel took about 670 ms
idle); their ratio is what counts. Images are the same to 0.04% (roulette).

Where the time goes (NVIDIA, 32 hits): a traversal that tests nothing costs
52 of the 76 ms; testing every box's splat, 4 more; SH colors about 11.

Tried and dropped:
- **Proxy meshes** (3DGRT's): an icosahedron around each splat in triangle
  Blases, back faces culled, so the hardware rejects misses; a zero-length
  query of the proxies' boxes finds those that hold a ray's start (their
  entry face lies behind it). Candidates fall from 139 to 40 a ray and the
  image matches, but rays cost 2× on NVIDIA, 6× on RADV and about 1.3× on
  Metal: 25M triangles make a deep, overlapping BVH, and a ray query returns
  to the kernel for a triangle candidate as for a box. Only a triangle's own
  distance can be committed, so the k-buffer's cutoff culls less. Without
  duplicate-free candidates the same splat was met up to three times.
- **8 hits**: more rounds, each a whole traversal (120 ms on NVIDIA).
- **No cutoff commit**: 136 ms on NVIDIA.
- **48 hits**: no better than 32.

## Spectral and RGB from one code path

All tracing code works on a `Spec` type:

```glsl
#if SPECTRAL
#define Spec vec4   // four wavelengths: one sampled, three rotated across 360–830 nm
#else
#define Spec vec3   // linear ACEScg
#endif
```

`tools/build_kernels.py` builds every tracing kernel twice, with `-DSPECTRAL=1`
and `-DSPECTRAL=0`. The Render node's Color menu chooses between them. The rules:

- **Wavelengths:** each path carries four hero-rotated wavelengths. They are drawn
  from pbrt's analytic visible-wavelength pdf, using one camera sample dimension.
- **Probabilities are scalars.** Lobes and lights are chosen from wavelength-free
  values (the luminance of their RGB parameters), so MIS weights never depend on the
  wavelength.
- **RGB parameters become spectra** through Jakob–Hanika sigmoid coefficients
  (c0, c1, c2, scale). The tables are built offline for ACEScg and Rec.709, each
  64³. In the RGB build the same parameters are read as plain RGB.
- **Dispersion** (OpenPBR's `transmission_dispersion_scale` and Abbe number): the
  index follows Cauchy's law through `specular_ior` at 587.6 nm. A path meeting a
  dispersive material ends its three secondary wavelengths (their lanes zero,
  their wavelengths negated so the film skips them) and lane 0 carries the
  sample ×4; dispersive indices are then the hero's, on camera and light paths
  alike, so MIS stays consistent. Light paths make the rainbow caustics.
- **Lights:** emitter radiance is normalized so OpenPBR's `emission_luminance`
  (nits) lands on Y. Blackbody temperature is evaluated per wavelength.
- **Film:** accumulates XYZ through the CIE 1931 2° curves at 1 nm, then converts
  to ACEScg. The RGB build accumulates ACEScg directly.

## Integrator: wavefront on queues

Each pass traces one sample per pixel. The path pool defaults to 2^20 paths, and
the image is covered in bands when it has more pixels than that. Per-path state
lives in one buffer as an array per field, the layout Cycles uses. It is bound
twice, once as `uint[]` and once as `vec4[]`, with field offsets in the
constants. That fits luce-gpu's 16 bindings until buffer device addresses
arrive. The state takes about 216 bytes per path: 128 main, 64 shadow and 24 of
queues.

Kernels:

| Kernel | Does |
| --- | --- |
| `schedule` | One thread. Turns queue counters into indirect dispatch arguments and clears the next ones, so the host never reads back between kernels. |
| `camera` | Fills free paths: pixel and filter sample, wavelengths, thin-lens ray. Skips converged pixels. |
| `intersect_closest` | Traces the BVH or a ray query. Applies Russian roulette. Sends each path to the surface queue or the miss queue. |
| `shade_surface` | OpenPBR: emission with MIS, a light-tree light sample (shadow ray in the same slot as its path), a BSDF sample, and the denoiser's albedo and normal at the first diffuse vertex. |
| `shade_miss` | Distant lights and background. Writes the path to the film and frees it. |
| `intersect_shadow` | Any hit, opaque only in v1. |
| `shade_shadow` | Adds an unoccluded light sample to its path's radiance. No atomics, since the slot is the path's own. |
| `film_convert` | Divides by the sample count and converts XYZ to ACEScg, writing an rgba32f image. |
| `adaptive_*` | Cycles' convergence test and filter. |

The per-bounce order is fixed:

```
schedule → intersect_closest → schedule → shade_surface, shade_miss
         → schedule → intersect_shadow → schedule → shade_shadow
```

A batch records K bounces, with `camera` refilling the pool at the top of each
batch. The host reads the live count one batch behind, without waiting. Each
batch stays well under 50 ms, following luce-gpu's rule for long work: submit
the next pass only once `done()` reports the previous one finished.

Paths need no RNG state. Sobol–Burley (Owen-scrambled) draws dimension
`16·bounce + k` from the pixel and sample index, with blue-noise ordering first
for the viewport.

## Lights

- **UsdLux shapes:** rect, disk, sphere, distant and dome. Rect and disk lights
  are hit by BSDF rays (lights.glsl's intersect_light).
- **Dome lights** (environment.lucb): an OpenEXR lat-long image (linear
  Rec.709, made ACEScg), its center down the light's -Z.
  - **Sampling:** texels are constant and sampled in proportion to luminance
    × sin θ (rows, then the row's texels, by binary search), so the pdf follows
    the radiance exactly. Like distant lights, domes are chosen by power share
    and MIS-weighted on misses.
  - **Spectra:** texels become spectra linearly: three smooth basis spectra
    that sum to the flat one, weighted by a 3 × 3 matrix of the ACEScg. White
    stays flat and colors are exact without a per-texel fit.
  - **Storage:** the data rides in the lights buffer after the light tree.
  - **Checked:** a white furnace under a uniform dome; a floor under a sky with
    a sun 1/512 of the texels agrees with the image's irradiance to 0.03%.
- **Light tree:** pbrt-v4's light-bounds importance over a median-split binary
  tree, one emitter a leaf. It covers the lights that have a place and the emissive
  triangles. Each emitter's trail gives its pdf for MIS (light_tree.lucb,
  lighttree.glsl). Cycles' min/max importance and wider leaves can come later.
- **MIS:** power-heuristic between light sampling and BSDF sampling. Russian
  roulette, indirect clamping and filter-glossy follow Cycles.

## OpenPBR

| Stage | Lobes |
| --- | --- |
| v1 (done) | EON diffuse (energy-preserving Oren–Nayar), F82-tint metal, GGX specular with anisotropy over albedo-scaled diffuse, coat with darkening, absorption and base roughening, emission, opacity cutout, rough dielectric transmission (no interior medium; shadow rays treat glass and cut-outs as opaque) |
| v2 | transmission depth, dispersion, fuzz, thin film |
| v3 | subsurface random walk |

**Node graphs** (shader_graph.lucb, shaders/shader_program.glsl, after
[research/SHADER-NODES-STUDY.md](research/SHADER-NODES-STUDY.md) and Cycles'
SVM):
- **The graph:** a material's `shader_graph` text parameter holds nodes,
  constants and links, one statement a line (or after a `;`). Its
  `openpbr_surface` node's linked inputs drive OpenPBR parameters; unlinked
  ones leave the material record as it is.
- **Compiling:** the host compiles each graph to a flat u32 program. An operand
  is a constant's bits or a NaN-tagged stack slot; the stack is 64 floats.
- **Running:** shade_surface runs the program once a hit, for camera and light
  paths alike, before the surface is read.
  - Driven parameters replace the record's in read_material and read_colors;
    driven colors go through the environment basis.
  - Programs see the surface's outward shading normal, so a bumped surface
    is the same seen from either side.
  - Only Fresnel and Layer Weight read the view (toward the camera's side of
    the path). Light paths run such a program toward the lens, connect, and
    stop. Camera paths give the light tracer zero density through such
    vertices past the first, so MIS stays consistent.
  - A scene without graphs drops the interpreter (HAS_PROGRAMS).
- **Nodes:**
  - inputs: uv, position, value, color;
  - textures and procedurals: image (PNG, JPEG, OpenEXR via the texture
    table), checker, Perlin fractal noise;
  - math and color: math, mix, map range, clamp, separate, combine, a
    two-stop ramp;
  - normals: normal map (tangent frame from the triangle's UVs); bump (the
    height subgraph compiled three times, a camera-pixel footprint apart);
  - facing: Fresnel, Layer Weight.
- **Checked:**
  - an image-driven floor lights as a grey material (0.02%);
  - a graph of every node kind wired to grey matches exactly;
  - linear height ramps, by position and by UV, bump to the tilted normal;
  - a view-reading material's image mean agrees with light paths on and off.

Energy compensation uses precomputed albedo tables (study §3.5):

- GGX E (32²) and its average;
- dielectric reflection (32×32×16);
- glass albedo (16³).

EON, fuzz and thin film are closed form and need no tables.

## Speed, later

- **Ray queries** (Metal `intersection_query` via spirv-cross; confirmed to compile),
  with BVH2 as the fallback.
- **GPU BVH refit and build** for interactive edits.
- **Sorting** the surface queue by material.
- **Path guiding:** a GPU hash grid of small von Mises–Fisher mixtures, trained on
  luminance. Russian roulette keeps the unguided throughput.
- **Viewport sampling:** ReSTIR DI for the viewport mode only.
- **Subgroup-wide queue appends** once luce-gpu has subgroups.
- **Wider nodes:** CWBVH 8-wide nodes.
- **Buffer device addresses** remove the 16-binding packing.

## Validation

- **Host:** the BVH covers every triangle once, and its boxes nest (`tests.lucb`).
- **GPU:** the probe kernel matches brute-force intersection on every pixel
  (`tests/gpu`).
- **To come:**
  - white-furnace tests per lobe;
  - spectral against RGB on non-dispersive scenes (they should agree within the
    upsampling error);
  - convergence against a reference Cornell box;
  - per-kernel timings once luce-gpu has timestamps.

## Status

v1 renders OpenPBR v1 (EON diffuse, GGX specular over it, F82 metal, coat,
rough dielectric transmission, emission, cut-out opacity) with per-face material
binding, UsdLux lights and emissive meshes. Light samples come from a light tree
over every emitter (power-chosen distant lights aside), MIS-weighted against BSDF
samples through trail-recomputed tree pdfs. It has an indirect clamp, per-kind
bounce limits, hardware ray queries with the software BVH as fallback, and
spectral and RGB builds.

GPU tests check it against closed forms or self-consistency:

- **White furnaces:** diffuse under specular, rough EON, rough metal, coat, smooth
  glass.
- **Cut-out opacity.**
- **Emission.**
- **Assigned materials.**
- **Rect light and emissive quad irradiance:** both against the closed form.
- **Spectral against RGB.**
- **Ray queries against the BVH.**

**Speed:** about 9 ms a 1080p spectral sample at 12 bounces on an M4 Max.

Since then:

- **Adaptive sampling:** Cycles' half-buffer metric with a 3 × 3 widened
  stop.
- **Cut-out shadows.**
- **Per-kernel GPU timings.**
- **Path guiding v1** (shaders/guide.glsl), after REALTIME-STUDY §3.1:
  - a world-space hash grid of 4-lobe vMF mixtures, with cells sized by the
    camera footprint;
  - trained on cosine-weighted incident luminance from each path's first
    vertices, fitted once a pass;
  - one-sample MIS with the BSDF, with Russian roulette on the unguided
    throughput.

  It is off by default. In a room lit by a ceiling spot it cuts error 1.22× at
  equal samples, and costs about 25% a sample (2.5 ms at 1080p). Making it pay
  everywhere comes next:
  - cheaper training: subgroup-aggregated atomics, or a fraction of paths once
    the fit is stable;
  - soft EM, product with the BSDF, MI reweighting of training passes, and
    illumination-aware cells.

Performance, after BACKENDS-STUDY.md:

- **Keep the portable layer.** Dispatch overhead is small, about 3.5 µs a
  dependent dispatch; one bounce already costs 80% of a 12-bounce sample.
- **Fast math** on the shading kernels: 9.0 to 8.3 ms.
- **Scene specialization:** material features, cut-outs, guiding and the light
  count become specialization constants. Shading drops 3.3 to 2.1 ms, the
  sample 8.4 to about 7.2 ms spectral and 7.8 to 6.6 ms RGB, at 1080p on an
  M4 Max.

- **No separate scheduling kernel.** Each kernel's last workgroup writes the
  next kernels' indirect arguments, so a round is 4 dispatches instead of 7:
  ~7.1 to 6.9 ms.
- **Mid-pass path refill: not done, as it doesn't pay here.** Going from 4 to 12
  bounces costs only 0.35 ms (5%), the most refill could recover; tail rounds
  are real work.
- **Material sorting: waits** for multi-material benchmark scenes. Every test
  scene has one material, where sorting only costs.

## Benchmark: the Cornell box

`tests/cornell` is the scene optimizations are judged on: a Cornell box (red
and green walls, white floor, ceiling and back, a warm ceiling area light) with
a blue clear-coated box, a smooth glass ball, a rough gold ball, an orange rough
plastic ball and a white cube. That's 11,938 triangles and seven materials,
with colour bleeding, refraction, a caustic under the glass, glossy
interreflection and coat.

![The Cornell box, 512 × 512, 128 samples](cornell.png)

`luc test` renders 512 × 512 at 128 samples into build/cornell.ppm and prints
the time. `LUCE_BENCH=1 build/luc-test/cornell/cornell` adds 1024 × 1024 timings
and the error after equal samples against a 4096-sample reference, with and
without guiding.

First numbers, on an M4 Max:

| Measure | Value |
| --- | --- |
| 1024 × 1024, spectral | 8.1 ms a sample |
| 1024 × 1024, RGB | 7.1 ms a sample |
| 256 × 256, 256 samples, unguided | MSE 0.0078 at 1.39 ms a sample |
| 256 × 256, 256 samples, guided | MSE 0.0076 at 2.08 ms a sample (worse at equal time) |

## Light tracing

Caustics made the Cornell box's fireflies: light focused through the glass
ball onto the floor, seen directly or after a bounce. Shadow rays cannot
follow refraction, so camera paths found that light only by a lucky BSDF ray
through the ball onto the small lamp. A clamp hid the fireflies by deleting
59% of the caustic. [research/FIREFLIES-STUDY.md](research/FIREFLIES-STUDY.md)
has the measurements and the literature.

Light paths fix it without bias (`RenderSettings.light_tracing`, light paths a
pixel a sample, 0.5 by default; the Render node's "Light paths"):

- **Where they run:** in the camera paths' wavefront. They take the pool's
  slots after the band's camera paths; `light_emit` starts them in the first
  band's pass, and they share the rounds. `shade_surface` shades both kinds in
  one dispatch, so their latencies overlap.
- **What they do:** start at an emitter chosen by power, at a point by area,
  in a cosine direction. At every surface they connect to the camera with a
  shadow ray, and `intersect_shadow` splats what gets through atomically on a
  fourth film plane. `film_convert` adds the splats over the light-tracing
  samples taken.
- **MIS over three techniques** (shaders/connect.glsl): the camera's BSDF hit
  on a light, its light sample, and the light path's camera connection, each
  weighed by the power heuristic over all three. Both sides carry the ratio of
  the other side's path density to their own, settling each vertex's factor
  once the direction beyond it is known. Light paths choose lobes by Fresnel
  at normal incidence (`light_lobes`), so camera paths compute their reverse
  pdfs without tables.
- **Transport:** light paths weigh each bounce by the BSDF as the camera
  evaluates it (radiance transport), with Veach's shading-normal correction.
- **Visibility reads the same both ways:** light paths cross flat lights the
  mirror way of camera rays, and a camera connection stops at a visible light
  the camera's ray would hit first.

Checked against closed forms (rect light and emissive quad with light paths)
and against path tracing: the all-diffuse Cornell box agrees to 0.01%, and the
lamp-in-a-shade room converges 22x lower MSE at equal samples.

## Benchmark: the Cornell box, now

512 × 512, 12 bounces, spectral, M4 Max (another GPU app running, so absolute
times are about 15% high). Error is relative MSE against a 16384-sample
light-traced reference; "trimmed" drops the worst 0.1% of pixels.

| Light paths a pixel | ms a sample | relMSE at 512 samples | relMSE x time | trimmed x time |
| --- | --- | --- | --- | --- |
| 0 (path tracing) | 3.14 | 0.0392 | 0.123 | 0.090 |
| 0.1 | 3.94 | 0.0086 | 0.034 | 0.016 |
| 0.25 | 4.25 | 0.0059 | 0.025 | 0.012 |
| 0.5 (default) | 4.79 | 0.0041 | 0.020 | 0.011 |
| 1 | 5.88 | 0.0033 | 0.020 | 0.011 |

Light tracing is about 6x more efficient at equal time (8x trimmed, 11x
outside the glass ball). What noise is left is in caustics seen through glass
(the ball's inside, the clear coat's reflection), which neither technique can
sample: manifold next-event estimation is the next step for those
(REALTIME-STUDY §3.6). Path guiding trained from light paths too (Vorba 2014)
was tried and did not pay: it doubled the error on the gold and the coat.

### Light paths from domes and the sun

Domes and distant lights with a size start light paths too (lights.glsl's
"from afar"). Each path:
- takes a direction by the light's own distribution;
- starts at a point on a disk facing that direction, centered on the
  triangles whose materials cast caustics (glass, metals, clear coats, sharp
  specular);
- reaches a first surface whose density is |cos| over the disk's area there.

The light's "point" is a direction, so both sides weigh it by solid-angle
pdfs. Under a studio dome (glass, gold, coat on a floor, 384 × 256), error at
about equal time drops 1.9x (3.6x trimmed, 4.6x on the floor).

## Since then

- **Depth of field:** a thin lens from the camera's f-stop and focus distance;
  light paths connect to a sampled lens point.
- **Dome lights with OpenEXR environments**, importance-sampled (see Lights).
- **Dispersion** (see Spectral): rainbow caustics come from light paths.
- **Textures:** base color, roughness, metalness and normal maps (see OpenPBR).

Timings move with the machine's load: another app was using the GPU while
these were taken. The Cornell box at 1024 × 1024, spectral, 12 bounces:
- about 10.5 ms a sample from camera paths alone;
- about 17 ms with light paths at 0.5 a pixel. Shading a light vertex costs
  about 1.7 times a camera vertex: it connects to the camera and evaluates the
  material twice.

### Against Cycles

The Cornell box was rebuilt for Cycles (Blender 5.1.2, Metal, Principled in
place of OpenPBR, 12 bounces, no adaptive sampling, denoising or clamp). The
driver stays local, in .donors/render/cycles_cornell.py. Results on the M4 Max,
512 × 512:

| | ms a sample | variance at equal time (1.6 s, relative) |
| --- | --- | --- |
| Cycles | 3.12 | 0.0074 |
| luce-render, path tracing, RGB / spectral | 2.96 / 3.11 | 0.0018 (spectral) |
| luce-render, light paths 0.5 (default), RGB / spectral | 4.5 / 5.2 | 0.00043 / 0.00059 |

Variance is measured between two seeds of each renderer, since their
materials differ.
- **Per sample:** speed is even.
- **At equal time:** the default is 13–17 times less noisy than Cycles.
- **Startup:** from launch to first sample takes 0.14 s; Cycles' first
  render spends 60 s compiling Metal kernels.

Next:
- Manifold sampling for caustics seen through glass.
- A denoiser.
- Instancing as a BLAS per prototype.
- Per-scene light-path budgets (efficiency-aware MIS, Grittmann et al. 2022).

## Interactive renders

A viewport renders through a view it places, and changes it on every orbit.
A Renderer therefore keeps three kinds of device state apart (loading.lucb),
and replaces each only when what it follows changes:

| State | Follows | Replaced by |
| --- | --- | --- |
| Geometry: `GpuScene`, the triangles' and splats' buffers, BVHs, BLAS and TLAS | meshes and clouds | a new renderer |
| Loaded: lights and their tree, materials, textures, programs, constants, specialized kernels | lights, materials, settings | `Renderer.reload` |
| View: path state, film, images | camera and size | `Renderer.restart` |

- **`restart(frame)`:** a new camera or size. The guide and view constants are
  uploaded; the next pass clears the film; passes still in flight finish
  uncounted, and their image shows until the next pass's. A smaller view fills
  the images' top-left corner (`shown_size`); only a larger one makes the view's
  buffers again.
- **`reload(scene, settings)`:** the same geometry with other lights or
  materials. `compile_scene(set, geometry = false)` makes such a scene without
  splats or BVHs; reload puts its triangles in the BVH's leaf order
  (`CompiledScene.order`) and uploads the per-triangle shading again.
- **`same_geometry(a, b)`** tells when that holds: the meshes and clouds have
  the same shape and placement, and are made of the same geocore array blocks
  (blocks are never changed in place).
- **`RenderSession`** compiles on a thread of its own (the bonsai capture takes
  about 2.6 s), queues a newer scene behind a running compile, and picks the
  result up in `draw`. `look` places a view, and `start_view` renders through
  it. Neither the albedo tables nor the BLAS build blocks the caller: later
  passes are ordered after them on the queue.

`tests/gpu/interactive.lucb` checks restarts through other views and sizes,
restarts with passes in flight, and a reload with a brighter light, each
against a renderer made afresh.
