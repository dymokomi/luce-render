# luce-render: source study of Cycles X, pbrt-v4, OpenPBR and OpenPGL

Status: research note, 2026-10-06. Input for the luce-render design. Nothing here is code to copy.
Every algorithm below is described in our own words. File references point at the scratch clones
so a reader can check the claims.

## 0. Sources and how to read the references

Shallow clones in `/Users/sedov/Dev/luce_dev/.donors/render/`. These are reference only and are never copied into our repos:

| Clone | Commit | What it gave us |
|---|---|---|
| `cycles/` (github blender/cycles) | `a456b76` | wavefront integrator, scheduler, light tree, sampler, BVH2, tables |
| `pbrt-v4/` (github mmp/pbrt-v4) | `b4ce968` | spectral machinery, RGB→spectrum, sensor, GPU wavefront |
| `OpenPBR/` (ASWF spec repo) | `f8d6d94` (v1.1.1, 2026-04-17) | spec text `index.html`, MaterialX graph `reference/open_pbr_surface.mtlx` |
| `OpenPBR-viewer/` (github portsmouth/OpenPBR-viewer) | `cc31927` | **GLSL path-tracer implementation of OpenPBR** by a spec author |
| `MaterialX/` (sparse: `libraries/pbrlib`, `libraries/bxdf`) | `d079210` | GLSL lobe library used by the MaterialX reference graph |
| `openpgl/` (github RenderKit/openpgl) | `b4e4b86` (0.8.0) | path guiding: VMM + kd-tree, training loop |

Notes:
- The OpenPBR repo has **no GLSL**. Its "reference implementation" is a MaterialX node graph
  (`reference/open_pbr_surface.mtlx`). That graph sits on MaterialX's GLSL lobe library (`MaterialX/libraries/pbrlib/genglsl/lib/*.glsl`).
  The closest thing to a GLSL reference BSDF is `OpenPBR-viewer/glsl/pathtracing/*.glsl`.
  Caution: the viewer also carries **unmerged spec PRs** (#247 decoupled specular_weight Fresnel, #253 coat darkening,
  #254 specular haze, #255 retroreflectivity). Those are marked `PR #…` in its source. Treat them as previews, not spec 1.1.1.
- Paths below are relative to each clone. `cycles/src/...` is abbreviated `C:`, `pbrt-v4/src/pbrt/...` is `P:`,
  `OpenPBR-viewer/glsl/pathtracing/...` is `V:`, and `openpgl/openpgl/...` is `G:`.

---

## 1. Cycles X wavefront integrator

### 1.1 Kernel list

The GPU kernels are the `DeviceKernel` enum, `C:kernel/types.h:1743-1825`. The path-iteration kernels come first, and
`DEVICE_GPU_KERNEL_INTEGRATOR_NUM` is set equal to `MEGAKERNEL`, so only the kernels before it carry queue counters:

| Kernel | Role |
|---|---|
| `INIT_FROM_CAMERA` | Map a work-tile index to (x, y, sample). Skip converged pixels. Generate the camera ray. Init state. Queue `INTERSECT_CLOSEST` (`C:kernel/integrator/init_from_camera.h:57-145`). |
| `INIT_FROM_BAKE` | Baking variant. |
| `INTERSECT_CLOSEST` | Trace the main ray and test analytic lights for MIS. Do **Russian roulette before shading**. Pick the next kernel (`C:kernel/integrator/intersect_closest.h:351-448`). |
| `INTERSECT_SHADOW` | Trace a shadow ray. Record up to `INTEGRATOR_SHADOW_ISECT_SIZE_GPU = 4` transparent hits (`types.h:58-64`). |
| `INTERSECT_SUBSURFACE`, `INTERSECT_VOLUME_STACK`, `INTERSECT_DEDICATED_LIGHT`, `INTERSECT_MNEE` | Feature-specific ray casts. |
| `SHADE_BACKGROUND` | Missed rays: world shader and dome light with MIS. |
| `SHADE_LIGHT_FORWARD` (and `SHADE_LIGHT_NEE`) | BSDF ray hit an analytic light. Add emission with the forward MIS weight. The NEE variant evaluates non-constant light shaders for shadow paths. |
| `SHADE_SURFACE` / `SHADE_SURFACE_RAYTRACE` | Evaluate the shader. Add emission. Run NEE, which **spawns a shadow path**. Sample the BSDF for the next bounce. The `_RAYTRACE` variant exists for AO/bevel nodes that trace rays. |
| `SHADE_VOLUME`, `SHADE_VOLUME_RAY_MARCHING` | Volumes. |
| `SHADE_SHADOW` | Apply transparency and volume attenuation along the shadow ray. Write the light contribution to the film. Re-queue `INTERSECT_SHADOW` if more than 4 transparent hits remain. |
| `SHADE_DEDICATED_LIGHT` | Shadow linking. |

The utility kernels follow (`types.h:1763-1776`):
- `QUEUED_PATHS_ARRAY`, `QUEUED_SHADOW_PATHS_ARRAY`, `ACTIVE_PATHS_ARRAY`, `TERMINATED_PATHS_ARRAY`
- `SORTED_PATHS_ARRAY`, `SORT_BUCKET_PASS`, `SORT_WRITE_PASS`
- `COMPACT_PATHS_ARRAY`, `COMPACT_STATES`, plus the shadow variants
- `RESET`, `PREFIX_SUM`
- adaptive sampling `CONVERGENCE_CHECK`/`FILTER_X`/`FILTER_Y`
- film convert kernels, denoiser pre/post kernels

### 1.2 IntegratorState: SoA layout and fields

The state is declared once with X-macros, in `C:kernel/integrator/state_template.h` (main path) and
`shadow_state_template.h` (shadow path). `state.h:46-94` expands the macros into an AoS struct for the CPU.
`state.h:137+` expands them into a GPU **struct of pointers, one device array per member**. The pointers live in constant memory
(`integrator_state`, copied in `path_trace_work_gpu.cpp:325-331`). Each member has a feature mask
(`KERNEL_FEATURE_*`). `alloc_integrator_soa()` (`path_trace_work_gpu.cpp:107-240`) allocates only the arrays
that the scene's features need. Some groups are "packed" (`KERNEL_STRUCT_BEGIN_PACKED`): `ray`, `isect` and
`shadow_ray` are stored as one struct per path so a single wide load fetches them.

Main path fields (`state_template.h:7-156`):
- `path`:
  - `render_pixel_index` u32. The buffer offset is multiplied out late so the field stays 32-bit.
  - `sample` u32
  - bounce counters as u16 each: `bounce`, `transparent_bounce`, `diffuse_bounce`, `glossy_bounce`, `transmission_bounce`, `volume_bounce`, `volume_bounds_bounce`, `portal_bounce`
  - `queued_kernel` u16, which is **the path's next kernel**
  - `rng_pixel` u32
  - `rng_offset` u16, the base dimension of the current bounce
  - `visibility` u8
  - `flag` u32
  - `mnee` u8
  - `optical_depth` f32
  - MIS: `mis_ray_pdf` f32 (BSDF pdf of the last bounce), `mis_ray_object`, `mis_origin_n` (packed float3, the last shading normal, needed by the light tree pdf)
  - `min_ray_pdf` (filter glossy)
  - `continuation_probability` f32
  - `throughput` (PackedSpectrum = 3 floats)
  - `unguided_throughput`
  - light-pass weights `pass_diffuse_weight` and `pass_glossy_weight`
  - `denoising_feature_throughput`
  - `shader_sort_key` u32
- `ray` (packed): `P` (packed float3), `dP`, `D`, `dD`, `tmin`, `tmax`, `time`, plus `previous_dt` for the light tree
- `isect` (packed): `t`, `u`, `v`, `prim`, `object`, `type`
- `subsurface`: albedo, radius, anisotropy, N
- `volume_stack[MAX_VOLUME_STACK_SIZE=32]`: object and shader
- `guiding`: an OpenPGL segment pointer, so CPU only (see §4)
- `shadow_link`

Shadow path fields (`shadow_state_template.h:7-112`):
- pixel, sample, rng, bounce counters, queued kernel, flag
- `throughput`, plus `unshadowed_throughput` for AO
- pass weights
- `packed_num_hits` (total hits, resume index and a skip-volume bit)
- `bsdf_eval_average`, used only without the light tree for light termination
- `shadow_ray` (packed): `P`, `D`, tmin/tmax/time/dP/dD, `self_light_object`/`prim`
- `shadow_isect[4]`: t/u/v/prim/object/type
- `shadow_volume_stack`

Takeaway: Cycles never stores a full shading point in state. `shade_surface` rebuilds `ShaderData` from
`isect` + `ray` (`shade_surface.h:29-41`, `integrate_surface_shader_setup`). Shadow paths are fully separate
state, so a shadow ray can be traced while its main path has already moved on.

### 1.3 Queue counters and kernel choice

Every main-path kernel must call exactly one of `integrator_path_init/next/terminate` (`state_flow.h:20-98`):
- `next` atomically decrements `num_queued[current]`, increments `num_queued[next]` and writes `queued_kernel`.
- `terminate` sets `queued_kernel = 0` and decrements the counter.
- The `_sorted` variants also bump `sort_key_counter[kernel][key]` (`state_flow.h:163-200`).

The counters live in `IntegratorQueueCounter { int num_queued[N]; int cache_miss; }` (`state.h:100-103`).

The host loop is `render_samples` (`path_trace_work_gpu.cpp:348-412`). It repeats:
1. **Refill.** `enqueue_work_tiles` (`:812-948`) adds camera paths only when the current most-queued kernel is
   `INTERSECT_CLOSEST`, or there is none. That keeps new and old paths aligned in one wavefront. It also requires
   `num_active < min_num_active_main_paths_` (the busy threshold).
2. **Pick a kernel.** `enqueue_path_iteration()` (`:459-517`) chooses the kernel with the **largest queue**
   (`get_most_queued_kernel`, `:414-434`). If that kernel creates shadow paths and the shadow pool lacks room,
   it first drains `SHADE_LIGHT_NEE`, then `INTERSECT_SHADOW`, then `SHADE_SHADOW`.
3. **Read back.** After every launch, `update_queue_counter_and_cache()` (`:324-346`) **copies the counters back and synchronizes**.
   That is one CPU↔GPU round trip per kernel launch, which is the main overhead of this design on Metal.

Building the index list for a launch (`enqueue_path_iteration(kernel, limit)`, `:519-610`):
- Sorted kernels (`SHADE_SURFACE*`) run a counting sort. `PREFIX_SUM` over `sort_key_counter`, then
  `SORTED_PATHS_ARRAY` scatters path indices into shader-ordered slots (`:612-674`).
- On Metal with "local atomic sort", `SORT_BUCKET_PASS` + `SORT_WRITE_PASS` sort inside partitions of
  `sort_partition_divisor` states. The key is `shader + max_shaders * (state / divisor)` (`state_flow.h:159-161`).
  This keeps state locality and shader coherence together. Partitioning is turned off above 300 shaders
  (`device/metal/queue.mm:338-351`).
- Other kernels launch over `max_active_main_path_index` directly when most states are queued for them.
  Otherwise they build a compact index list with `QUEUED_PATHS_ARRAY` (a parallel active-index pass using
  warp ballots + one atomic per warp, `kernel/device/gpu/parallel_active_index.h`).

### 1.4 Compaction, shadow splitting, termination

- **Shadow paths.** NEE calls `integrator_shadow_path_init` (`state_flow.h:119-132`). It takes a fresh
  shadow slot with `atomic_fetch_and_add(next_shadow_path_index)`, so the shadow pool is bump-allocated, and queues
  `INTERSECT_SHADOW` (or `SHADE_LIGHT_NEE` for non-constant light shaders). The main path keeps going. Shadow
  state copies everything it needs: throughput × BSDF × light / pdf × MIS weight, the ray, pixel and rng
  (`shade_surface.h:205-311`).
- **Shadow compaction** (`compact_shadow_paths`, `:718-757`). It runs when the bump index exceeds twice the live
  shadow count (and is at least 32): terminated slots are collected, high live slots are moved into them, and the index is reset.
- **Main compaction** (`compact_main_paths`, `:692-716`, `compact_paths`, `:759-810`). Before new tiles are added,
  live states above `num_active` are moved into terminated holes below it. Three kernels do this:
  `TERMINATED_PATHS_ARRAY` lists the holes, `COMPACT_PATHS_ARRAY` lists the movers, and `COMPACT_STATES` copies every SoA member.
  New camera paths then go into `[num_active, num_active + new)`, which keeps the index range dense.
  The comment at `:133-136` explains why the busy threshold is capped at half the pool: regeneration must
  wait until half the states are free, or compaction breaks.
- **Termination.** RR runs in `intersect_closest` *before* shading
  (`integrator_intersect_terminate`, `intersect_closest.h:29-86`). A path that loses RR at a non-emissive surface is killed
  without launching `shade_surface`. If the surface is emissive, the path gets `PATH_RAY_TERMINATE_ON_NEXT_SURFACE`,
  is shaded for emission only, and then stops (`shade_surface.h:650-666`). A terminated state has `queued_kernel == 0`.
  It is reclaimed by compaction or by the next tile refill.

### 1.5 Work tiles and paths in flight

- **Pool size.** `max_num_paths_ = queue->num_concurrent_states(state_size)`.
  - Metal: start at **4,194,304 states**, reduce if memory is short, and double at most twice if
    the free working set allows (`device/metal/queue.mm:266-330`).
  - Busy threshold: `num_concurrent_busy_states = states/4` ("1:4 busy:total gives best performance", `:332-336`).
  - CUDA: `max(threads, 65536) × 16` (`device/cuda/queue.cpp:30-51`).
- **Tile scheduler.** `WorkTileScheduler` (`C:integrator/work_tile_scheduler.cpp:51-137`) cuts the image × sample range
  into tiles of `w × h × num_samples` states. One "scheduling unit" is `max_num_paths/8` (`path_trace_work_gpu.cpp:357`).
- **Tile shape.** `tile_calculate_best_size` (`integrator/tile.cpp:39-99`) prefers small power-of-two square tiles with
  **several samples per tile**: `num_samples ≈ pow2(sqrt(spp/2))`. Tiles are handed out greedily until the pool is full.
- **Thread mapping.** The camera kernel maps a work index to pixel/sample with `get_work_pixel`
  (`kernel/device/gpu/work_stealing.h:14-38`). Several samples of the **same pixel are in flight together**.
  That is why Cycles' film writes are **float atomics** (`kernel/film/write.h:58-110`, `atomic_add_and_fetch_float`).

### 1.6 Adaptive sampling

- **Two buffers.** The combined pass holds all samples. An auxiliary pass holds **twice the contribution of
  "class A" samples only** (`kernel/film/light_passes.h:189-197`). The class is decided by
  `popcount(sample & 0xaaaaaaaa) & 1` (`kernel/sample/pattern.h`, `sample_is_class_A`). This follows the
  Christensen PMJ class split and works for Owen-scrambled Sobol.
- **Convergence check** (`kernel/film/adaptive_sampling.h:31-77`). The per-pixel error is the sum of |I − A| over RGB, times exposure / spp.
  It is divided by `sqrt(intensity)` when intensity < 1, otherwise by the intensity itself. The pixel has converged when the error is below
  the threshold. A converged pixel sets `aux.w = 1`, and `init_from_camera` then skips it (`film_need_sample_pixel`, `:15-25`).
- **Filter.** A separable "dilate unconverged" pass, `FILTER_X` then `FILTER_Y` (`:82-147`). A converged
  neighbour of an unconverged pixel is reset to unconverged.
- **Schedule.** The check runs every `adaptive_step` samples after `min_samples`, aligned to step boundaries
  (`C:integrator/adaptive_sampling.cpp:30-54`).

### 1.7 Light tree

- **Origin.** It is a modified version of Conty Estevez & Kulla, "Importance Sampling of Many Lights with Adaptive Tree Splitting" (HPG 2018)
  (`C:kernel/light/tree.h:1-13`). There is **no splitting**: one light is chosen per shading point. To compensate,
  every node gets a **max and a min importance**, and the selection probability is the average of the two normalized
  probabilities.
- **Node** (`kernel/types.h:1555-1593`, 64 B):
  - bbox
  - bounding cone (axis, θ_o normal spread, θ_e emission spread)
  - energy
  - type, `num_emitters`
  - a union of {leaf first_emitter | inner left/right | instance reference}
  - `bit_trail` and `bit_skip`
  Mesh emitters reference a sub-tree per object ("instance" nodes), so instanced emissive meshes share one sub-tree.
- **Build** (`C:scene/light_tree.cpp:516-628`). Binned top-down build with **12 buckets** (`light_tree.h:225`) per axis.
  - Cost: the **SAOH** measure `energy × bbox_area × orientation_measure(θ_o, θ_e)`
    (`light_tree.h:100-110`, `light_tree.cpp:14-27`).
  - A node splits if the best split beats the leaf cost or holds more than `max_lights_in_leaf = 8` (`scene/light.cpp:1070`).
  - Distant lights get their own node.
  - Per-emitter energy is |strength| averaged, times area for triangles (`light_tree.cpp:116, 237-239`).
- **Importance** (`tree.h:125-239`). For a shading point P with normal N it uses:
  - the subtended half-angle of the node's bounding sphere, θ_u
  - the incidence bound θ_i' relative to N (zero if the node is behind an opaque surface)
  - the outgoing bound θ' from the cone (zero if outside θ_o + θ_e)
  The max importance is `energy · cos θ_i' · cos θ' / d_min²`. The min importance is the analogous bound with max angles over d_max².
- **Traversal** (`light_tree_sample`, `tree.h:743-811`). At inner nodes, the left probability is the average of the
  max-based and min-based probabilities (`get_left_probability`, `:690-725`). The same random number is
  **rescaled and reused** at every level (`sample_reservoir`, `:551-590`). At a leaf, two streaming reservoirs (max and min)
  pick one of ≤8 emitters (`:593-687`).
- **PDF for MIS** (`light_tree_pdf`, `:814-930`). It re-walks the tree along the emitter's **bit trail**, so the
  forward-hit MIS weight needs the previous vertex's P, N (`mis_origin_n`) and `previous_dt`. That is why those fields
  are in the state.
- **When it is used.** `use_light_tree` defaults to true (`scene/integrator.cpp:145`). Without it Cycles uses a flat
  power distribution. Note: analytic lights are hit-tested by a **linear loop over all lights** in
  `intersect_closest` (`kernel/light/light.h:233-260`), and they are not in the BVH.

### 1.8 MIS, Russian roulette, clamping

- **MIS.** The **power heuristic** (β=2) is used everywhere (`kernel/sample/mis.h`, `light/sample.h:327-360`).
  - NEE weight: `power(light_pdf, bsdf_pdf)`.
  - Forward weight: `power(mis_ray_pdf, light_pdf · selection_pdf)` (`light/sample.h:469-560`).
  - The first bounce from the camera sets `PATH_RAY_MIS_SKIP`.
  - Light pdf in solid angle: spherical-rectangle sampling for rect lights (Ureña et al., `light/area.h:19-104`),
    uniform-area sampling for ellipses/disks (`area.h:250-328`), cone sampling for spheres seen from outside
    (`light/point.h:18-91`), cone sampling for the sun (`sun.h:34-55`).
- **RR.** The continuation probability is `min(sqrt(max(|throughput|)), 1)`, applied after `min_bounce`
  (or `transparent_min_bounce`) (`path_state.h:277-304`). The sqrt "roughly matches a view transform" so paths live a bit longer.
  The roll uses dimension `PRNG_TERMINATE`, and throughput is divided by the probability in shade_surface.
- **Clamping** (`film/light_passes.h:125-146`). Non-finite values become 0. If `sum(|L|)` exceeds
  `sample_clamp_direct` (bounce 0) or `sample_clamp_indirect`, L is scaled down so its sum equals the limit.
- **Filter glossy** (`integrator/surface_shader.h:185-212`). The state tracks `min_ray_pdf`, the smallest BSDF pdf on the path.
  When `filter_glossy · min_ray_pdf < 1`, every closure is blurred with roughness `sqrt(1 − blur_pdf)/2`. It is a firefly control, and it is biased.

### 1.9 Sampler: Sobol–Burley and dimensions

- **Generator** (`kernel/sample/sobol_burley.h`). Burley-2020 Owen-scrambled, shuffled Sobol over **4 base dimensions**.
  The direction table is `sobol_burley_table[4][32]` (`kernel/tables.h:81`). Higher dimensions come from
  **padding**: every request is a 1D/2D/3D/4D tuple whose seed is hashed with the dimension index, and its index
  is Owen-shuffled per dimension set (`:99-201`). Owen scrambling is a cheap hash, `reversed_bit_owen`
  (`sample/util.h:18-27`). An index mask bounds the sequence length at low spp.
- **Pixel decorrelation** (`pattern.h`, `path_rng_pixel_init`). The white-noise patterns use a per-pixel hash seed.
  The blue-noise patterns run one global sequence, and each pixel starts at a slot given by a base-4 Owen-scrambled Morton index of (x, y).
  That is the Psychopath "dithered blue noise" scheme. A `BLUE_NOISE_FIRST` mode gives sample 0 its own 1-spp
  blue-noise sequence for viewport navigation.
- **Dimension allocation** (`types.h:72-119`). Each bounce owns **16 dimensions** (`PRNG_BOUNCE_NUM`).
  `rng_offset += 16` per bounce. The camera bounce uses 0 = filter (2D) and 1 = lens/time.
  The shading bounce uses 0 = terminate, 1 = light (3D: select + position), 2 = light terminate, 3 = BSDF (3D: lobe + direction),
  4 = AO, 5 = bevel, 6 = guiding, and 10/11 = guiding RIS. Branches such as SSS re-hash `rng_offset` (`path_state.h:356`).

### 1.10 BVH: hardware versus BVH2

- **Hardware paths.** Metal uses **MetalRT** when available, else BVH2 (`device/metal/device_impl.mm:50-53`).
  MetalRT runs `metal::raytracing::intersector` with intersection-function tables inside compute kernels (`kernel/device/metal/compat.h:314-328`).
  That is exactly the "ray query from compute" model we plan. OptiX, HIP-RT and Embree are the other back ends.
- **Software BVH2 layout** (`bvh/bvh2.cpp:96-174`, `bvh/bvh2.h:20-21`). This is our fallback blueprint.
  - **Inner node = 4 × float4 = 64 B**, and it holds **both children's boxes**:
    - `[0]` = (vis0, vis1, child0, child1)
    - `[1]` = (c0.min.x, c1.min.x, c0.max.x, c1.max.x)
    - `[2]` = same for y
    - `[3]` = same for z
  - Negative child index = leaf.
  - **Leaf = 1 × int4** = (prim_start, prim_end, visibility, prim_type). An object (instance) leaf stores `~object`.
  - The triangle vertex data lives in a separate pre-transformed `prim_tri_verts` array.
- **Build parameters** (`bvh/params.h:100-140`). Binned SAH with **spatial splits** (32 spatial bins, α=1e-5),
  `max_triangle_leaf_size = 8`, node/prim cost 1:1, max depth 64.
- **Traversal** (`kernel/bvh/traversal.h`). A stack-based while-while loop:
  - The stack is `int[192]` (`kernel/bvh/types.h:20`).
  - Both child slabs are tested at once (`bvh_aligned_node_intersect`, `nodes.h`). If both are hit, the farther child is pushed.
  - Leaves loop over triangles with the Embree-style **Plücker/edge-function watertight test** (`util/math_intersect.h:163-250`).
  - Instances are handled by transforming the ray at an object leaf and pushing a sentinel.
  - **Self-intersection is avoided by prim/object id skip**, not by ray offsets (`intersection_skip_self_shadow`).
- **Shadow rays** set `PATH_RAY_VISIBILITY_SHADOW_OPAQUE` for any-hit early exit.

### 1.11 Denoising hooks and film

- **Denoise features** (`film/denoising_passes.h:37-220`). At each surface the kernel computes a closure-weighted
  normal and albedo. Diffuse-like closures (roughness ≥ 0.15 via smoothstep) write the features. Near-specular
  closures *defer* them to the next bounce through `denoising_feature_throughput`, so a mirror shows the reflected
  object's albedo and normal. Depth is also written. OIDN and OptiX consume albedo + normal (`integrator/denoiser_oidn*.cpp`).
- **Film.** One interleaved float buffer: `pass_stride` floats per pixel, with each pass at an offset
  (`kernel_data.film.pass_*`). Writes are float atomics. A separate `FILM_CONVERT_*` kernel family turns accumulated
  sums into display values (divide by the per-pixel sample-count pass).

### 1.12 Recommendations for luce-render (from §1)

1. **Kernels.**
   - Keep Cycles' split: `init_camera`, `intersect_closest`, `shade_surface`, `shade_miss`, `shade_light_hit`, `intersect_shadow`, `shade_shadow`, `film_*`.
   - Fold `shade_light_hit` into `shade_surface` when area lights are real geometry (see §5).
   - Put **RR in `intersect_closest`** as Cycles does. Killing non-emissive losers before material fetch saves a lot.
2. **Push queues, not scan queues.** Replace "scan every state for `queued_kernel == K`" with **per-kernel index queues filled by atomic append**,
   pbrt style (§2.6). Every producer writes `queue_K[atomicAdd(count_K,1)] = path`. This removes the active-index pass and fits our
   int atomics directly.
3. **No per-launch readback.** We have **indirect dispatch**. Add one tiny `schedule` kernel (1 thread) that turns counters into
   `DispatchIndirect` args and clears the next counters. The host records a **fixed sequence** of bounce iterations
   (camera → [closest → surface/miss → shadow → shade_shadow] × K) with zero host round trips, and reads counters
   asynchronously once per batch. That is Cycles' "largest queue first" traded for a fixed order, which is what pbrt does,
   and on Apple GPUs the sync cost beats the occupancy loss.
4. **Shadow slot = main-path slot.** With fixed order, every shadow ray is resolved before its parent shades again. A main path owns
   at most one live shadow, so the shadow pool needs **no allocator and no compaction**, and the contribution can be added to the
   parent's per-path accumulator without atomics. Keep Cycles' separate shadow SoA so `intersect_shadow` touches only shadow data.
5. **Pool sizing.**
   - Start at **2^20 paths** and scale by the device working set as Cycles' Metal queue does (1M–4M).
   - Refill when live < pool/4.
   - Compact main paths on refill (three-kernel compaction as in §1.4) or use a **free-list** of terminated indices.
     The free-list is simpler with push queues: terminated paths push their index to a `free` queue, and camera init pops it.
6. **Film.**
   - Use **one sample per pixel per refill unit** (sample-major tiles, not Cycles' samples-per-tile). Then a pixel has at most one live
     path and per-path accumulation needs no atomics. If we later keep several samples per pixel in flight, the final `film += L`
     uses a CAS-loop float add on `uint` (int atomics), and native float atomicAdd when the device has it.
   - Use **filter importance sampling** (Cycles' 1024-entry inverse-CDF `filter_table`, `kernel/types.h:34`, `camera.h:463-465`),
     so each sample lands in exactly one pixel and has weight 1.
7. **Adaptive sampling.** Copy the scheme: class-A half buffer, the error metric with the sqrt normalization, the separable dilate,
   every N samples. It is cheap and robust. Compute the error in **ACEScg RGB after XYZ conversion**, not per-lambda.
8. **Sampler.**
   - Sobol–Burley with 4D padding: 128 u32 of table, all integer ops, ideal for GLSL.
   - Use Cycles' 16-dims-per-bounce allocation plus **one extra camera dimension for the wavelength** (see §2/§5).
   - Blue-noise-first (Morton + base-4 Owen) for the viewport.
9. **Light sampling.** Light tree from day one for meshes with emission (OpenPBR `emission_luminance`) plus UsdLux lights.
   Use Cycles' min/max-importance average and ≤8 emitters per leaf, and store a bit trail per emitter for the MIS pdf.
   Analytic lights go into the BVH as primitives (or a tiny separate light BVH) rather than Cycles' O(N) loop.
10. **Power heuristic, clamp, filter glossy.** Same semantics. Clamp per sample in the film write (on the RGB/XYZ value in spectral mode).
11. **BVH fallback.** Cycles BVH2 layout (64 B node holding both child AABBs, 16 B leaf) and the watertight edge-function triangle test.
    Self-hit avoidance by (instance, prim) id skip works for both the software BVH and ray queries (filter in the candidate loop).

---

## 2. pbrt-v4 spectral rendering and its GPU wavefront

### 2.1 SampledWavelengths, SampledSpectrum

- **Lanes.** `NSpectrumSamples = 4`, `Lambda_min/max = 360/830 nm`, `CIE_Y_integral = 106.856895` (`P:util/spectrum.h:30-38`).
  `SampledSpectrum` is a 4-float array with element-wise math. `SampledWavelengths` holds `lambda[4]` and `pdf[4]` (`:269-349`).
- **`SampleVisible(u)`** (`:331-345`). It is **stratified hero sampling**: lane i uses `u_i = frac(u + i/4)`, maps it through an
  analytic inverse CDF and stores the per-lane pdf.
  - pdf: `p(λ) = 0.0039398042 / cosh²(0.0072 (λ − 538))` on [360, 830]
  - inverse: `λ(u) = 538 − 138.888889 · atanh(0.85691062 − 1.82750197 u)` (`P:util/sampling.h:163-171`)
  This bell centered on 538 nm roughly follows ȳ, which cuts chroma noise against uniform sampling. There is no table.
- **`SampleUniform`** (`:283-302`) is the equal-spaced rotation variant, kept for comparison.
- **`TerminateSecondary()`** (`:313-321`). It zeroes `pdf[1..3]` and divides `pdf[0]` by 4. The estimator divides by pdf
  and averages over 4 lanes, so the surviving hero lane carries the full weight. It is called when the **IOR depends on wavelength**:
  `DielectricMaterial::GetBxDF` and `ThinDielectric` (`P:materials.h:183-187, 228-232`), only if `eta` is not a
  `ConstantSpectrum`. It is applied **at the vertex, before lobe sampling**, so reflections at a dispersive surface also drop
  secondaries. Our rule: drop them only when the material actually disperses (`transmission_dispersion_scale > 0`).

### 2.2 RGB → spectrum: RGBSigmoidPolynomial and RGBToSpectrumTable (Jakob & Hanika 2019)

- **Model** (`P:util/color.h:332-365`). A smooth reflectance is `s(c0 λ² + c1 λ + c2)` with λ in nm and the sigmoid
  `s(x) = ½ + x / (2 sqrt(1 + x²))`. That is algebraic, no exp, and it stays in [0, 1] by construction. `MaxValue` checks
  the endpoints and the parabola vertex `−c1/(2c0)`.
- **Table** (`color.h:369-400`). `res = 64`.
  - Coefficients `float[3][64][64][64][3]`: the first index is **which RGB channel is the max**, then z (the max value,
    on a non-uniform grid `zNodes`), then y, then x.
  - Size: 3·64³·3·4 B = **9.4 MB per color space**.
  - Gamuts pbrt ships: sRGB, DCI-P3, Rec2020, ACES2065-1 (`color.cpp:70-90`).
- **Lookup** (`P:util/color.cpp:31-68`).
  - Gray input (r = g = b): closed form `c = (0, 0, (v − ½)/sqrt(v(1 − v)))`.
  - Otherwise: find the max channel m, set `z = rgb[m]`, `x = rgb[m+1]·63/z`, `y = rgb[m+2]·63/z`, binary-search z in `zNodes`,
    then **trilinear interpolation of the coefficients**.
- **Builder** (`P:cmd/rgb2spec_opt.cpp`, offline).
  - CIE data has 95 samples at 5 nm (`:40-42`). It is upsampled to `CIE_FINE_SAMPLES = 3·94+1 = 283` nodes and
    integrated with **Simpson 3/8 weights** (`:343, 463-486`).
  - `init_tables` precomputes `rgb_tbl[k][i]` = (XYZ→RGB matrix · CMF) × **illuminant of the color space** × weight
    (D65 for sRGB/Rec2020/P3, D60 for ACES, D50 for ProPhoto, E for XYZ) (`:408-486`).
  - The residual is **CIELAB distance** between the target RGB and the RGB of `sigmoid(poly(λ_normalized))` (`:488-514`).
  - Solver: Gauss–Newton with a finite-difference Jacobian and 15 iterations (`:516-560`).
  - Grid: z-nodes `scale[k] = smoothstep(smoothstep(k/63))` (`:819-820`), so they are dense near 0 and 1.
  - **Warm start**: for each (l, x, y) the solve starts at `k = res/5` and walks up to 1, then from `res/5` down to 0, each solve seeded
    with the previous solution (`:825-870`).
  - The fit runs in normalized λ ∈ [0, 1] and is converted back to coefficients in nm with the stated algebra (`:845-849`).
- **Spectrum classes** (`P:util/spectrum.h:531-640`, `spectrum.cpp:230-247`):
  - `RGBAlbedoSpectrum`: rgb ∈ [0, 1] → coefficients.
  - `RGBUnboundedSpectrum`: `scale = 2·max(rgb)`, coefficients of `rgb/scale`, value `scale · s(poly)`.
    Used for unbounded inputs such as scattering coefficients.
  - **`RGBIlluminantSpectrum`**: same as unbounded, then **multiplied by the color space's standard illuminant**
    (`cs.illuminant`, D65 for sRGB). That way RGB (1, 1, 1) emission is the white point of the space, not a flat spectrum.
    Lights also get `scale /= SpectrumToPhotometric(illuminant)` = ∫ȳ·illum (`P:lights.cpp:200, 266, 664, 911`).
    So an RGB (1, 1, 1) emitter with scale 1 has **luminance Y = 1** in sensor units.

### 2.3 Blackbody, CMFs, XYZ and the sensor

- **Blackbody.** `Blackbody(λ, T)` is Planck's law in SI units (`spectrum.h:69-80`). `BlackbodySpectrum` normalizes it by the value at
  Wien's peak `λ_max = 2.8977721e-3/T` (`:497-527`). For lights the photometric normalization above applies again.
- **CMFs.** CIE 1931 2° X̄Ȳ Z̄ tabulated at **1 nm from 360 to 830 (471 samples)** (`spectrum.cpp:263+`, `nCIESamples = 471`).
- **`SampledSpectrum::ToXYZ`** (`spectrum.cpp:205-216`). It computes `XYZ = average_i( X̄(λ_i) L_i / pdf_i ) / CIE_Y_integral` (same for Y, Z).
  This is a 4-sample Monte Carlo estimate of ∫X̄L dλ, and it is why lanes carry their pdf.
- **Sensor** (`P:film.h:36-100`).
  - The default "cie1931" sensor uses `r̄ = X̄, ḡ = Ȳ, b̄ = Z̄`.
  - `ToSensorRGB = imagingRatio · average(c̄(λ) · L/pdf)`, **without** the 1/CIE_Y_integral. Together with the light normalization
    this means sensor Y equals photometric luminance relative to scale 1.
  - `imagingRatio = exposureTime · ISO/100` (`film.cpp:233`).
  - `XYZFromSensorRGB` is a white-balance (von Kries) matrix when `whitebalance` is set. For real camera curves it is solved by least
    squares over 24 ColorChecker swatches.
  - `RGBFilm::AddSample` (`film.h:239-255`) converts per sample to sensor RGB, **clamps by max component**
    (`maxcomponentvalue`), and accumulates `weight · rgb` + `weight`.
  - At output it applies `outputRGBFromSensorRGB` (`:260-272`), which is sensor → XYZ → output color space.

### 2.4 Spectral MIS in pbrt

pbrt carries **rescaled path pdfs** `r_u` (unidirectional) and `r_l` (light) as **SampledSpectrum** in every work item
(`P:wavefront/workitems.soa`). Contributions are `beta·L / average(r_u + r_l)` (`wavefront/intersect.h:38`,
`integrator.cpp:516-563`). Averaging over lanes is the **balance heuristic over the 4 wavelengths' sampling techniques**,
i.e. hero-wavelength spectral MIS. For surfaces with λ-independent pdfs all lanes are equal and this reduces to the
ordinary balance heuristic. RR uses `beta · etaScale / average(r_u)` (`surfscatter.cpp:212-222`).

### 2.5 Recommendation notes for spectral (feeds §5)

- **Lane semantics.** One generic `Spec` type: a 4-lane vec4 under `-DSPECTRAL`, an RGB vec3 otherwise.
  - Spectral: wavelengths from `SampleVisible(u_λ)` with u_λ taken as **one extra camera dimension**.
  - RGB: λ is unused and lane 3 is absent.
  - Keep **all direction pdfs scalar**: choose lobes and lights with λ-independent probabilities, e.g. luminance of the RGB parameters
    or the mean over lanes. Then the power heuristic stays scalar and identical in both variants. pbrt's spectral `r_u/r_l` is only
    needed when pdfs depend on λ, which for us happens only with chromatic media (later). Dispersion is handled by `TerminateSecondary`.
- **Upsampling.**
  - Material constants: convert on the host to `(c0, c1, c2, scale)` once and store them in the material block.
  - Textures: v1 converts texels to coefficient textures at load. Coefficients interpolate well enough for bilinear filtering,
    and this is the approach Jakob–Hanika recommend.
  - Evaluation: `s(poly(λ))` per lane is about 8 flops, which makes spectral reflectance nearly free.
- **Tables.** Build **our own coefficient tables offline** (a Luce tool, Gauss–Newton in CIELAB, 64³ per max channel):
  - **ACEScg (AP1, D60 white)**, because OpenPBR assumes ACEScg by default (spec "Metadata").
  - **Rec.709/sRGB (D65)** for imported textures.
  pbrt ships none for ACEScg. Note that AP1's green primary lies near the spectral locus, so expect larger fit error there and clamp.
- **Emission.** `RGBIlluminantSpectrum` semantics with the **working space's illuminant (D60 for ACEScg)**,
  normalized so that rgb (1, 1, 1) at luminance 1 gives Y = 1. Then OpenPBR `emission_luminance` (cd/m²) maps directly to Y, and
  UsdLux `intensity·2^exposure·color` (or blackbody `colorTemperature`, normalized to Y = 1) uses the same unit.
- **Film.** Accumulate **XYZ** per pixel in spectral mode: `average(cmf(λ_i)·L_i/pdf_i)`, CMF from a 471×3 float table
  with linear interpolation. Convert to ACEScg (Bradford adapted to D60) in film-convert. In the RGB variant, accumulate ACEScg directly.
  The film format is the same in both: rgba32f with sample count in a separate r32f/r32ui, plus the class-A aux.

### 2.6 pbrt-v4 GPU wavefront, compared to Cycles

- **Structure** (`P:wavefront/integrator.cpp:336-436`). Synchronous loops: for each sample index, for each band of scanlines
  (`scanlinesPerPass` sized so `maxQueueSize = 1024·1024` pixel samples per pass, `:227-236`):
  1. generate camera rays
  2. for each depth: reset queues, `GenerateRaySamples`, `IntersectClosest`, `SampleMediumInteraction`, `HandleEscapedRays`,
     `HandleEmissiveIntersection`, `EvaluateMaterialsAndBSDFs`, `TraceShadowRays`, `SampleSubsurface`
  3. `UpdateFilm`
- **Queues** (`P:wavefront/workqueue.h:42-135`). `WorkQueue<T>` is an SoA of **full work items** with an atomic size.
  `Push` = `atomicAdd(size, 1)` then write all fields.
  - `ForAllQueued` launches `maxQueueSize` threads, and each exits if `index >= q->Size()`. **No host readback**: the GPU reads the size.
  - `MultiWorkQueue` keeps one queue per material type (basic vs universal evaluator), which gives coarse "sorting by shader".
  - Two ray queues alternate by depth.
  - Work items: `RayWorkItem`, `EscapedRayWorkItem`, `HitAreaLightWorkItem`, `ShadowRayWorkItem`, `MaterialEvalWorkItem<M>`,
    `GetBSSRDFAndProbeRayWorkItem`, `SubsurfaceScatterWorkItem`, `MediumSample/ScatterWorkItem` (`workitems.soa`).
  - `PixelSampleState` holds per-pixel λ, accumulated `L`, `cameraRayWeight`, filter weight, `VisibleSurface` (G-buffer) and `RaySamples`.
- **Samples.** `GenerateRaySamples` (`P:wavefront/samples.cpp:20-68`) fills a per-pixel `RaySamples` struct per depth:
  direct (uc, u), indirect (uc, u, rr), subsurface. The dimension is `6 + 7·depth (+3·depth with SSS)`.
- **Radiance** goes into `pixelSampleState.L[pixelIndex]` **without atomics**. One path per pixel per pass, and shadow rays at
  depth d are resolved before depth d+1, so no two writers ever share a pixel. Our §1.12 item 4 follows this.
- **Geometry.** OptiX only on GPU (`wavefront/aggregate.cpp`). There is **no software GPU BVH** in pbrt.
- **Light sampler.** Default `"bvh"` (`wavefront/integrator.cpp:182-185`), `BVHLightSampler` (`P:lightsamplers.h:102-260`).
  - One light per leaf.
  - **32-byte node**: `CompactLightBounds` holds the octahedral-encoded cone axis, `phi` (power), 15-bit quantized cos θ_o and cos θ_e, a two-sided bit,
    and a **16-bit quantized AABB** relative to the scene bounds. A 31-bit child/light index plus a leaf bit complete the node.
  - Uses a bit-trail map for the PMF.

| | Cycles X | pbrt-v4 wavefront |
|---|---|---|
| state | persistent per-path SoA, kernels pass indices | data copied into per-stage work-item queues |
| scheduling | largest queue first, host readback every launch | fixed stage order per depth, GPU-side sizes, no readback |
| regeneration | yes (tiles refill when live < 1/4) | no (one pass = one sample of a scanline band) |
| shadow rays | separate shadow state pool, async | queue resolved within the same depth |
| sorting | counting sort by shader (partitioned on Metal) | per-material-type queues |
| film | float atomics, several samples per pixel live | per-pixel L, one path per pixel |
| memory/path | larger (state + no copies) | larger per stage (copies), but only live stages |

**Verdict for luce-render.** Take Cycles' *persistent SoA state with index queues* (smaller traffic than copying work items),
pbrt's *fixed stage order with GPU-resident queue sizes* (indirect dispatch instead of `index >= size` early-out),
and pbrt's *one path per pixel per pass*, which needs no float atomics. Add Cycles-style refill later, once the fixed-order loop is
measured. Refill matters mostly for long-tail bounce depths.

---

## 3. OpenPBR Surface (v1.1.1)

### 3.1 Parameters and defaults (`OpenPBR/reference/open_pbr_surface.mtlx:8-88`)

| Group | Parameter (default) |
|---|---|
| base | `base_weight` (1.0), `base_color` (0.8, 0.8, 0.8), `base_diffuse_roughness` (0.0), `base_metalness` (0.0) |
| specular | `specular_weight` (1.0), `specular_color` (1, 1, 1), `specular_roughness` (0.3), `specular_ior` (1.5), `specular_roughness_anisotropy` (0.0) |
| transmission | `transmission_weight` (0), `transmission_color` (1, 1, 1), `transmission_depth` (0), `transmission_scatter` (0, 0, 0), `transmission_scatter_anisotropy` (0), `transmission_dispersion_scale` (0), `transmission_dispersion_abbe_number` (20) |
| subsurface | `subsurface_weight` (0), `subsurface_color` (0.8, 0.8, 0.8), `subsurface_radius` (1.0), `subsurface_radius_scale` (1.0, 0.5, 0.25), `subsurface_scatter_anisotropy` (0) |
| fuzz | `fuzz_weight` (0), `fuzz_color` (1, 1, 1), `fuzz_roughness` (0.5) |
| coat | `coat_weight` (0), `coat_color` (1, 1, 1), `coat_roughness` (0), `coat_roughness_anisotropy` (0), `coat_ior` (1.6), `coat_darkening` (1.0) |
| thin film | `thin_film_weight` (0), `thin_film_thickness` (0.5 µm), `thin_film_ior` (1.4) |
| emission | `emission_luminance` (0, cd/m²), `emission_color` (1, 1, 1; HDR allowed since 1.1.1) |
| geometry | `geometry_opacity` (1), `geometry_thin_walled` (false, uniform), `geometry_normal`, `geometry_coat_normal`, `geometry_tangent`, `geometry_coat_tangent` |

Scalar count: 55 (9 color3 + 28 scalars).
- RGB variant: about 64 floats packed, so 256 B.
- Spectral variant: every color3 becomes (c0, c1, c2, scale), so 9×4 + 28 = 64 floats, also 256 B.
  Lay both out identically with colors as vec4.

Units and conventions from the spec:
- Emission is photometric (cd/m²). "The conversion to radiometric units is the responsibility of the renderer."
- The default color space is **ACEScg**.
- There is a length-unit-to-meters factor in metadata (it matters for `transmission_depth`, `subsurface_radius` and `thin_film_thickness`).

### 3.2 Layer structure

In words (spec "Model" section, `index.html`; mixture derivation in "Non-thin-walled mode"):
- **Fuzz** (coverage F) sits over **coat** (coverage C, an absorbing dielectric slab of `coat_color` = T² at normal incidence, with `coat_ior`).
- The coat sits over the **base substrate**, which mixes **metal** (M) with a **dielectric base** (1 − M).
- The dielectric base mixes **translucent** (T: specular BTDF into a homogeneous medium) with an **opaque base** (1 − T).
- The opaque base mixes **subsurface** (S) with **glossy-diffuse** (1 − S), a dielectric specular interface over an EON diffuse slab.
- Thin film (weight W) modifies the Fresnel of the base dielectric and metal interfaces.
- Emission lives under the coat and fuzz and is attenuated by them.
- `geometry_opacity` is a stochastic cut-out with the "nothing" (transparent) BSDF.
- `geometry_thin_walled` turns the bulk into a thin sheet:
  - transmission does not refract (straight-through, with only the roughness lobe)
  - subsurface becomes a diffuse BRDF/BTDF pair
  - there is no interior medium

The mixture weights as the viewer implements them (`V:openpbr_surface.glsl:51-159`), in the non-reciprocal albedo-scaling approximation. E_x is the directional albedo of lobe x at ω_o:

```
w_fuzz        = F
w_coated_base = lerp(1, 1 − E_fuzz, F)
w_coat        = w_coated_base · C
w_substrate   = w_coated_base · lerp(1, darkening · coat_color · (1 − E_coat), C)
w_metal       = w_substrate · M
w_dielectric  = w_substrate · (1 − M)
w_spec_brdf   = w_dielectric · specular_color · specular_weight
w_spec_btdf   = w_dielectric · T                              (transmission Fresnel omitted, spec 1119)
w_diffuse     = w_dielectric · (1 − T) · (1 − S) · base_weight · base_color · (1 − E_spec)
w_sss         = w_dielectric · (1 − T) · S                    (bulk: an interior medium behind a spec BTDF)
thin-walled SSS: diffuse BRDF += ½·S·subsurface_color·(1 − g)·(1 − E_spec), diffuse BTDF = ½·S·subsurface_color·(1 + g)·(1 − E_spec)
```

**Lobe selection.** Probability ∝ `|w_x · E_x|`. The 8 lobes are fuzz, coat, metal, spec BRDF, spec BTDF, diffuse BRDF, diffuse BTDF and SSS.
After sampling one lobe, **all other lobes are evaluated** at the sampled direction, and the returned pdf is `Σ p_x pdf_x`
(one-sample MIS over lobes, `:161-328`). That is the standard uber-BSDF pattern, and we should keep it.

### 3.3 Lobes, sampling and the specific models

- **Microfacet.**
  - NDF: anisotropic GGX with the spec's mapping (spec "Microfacet model"):
    `α_t = r²·sqrt(2/(1 + (1 − a)²))`, `α_b = (1 − a)·α_t`, so that `α_t² + α_b² = 2α²`.
    The viewer clamps α ≥ 1e-4 instead of special-casing perfect specular (`V:specular_brdf.glsl:7-16`).
  - Shadowing: Smith G2 (height-correlated in MaterialX, `mx_microfacet_specular.glsl:83-89`).
  - Sampling: **visible normals via spherical caps** (Dupuy & Benyoub 2023; `V:main.glsl` `ggx_ndf_sample`).
    The pdf is `D_V(m)/(4|ω·m|)`. For ω_i below the surface the hemisphere is mirrored.
- **Coat roughening of the base** (spec "Roughening", mtlx `:100-131`):
  `r_B' = lerp(r_B, min(1, r_B⁴ + 2 r_C⁴)^¼, C)`.
- **Dielectric Fresnel.**
  - Exact unpolarized formula (`V:main.glsl` `FresnelDielectricReflectance`).
  - IOR ratio: `η_s` mixes `specular_ior/1` with the coat-relative ratio (inverted if < 1) by `coat_weight` (`V:specular_brdf.glsl:33-46`).
  - `specular_weight` modulates F0 linearly by solving for a modified IOR ε. If TIR is possible, it uses the Fresnel at the refracted angle with
    the modified ratio, so refraction directions do not change (`main.glsl:384-411`, PR #247, not yet in 1.1.1).
    In 1.1.1 the MaterialX graph simply scales the reflection.
- **Metal: F82-tint** (Kutz et al., "Novel aspects of the Adobe Standard Material"):
  `F(μ) = Schlick(F0, μ) − μ(1 − μ)⁶ · (1 − F82_tint) · Schlick(F0, μ̄) / (μ̄(1 − μ̄)⁶)` with `μ̄ = 1/7`.
  - F0 = `base_weight · base_color`, tint = `specular_color`, result clamped to [0, 1] (`main.glsl:421-429`).
  - The BRDF is multiplied by `min(1, specular_weight · F)` (`V:metal_brdf.glsl:52-64`).
  - Cycles has the same model (`C:kernel/closure/bsdf_microfacet.h:56, 382-406`).
- **Coat.** A dielectric microfacet BRDF with `coat_ior` and independent roughness/anisotropy (`V:coat_brdf.glsl`).
  - **Darkening** (spec "Darkening", `V:openpbr_surface.glsl:93-119`):
    - `Δ = (1 − K)/(1 − E_b·K)`
    - `K = lerp(K_s, K_r, r_b)`, with `K_r = 1 − (1 − E_F(η_c))/η_c²` (rough base) and `K_s = F(μ_o, η_c)` (smooth base)
    - `E_F` = hemispherical Fresnel average via the closed fit `E_F(η) = ln((10893η − 1438.2)/(−774.4η² + 10212η + 1))` (`main.glsl:431-443`)
    - `r_b` = base roughness estimate, `E_b` = normal-incidence base albedo
    - Applied as `lerp(1, Δ, C·coat_darkening)`.
  - **Absorption**: `coat_color^(½(1/μ_i^t + 1/μ_o^t))` with refracted cosines (spec "View-dependent absorption").
    The viewer uses the normal-incidence value `coat_color`.
- **Energy compensation.**
  - The spec asks implementations to compensate microfacet multiple scattering
    ("An implementation should ideally account for this", spec "Microfacet model"), citing Kulla–Conty 2017 and Turquin 2019.
  - The viewer does **not**: no MS term, and it estimates lobe albedos by **4-sample Monte Carlo per vertex**
    (`V:specular_brdf.glsl:275-298`), which is noisy and costly.
  - MaterialX uses an **analytic rational fit** of GGX directional albedo `(NdotV, α) → (A, B)` with `E = F0·A + F90·B`
    (`MaterialX/.../mx_microfacet_specular.glsl:91-109`). Its compensation is `1 + F_avg·(1 − E)/E` (`:517-522`).
  - Cycles uses **tables**, which are what we want (see §3.5).
- **Diffuse: EON** (Portsmouth, Kutz, Hill 2024, "energy-preserving Oren–Nayar"; `V:diffuse_brdf.glsl`).
  - Single-scatter lobe: Fujii's improved ON, `f_ss = ρ/π · A_F (1 + r·s/t)`.
  - Multi-scatter lobe: `f_ms = ρ_ms/π · (1 − E_F(μ_o))(1 − E_F(μ_i))/(1 − Ē_F)` with `ρ_ms = ρ²Ē_F/(1 − ρ(1 − Ē_F))`.
  - Fully **analytic**, exact or polynomial-approximated `E_F` (`:47-117`).
  - Sampling: a mixture of uniform-hemisphere and a **clipped LTC** fitted with closed-form coefficients (`:121-196`).
  - Directional albedo is analytic (`E_EON`, `:105-114`). **No tables.**
- **Fuzz: Zeltner et al. 2022 LTC sheen** (enabled in 1.1).
  - Evaluation and sampling via an LTC with **rational fits** for the inverse matrix entries (aInv, bInv).
  - Directional albedo from a **Gaussian fit** (`V:fuzz_brdf.glsl:9-25`; identical constants in `MaterialX/.../mx_microfacet_sheen.glsl:96-115`).
  - The value is `fuzz_color · albedo · D_ltc(ω_i)/μ_i`. Roughness is clamped to [0.01, 1]. **No tables.**
- **Translucent base / transmission.**
  - Rough dielectric BTDF (`V:specular_btdf.glsl`). The interior medium has `μ_t = −ln(transmission_color)/transmission_depth`,
    albedo = `transmission_scatter`, HG anisotropy (`openpbr_surface.glsl:250-264`).
  - With depth 0, `transmission_color` tints the interface directly.
  - **Dispersion**: Cauchy `n(λ) = A + B/λ²` from `n_d = specular_ior` and `V_d = abbe/scale`, using the Fraunhofer C/d/F lines
    (`V:main.glsl:540-551`, spec "Dispersion"). This is natural in spectral mode with `TerminateSecondary`.
- **Subsurface.**
  - The spec defines it as a volumetric random walk inside the bulk, with MFP = `subsurface_radius · radius_scale`
    and `subsurface_color` as the multiple-scatter albedo.
  - The viewer maps color → single-scatter albedo with the Hyperion-style fit `1 − s²`,
    `s² = exp(−11.43A + 15.38A² − 13.91A³)`, adjusted for anisotropy (`openpbr_surface.glsl:266-276`).
  - Thin-walled mode uses the two diffuse lobes above.
- **Thin film.**
  - Airy summation with complex arithmetic (Kutz & Portsmouth), evaluated **per wavelength** over a dielectric or conductor substrate.
    The conductor's (n, k) comes from F0/edge-tint via Gulbrandsen (`V:thin-film.glsl`).
  - In spectral mode this is per lane with no special handling, since the direction does not depend on λ.
  - Cycles in RGB mode needs a Fourier-integrated CMF table `table_thin_film_cmf[512][6]` (`C:scene/shader.tables:1287`, `scene/shader.cpp:955-975`).
    **Spectral removes that table.**

### 3.4 Staged subset

| Stage | Lobes / features |
|---|---|
| **v1** | base diffuse (EON, analytic), metal (F82-tint), dielectric specular BRDF with albedo-scaled diffuse below it, specular roughness + anisotropy, **coat** (BRDF + darkening + normal-incidence absorption + roughening of the base), emission (luminance + color), `geometry_opacity` (stochastic, opaque/cutout only), thin-walled flag for opacity cutouts and leaves, energy compensation from tables. Transmission v1-lite: **smooth and rough dielectric BTDF with interface tint** (`transmission_color`, depth = 0), no interior medium. |
| **v2** | `transmission_depth` (Beer–Lambert interior via a one-entry "current interior" in path state), dispersion (spectral + TerminateSecondary), **fuzz** (Zeltner LTC, analytic), **thin film** (spectral Airy), coat view-dependent absorption, nested-dielectric priorities, thin-walled transmission and subsurface as diffuse BTDF. |
| **v3** | Bulk subsurface random walk (needs an interior medium with a scattering albedo, Cycles' `subsurface_random_walk.h` as heuristics reference), `transmission_scatter` (scattering interior), separate coat normal/tangent, reciprocal or position-free layering refinements. |

### 3.5 Precomputed tables we need

Build them offline with a Luce tool by Monte Carlo over our own BSDF code, so the tables match our evaluation exactly.
Cycles' generator `C:app/cycles_precompute.cpp:20-230` shows the method: sample the BSDF, average `eval/pdf`, 2^20–2^26 samples per cell.

| Table | Axes / res | Use | Cycles analogue |
|---|---|---|---|
| `E_ggx(μ, r)` | 32×32 | metal / any conductor MS compensation (Kulla–Conty `1 + F_avg(1 − E)/E` style) | `ggx_E` 32×32 |
| `Ē_ggx(r)` | 32 | the average for F_ms | `ggx_Eavg` 32 |
| `E_spec_refl(μ, r, z(η))` | 32×32×16 | dielectric **reflection-only** albedo with exact Fresnel. Drives `1 − E_spec` for diffuse below specular, coat `1 − E_coat`, lobe selection. η>1 and η<1 halves (or z mapped over both). | `ggx_gen_schlick_ior_s` 16³, which stores the interpolation factor s with E = lerp(F0, 1, s)·E_ggx |
| `E_glass(μ, r, z(η))` | 16×16×16 (+ inverse-η table) | rough dielectric BSDF (R+T) MS compensation | `ggx_glass_E`, `ggx_glass_inv_E` |
| `Ē_glass(r, z)` | 16×16 (+ inverse) | average for the above | `ggx_glass_Eavg`, `ggx_glass_inv_Eavg` |
| F82 average | analytic or 32×32 (F0 is per lane, so tabulate s(μ, r) as Cycles does) | metal F_ms | reuses `ggx_gen_schlick_s` at exponent 5 (`bsdf_microfacet.h:537-548`) |
| RGB→sigmoid coefficients | 3×64×64×64×3, ×2 gamuts (ACEScg, Rec.709) | spectral upsampling | pbrt `*ToSpectrumTable_Data` |
| CIE 1931 2° CMF | 471×3 (1 nm) | film | pbrt `CIE_X/Y/Z` |
| D60, D65 illuminants | 1 nm | illuminant spectra for emission | pbrt `Spectra::D` |
| Sobol–Burley directions | 4×32 u32 | sampler | `sobol_burley_table` |
| pixel filter inverse CDF | 1024 | filter importance sampling | `filter_table` |

No tables are needed for EON, Zeltner fuzz or `E_F(η)` (all closed form), nor for thin film in spectral mode. Total table memory is about 19 MB,
dominated by the two RGB→spectrum gamuts. Store them in one `tables` storage buffer with a header of offsets.
Index them by **z = sqrt(|η − 1|/|η + 1|)** as Cycles does (`bsdf_microfacet.h:457`), so precision concentrates around η ∈ [1, 2].

---

## 4. Path guiding and caching

### 4.1 OpenPGL and what Cycles does with it

- **CPU only in Cycles.** `__PATH_GUIDING__` is defined only when `!__KERNEL_GPU__` (`C:kernel/features.h:158-163`).
  OpenPGL itself offers CPU device types only (`G:include/openpgl/config.h:32-39`), and its README says GPU back ends are "planned".
- **Cycles integration** (`C:integrator/path_trace.cpp:1440-1560`).
  - A `Field` with a **kd-tree spatial structure** (max depth 16) and either **VMM** or a **directional quad-tree** per leaf.
  - Kernels record path segments into a `SampleStorage` while rendering.
  - After each iteration, `Field::Update` refits once at least 1024 samples are stored.
  - During training the scheduler limits each update to 4 spp (`guiding_prepare_structures`).
- **Use in shading.** `surface_shader_prepare_guiding` and `guiding_*` in `C:kernel/integrator/guiding.h` handle it:
  - guided-vs-BSDF selection by a probability (`surface_guiding_sampling_prob`)
  - RIS between BSDF and guiding (`PRNG_SURFACE_RIS_GUIDING_*`)
  - `unguided_throughput` kept so RR is not biased by guiding pdfs
- **OpenPGL data model.**
  - Kd-tree nodes are 8 B: split position + 2-bit dimension + 30-bit child or data index (`G:spatial/kdtree/KDTree.h:20-140`).
  - A leaf splits beyond `maxSamples = 32000` samples (`config.h:30, 42-48`).
  - Each leaf holds a **parallax-aware von Mises–Fisher mixture**: up to `PGL_VMM_MAX_COMPONENTS = 32` lobes (init K = 16), each with a weight,
    κ ≤ 320000, a mean direction, a **distance** (parallax compensation re-centers lobes when the query point moves inside the cell), and a pivot
    (`G:directional/vmm/ParallaxAwareVonMisesFisherMixture.h:34-77`, `include/openpgl/common.h:29-30`).
  - Fitting is **weighted online EM** with chi-square-driven **split and merge** of components (`directional/vmm/*Factory.h`, `VMMChiSquare*`).
  - Training samples are `{position, direction, weight = radiance/pdf, pdf, distance, flags}` (`G:include/openpgl/data.h:21-66`).
  - Region lookup is **stochastic nearest-neighbor** among the 4 of 8 precomputed neighbours, using one random number
    (`G:field/Field.h:114-130`, `spatial/KNN.h:16-17`). This hides cell boundaries.
- **Verdict.** OpenPGL's EM fitting with split/merge and a CPU kd-tree is a **host-side, CPU training** design. It would need a GPU port of
  EM, split/merge and the tree. Not a fit for a GPU-only first implementation.

### 4.2 Other options, from the literature (not studied in source here)

- **ReSTIR DI** (Bitterli et al. 2020). Reservoir resampling of light samples with spatial and temporal reuse.
  - Great for interactive many-light *direct* lighting.
  - Reuse makes it biased unless the unbiased variants are used, and it is temporal by design.
  - For a final-frame production tracer the **light tree** already does the per-vertex job. ReSTIR's value is in the viewport.
- **ReSTIR GI / PT** (Ouyang 2021; Lin et al. 2022 "GRIS").
  - Resample whole path suffixes and reconnect them.
  - High complexity: shift mappings, Jacobians, and interactions with rough/specular lobes and spectral lanes.
  - A later research item.
- **Neural radiance caching** (Müller et al. 2021).
  - Online-trained small MLP queried at path terminations.
  - Needs fast fused MLP training (tensor/simdgroup-matrix paths). luce-nn is paused.
  - Biased (cache termination), and best suited to real-time. Not first.
- **Practical Path Guiding** (Müller et al. 2017; SD-tree).
  - Spatial binary tree with a directional **quad-tree** per leaf, trained in doubling passes from radiance splatted by atomics.
  - GPU ports exist and fit wavefronts well. Training is atomic adds into quad-tree cells, and refinement is a cheap per-iteration rebuild.
  - Cycles/OpenPGL also offer the directional quad-tree (`GUIDING_TYPE_DIRECTIONAL_QUAD_TREE`).
- **Hash-grid vMF mixtures**, e.g. MCMM real-time guiding (Dittebrandt et al. 2023).
  - A spatial **hash grid** (cells keyed by quantized position + normal) of **small vMF mixtures** (1–4 lobes).
  - Updated **online in the shading kernel** with running sufficient statistics (no EM pass).
  - Fully GPU-resident, constant memory, no tree build. The closest match to a wavefront kernel with int atomics.

### 4.3 Recommendation

1. **v1: no guiding.** Get light tree + MIS + RR + adaptive sampling right first. That covers the "many lights" half of the speed story.
2. **v2: GPU hash-grid directional guiding** (MCMM-style vMF lobes, or a small fixed-resolution directional grid per cell).
   - Cells are keyed by (quantized world position at a scale chosen from the camera footprint, octahedral normal bin).
   - Training statistics come from completed path segments. The kernel writes `(cell, direction, scalar luminance/pdf)` records into an
     append buffer, and a separate `guide_update` kernel folds them into the cell lobes. One writer per cell avoids float atomics.
   - Sampling: one-sample MIS between BSDF and guide with a guide probability per cell (Cycles' `surface_guiding_sampling_prob` idea).
     Keep **unguided throughput for RR** as Cycles does.
   - Spectral: train on **luminance (Y) of the 4-lane contribution**. Guiding pdfs stay λ-independent, which preserves the scalar-pdf rule from §2.5.
3. **v3: evaluate an SD-tree variant**, which is better for glossy and caustic paths. If the owner wants it, try **ReSTIR DI for the luced-3d viewport mode** only.

---

## 5. Proposed luce-render architecture

### 5.1 Constraints from luce-gpu (now)

- Compute only.
- **Set 0, bindings 0..15**, so at most 16 buffers or images per kernel.
- **≤128 B push constants.**
- Storage buffers without BDA.
- Images: rgba32f, r32f, r32ui, rg32f.
- Int atomics, optional float atomicAdd.
- Indirect dispatch and automatic barriers.

Later: subgroups, timestamps, BDA, ray queries (GL_EXT_ray_query → Metal `intersection_query`) with a software BVH fallback.

Two consequences:
- **SoA must live in a few big buffers.** With 16 bindings and no BDA, one buffer per field (Cycles' approach) is impossible.
  Use **one `PathState` buffer**, every field an array at a known base offset, the offsets passed in a constants block.
  Bind the same VkBuffer twice under two views: `uint words[]` for 4-byte fields and `vec4 quads[]` for 16-byte fields.
  This is legal aliasing in Vulkan, and spirv-cross maps it to two Metal buffer arguments.
- **Never store `vec3` arrays.** In std430 a `vec3[]` element has a 16-byte stride, so store vec4 or split floats.

### 5.2 Kernels (one SPIR-V module per kernel, two variants each via `-DSPECTRAL`)

| # | Kernel | Reads → writes | Notes |
|---|---|---|---|
| 0 | `reset` | — → queues, counters | per frame or progressive restart |
| 1 | `schedule` | counters → indirect args, next counters cleared | **1 thread**. Replaces Cycles' host-side `get_most_queued_kernel` and readback |
| 2 | `camera` | free queue / tile params → main state, `Q_closest` | sample pixel (filter IS), sample λ (spectral), USD camera (thin lens, later motion), skip converged pixels |
| 3 | `intersect_closest` | `Q_closest` → isect, `Q_surface` / `Q_miss`, `free` | BVH or ray query. **RR here** (Cycles). Lights are BVH geometry, so a light hit is a surface hit with an emissive material id |
| 4 | `shade_surface` | `Q_surface` → state, shadow state, `Q_shadow`, `Q_closest` | OpenPBR prepare (lobe weights + albedos from tables), emission with forward MIS (light-tree pdf), denoise features at first non-specular vertex, NEE (light tree → sample → BSDF eval → **shadow slot = path index**), BSDF sample, throughput, `TerminateSecondary`, bounce counters |
| 5 | `shade_miss` | `Q_miss` → state, `free` | background / distant lights with MIS. Dome later. Writes path L to film and frees the path |
| 6 | `intersect_shadow` | `Q_shadow` → `Q_shade_shadow` (visible) | any-hit early exit. v1: opaque only (opacity cutouts via stochastic any-hit test). v2: transparent hits |
| 7 | `shade_shadow` | `Q_shade_shadow` → main L accumulator | adds the unoccluded contribution into the **parent's** `L` (no atomics, §1.12-4) |
| 8 | `terminate` | `Q_terminate` → film, `free` | `film[pixel] += toFilm(L, λ)`, clamp, class-A aux, sample count. Can fold into 3/5 |
| 9 | `adaptive_check`, `adaptive_filter_x/y` | film → aux | Cycles §1.6 |
| 10 | `film_convert` | film → display image (rgba32f) | divide by count, spectral XYZ → ACEScg, exposure. Tone mapping stays in luce-color/ACES 2.0 downstream |
| later | `sort_surface` (counting sort by material/texture key), `guide_update`, `compact` | | |

Per-bounce dispatch sequence, recorded once per batch of K bounces:
`schedule, intersect_closest, schedule, shade_surface, shade_miss, schedule, intersect_shadow, schedule, shade_shadow`.
Then `camera` refill at the top of each batch. The host reads `live_count` asynchronously, one batch behind, to decide refills and
when a sample pass is done. A hard cap on bounces (`max_bounce`) bounds a batch.

### 5.3 Path state (per path, SoA inside one buffer)

| Field | Type | Bytes |
|---|---|---|
| ray origin + tmax | vec4 | 16 |
| ray direction + ray-cone spread (or time) | vec4 | 16 |
| throughput | `Spec` stored as vec4 | 16 |
| radiance accumulator L | vec4 | 16 |
| wavelengths λ0..λ3 (spectral only; pdf recomputed from λ via the analytic `VisibleWavelengthsPDF`, plus 1 flag bit for "secondaries terminated") | vec4 | 16 |
| isect: t, prim, instance, packed barycentrics (2×unorm16) | uvec4 | 16 |
| pixel index | u32 | 4 |
| sample index | u32 | 4 |
| bounce counters packed (total, diffuse, glossy, transmission, transparent: 6 bits each) | u32 | 4 |
| path flags (MIS skip, specular, inside-dielectric, denoise-features-pending, …) | u32 | 4 |
| `mis_ray_pdf` (scalar, §2.5) | f32 | 4 |
| `mis_origin_n` (octahedral 2×16) | u32 | 4 |
| `min_ray_pdf` (filter glossy) | f32 | 4 |
| interior material id (v2 transmission depth) | u32 | 4 |
| **main total** | | **128 B** |

- RNG state: none. `(pixel, sample, bounce)` determine every Sobol–Burley dimension, and `rng_offset = 16·bounce` (+1 camera dimension for λ).
- RGB variant: drop λ (−16 B, 112 B). Keep the layout identical otherwise.

Shadow state (same index as the parent):

| Field | Bytes |
|---|---|
| origin + tmax (vec4) | 16 |
| direction + self-hit (instance, prim) packed (vec4 / uvec4 with floatBitsToUint) | 16 |
| unoccluded contribution (`Spec`, vec4) | 16 |
| self (instance, prim) if not packed above, flags | 8 |
| **shadow total** | **≈56–64 B** |

Queues: `Q_closest`, `Q_surface`, `Q_miss`, `Q_shadow`, `Q_shade_shadow`, `free` at 4 B each, ≈24 B/path. Counters and indirect args are a few hundred bytes.

**Budget ≈ 128 + 64 + 24 = 216 B per path.**
- 1M paths (2^20): ~216 MB
- 2M: ~432 MB
- 4M (Cycles' Metal default): ~864 MB

Recommend **2^20 default**, raised to 2^21–2^22 on large-memory Apple GPUs after measuring occupancy. Cycles notes that more than 4M states brings no notable gain on M1, and it caps growth at two doublings because of diminishing returns (`device/metal/queue.mm:270-310`).

Path ordering: sample-major, **one sample per pixel per pass** (the tile is the whole image, or bands when pixels > pool, like pbrt's
`scanlinesPerPass`). Within the image, index by a Morton or 8×8-tile swizzle so neighbouring threads share BVH and texture caches.
Add Cycles-style partial-tile refill only after measuring.

### 5.4 Scene buffers

| Buffer | Content and layout |
|---|---|
| `geo_tri` | per triangle, **3 world- or object-space vertices** as 3×vec4 (48 B). Pre-gathered for intersection, the Cycles `prim_tri_verts` approach. Optionally switch to indexed + 16-bit quantization later |
| `geo_attr` | per vertex: oct-normal (4 B), uv (8 B), tangent oct + sign (4 B). Per triangle: material id (u16), smoothing flags |
| `bvh_nodes` | BLAS per mesh + TLAS over instances, Cycles BVH2 node (64 B: both child AABBs SoA'd by axis + child indices + visibility mask) and 16 B leaves. Host build in Luce Base: binned SAH, ≤4–8 triangles per leaf, spatial splits later. v2: GPU builder (LBVH/PLOC) for interactive edits, and **CWBVH-style 8-wide compressed nodes** once subgroups land |
| `instances` | 3×4 object→world + 3×4 world→object (96 B), BLAS root, material override, visibility mask, light-tree sub-tree root |
| `lights` | UsdLux rect/disk/sphere/distant (dome later), 96 B: type, position, u/v axes with half-sizes (rect/disk) or radius, normal, cone/spread (`shaping:cone:angle` later), emission (sigmoid c0..c2 + scale **or** blackbody T + scale), normalize flag, visibility. Rect and disk lights are also **emissive BVH primitives** so BSDF rays hit them (two triangles or an analytic disk test in the leaf) |
| `light_tree` | 32 B nodes (pbrt `CompactLightBounds` layout: oct axis, φ, quantized cos θ_o/θ_e, quantized AABB, child/emitter index) **with Cycles' min/max-importance selection** and ≤8 emitters per leaf. Emitters: UsdLux lights + emissive triangles (energy = luminance × area). Per-emitter bit trail for the MIS pdf. Emissive instanced meshes reference shared sub-trees (Cycles' instance nodes) |
| `materials` | **OpenPBR param block, 256 B** (64 vec4-packed floats; colors as vec4 = RGB+pad in the RGB variant or sigmoid c0, c1, c2, scale in the spectral one) + a 64 B texture-binding block (16 × u32 texture slot/transform index) |
| `textures` | v1 without bindless: a **texel pool storage buffer** (or a few large rgba32f storage images as atlases) with manual bilinear. Spectral textures are stored as coefficient texels (c0, c1, c2, scale) |
| `tables` | §3.5 tables with a header of offsets |
| `film` | rgba32f accum (RGB, or XYZ in spectral), r32ui sample count, rgba32f class-A aux, albedo + normal (rgba32f each) for OIDN-style denoising |
| `guide` (v2) | hash-grid cells |

Binding plan: 0 constants, 1 state u32 view, 2 state vec4 view, 3 queues/counters/args, 4 film, 5 geo_tri, 6 geo_attr + instances (u32 view),
7 bvh, 8 materials, 9 lights + light tree, 10 textures, 11 tables, 12 guide, 13 aux/debug. That is 14 of 16.
Push constants hold the batch/bounce index, sample index, frame seed and kernel variant flags (≤128 B).

### 5.5 Spectral/RGB generic code

```glsl
#ifdef SPECTRAL
  #define Spec vec4                     // 4 hero-rotated wavelengths
  Spec spec_from_coeffs(vec4 c, vec4 lambda);     // scale * sigmoid(c0 λ² + c1 λ + c2) per lane
#else
  #define Spec vec3
  Spec spec_from_coeffs(vec4 c, vec4 lambda) { return c.rgb; }
#endif
float spec_luminance(Spec s, vec4 lambda);  // Y-weighted / pdf in spectral; dot(ACEScg→Y) in RGB
```

Rules:
- every pdf is a scalar
- lobe and light selection use `spec_luminance` of the λ-independent RGB parameters
- dispersion and λ-dependent refraction call `terminate_secondary()`, which sets a flag, zeroes lanes 1..3 of the throughput, and multiplies lane 0 by 4
  (pbrt's pdf adjustment, folded into the throughput)
- the film write does `average(cmf(λ_i)·L_i / pdf(λ_i))`, with lane pdf = 0 for terminated lanes
- thin film and F82 run per lane in spectral mode and per channel in RGB

### 5.6 Staging

- **v1: "correct and fast enough."**
  - Kernels: camera, intersect_closest (software BVH2, CPU-built), shade_surface, shade_miss, intersect_shadow, shade_shadow, terminate, film_convert, schedule.
  - Fixed-order indirect loop, 1 spp per pixel per pass, 2^20 paths.
  - Sobol–Burley + blue-noise-first, filter IS.
  - UsdLux rect/disk/sphere/distant + emissive OpenPBR meshes through the light tree, power-heuristic MIS, RR, clamping.
  - OpenPBR v1 subset (§3.4) with tables, both `-DSPECTRAL` and RGB builds, XYZ/ACEScg film.
  - Adaptive sampling, albedo + normal AOVs for the denoiser.
  - Validation: furnace tests per lobe (white furnace for EON/metal/coat with compensation), RGB-vs-spectral A/B on non-dispersive scenes (they should match within upsampling error), Cornell-style convergence against a reference.
- **v2: speed and materials.**
  - Ray queries (Metal `intersection_query` via spirv-cross) with BVH2 as fallback.
  - GPU BVH refit/build for editing.
  - Counting sort of `Q_surface` by material key.
  - Hash-grid path guiding.
  - OpenPBR v2 subset: transmission depth, dispersion, fuzz, thin film.
  - Dome light (importance map + light-tree distant node), subgroup-based queue append (one atomic per subgroup, Cycles `parallel_active_index.h`), timestamps for per-kernel profiling, Cycles-style partial refill.
- **v3: production.**
  - Bulk subsurface and scattering interiors (volume stack beyond one entry), BDA (to remove the 16-binding packing), motion blur, light linking,
    SD-tree guiding evaluation, CWBVH, many-instance scenes, Vulkan back end on Windows/Linux with the same SPIR-V.
