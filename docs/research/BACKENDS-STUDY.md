# luce-render: how Cycles handles GPU backends, and what that means for us

Status: research note, 2026-10-07. This is the first step of the optimization research.
The owner's question: do we pay for luce-gpu's lowest-common-denominator abstraction
(GLSL 4.5 → SPIR-V → spirv-cross → MSL), and how does Cycles share one kernel across
CPU/CUDA/OptiX/HIP/HIP-RT/Metal/oneAPI without that cost? This note is not code to copy.
Every mechanism is described in our own words, with file:line references so a reader
can check each claim.

Our baseline, as measured: about 9 ms per 1080p spectral sample on an M4 Max with 12 bounces.
`shade_surface` takes about 4.3 ms and `intersect_closest` about 1.3 ms. A pass has about
90 dispatches, and each dispatch seems to cost about 30 µs of overhead.

## 0. Sources

| Source | Where | Version |
|---|---|---|
| Cycles standalone | `/Users/sedov/Dev/luce_dev/.donors/render/cycles` | `a456b76` (2026-09-08), shallow |
| luce-gpu (ours) | `/Users/sedov/Dev/luce_dev/luce-gpu/src/gpu/metal/*.lucb` | working tree |
| luce-render (ours) | `/Users/sedov/Dev/luce_dev/luce-render` | working tree |
| Blender commit da4ef05 (intersection-kernel specialization, 2022-07) | <https://archive.blender.org/lists/bf-blender-cvs/2022-July/175756.html> | measured numbers |
| Blender commit a7cc6e0 ("Additional Metal kernel specialisation exposed through UI", 2023-01) | <https://lists.blender.org/pipermail/bf-blender-cvs/2023-January/182555.html> | |
| Blender commit e270a19 (DONT_SPECIALIZE markup) | <https://lists.blender.org/pipermail/bf-blender-cvs/2023-January/183184.html> | |
| Blender commit b82de02 ("Enable inlining on Apple Silicon for 1.1x speedup", 2022-04) | <https://lists.blender.org/pipermail/bf-blender-cvs/2022-April/172816.html> | |
| Blender commit 654e1e9 ("Use local atomics for faster shader sorting", 2023-02) | <https://developer.blender.org/rB654e1e9> | |
| Cycles X announcement (OpenCL removal) | <https://code.blender.org/2021/04/cycles-x/> | |
| Apple docs: `MTLDispatchType.concurrent` | <https://developer.apple.com/documentation/metal/mtldispatchtype/concurrent> | |
| Apple forum: concurrent dispatch on Apple GPUs | <https://developer.apple.com/forums/thread/721349> | |
| Blender 4.0 release notes (MetalRT default on M3) | <https://wiki.blender.org/release_notes/4.0/> | |

`C:` means `cycles/src/`. `G:` means `luce-gpu/src/gpu/`. `R:` means `luce-render/`.

---

## 1. Architecture: one kernel source, many backends

### 1.1 The shape

Cycles has **one kernel**, written in a restricted C++ dialect. It is about 55 k lines of
headers under `C:kernel/` (integrator, closures, SVM, geometry, BVH, lights, film), not
counting `kernel/device/` and OSL. Each backend contributes two things:

1. **A thin "device kernel" shim** in `C:kernel/device/<backend>/`. It holds `compat.h`
   (qualifier and math macros), `globals.h` (how `kernel_data` and the data arrays are
   reached), sometimes `config.h` (block sizes and register caps), and one entry file
   (`kernel.cu`, `kernel.metal`, `kernel.cpp`) that includes the shared entry-point
   header `C:kernel/device/gpu/kernel.h`.
2. **A host "device" layer** in `C:device/<backend>/`. It holds `Device` (memory and kernel
   loading), `DeviceQueue` (enqueue and synchronize) and, for RT backends, BVH builds.

The **scheduler** (`C:integrator/path_trace_work_gpu.cpp`) is shared by every GPU backend.
It only talks to the abstract `DeviceQueue` (`C:device/queue.h:89-198`).

### 1.2 Macro layer (the "ccl_" dialect)

Shared code never names an address space or a qualifier directly. Each backend maps the macros:

| Macro | CUDA (`C:kernel/device/cuda/compat.h`) | Metal (`C:kernel/device/metal/compat.h`) | CPU (`C:util/defines.h`) |
|---|---|---|---|
| `ccl_device` | `__device__ __inline__` (:33) | empty (:35) | `static inline` (:25) |
| `ccl_device_inline` | `__device__ __inline__` (:35) | `always_inline` (:36) | `__forceinline` / `always_inline` (:32, :48) |
| `ccl_device_noinline` | noinline | **inline on Apple GPUs** (`__KERNEL_METAL_APPLE__`, :38-42) | noinline |
| `ccl_global` | empty (:41) | `device` (:48) | empty |
| `ccl_private` | empty (:47) | `thread` (:54) | empty |
| `ccl_gpu_shared` | `__shared__` | `threadgroup` (:53) | n/a |
| `ccl_gpu_ballot` | `__ballot_sync` | `simd_ballot` (:79) | n/a |
| `ccl_gpu_syncthreads` | `__syncthreads` | `threadgroup_barrier(mem_threadgroup)` (:80) | n/a |

Math is mapped as well. On Metal, `sinf/cosf/tanf/expf/sqrtf/logf` become `fast::` variants,
with the comment "Use native functions with possibly lower precision for performance, no
issues found so far" (`C:kernel/device/metal/compat.h:285-293`).

### 1.3 Globals: how a kernel reaches scene data

All backends use the same idea: **one parameter block holding raw device pointers to every
data array, plus the `KernelData` constants and the integrator-state SoA pointers**.
X-macro files generate it: `C:kernel/data_arrays.h` and `C:kernel/data_template.h`.

- **CUDA:** `__constant__ KernelParamsCUDA kernel_params;`. `kernel_data` is
  `kernel_params.data` and `kernel_data_fetch(name, i)` is `kernel_params.name[i]`
  (`C:kernel/device/cuda/globals.h:26-46`).
- **OptiX:** the same block becomes the OptiX launch-params variable (`pipelineLaunchParamsVariableName =
  "kernel_params"`, `C:device/optix/device_impl.cpp:395`).
- **Metal:** `struct KernelParamsMetal { <pointer per array>; IntegratorStateGPU integrator_state;
  KernelData data; }` (`C:kernel/device/metal/globals.h:17-25`). The accessor macros route
  through `launch_params_metal` (:34-38). The host keeps this struct in **one shared
  MTLBuffer** and writes `gpuAddress` values straight into it
  (`C:device/metal/device_impl.mm:205-210`, `:1023-1040`).

So **Cycles is bindless on every GPU**. Kernels see one pointer table, never per-kernel bindings.

### 1.4 Entry points: how integrator kernels are instantiated per device

`C:kernel/device/gpu/kernel.h` defines every GPU entry point once, using backend macros:

```
ccl_gpu_kernel(GPU_KERNEL_BLOCK_NUM_THREADS, GPU_KERNEL_MAX_REGISTERS)
    ccl_gpu_kernel_signature(integrator_shade_surface, const ccl_global int *path_index_array,
                             ccl_global float *render_buffer, const int work_size)
{ ... state = path_index_array ? path_index_array[i] : i;
      ccl_gpu_kernel_call(integrator_shade_surface(nullptr, state, render_buffer)); }
```

(`C:kernel/device/gpu/kernel.h:313-327`. Intersect_closest is at `:158-171`.)

- **CUDA/HIP:** `ccl_gpu_kernel` expands to `extern "C" __global__ __launch_bounds__(threads,
  regs-derived min blocks)` (`C:kernel/device/cuda/config.h:50-58`). The block size and register
  cap depend on the arch: 256 threads with 64 registers for sm ≤ 6.x, and 384 threads with
  168 registers for sm 7.x–12.x (`:16-46`). HIP uses 1024 threads with 64 registers
  (`C:kernel/device/hip/config.h:21-22`). `kernel.cu` only includes compat, config, globals
  and `gpu/kernel.h` (`C:kernel/device/cuda/kernel.cu`).
- **Metal:** MSL has no global kernel-parameter block and no free functions reaching it, so the
  Metal shim **wraps the whole shared kernel in a class**. `context_begin.h` opens `class
  MetalKernelContext { constant KernelParamsMetal &launch_params_metal; constant
  MetalAncillaries *metal_ancillaries; ...` (`C:kernel/device/metal/context_begin.h:13-24`).
  Every integrator header is included *inside* that class (`C:kernel/device/gpu/kernel.h:16-52`).
  `ccl_gpu_kernel_signature` then generates, per kernel, a parameter struct, a `kernel void
  cycles_metal_<name>(device const kernel_gpu_<name> *params, constant KernelParamsMetal&,
  constant MetalAncillaries*, threadgroup atomic_int*)` entry, and a `run()` that calls into the
  context (`C:kernel/device/metal/compat.h:118-183`). On macOS 14+ the thread-position builtins
  come from program-scope globals (`__METAL_GLOBAL_BUILTINS__`, `:398-…`,
  `C:device/metal/device_impl.mm:369-372`).
- **OptiX:** RT-using and shading kernels are separate `__raygen__kernel_optix_integrator_*`
  programs that call the same shared function (`C:kernel/device/optix/kernel.cu:25-70`,
  `C:kernel/device/optix/kernel_shader_raytrace.cu`). Every other kernel (sorting, compaction,
  film) is the plain CUDA module (`C:device/optix/queue.cpp:57-59`).
- **oneAPI:** SYCL. Kernel templates in `C:kernel/device/oneapi/kernel_templates.h`, plus a
  context class like Metal's (`context_begin.h`). Embree-on-GPU features are specialization
  constants (`C:kernel/device/oneapi/kernel.cpp:273, 310, 374`).
- **CPU:** `C:kernel/device/cpu/kernel_arch_impl.h:56-70` defines per-ISA wrappers.
  `kernel.cpp` and `kernel_avx2.cpp` compile that same header once per ISA. The CPU runs the
  megakernel path, not the wavefront.

### 1.5 Size of the device-specific layer (`wc -l`, all files)

| Backend | Kernel shim `C:kernel/device/<b>/` | Host layer `C:device/<b>/` | Notes |
|---|---:|---:|---|
| shared GPU entry (`gpu/`) | 1 981 | — | entry points, active-index, prefix sum, sort, work stealing |
| CPU | 2 031 | 818 | per-ISA instantiation |
| CUDA | 506 | 2 389 | |
| OptiX | 1 296 | 2 610 | + CUDA host it inherits |
| HIP | 361 | 2 300 | |
| HIP-RT | 1 131 | 1 183 | |
| **Metal** | **1 896** | **5 815** | largest: PSO cache, specialization, MetalRT, residency |
| oneAPI | 1 608 | 2 730 | |
| shared kernel (`kernel/` minus `device/`, minus OSL) | ≈ 55 500 | — | |

So the backend-specific code is **≈ 3 % of the kernel** per backend, and **≈ 7 700 lines for
Metal in total**. Almost all of Metal's size is host-side policy (pipeline cache,
specialization, MetalRT tables), not kernel code.

### 1.6 What is device-specific vs shared

Shared: all integrator logic, closures, SVM, lights, film, BVH2 traversal, path-state SoA
layout, queue counters, active-index/compaction/sort *algorithms* (`gpu/parallel_*.h`), and the
whole scheduler.

Device-specific:
- qualifiers, math intrinsics, builtins;
- the kernel-parameter carrier (constant block / launch params / pointer table);
- texture sampling (Metal: bindless `texture2d` handles in an ancillary table,
  `C:kernel/device/metal/context_begin.h:26-50`);
- the RT API (OptiX `optixTrace`, MetalRT `intersector` with intersection-function tables,
  HIP-RT, Embree-GPU);
- block sizes and register caps;
- and, on Metal only, **runtime compilation from source with scene specialization** (§3.5).

---

## 2. Device and queue layer, and the scheduler

### 2.1 `DeviceQueue` API

`C:device/queue.h:89-198` has these methods:
- `enqueue(kernel, work_size, args)`: a 1-D launch, with arguments as a list of int, float and
  device pointers;
- `synchronize()`;
- `zero_to_device`, `copy_to_device`, `copy_from_device`, all in queue order;
- three sizing hooks: `num_concurrent_states(state_size)`, `num_concurrent_busy_states`,
  `num_sort_partitions`;
- `supports_local_atomic_sort()`.

There is **no indirect dispatch** in the API, and no events beyond `synchronize`.

### 2.2 The scheduler loop: host readback after every step

`PathTraceWorkGPU::render_samples` (`C:integrator/path_trace_work_gpu.cpp:348-412`) loops:

1. `enqueue_work_tiles` (`:812-905`). It only adds new camera paths when the most-queued
   kernel is `INTERSECT_CLOSEST` or nothing is queued (`:818-821`), so new and old paths stay
   aligned. It adds them only when active paths fall below `min_num_active_main_paths_`
   (`:863`). Before the tiles it may compact the main paths (`:890`).
2. If work was added: `update_queue_counter_and_cache()` copies `IntegratorQueueCounter` to the
   host and **calls `queue_->synchronize()`** (`:324-331`).
3. `enqueue_path_iteration()` (`:459-517`) picks **the kernel with the most queued paths**
   (`get_most_queued_kernel`, `:414-435`). If that kernel spawns shadow paths and the shadow
   pool is short, it runs a shadow kernel first (`:483-506`). Then it enqueues the chosen kernel.
4. Again `update_queue_counter_and_cache()`, which is a **full host round trip**.

So Cycles makes **one blocking host↔GPU round trip per wavefront step**, and sometimes two:
`compact_paths` also calls `synchronize()` to read the compaction count (`:794-797`). On Metal,
`synchronize()` ends the encoder, signals a shared event, commits the command buffer and
waits on a semaphore (`C:device/metal/queue.mm:672-710`). The CUDA queue does
`cuStreamSynchronize` (`C:device/cuda/queue.cpp:153`).

There are **no CUDA graphs** (no `Graph` symbol anywhere in `C:device/`). There is **no
indirect dispatch**. Our design (a GPU `schedule` kernel writing indirect arguments, one
command buffer per batch, readback one batch behind) is already *more* submission-efficient
than Cycles.

### 2.3 Launches per step

One step = one integrator kernel plus 0–3 helper kernels and 0–2 fills:

- **Not sorted, queue sparser than the index range:** zero `num_queued_paths` (a blit fill),
  then `QUEUED_PATHS_ARRAY` or `QUEUED_SHADOW_PATHS_ARRAY` to build an index list
  (`:536-548`, `:676-690`), then the kernel.
- **Not sorted, dense:** the kernel alone runs over `[0, max_active_index)` with
  `path_index_array = null` (`:524-537`).
- **Sorted (`SHADE_SURFACE`, `SHADE_SURFACE_RAYTRACE` only, `:1308-1312`):**
  - with local atomic sort: `SORT_BUCKET_PASS` then `SORT_WRITE_PASS` (`:618-633`);
  - otherwise `PREFIX_SUM` (one thread!), a fill, then `SORTED_PATHS_ARRAY` (`:636-673`).
  Shading kernels count their sort keys as paths are queued; the counters are
  `integrator_state_gpu_.sort_key_counter`.
- **Compaction**, when the active range is sparse (`:692-716`, `:759-810`):
  `TERMINATED_PATHS_ARRAY`, `COMPACT_PATHS_ARRAY`, a **readback**, then `COMPACT_STATES`, which
  moves whole SoA states.

Cycles does not think in "launches per sample": paths of many samples are in flight, and tiles are
regenerated continuously. With the default 1:4 busy:total ratio (§2.4), a scene with roughly
5–7 kernels per bounce makes on the order of 2–4 launches per step and one sync per step.

### 2.4 Queue and state sizing

- **Metal** (`C:device/metal/queue.mm:266-336`):
  - The base is **4 194 304 states** (2^22). On anything except M1 it doubles up to twice
    (so up to 2^24) while the state stays under 50 % of `recommendedMaxWorkingSetSize`
    (`:302-321`). The comment there: "Enlarging the state size allows us to keep dispatch sizes
    high and minimize work submission overheads".
  - Busy states are a quarter of the total: "A 1:4 busy:total ratio gives best rendering
    performance" (`:332-336`).
  - The scheduler caps the main pool at half of all states (`C:integrator/path_trace_work_gpu.cpp:133-136`).
- **CUDA** (`C:device/cuda/queue.cpp:30-62`): `max(SMs × threads/SM, 65536) × 16` states,
  with busy = `4 × SMs × threads/SM`. **HIP** is the same formula (`C:device/hip/queue.cpp:30-59`).
- **Sort partitions:** `max_paths / 65536` partitions when the scene has fewer than 300
  shaders (`C:device/queue.h:104-116`; Metal `C:device/metal/queue.mm:338-351`,
  `C:device/metal/util.mm:68-79`). The comment: partitioning "results in an overall render
  time speedup of up to 15%" on M1/M2.

**Takeaway:** Cycles keeps **2–8× more paths in flight than our 2^21** on Apple GPUs. It refills
when the pool is three-quarters empty, so every dispatch stays large and per-dispatch overhead
is amortized over millions of threads. That is its main answer to launch overhead.

### 2.5 Sorting and compaction kernels

- **Active index** (`C:kernel/device/gpu/parallel_active_index.h:93-148`): each thread tests a
  predicate. A warp ballot plus popcount gives the in-warp offset, one shared array holds the
  per-warp counts, and one global atomic per block reserves the output range. Block size
  512 (1024 on HIP) (`C:kernel/device/gpu/block_sizes.h`). Threadgroup memory is
  `(threads+1)×4` bytes, rounded up to 16 on Metal (`C:device/metal/queue.mm:551-562`).
- **Local atomic sort** (Metal default on, `C:util/debug.h` `use_local_atomic_sort = true`;
  kernels in `C:kernel/device/gpu/parallel_sorted_index.h:18-136`). There is one threadgroup
  of 1024 per 65 536-state partition.
  - The bucket pass counts the sort keys of that partition into `threadgroup` atomics (one int
    per shader).
  - The write pass adds the earlier partitions' totals, then scatters the indices with local
    atomics.
  - Threadgroup memory is `max_shaders × 4` bytes, asserted ≤ 32 KiB (`C:device/metal/queue.mm:564-578`).
  - The commit message reports a "2-3%" overall gain over the global prefix-sum sort
    (rB654e1e9). The earlier partitioning step reports "up to 15%".

---

## 3. The Metal backend in depth

### 3.1 Compilation: MSL source at run time, three pipeline tiers

- At run time the kernel source is assembled as one string: `#include "kernel/device/metal/kernel.metal"`
  with includes inlined (`C:device/metal/device_impl.mm:441-452`). Blender ships the kernel
  headers as text (`C:kernel/CMakeLists.txt:392-402` installs them under `source/kernel/...`).
- `preprocess_source` prepends `#define`s for the features in use:
  - MetalRT, motion, extended limits;
  - local atomic sort, NanoVDB, macOS version;
  - and, for specialized tiers, `__KERNEL_USE_DATA_CONSTANTS__` (`:341-439`).
- `newLibraryWithSource` builds the library on a background GCD queue, with
  **`fastMathEnabled = YES`** and MSL 3.2 on macOS 15 (`:568-605`). Front-end compiles "can
  take a few seconds".
- Pipelines (PSOs) come from a `ShaderCache` per `MTLDevice`. It runs
  `maximumConcurrentCompilationTaskCount − 1` compiler threads (`C:device/metal/kernel.mm:301-325`),
  and the device asks for `setShouldMaximizeConcurrentCompilation:YES` (`device_impl.mm:93-97`).
- There are **three tiers** (`C:device/metal/kernel.mm:28-41`): `PSO_GENERIC`,
  `PSO_SPECIALIZED_INTERSECT` and `PSO_SPECIALIZED_SHADE`. The user's "Kernel Optimization Level"
  picks the ceiling (`device_impl.mm:145-157`). The interactive viewport never goes beyond
  intersect (`device_impl.mm:965-972`).
- **Hot swap:** `get_best_pipeline` returns the most specialized loaded PSO whose MD5 matches the
  current scene. Rendering starts on generic PSOs and swaps when the specialized ones finish
  compiling in the background (`kernel.mm:367-400`). Up to three variants per tier stay cached
  (`kernel.mm:222-236`).
- **Binary archives:** generic PSOs and all shade PSOs (slow to compile) are serialized as
  `MTLBinaryArchive` files keyed by kernel MD5, OS version and threadgroup size, under
  `kernels/<gpu>/<kernel>/<tier>/<md5>.bin`. They are loaded with `FailOnBinaryArchiveMiss`
  (`kernel.mm:402-436`, `:668-721`). They are skipped for MetalRT intersection kernels, because
  "Binary linked functions aren't supported in binary archives".

### 3.2 One `MTLComputePipelineState` per kernel, and its descriptor

`MetalKernelPipeline::compile` (`C:device/metal/kernel.mm:570-860`) does the following:
- Gets the function `cycles_metal_<kernel>` with **`MTLFunctionConstantValues`** set from the
  scene's `KernelData` (specialized tiers) or zeros (generic) (`:572-587`, `:438-470`).
- Marks buffers 0, 1 and 2 as **`MTLMutabilityImmutable`** (`:647-649`). That is a driver hint
  that these argument buffers are not written by the kernel.
- Sets `maxTotalThreadsPerThreadgroup` from a tuning table and
  **`threadGroupSizeIsMultipleOfThreadExecutionWidth = true`** (`:651-652`).
- Sets `maxCallStackDepth = 1`, or 2 for MetalRT kernels with intersection functions (`:661-664`).
- For MetalRT intersection kernels, adds **linked functions**: the intersection functions for
  triangles, curves and points in eight tables (default, shadow, shadow-all, volume, local ×4)
  (`:600-642`, `:656-660`). Each `MTLIntersectionFunctionTable` is built once per pipeline swap,
  with `launch_params_buffer` bound at index 1 one time (`:506-536`).
- If the table gave no threadgroup size: `num_threads_per_block = round_down(maxTotalThreadsPerThreadgroup,
  threadExecutionWidth)` (`:820-824`).

### 3.3 Resource binding model

Each dispatch binds **three buffers** (`C:device/metal/queue.mm:471-513`):

| Index | What | How |
|---|---|---|
| 0 | the kernel's own arguments, packed C-style; device pointers are patched to `gpuAddress` (`patch_resource`, `:382-390`) | `setBytes` (≤ 512 bytes, `:471-491`) |
| 1 | `KernelParamsMetal`: every data-array GPU address, `IntegratorStateGPU` SoA pointers, `KernelData` | `setBuffer(launch_params_buffer)`, one persistent shared buffer |
| 2 | `MetalAncillaries`: texture table address, TLAS, BLAS array, 8 intersection-function tables | `setBytes` of resource IDs and addresses (`:493-509`) |

There are no per-resource `setBuffer`/`setTexture` calls. Textures are a bindless table of
`texture2d` resource IDs in `image_bindings` (`:407-433`). Samplers are constexpr in the
shader (`C:kernel/device/metal/compat.h:373-396`). This is effectively **buffer device
addresses everywhere**, which our luce-gpu Metal path only half has (G:metal/compute.lucb:118-130
makes address-reached buffers resident).

**Residency:**
- On macOS 15+ one **`MTLResidencySet`** holds every allocation. It is attached to the command
  queue, and only re-committed when allocations change (`device_impl.mm:175-200`, `:253-317`,
  called each enqueue at `queue.mm:585`).
- Without it, every new encoder calls `useResource` for every allocation (`queue.mm:786-816`).
- MetalRT dispatches also `useResource` the TLAS, the BLASes and the function tables
  (`:515-541`).
- The comment says the set exists "avoiding the overhead of per-encoder useResource calls on
  every dispatch".

### 3.4 Hazard tracking, encoders, barriers

- **Hazard tracking:** Cycles never sets a hazard mode. There is no `HazardTrackingModeUntracked`
  anywhere in `C:device/metal/`. Buffers are created with `MTLResourceStorageModeShared`, or
  `Private` for device-only memory (`device_impl.mm:684-700`). **All resources are tracked.**
- **Barriers:** there is **no `memoryBarrierWithScope`** anywhere in the Metal backend (grep).
- **Encoders** (`C:device/metal/queue.mm:818-872`):
  - The queue keeps one open compute encoder. Path-iteration kernels (enum values below
    `DEVICE_GPU_KERNEL_INTEGRATOR_NUM`, i.e. init/intersect/shade, `C:kernel/types.h:1825`) get a
    **`MTLDispatchTypeConcurrent`** encoder. Helper kernels (index arrays, sort, compaction,
    prefix sum, film) get a **`MTLDispatchTypeSerial`** encoder.
  - Changing type closes the encoder and opens a new one. So does any blit (`zero_to_device`
    uses a blit encoder, `:712-743`).
  - In profiling mode every dispatch gets its own encoder with counter-sample attachments at
    stage boundaries (`:822-858`).
- **What ordering actually holds:** Apple says that within a concurrent encoder, "If you encode
  multiple commands that access a single resource, you're responsible for synchronizing"
  (Apple docs, MTLDispatchType.concurrent). Cycles gets away without barriers because of how the
  scheduler shapes the work:
  - each step ends in `synchronize()` (a commit and wait);
  - the helper kernels that feed a path kernel run in a *serial* encoder or a blit, which ends
    the concurrent encoder;
  - so **a concurrent encoder in practice holds exactly one path kernel**;
  - ordering between encoders comes from Metal's hazard tracking of tracked resources.

  The concurrent type is therefore near moot. It mainly says "no implicit serialization inside
  this encoder". An Apple engineer: "on Apple GPUs, the case where concurrent dispatches improve
  overall compute performance is rare" (Apple forum thread 721349).
- **Command buffers:** one open command buffer accumulates until `synchronize()`
  (`queue.mm:841-844`, `:672-710`). `copy_to_device` and `copy_from_device` are no-ops because of
  unified memory (`:745-770`).

**How they avoid per-dispatch overhead:** mostly by **not having many dispatches**. Pools are
very large (§2.4), so each dispatch runs millions of threads. Each dispatch has three cheap
`setBytes`/`setBuffer` calls. There is no per-dispatch `useResource` (residency set). One
persistent pointer-table buffer is written in place on the host (UMA), which avoids any
upload. Cycles accepts **a CPU↔GPU round trip per step** and amortizes it with size.

### 3.5 Kernel specialization (function constants)

This is the biggest Metal-specific optimization, and the one with published numbers.

- **Mechanism:**
  - `C:kernel/device/metal/function_constants.h:5-20` declares one
    `constant T kernel_data_<parent>_<name> [[function_constant(N)]]` for every `KernelData` member
    (from the X-macro `data_template.h`, 171 members), plus `kernel_features`.
  - For specialized tiers, the host rewrites the source text. `kernel_data.parent.` becomes
    `kernel_data_parent_` by a **same-length string replacement**, so source positions survive
    (`C:device/metal/device_impl.mm:389-423`).
  - Members marked `KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE` stay memory loads. These are the seed
    and Sobol/blue-noise table sizes (`C:kernel/data_template.h:199-214`); they change per frame
    or don't help.
  - With `__KERNEL_USE_DATA_CONSTANTS__` the struct fields become `__unused_*` so nothing can read
    them by mistake (`C:kernel/types.h:1324-1328`).
  - `GetConstantValues` fills `MTLFunctionConstantValues` from the live `KernelData` (`C:device/metal/kernel.mm:438-470`).
  - The PSO identity is the MD5 of the constant values plus the source (`device_impl.mm:487-530`).
    Changing a scene setting such as max bounces therefore picks a new PSO.
- **What specialization kills:**
  - Integrator settings become compile-time constants: bounce limits, caustics, clamping,
    light-sampling mode, guiding flags, volume flags, film passes.
  - **Unused SVM nodes:** `SVM_CASE(node)` turns into `case node: if (!kernel_data_svm_usage_node) break;`,
    so node implementations the scene never uses are dead-stripped from `svm_eval_nodes`
    (`C:kernel/svm/svm.h:91-99`; usage collected in `C:scene/svm.cpp:169`).
  - Feature early-outs, e.g. `volume_stack_enter_exit` returns at once when the scene has no
    volumes: "scenes without volumetric features can render 1 or 2% faster by dead-stripping this
    function" (`C:kernel/integrator/volume_stack.h:73-81`).
  - MetalRT geometry types: `assume_geometry_type(triangle | (have_curves ? curve : none) | ...)`
    becomes constant (`C:kernel/device/metal/bvh.h:176-181`).
- **Scope:** only integrator kernels, with intersect kernels in the INTERSECT tier and shade
  kernels in the SHADE tier (`kernel.mm:272-286`).
- **Measured** (commit da4ef05, M1 Max, macOS 13, samples/min):

  | Scene | Generic | + intersect specialized | + shade specialized |
  |---|---:|---:|---:|
  | Sponza | 830.6 | 929.6 (+12 %) | 1142.4 (+38 %) |
  | BMW27 | 1486.1 | 1671.0 (+12 %) | 1825.8 (+23 %) |
  | Junkshop | 205.4 | 212.0 (+3 %) | 257.7 (+25 %) |

  The later SVM-node stripping commit a7cc6e0 reports "≈13 % uplift in isolation" and lower
  register use. The cost is compile time: shade PSOs take tens of seconds, hence the background
  compile, the generic fallback and the binary archives.

### 3.6 Inlining and register pressure

Commit b82de02 made `ccl_device_noinline` *inline* on Apple GPUs (`C:kernel/device/metal/compat.h:38-42`):
"~1.1x speedup and 10% spill reduction for integrator_shade_surface", at the cost of
"~4.5 minutes" compile on M1 Max. That is why archives and background compiles matter.
`ShaderCache` also revived binary archives in the same change.

### 3.7 Threadgroup sizes per kernel and GPU family

`ShaderCache` has an occupancy LUT of `{threads_per_threadgroup, num_threads_per_block}` per
kernel (`C:device/metal/kernel.mm:46-99`):

| Kernel | M1 | M2 | M2 Max/Ultra | M3 **and unknown (M4+)** |
|---|---|---|---|---|
| intersect_closest | 512 / 128 | 64 / 64 | 1024 / 64 | 64 / 64 |
| intersect_shadow | 384 / 128 | 64 / 64 | 704 / 704 | 64 / 64 |
| shade_surface | 576 / 384 | 448 / 384 | 768 / 576 | 64 / 64 |
| shade_shadow | 384 / 32 | 256 / 256 | 32 / 32 | 64 / 64 |
| queued/sorted arrays | 512–1024 | 1024 | 896 / 768 | 64 / 64 |
| sort bucket / write | 1024 | 1024 | 1024 | 1024 |

The M3 note reads: "Peak occupancy is achieved through Dynamic Caching on M3 GPUs." The family
check is a substring match on the device name (`C:device/metal/util.mm:53-66`). An **M4 Max
reports `APPLE_UNKNOWN`**, which is ordered last in the enum *so that future GPUs get M3
behaviour* (`C:device/metal/util.h:23-29`). So on our M4 Max, **Cycles dispatches every
integrator kernel at 64 threads per threadgroup**, which is what we already do
(`R:shaders/*.comp`, `local_size_x = 64`).

### 3.8 MetalRT vs BVH2

- On macOS 14+ `use_hardware_raytracing = device.supportsRaytracing`. **MetalRT is the default
  on M3 and later** (`C:device/metal/device.mm:88-99`); the Blender 4.0 notes describe
  "significant path-tracing speedups and faster BVH builds". The BVH layout is `BVH_LAYOUT_METAL`
  or Cycles' own BVH2 (`device_impl.mm:50-53`).
- The MetalRT kernels use the **`intersector<triangle_data, curve_data, instancing ...>` object
  with intersection-function tables**, not `intersection_query` (`C:kernel/device/metal/compat.h:313-330`,
  `bvh.h:168-210`). Custom intersection functions do self-intersection rejection,
  shadow-transparency recording, local (SSS) hits, curves and points (`kernel.metal`).
- Every query calls `force_opacity(non_opaque)` so the functions run, plus `assume_geometry_type`.
  The shadow query uses `accept_any_intersection(true)` (`bvh.h:290-301`).
- Motion blur uses `instance_motion, primitive_motion` tags only when the scene has motion
  (`compat.h:299-305`). On Apple9 with macOS ≥ 15.6 it uses per-component motion interpolation
  (`device_impl.mm:110-119`).

### 3.9 Other Apple-specific choices

- `fastMathEnabled = YES` for the whole library (§3.1) and `fast::` trig (§1.2).
- Unified memory: no staging copies, and the host writes the pointer table in place.
- The command-queue error option `EncoderExecutionStatus` gives per-kernel fault messages
  (`queue.mm:28-29`).
- `CYCLES_METAL_PROFILING` prints a per-kernel table of dispatch count, threads and time using
  stage-boundary counters (`queue.mm:217-263`, `:650-670`). Note that it **closes the encoder
  per kernel** in that mode, which changes the timing it measures (`:822-825`).

---

## 4. CUDA, OptiX and HIP, briefly

- **Compile:**
  - Kernels are precompiled offline to `cubin` for each `sm_XY`, with a PTX fallback for
    newer arches (`C:device/cuda/device_impl.cpp:265-290`).
  - Runtime compile (adaptive compilation, `-D__KERNEL_FEATURES__=…`) is possible but off by
    default (`:208-231`).
  - Flags include `--use_fast_math` (`:225`).
  - HIP uses precompiled `fatbin` per `gfx` arch with `-ffast-math` (`C:device/hip/device_impl.cpp:247-275`).
    oneAPI also uses `-ffast-math` (`C:kernel/device/oneapi/CMakeLists.txt:71`).
- **OptiX:**
  - PTX modules are built with `optixModuleCreateWithTasks` at `OPTIX_COMPILE_OPTIMIZATION_LEVEL_3`
    (`C:device/optix/device_impl.cpp:205-215`, `:373-383`).
  - It uses **no bound values** (`boundValues = nullptr`, `:385`), so there is no OptiX-side
    specialization.
  - The pipeline uses 8 payload registers and 2 attributes, with single-level instancing as the
    default (`:388-397`).
  - Separate pipelines exist for shading and intersection.
  - Each `enqueue` writes the per-launch parameters with `cuMemcpyHtoDAsync` into the
    launch-params block (`C:device/optix/queue.cpp:70-97`), then calls `optixLaunch` with the
    raygen record for that kernel (`:100-225`).
- **Launch overhead:** CUDA uses plain `cuLaunchKernel` on a non-blocking stream
  (`C:device/cuda/queue.cpp:21`, `:128`). There are **no CUDA graphs** and no indirect launches.
  The strategy is the same as Metal: huge state pools (`SMs × threads × 16`) and one
  `cuStreamSynchronize` per step.
- **Shared memory:**
  - Only the active-index, compaction and sort kernels use it: `(block+1)×4` bytes for per-warp
    offsets (`C:device/cuda/queue.cpp:108-122`; algorithm in `gpu/parallel_active_index.h`).
  - The local-atomic sort is **only enabled on Metal** (`supports_local_atomic_sort` defaults to
    false, `C:device/queue.h:120-123`). CUDA uses the global prefix-sum path.
  - Integrator kernels use no shared memory. They are register-bound, and capped via
    `__launch_bounds__` (§1.4).

## 5. Vulkan

**Cycles has no Vulkan compute backend.** The device directories are cpu, cuda, optix, hip,
hiprt, metal, oneapi, multi and dummy (`C:device/`). Vulkan shows up only as a *display
interop* target: CUDA, HIP and oneAPI can write the render result into a Vulkan buffer that
Blender's UI draws (`C:device/cuda/graphics_interop.cpp:62-69`, `C:device/CMakeLists.txt:227`).
Blender's own Vulkan backend is the GPU module for drawing (UI, EEVEE, viewport), not Cycles.

Why not, in their words and in ours:
- Cycles X dropped OpenCL because "the combination of the limited Cycles split kernel
  implementation, driver bugs, and stalled OpenCL standard has made maintenance too difficult".
  They said they would target vendor APIs instead: HIP, Metal, SYCL (code.blender.org Cycles X post).
- *Our inference, not stated by Blender:* the kernel is ≈55 k lines of C++ (templates, lambdas
  via functor structs, pointers into SoA arrays, recursion-free but deep call graphs). There is
  no production C++→SPIR-V-for-Vulkan compute compiler. Vulkan compute also lacks things Cycles
  leans on: OptiX/MetalRT-style intersection callbacks; until recently, raw device pointers
  everywhere (BDA is now core in 1.2); and vendor compilers of equal quality to nvcc or
  Apple's MSL compiler. Every vendor already offers a better native path, so a Vulkan backend
  would only add a fifth target for the same GPUs.

---

## 6. Where we stand against Cycles, item by item

| Concern | Cycles | luce-render / luce-gpu today |
|---|---|---|
| Kernel language | one C++ source, macros, native compiler per backend | GLSL 4.5 → SPIR-V (Vulkan); SPIR-V → spirv-cross → MSL → Apple compiler (Metal) |
| Submission | host picks next kernel, **commit + wait per step** | GPU `schedule` writes indirect args; whole batch in one command buffer; readback one batch behind |
| Paths in flight (Apple) | 2^22–2^24, refilled at 1:4 busy | 2^20–2^21, refilled at the top of each batch |
| Binding | 3 bindings; one pointer table holding everything | up to 16 set-0 bindings + 128 B uniforms per dispatch; BDA for some buffers |
| Residency | `MTLResidencySet` once (macOS 15) | `useResources` for addressed buffers on each new compute encoder (G:metal/compute.lucb:118-130) |
| Hazards | tracked; serial and concurrent encoders; no explicit barriers | tracked (G:metal/compute.lucb:9, buffer.lucb:9); default (serial) `computeCommandEncoder` (compute.lucb:119) |
| Specialization | function constants from `KernelData`, background compile, hot swap | none (one pipeline per kernel; feature variants only by separate kernels, e.g. `*_rgb`) |
| Fast math | on everywhere (Metal, CUDA, HIP, oneAPI) | **off** for every luce-render kernel (`R:src/render/kernels.lucb`, `*_fast_math = false`; luce-gpu defaults to IEEE, G:metal/compute.lucb:52-58) |
| Threadgroup | 64 on M3/M4 for integrator kernels | 64 |
| Sorting | shade_surface sorted by shader, 65 536-state partitions, local atomics | not yet ("Speed, later" in `R:docs/DESIGN.md`) |
| RT | MetalRT `intersector` + function tables on M3+; BVH2 elsewhere | `GL_EXT_ray_query` → spirv-cross → `intersection_query`; BVH2 fallback |
| Profiling | stage-boundary counters, one encoder per kernel in profile mode | one **command buffer** per timestamp (G:metal/compute.lucb:154-181), and a timestamp after **every** dispatch in profile mode (`R:src/render/integrator.lucb:470-478`) |

Two things in the table matter for the overhead number:

1. **Our "≈30 µs per dispatch" may be partly a measuring artifact.** In profile mode, luce-gpu
   ends the command buffer at every timestamp and luce-render stamps every dispatch. So a
   profiled pass is ~90 committed command buffers, not one. Cycles' profile mode has the same
   bias at a smaller grain (per encoder). The real per-dispatch cost must be measured with
   profiling off, by comparing:
   - pass GPU time (`GPUEndTime − GPUStartTime` of the one command buffer), against
   - the sum of the kernel times.
2. **Half our dispatches are 1-thread `schedule` kernels.** Each one forces a full drain-and-fill
   of the GPU (serial encoder plus tracked hazards), with nothing to overlap. Cycles pays a
   *host* round trip at the same point instead. Neither is free, but ours can be removed (§7.3).

---

## 7. Lessons for luce-render, ranked by expected payoff ÷ effort

Payoff estimates are for the 9 ms/sample M4 Max baseline. "Evidence" says what Cycles does and
what numbers exist; where there is no number, the estimate is ours and is labelled.

### 7.1 Measure the dispatch overhead without profiling distortion: trivial effort, decides 7.3/7.4

**Do:**
- Add an unprofiled timing mode: one command buffer per pass, read only its GPU start and end
  time.
- Then time a pass with the bounces cut to 1, 2, 4 and 12, and a pass where `schedule` is
  replaced by fixed-size direct dispatches. That separates the drain cost from the kernel cost.

**Cycles:** stage-boundary counters, per encoder (`C:device/metal/queue.mm:822-858`). It is still
biased, and its own comment admits closing encoders only to time them.

**Why first:** if the true overhead is, say, 5–10 µs, then 90 dispatches cost ≤1 ms and the
work lies in `shade_surface`, not in the abstraction.

### 7.2 Fast math for the hot kernels: one flag, likely 5–15 % on shade_surface (estimate)

**Cycles:** every backend compiles with fast math: Metal `fastMathEnabled = YES`
(`C:device/metal/device_impl.mm:570`), CUDA `--use_fast_math` (`C:device/cuda/device_impl.cpp:225`),
HIP and oneAPI `-ffast-math`. On top of that, Metal forces the `fast::` variants of
sin/cos/tan/exp/sqrt/log (`C:kernel/device/metal/compat.h:285-293`). Their comment: "no issues
found so far".

**Us:** every luce-render kernel is built with `fast_math = false`.

**Evidence:** Cycles publishes no isolated number. Our 5–15 % is an estimate for ALU-heavy
shading on Apple GPUs. Fast math enables FMA contraction, reciprocal division and fast
transcendentals, and the spectral code (CIE fits, Fresnel, GGX, EON) is transcendental-heavy.

**Risk:**
- NaN and Inf handling: `isnan`/`isinf` guards may be optimized out under fast math. Keep the
  guards as bit tests (`floatBitsToUint`), which spirv-cross emits as `as_type`.
- Light-tree and MIS pdf ratios: validate against the reference images.

**Do:** turn it on for `shade_surface`, `shade_miss`, `intersect_*` and `camera`. Run the image
tests and compare the timing.

### 7.3 Remove the `schedule` dispatches: small to medium effort, up to ~45 of 90 dispatches

**Cycles:** nothing comparable. It reads the counters on the host and picks the most-queued
kernel (`C:integrator/path_trace_work_gpu.cpp:414-435`), paying a round trip instead.

**Do:** let the producing kernel write the next indirect arguments itself. This is the
"last threadgroup done" pattern:
- each threadgroup increments a done-counter after its queue appends (device-scope atomic plus
  a memory fence);
- the group that sees `count == groups−1` converts the queue counters into `(groups,1,1)` and
  clears the next counters, i.e. the current `schedule` body;
- `intersect_closest` writes the arguments for `shade_surface` and `shade_miss`, and so on.

That turns `schedule → K → schedule → K` into `K → K`, which halves the number of
drains. On Vulkan this needs the indirect buffer written with a storage-buffer write and a
`DRAW_INDIRECT` barrier, which luce-gpu already inserts between dispatches.

**Payoff:** (number of schedules removed) × (true per-drain cost from 7.1). At 30 µs that is
≈1.3 ms of 9 ms. At 5 µs it is ≈0.2 ms.

### 7.4 Keep dispatches large: path regeneration within the pass and a bigger pool, medium effort

**Cycles:**
- Pools of 2^22–2^24 states on Apple GPUs, "to keep dispatch sizes high and minimize work
  submission overheads" (`C:device/metal/queue.mm:311-321`).
- New camera paths are injected whenever active paths drop below a quarter, aligned to the
  `intersect_closest` stage (`C:integrator/path_trace_work_gpu.cpp:812-905`).
- "Align kernels of existing and new paths" was measured as a gain in the Cycles X branch
  (commit e0716af, May 2021).

**Us:** one sample per pixel per pass and a fixed bounce loop. By bounce 6–12 the surviving paths
are a few percent of 2^21, yet each bounce still pays ≈9 drains for tiny grids that cannot
fill an M4 Max.

**Do:** let `camera` refill free slots at each `intersect_closest` boundary while samples remain
(a sample-index counter, not "one sample per pass"). Size the pool from the working set, as
`num_concurrent_states` does. Then a pass ends after N *path starts*, not N bounces.

**Payoff (estimate):** removes most late-bounce tail dispatches, and raises occupancy for
everything after bounce 3. Together with 7.3 this is how the "~90 dispatches" stops mattering.
It needs per-slot sample bookkeeping, because film accumulation already uses atomics or
per-pixel ownership.

### 7.5 Specialization constants: medium effort, +12–38 % in Cycles

**Cycles:** function constants for all 171 `KernelData` members except seed and table sizes.
They dead-strip SVM nodes, volume code and curve geometry types, and compile in the background
with a generic fallback (§3.5). Measured +12 % (intersect) to +23–38 % (shade) on M1 Max, and
≈13 % from SVM node stripping alone.

**Us:**
- GLSL `layout(constant_id = N) const …` becomes SPIR-V `OpSpecConstant`. **spirv-cross already
  maps specialization constants to MSL `[[function_constant(N)]]`**, so the path exists end to
  end.
- What is missing is a luce-gpu API: `Kernel.create(..., constants: [(id, value)])` →
  Vulkan `VkSpecializationInfo` and Metal `MTLFunctionConstantValues` with
  `newFunctionWithName:constantValues:error:`.

**Candidates:**
- max bounces;
- lobe enables (coat, transmission, anisotropy, emission present);
- light-tree vs uniform light picking;
- guiding on/off;
- the spectral vs RGB build (today two separate kernels);
- adaptive sampling on/off;
- film passes (albedo/normal AOVs).

We have one material model (OpenPBR), not an SVM interpreter, so expect the lower end
(≈10 %). The gains come from dropping unused lobes and from register pressure.

**Compile cost:** keep generic pipelines, and build specialized ones on a background thread
with hot swap (`C:device/metal/kernel.mm:367-400`). Use `MTLBinaryArchive` as the cache.

### 7.6 Shader sort with partitions: medium effort, +2–15 % in Cycles (scene-dependent)

**Cycles:**
- It sorts the `shade_surface` queue by shader every step. Partitions of 65 536 states give
  "up to 15%" on M1/M2 (`C:device/metal/util.mm:74-77`).
- Local-atomic bucketing adds "2-3%" (rB654e1e9).
- It is turned off when the scene has more than 300 shaders.

**Us:** `shade_surface` is 48 % of the frame. With one OpenPBR kernel the divergence comes from
lobe mix and texture paths, not from different shader programs. So sort by a *material key*
(material id, or a lobe-set bitmask), not by kernel. The partitioned bucket sort maps onto GLSL
`shared` atomics with no backend help.

**Evidence for us:** none until tried. Coherence gains need many materials. Partitioning also
improves state-fetch locality, which helps even in single-material scenes.

### 7.7 Inspect the spirv-cross MSL of `shade_surface` for spills: small effort, diagnostic

**Cycles:** forced inlining gave "~1.1x" and "10% spill reduction" on `shade_surface` (b82de02).

**Us:** Xcode's Metal debugger and GPU counters (`compute.lucb` already supports timestamps)
show register spills and occupancy per pipeline. Check them on the spirv-cross output:
- Our big `vec4[]`/`uint[]` double-binding of the state buffer could hinder alias analysis.
- So could function-local arrays that spirv-cross turns into `thread` arrays with dynamic
  indexing.

**Do:** dump `shade_surface.msl` and look for:
- `spvUnsafeArray` used with dynamic indices;
- `thread` copies of large structs;
- missing `restrict`.

Fix these at the GLSL level. That is cheaper than any new toolchain, and it tells us whether the
cross-compile *output* is the problem (see 7.10).

### 7.8 Argument tables and residency sets: small effort, CPU-side only

**Cycles:** three bindings per dispatch; one pointer-table buffer with all addresses; a residency
set on macOS 15 (§3.3), plus `MTLMutabilityImmutable` on the argument buffers.

**Us:** we encode once per batch with indirect dispatch, so the CPU encode cost is already off the
critical path. Benefits:
- the 16-binding/128 B limits go away (DESIGN.md already plans BDA);
- one `MTLResidencySet` replaces `useResources` on every encoder;
- `immutable` buffer hints on uniform/table buffers.

Expect no measurable GPU time win, but simpler kernels. This is a luce-gpu change, useful to
every client.

### 7.9 Hazard tracking, encoders, barriers: low payoff on Apple, but a cheap experiment

**Cycles:** tracked resources, no explicit barriers, a concurrent encoder per path kernel.
Apple: concurrent dispatch rarely helps on Apple GPUs (forum 721349).

**Us:** one serial encoder, tracked. Within a *serial* compute encoder Metal already orders every
dispatch after the previous one. Tracked hazards add CPU-side bookkeeping, but no extra GPU
stall beyond that ordering.

**Experiment:** allocate the integrator buffers `MTLResourceHazardTrackingModeUntracked`, use one
`MTLDispatchTypeConcurrent` encoder, and insert `memoryBarrierWithScope:MTLBarrierScopeBuffers`
only where a true dependency exists. Most of our dispatches *are* dependent, so the expected gain
is small. The one real win would be letting independent kernels overlap: `shade_miss` alongside
`shade_surface`, `intersect_shadow` alongside the next bounce's camera refill.

**Rank:** below 7.1–7.6. luce-gpu's automatic barriers are not the bottleneck Cycles avoids.
Cycles keeps the same model.

### 7.10 Threadgroup size: nothing to do

Cycles uses 64 on M3 and M4-class GPUs for every integrator kernel (`C:device/metal/kernel.mm:51-58`),
and so do we. A 32/64/128 sweep of `shade_surface` costs minutes and may show ±3 %. Set
`threadGroupSizeIsMultipleOfThreadExecutionWidth` (`C:device/metal/kernel.mm:652`) in luce-gpu
when the group is a multiple of 32. It is a free compiler hint.

### 7.11 MetalRT path: check, don't rewrite

**Cycles:** an `intersector` with function tables, because it needs custom any-hit behaviour
(transparency, self-hit rejection, curves). It forces `non_opaque`.

**Us:** v1 shadows are opaque-only triangles, so `intersection_query` with **opaque triangles and
`assume_geometry_type(triangle)`** is the lighter call. Confirm that spirv-cross emits the
opaque/`geometry_type::triangle` hints (`gl_RayFlagsOpaqueEXT`, cull flags), and that we use
`gl_RayFlagsTerminateOnFirstHitEXT` for shadows. That is Cycles' `accept_any_intersection(true)`
(`C:kernel/device/metal/bvh.h:301`). Our `intersect_closest` is 1.3 ms, so only a small win is
left there.

### 7.12 Kernel-language strategy: don't rewrite the kernels to chase speed

What the evidence says:
- Every Cycles Metal gain with a number comes from **what** the compiler is told:
  - specialization: +12–38 %;
  - inlining: 1.1×;
  - sort partitioning: up to 15 %;
  - local sort: 2–3 %;
  - fast math: on everywhere.
- None comes from MSL being "native" rather than generated. Apple's compiler sees MSL either
  way: ours comes from spirv-cross, theirs from a preprocessor.
- Cycles' single source works because it is **C++ with a macro shim per backend, compiled by each
  vendor's own compiler**. Metal compiles from source at run time, and that is what makes the
  scene specialization cheap to express.

Options for us:

| Option | Speed impact | Cost | Verdict |
|---|---|---|---|
| (a) Keep GLSL → SPIR-V (Vulkan) → spirv-cross → MSL | same compiler backend on Apple; risk only in spirv-cross codegen idioms (7.7) | none | **keep for now**; add spec constants (7.5) and fast math (7.2) |
| (b) Hand-write MSL for hot kernels | no evidence of a gain over well-shaped spirv-cross output | two sources of `shade_surface` to keep in step | **no**: Cycles itself refuses per-backend kernel forks |
| (c) A C-like kernel subset compiled per backend (the Cycles model) | equal to (a) for codegen; easier `restrict`, address spaces, templates | new front-end or C++ toolchain dependency | only if GLSL's expressiveness blocks us (templates for lobe variants, pointers into SoA) |
| (d) Luce Base emits MSL + SPIR-V (+PTX later) | same as (c); one language from host to kernel; specialization natural as compile-time constants | large: a GPU backend in the compiler (address spaces, no recursion, subgroup ops, ray-query intrinsics) | strategic, not an optimization; revisit after 7.1–7.6 show what is left |

**Recommendation:** treat the kernel language as a *productivity* decision, not a performance
one. For performance, take the Cycles levers in order: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6. Re-measure
after each. If, after that, the spirv-cross MSL of `shade_surface` still spills where an
equivalent hand-shaped MSL does not (a one-kernel A/B experiment), then (c) or (d) has evidence
behind it.

---

## 8. Summary table of recommendations

| # | Change | Effort | Expected payoff | Evidence |
|---|---|---|---|---|
| 7.1 | Unprofiled pass timing; isolate drain cost | trivial | decides 7.3/7.4 | our profile mode splits a command buffer per dispatch |
| 7.2 | Fast math on hot kernels | trivial | 5–15 % shade (est.) | Cycles: on in all backends + `fast::` trig |
| 7.3 | Fold `schedule` into producers (last-group-done) | small–medium | −~45 drains/pass; ≤1.3 ms if 30 µs/drain is real | none in Cycles (host round trip there) |
| 7.4 | Regenerate paths within the pass; bigger pool | medium | removes late-bounce tiny dispatches; higher occupancy | Cycles: 2^22–2^24 states, 1:4 refill, wavefront alignment |
| 7.5 | Specialization constants via luce-gpu | medium | ≈10 % (est., one material model) | Cycles: +12–38 % on M1 Max |
| 7.6 | Partitioned material sort for shade_surface | medium | 2–15 %, scene-dependent | Cycles: up to 15 % partitioning, 2–3 % local atomics |
| 7.7 | Spill audit of spirv-cross MSL | small | diagnostic; may unlock 5–10 % | Cycles: inlining 1.1×, 10 % fewer spills |
| 7.8 | Pointer table + residency set + immutable hints | small | CPU-side, simpler bindings | Cycles: 3 bindings, residency set |
| 7.9 | Untracked + concurrent + explicit barriers | small | small on Apple | Apple: concurrent rarely helps; Cycles keeps tracked |
| 7.10 | Threadgroup sweep / multiple-of-width hint | trivial | ±3 % | Cycles: 64 on M3/M4 |
| 7.11 | Opaque/triangle ray-query hints | trivial | small | Cycles: `assume_geometry_type`, any-hit for shadows |
| 7.12 | Kernel language: keep GLSL now, decide on evidence | — | none directly | Cycles' gains all come from compiler inputs, not source language |
