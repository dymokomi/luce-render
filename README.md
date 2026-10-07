# luce-render

A spectral path tracer that runs on the GPU, for luced-3d and any Luce program
that holds a luce-geocore `GeometrySet`. The host side, in Luce Base, compiles
the scene into GPU buffers and records compute passes on luce-gpu. All tracing
happens in GLSL compute kernels: Metal on macOS, Vulkan on Windows and Linux.
There is no CPU renderer.

It is early. The scene compiler, the BVH and the primary-ray probe work today.
The integrator, OpenPBR materials and lights come next; docs/DESIGN.md has the
plan.

```luce
from luce_render import render
from luce_gpu import gpu

var scene = render.compile_scene(set)              # triangles, BVH, cameras, lights
defer scene.close()
var on_gpu = render.GpuScene.upload(device, &scene)
defer on_gpu.destroy()
let frame = render.ray_frame(scene.objects.object(0), 1920, 1080)
render.probe(device, &on_gpu, &frame, probes)      # each pixel's polygon and distance
```

## What it reads

- **Geometry:** every polygon mesh in the set and in its visible instances. Analytic
  CAD needs a Tessellate node first; `CompiledScene.skipped` counts what was left out.
- **Cameras and lights:** luce-geocore's scene component, shaped like USD's
  UsdGeomCamera and UsdLux (rect, disk, sphere and distant lights). A camera looks
  down its local -Z, and its film fits the image horizontally.

## Kernels

GLSL sources are in `shaders/`. `python3 tools/build_kernels.py` embeds them into
`src/render/kernels.lucb` through luce-gpu's `embed_shaders.py`. Tracing kernels
will be built twice: spectral (`SPECTRAL=1`) and RGB (`SPECTRAL=0`).

## Tests

`luc test` runs the host tests and `tests/gpu`. That program compares the GPU's
BVH traversal with brute-force intersection, and prints `skip: no GPU` on machines
without one.
