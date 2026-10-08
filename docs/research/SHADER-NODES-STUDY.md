# luce-render: shader nodes, after Cycles' SVM

Status: research note, 2026-10-07. Artists will build materials from shader nodes inside a Material node in luced-3d,
and luce-render will compile the graph for the GPU. This note reads how Blender Cycles does it with SVM (its Shader
Virtual Machine) and proposes an SVM-like design for our OpenPBR-only kernels.

Source read: the local checkout at `/Users/sedov/Dev/blender-main/intern/cycles` (`BLENDER_VERSION 503`). Paths below
are relative to `intern/cycles`. The checkout is for reference only. Nothing here is code to copy.

One thing differs from older descriptions of SVM. This checkout no longer packs nodes into `uint4`s. The program is a
flat `uint` array (`kernel/data_arrays.h:71`, `KERNEL_DATA_ARRAY(uint, svm_nodes)`) of typed node structs, and inputs
that are not linked carry their constant inline. §1.4 describes the current encoding.

---

## 1. The pipeline

### 1.1 The graph: `scene/shader_graph.{h,cpp}`

- **`ShaderNode`** (`shader_graph.h:145`) holds `inputs` and `outputs`, an `id`, a `bump` tag
  (`SHADER_BUMP_NONE/CENTER/DX/DY`), and a `special_type` (proxy, autoconvert, closure, combine-closure, bump, output,
  ...). Each node implements:
  - `compile(SVMCompiler&)`, which emits the node's code;
  - `constant_fold(const ConstantFolder&)`;
  - `simplify_settings`;
  - `expand`, which can replace the node with several nodes;
  - `equals`, used for deduplication;
  - `get_feature()`, which returns its `KERNEL_FEATURE_NODE_*` bits;
  - flags such as `has_surface_emission` and `has_spatial_varying`.
- **`ShaderInput` / `ShaderOutput`** (`shader_graph.h:97,131`) are `ShaderIO`s. Each has a `SocketType` and a
  `stack_offset` that the compiler fills in. An input has at most one `link`. An output has a list of `links`. An input
  with no link uses its node's stored value.
- **`SocketType::Type`** (`graph/node_type.h:29`): the shading types are `FLOAT`, `INT`, `COLOR`, `VECTOR`, `POINT`,
  `NORMAL` and `CLOSURE`. Socket flags say what an unlinked input defaults to: `LINK_TEXTURE_GENERATED`,
  `LINK_TEXTURE_UV`, `LINK_NORMAL`, `LINK_POSITION`, `LINK_INCOMING`, `LINK_TANGENT` and so on (`DEFAULT_LINK_MASK`).

### 1.2 Preparing the graph

`ShaderGraph::finalize` (`shader_graph.cpp:374`) runs once per graph, after which the graph is frozen:

1. **`simplify`** (`:362`) runs `expand()`, then `default_inputs()`, then `clean()`, then `refine_bump_nodes()`.
   - **Proxies** come from Blender's node groups and reroutes, which `blender/shader.cpp:1362` turns into
     `ConvertNode`s from a type to the same type (`SHADER_SPECIAL_TYPE_PROXY`). `remove_proxy_nodes` (`:459`) links
     around them. Where a proxy's input was unlinked, it copies the value into each target socket, and it removes any
     autoconvert nodes that only fed default-linked sockets.
   - **`default_inputs`** (`:864`) connects each unlinked input that carries a `LINK_*` flag. One shared
     `TextureCoordinateNode` serves the Generated, UV and Normal flags, and one `GeometryNode` serves position, normal,
     tangent and incoming. The Incoming case for texture coordinates adds a world-to-object `VectorTransformNode`.
     This is how an Image Texture with nothing plugged into Vector ends up reading UVs.
   - **`clean`** (`:801`) runs these in order:
     - `constant_fold` (`:533`) visits nodes in topological order with a queue. Each node's `constant_fold` can call
       `ConstantFolder::make_constant`, `bypass`, `discard` or `fold_math`/`fold_mix`/... (`scene/constant_fold.h`). A
       folded value is written into each downstream socket, and `constant_folded_in` is set.
     - `simplify_settings`, which disconnects inputs that cannot matter. For example
       `PrincipledBsdfNode::simplify_settings` (`shader_nodes.cpp:2695`) drops the coat inputs when the coat weight is
       zero, and the emission inputs when emission is black.
     - `deduplicate_nodes` (`:600`) merges nodes of the same type, settings and input links, bottom-up.
     - `optimize_volume_output`.
     - `break_cycles` from the output and AOV nodes. Nodes it never reaches are disconnected and removed.
2. **Bump from displacement** (`bump_from_displacement`, `:1005`) runs only when displacement is set to bump. §3.5
   describes it.
3. **`transform_multi_closure`** (`:1101`) rewrites the tree of Mix Shader and Add Shader nodes above the output into
   plain float math. For each Mix Shader it adds a `MixClosureWeightNode` with outputs Weight1 = w·(1−fac) and
   Weight2 = w·fac. An Add Shader passes `w` to both sides. Each BSDF leaf gets the product of the weights on its path
   in its hidden `SurfaceMixWeight` input. The kernel never builds a closure tree. Each BSDF node appends its closures
   with its weight already applied.

### 1.3 The compiler: `scene/svm.{h,cpp}`

`SVMCompiler::compile` (`svm.cpp:1035`) emits one program per shader. It starts with a `NODE_SHADER_JUMP` header, then
compiles four parts with `compile_type` (`:897`): bump (only when bump comes from displacement), surface, volume and
displacement. The bump part ends without `NODE_END`, so execution falls through into the surface part.

- **Order.** `generate_multi_closure` (`:706`) walks the closure tree from the output's Surface input. At a Mix Shader
  with a linked factor it emits the factor's dependencies, then the nodes both branches share, then
  `NODE_JUMP_IF_ONE` (factor ≥ 1) around branch 1 and `NODE_JUMP_IF_ZERO` around branch 2, patching each jump
  distance after its branch. As a result, nodes that only feed a zero-weight closure are skipped at run time. `generate_closure_node` (`:619`)
  emits a BSDF's dependencies with `generate_svm_nodes` (`:588`). That function loops until every node whose inputs are
  done has been emitted, which is a plain topological order. It then sets `mix_weight_offset` and compiles the BSDF.
- **Stack allocation.**
  - `stack_find_offset` (`:204`) is a first-fit search over `Stack::users[SVM_STACK_SIZE]` reference counts.
  - A float or int takes 1 slot. A color, vector, point or normal takes 3. A closure takes 0. A value with derivatives
    takes 3× its size (`stack_size`, `:196`).
  - `stack_assign(ShaderOutput*)` gives an output a slot the first time it is needed.
  - `stack_assign(ShaderInput*)` (`:252`) reuses the slot of the linked output. For an unlinked input it emits a
    `NODE_VALUE_F` or `NODE_VALUE_V` that loads the default into a new slot.
  - After a node is emitted, `generate_node` (`:561`) frees two kinds of slots. `stack_clear_users` (`:384`) frees an
    output's slot once every consumer of that output is done. `stack_clear_temporary` frees the default-value slots.
  - Running out of stack logs "out of SVM stack space" and compiles an empty shader. `Summary::peak_stack_usage`
    reports the peak.
- **Emitting.** A node's `compile` calls `compiler.add_node(this, NODE_X, SVMNodeX{...})`. The struct is built from
  `compiler.input_float("Name")`, `input_float3`, `input_int`, `input_link` and `output("Name")` (`svm.h`,
  `svm.cpp:298-367`). `add_node` writes the type word, then the struct's words. It also records the node type in
  `svm_usage`, which feeds the Metal specialization described in §5.
- **Concatenation.** `SVMShaderManager::device_update_specific` (`svm.cpp:55`) compiles all shaders in parallel. It
  then lays out one global array:
  - first, a jump table of `num_shaders` entries, each `NODE_SHADER_JUMP` plus `{offset_surface, offset_volume,
    offset_displacement}`;
  - then every shader's body, with its local offsets rebased to global ones.

  The kernel starts at `(sd->shader & SHADER_MASK) * (1 + sizeof(SVMNodeShaderJump)/4)` (`kernel/svm/svm.h:112`) and
  jumps from there.

### 1.4 The interpreter: `kernel/svm/svm.h`, `util.h`, `types.h`, `node_types.h`

- **`svm_eval_nodes<node_feature_mask, ShaderType>`** (`svm.h:101`) keeps `float stack[SVM_STACK_SIZE]` (255 floats)
  in local memory. Its loop reads a type word with `kernel_data_fetch(svm_nodes, offset++)` and switches on it. Each
  case calls `svm_node_get<SVMNodeX>(kg, &offset)` (`util.h`), which returns a reference into the program and advances
  `offset` by the struct's size. A node with trailing data, such as a ramp table or curves, returns the new offset
  itself (`svm_node_rgb_ramp`, `ramp.h:95`). `NODE_END` returns.
- **Node set.** `kernel/svm/node_types_template.h` lists the node types. The enum order must match the order of the
  `switch` cases, which gives some GPU backends a jump table. `SHADER_NODE_TYPE_DERIVATIVE(X)` adds an `X_DERIVATIVE`
  variant right after `X`.
- **Operand encoding** (`types.h`):
  - `SVMStackOffset` is a `uint8_t`. 255 (`SVM_STACK_INVALID`) means "not on the stack".
  - `SVMInputFloat` is one word that holds either the constant's bits or a NaN pattern
    `SVM_INPUT_STACK_OFFSET_MASK (0x7FC00000) | offset`.
  - `SVMInputFloat3` holds three such words, and only `x` carries the offset tag.
  - `stack_load(stack, SVMInputFloat)` (`util.h`) tests the tag, then reads either the stack or the constant.
  - `SVMCompiler::input_float` (`svm.cpp:308`) replaces any non-finite constant with 0, so a constant can never be
    mistaken for a tag.

  So an unlinked input costs no instruction and no stack slot. Only sockets that must always be links (normals, through
  `input_link`) still load defaults through value nodes.
- **Derivatives.** `mark_nodes_requiring_derivatives` (`svm.cpp:860`) starts from image texture nodes that need
  filtering and marks every upstream node. Marked nodes compile to their `_DERIVATIVE` variant, which stores dual
  numbers (`dual1`/`dual3`: value, d/dx, d/dy) in 3 or 9 slots. A marked node with no derivative variant gets its
  derivative slots zeroed (`stack_zero_incomplete_derivatives`).

### 1.5 Closures

- `svm_node_closure_bsdf` (`kernel/svm/closure.h:219`) reads `mix_weight` from the stack, or uses 1 when the slot is
  invalid. When the weight is 0 it skips the node's data and returns (`svm_node_closure_bsdf_skip`). Otherwise it calls
  `bsdf_alloc(sd, size, weight)`, which appends a `ShaderClosure` to `sd->closure[]` (at most
  `kernel_data.max_closures`, ≤ `MAX_CLOSURE` 64). Closures below `CLOSURE_WEIGHT_CUTOFF` are dropped.
- **The Principled BSDF** (`case CLOSURE_BSDF_PRINCIPLED_ID`, `closure.h:273`; data `SVMNodePrincipledBsdfData`,
  `node_types.h:1131`) is one node with about 30 inputs. It expands in the kernel into up to 7 closures: transparent
  (for alpha), sheen, coat (GGX), metal (F82), glass or transmission, dielectric specular, and diffuse or BSSRDF.
  Each layer's weight is the albedo-scaled remainder of the layer above. `principled_bsdf_emission` (`:94`) handles
  alpha, sheen, coat and emission, and is shared with the emission-only path.
- **Mix and Add Shader.** `NODE_MIX_CLOSURE` (`closure.h:1499`) stores `in_weight·(1−fac)` and `in_weight·fac` in the
  two branch weight slots. Plain color weights on a closure use `NODE_CLOSURE_WEIGHT` and `NODE_CLOSURE_SET_WEIGHT`.
  The shader's result is the list of weighted closures, not one sampled closure.

---

## 2. Types and conversions

- Stack values are only `float` and `float3`. An int is stored as float bits (`stack_load_int`). Color, vector, point
  and normal are the same `float3` on the stack. The distinction exists in the graph for UI, implicit links and
  transforms.
- `ShaderGraph::connect` (`shader_graph.cpp:248`) inserts an automatic `ConvertNode` on a type mismatch
  (`SHADER_SPECIAL_TYPE_AUTOCONVERT`). The kernel side is `kernel/svm/convert.h`:
  - float to vector or color: splat;
  - color to float: `linear_rgb_to_gray` (luminance with the scene's coefficients);
  - vector to float: `average`;
  - to int: truncation.
- A closure cannot convert to anything else. Linking a float or color into a closure socket inserts an
  `EmissionNode`, using Strength for a float and Color for a color.
- An unlinked input takes its constant from the node's stored value, written inline as in §1.4. If a constant was
  folded into a socket that must be a link, `input_link` writes it to the stack.
- Colors stay RGB through the whole graph. They become `Spectrum` only at closure setup (`rgb_to_spectrum`). Cycles
  renders in RGB, so that conversion is the identity.

---

## 3. Nodes for a first version, ranked

Cost rank: **cheap** means a few ALU ops. **Medium** means loops or table reads. **Kernel** means the node needs
support outside the interpreter.

1. **Texture Coordinate, UV Map, Attribute, Vertex Color** (`tex_coord.h`, `attribute.h`, `geometry.h`). Cheap, but
   these nodes read the attribute tables per primitive (`find_attribute`, `primitive_surface_attribute`). They are
   also the leaves that the bump trick shifts (§3.5).
2. **Image Texture** (`kernel/svm/image.h`; `scene/shader_nodes.cpp:429`). Medium, and needs kernel support.
   - The node holds an image `handle`. `handle.kernel_id()` goes into `SVMNodeTexImage.id` with the projection, the
     flags (`NODE_IMAGE_COMPRESS_AS_SRGB`, `NODE_IMAGE_ALPHA_UNASSOCIATE`) and the stack slots.
   - `svm_image_texture` calls `kernel_image_interp_with_udim`, which maps a UDIM tile to a texture id.
   - `kernel_image_interp` (`kernel/device/gpu/image.h:88`) reads `image_textures[id]` to get a `KernelImageInfo`.
     Through tile descriptors it can reach a tiled, cached image, and then it uses the UV derivatives. It then samples
     with hardware bilinear filtering, or with bicubic built from 4 bilinear taps.
   - Box projection (`svm_node_tex_image_box`) blends up to 3 lookups.
   - The node's `TextureMapping` adds a `NODE_TEXTURE_MAPPING` op before the lookup.
3. **Math, Vector Math, Mix (float, vector, color), Map Range, Clamp, Invert, Separate/Combine XYZ and color, Value,
   RGB, Mapping** (`math.h`, `math_util.h`, `mix.h`, `map_range.h`, `clamp.h`, `sepcomb_*.h`, `mapping.h`). All cheap.
   All of them fold when their inputs are constant (`fold_math`, `fold_vector_math`, `fold_mix_color`,
   `fold_mapping`).
4. **Color Ramp and Float/RGB Curves** (`ramp.h`, `ramp_util.h`). Cheap. The ramp is baked on the CPU into a table of
   `table_size` float4s that follows the node in the program (`RAMP_TABLE_SIZE` entries). The kernel interpolates
   linearly or picks the nearest entry (`rgb_ramp_lookup`, `float_ramp_lookup`). No ramp logic runs on the GPU.
5. **Normal Map** (`tex_coord.h:220`, `svm_node_normal_map`). Cheap. It needs a tangent attribute and its sign. It
   scales tangent-space x and y by strength and mixes z toward 1, then transforms to world space. Without tangents it
   falls back to the unperturbed normal.
6. **Bump** (`kernel/svm/displace.h:20`, `svm_node_set_bump`). Medium. It triples the height subgraph and needs ray
   differentials (§3.5).
7. **Procedurals.** Their cost scales with their settings.
   - `checker.h` and `gradient.h`: cheap.
   - `noisetex.h` with `noise.h` and `fractal_noise.h` (Perlin fBM and multifractals): cost scales with Detail octaves,
     clamped to 15 (`noisetex.h:259`). 4D noise costs more.
   - `voronoi.h`: F1 and F2 visit 3³ = 27 cells in 3D. Smooth F1 visits 5³ = 125. Distance to edge makes two 27-cell
     passes. Smooth F1, distance to edge and n-sphere radius sit behind `KERNEL_FEATURE_NODE_VORONOI_EXTRA`.
   - `wave.h`, `magic.h` and `brick.h`: medium. `gabor.h`: expensive.
   - `white_noise.h`: cheap.
8. **Fresnel and Layer Weight** (`fresnel.h`). Cheap, but view-dependent: they read `sd->wi` (§4.3).
9. **Principled BSDF to closures.** §1.5 covers this. For us it maps to "parameters of one OpenPBR" (§6).

Nodes that need kernel support beyond the interpreter:
- AO, Bevel and Raycast trace rays from inside the shader (`__SHADER_RAYTRACE__`, `KERNEL_FEATURE_NODE_RAYTRACE`). A
  shader that uses them is routed to a separate kernel, `SHADE_SURFACE_RAYTRACE` (`kernel/integrator/intersect_closest.h:151`).
- Light Path reads integrator state.
- AOV output writes to the render buffer.
- Wireframe reads the triangle's edges.

### 3.5 The bump trick

A bump needs the height at the shading point and at two points offset along the surface. Cycles gets those values
without re-running the shader with a moved `sd->P`:

- **In the compiler.** `refine_bump_nodes` (`shader_graph.cpp:946`) handles each Bump node. `bump_from_displacement`
  (`:1005`) handles a displacement output set to bump. Both use `find_dependencies` and `copy_nodes` to make 2 extra
  copies of the whole height subgraph. Each copy is tagged `SHADER_BUMP_CENTER`, `DX` or `DY` and stores
  `bump_filter_width`. The three height outputs feed the Bump node's `SampleCenter`, `SampleX` and `SampleY` inputs.
- **In the leaves.** In the DX and DY copies, the nodes that produce coordinates compile with
  `bump_offset = NODE_BUMP_OFFSET_DX/DY` (`shader_bump_to_node_bump_offset`, `shader_nodes.cpp:3848`). The kernel then
  shifts the value by its own derivative: `data.val += data.dx * bump_filter_width` (`kernel/svm/geometry.h:69`).
  Every node downstream runs unchanged on the shifted coordinate.
- **In the Bump node.** `svm_node_set_bump` builds the surface gradient from `dP.dx` and `dP.dy`:
  - `Rx = cross(dPdy, N)` and `Ry = cross(N, dPdx)`;
  - `det = dot(dPdx, Rx)`;
  - `N' = normalize(filter_width·|det|·N − scale·sign(det)·((h_x−h_c)·Rx + (h_y−h_c)·Ry))`.

  It then mixes `N'` with `N` by strength.
- **Bump with displacement.** When the shader also displaces geometry, `NODE_ENTER_BUMP_EVAL` and
  `NODE_LEAVE_BUMP_EVAL` (`kernel/svm/bump.h`) save `P`/`dP` in `SVM_BUMP_EVAL_STATE_SIZE` (10) stack slots. The bump
  part then runs on the undisplaced position, and the saved state is restored before the surface part.

The cost is three evaluations of the height subgraph, all in the same straight-line program. The approach needs ray
differentials, because the offset is one pixel footprint (`dPdx`, `dPdy`) times the filter width.

**What luce-render does.** The compiler's offset copies are emitted afresh for each axis (`offset_slot`), and the uv and
position ops carry the axis and filter-width operand. The footprint is not the ray's differential. It is the camera
pixel's width at the point, along two tangents of the shading normal. That is a function of the point alone, so camera
paths, light paths and their connections all see one bumped surface, and MIS stays consistent. Cycles' footprint
follows each ray, so a surface seen in a reflection bumps more smoothly there. With `filter_width` folded into `dP`, the
gradient formula is the same as Cycles'.

---

## 4. Where shaders are evaluated

### 4.1 Surface hit: evaluate once, then sample and evaluate many times

- `integrate_surface` (`kernel/integrator/shade_surface.h:744`) calls `surface_shader_eval` once (`:772`). That runs
  `svm_eval_nodes` (`kernel/integrator/surface_shader.h:1151`) and fills `sd->closure[]`. Each closure keeps its
  weight and its `sample_weight`, the albedo it was given for choosing it.
- **NEE** (`integrate_surface_direct_light`, `:319`) calls `surface_shader_bsdf_eval(kg, state, sd, ls.D, ...)`
  (`surface_shader.h:361`). That loops over every closure (`_surface_shader_bsdf_eval_mis`) and sums the evaluations
  and the pdfs weighted by `sample_weight`.
- **The BSDF bounce** (`integrate_surface_bsdf_bssrdf_bounce`, `:482`) picks one closure by `sample_weight`
  (`surface_shader_bsdf_bssrdf_pick`), samples it (`surface_shader_bsdf_sample_closure`, `:892`), then evaluates the
  other closures for the same direction to get the mixture's eval and pdf.
- The node program never re-runs at the same vertex. Every later direction query uses the stored closures. They depend
  on `sd->wi` only through what the nodes computed when the shader was evaluated.

### 4.2 Other entry points, each with a smaller feature mask

The mask is a template argument, so code that cannot be reached compiles away (`kernel/features.h:87-114`).

| Where | Mask | Closures stored | Purpose |
|---|---|---|---|
| `shade_surface.h:772` | `KERNEL_FEATURE_NODE_MASK_SURFACE` (all) | up to `max_closures` | the path vertex |
| `shade_shadow.h:115` | `..._SURFACE_SHADOW` (no raytrace, no AOV) | 0 | transparent shadows: `surface_shader_transparency` |
| `light/sample.h:92`, `shade_light.h:194` | `..._SURFACE_LIGHT` (emission only) | 0 | emission of a mesh light at a sampled point |
| `shade_background.h:79` | `..._SURFACE_BACKGROUND` | 0 | world |
| `displacement_shader.h:39` | `..._DISPLACEMENT` | - | geometry displacement before rendering |

- **Shadow rays.** Transparent shadow hits run the full surface program each time. With
  `PATH_RAY_VISIBILITY_SHADOW`, `max_closures` is 0 (`surface_shader.h:1166`), so the BSDFs allocate nothing. Only
  `closure_transparent_extinction` accumulates. Under the emission-only mask, `svm_node_closure_bsdf` keeps only the
  Principled node's emission (`closure.h:241-266`).
- **Lights.** A shader with emission gets an estimate at compile time (`Shader::estimate_emission`,
  `scene/shader.cpp:124`). The estimate is constant only when the emission color and strength are unlinked. A constant
  shader sets `SD_HAS_CONSTANT_EMISSION`, which lets NEE skip evaluation (`surface_shader_constant_emission`).
  `emission_sampling` (NONE/AUTO/FRONT/BACK/FRONT_BACK) decides whether a mesh becomes a light and sets
  `SHADER_USE_MIS`. A light that does not use MIS gets BSDF pdf 0 in NEE (`surface_shader.h:378`).

### 4.3 View-dependent inputs and consistency

Fresnel and Layer Weight compute `fresnel_dielectric_cos(dot(sd->wi, N), eta)` and `|dot(sd->wi, N)|`. Geometry
Incoming and Texture Coordinate Reflection also read `wi`. Cycles stays consistent because it is unidirectional:

- At a path vertex, `wi` is always the direction back along the camera path. NEE and the BSDF bounce at that vertex
  reuse the same closures, so the evaluated BSDF, the sampled BSDF and their pdfs come from one closure set.
- Emitters are evaluated with `wi = -ray direction` both when a BSDF ray hits them and when NEE samples them
  (`shader_setup_from_sample(..., -ray_D, ...)`). Both MIS strategies therefore see the same emitted value.
- Shadow rays evaluate with the shadow ray's own direction. A view-dependent transparency is still a valid function of
  that ray.

Cycles never evaluates a surface for a light-path vertex, so it never has to reconcile two evaluations of the same
point from different directions. Our renderer does (§6.4).

---

## 5. Performance

- **Feature masks and specialization.**
  - `svm_eval_nodes` is a template on `node_feature_mask`. `IF_KERNEL_NODES_FEATURE(X)` is an `if constexpr`, so the
    shadow and light variants leave out BSDF setup, bump, raytrace and AOV code.
  - On Metal, Cycles also builds scene-specialized pipelines (`__KERNEL_USE_DATA_CONSTANTS__`,
    `device/metal/device_impl.mm:429`). Each `SVM_CASE(node)` tests `kernel_data_svm_usage_##node`, a function
    constant filled from the `svm_usage` flags the compiler set. Node types the scene never uses disappear from the
    switch.
  - Kernel-wide features (`KERNEL_FEATURE_NODE_RAYTRACE`, `VORONOI_EXTRA`, `PRINCIPLED_HAIR`, ...) select which
    kernels are built at all.
- **Compile-time folding.** §1.2 covers constant folding, `simplify_settings`, deduplication and dead-node removal.
  Mix Shader with fac 0 or 1 is bypassed (`MixClosureNode::constant_fold`, `shader_nodes.cpp:5349`). Inline constants
  (§1.4) remove most value loads.
- **Run-time skipping.** `NODE_JUMP_IF_ZERO` and `NODE_JUMP_IF_ONE` skip whole closure branches. A BSDF with weight 0
  returns at once. `CLOSURE_WEIGHT_CUTOFF` drops tiny lobes.
- **MIS flags.** `SHADER_USE_MIS` is set per shader from `emission_sampling` and the emission estimate.
  `SD_HAS_TRANSPARENT_SHADOW` (`kernel/types.h:936`) lets shadow rays skip shading at opaque hits.
- **Shader sorting.**
  - `integrator_path_next_sorted` queues each path for the next kernel with a key, the shader id. A state-index
    partition is added for locality: `INTEGRATOR_SORT_KEY = key + max_shaders · (state / sort_partition_divisor)`
    (`kernel/integrator/state_flow.h:160`).
  - The GPU work then sorts states by key (`integrator/path_trace_work_gpu.cpp:86`), so threads in a warp run the same
    program.
  - Shaders that need ray tracing go to their own kernel.
- **Sizes.** `SVM_STACK_SIZE` is 255 floats (1 KB of local memory per thread), with `uint8_t` offsets. `MAX_CLOSURE`
  is 64 fixed-size `ShaderClosure`s and `CAUSTICS_MAX_CLOSURE` is 4. Shadow and emission evaluations use a
  `ShaderData` without the closure array (`ShaderDataTinyStorage`, `kernel/types.h:1093`).

---

## 6. Recommendation for luce-render

### 6.1 What carries over, and what does not

luce-render has one surface model, OpenPBR (`shaders/openpbr.glsl`). There is no closure list, no Mix Shader or Add
Shader, and no per-closure sampling. A material node graph therefore reduces to **computing the OpenPBR parameters at
a point**. What carries over from Cycles: the graph passes (proxies, defaults, folding, dedup, dead-node removal), the
flat program with inline constants, the stack allocator, the jump table of per-material entry points, the feature
masks, the bump trick, and sorting by material. What does not: the closure machinery (`transform_multi_closure`, mix
weights, jump-if-zero around closures).

The terminal node is **OpenPBR Surface**. It has OpenPBR's parameters (base, specular, metalness, transmission, coat,
fuzz later, emission, opacity, thin walled, normal, coat normal, tangent), each one a socket.

### 6.2 The compiled form

- **Where the compiler runs.** Compilation runs in Luce or Base, on the CPU, when a material changes.
  `src/render/materials.lucb` already writes the material record.
- **Constants go into the record.** After folding, every OpenPBR parameter whose socket is unlinked or folded to a
  constant is written into the material record, exactly as today. A material with no links has no program. Its
  `read_material` and `read_colors` stay on the current fast path, so existing scenes cost nothing new.
- **Linked parameters get a program.** A program writes the linked parameters into a small override block.
  - The program lives in one `uint` SSBO, the "programs" buffer.
  - The material record gains a row of entry offsets: `full`, `opacity`, `emission`, `normal`. Each offset is
    `NONE` when that entry has no work. These play the role of Cycles' surface, volume and displacement jump table.
  - Each entry is the dependency slice of the graph that feeds those parameters. `intersect_shadow.comp` then runs only
    the opacity slice, which matches Cycles' `SURFACE_SHADOW` mask. The emission add in `shade_surface.comp` and the
    light code run only the emission slice.
- **Encoding.** Use Cycles' current form:
  - a `uint` opcode;
  - operand words that are either constant bits or `0x7FC00000 | slot` (NaN-tagged), with non-finite constants
    replaced at compile time;
  - `uint8` output slots;
  - trailing data inline for ramps and curves (a baked LUT of 256 entries or fewer, linear interpolation).

  Keep a small stack, `float stack[64]`, and fail compilation with a clear message in luced-3d when a graph needs
  more. Report peak usage the way `Summary::peak_stack_usage` does.
- **Interpreter.** Use one GLSL function, `run_material(entry, inout Inputs, inout Overrides)`, a `switch` in a loop.
  Order the opcode enum to match the case order, as `node_types_template.h` does.
  - Use specialization constants for node groups, the way we already use `HAS_COAT`, `HAS_TRANSMISSION` and the rest
    (`shaders/common.glsl:24`). Examples: `HAS_PROGRAMS`, `HAS_NOISE`, `HAS_VORONOI`, `HAS_BUMP`. A scene without them
    drops those cases when the driver lowers the SPIR-V or the MSL function constants. This is Cycles' Metal
    `svm_usage` idea, and needs no shader source compilation at run time.
  - With `HAS_PROGRAMS` false, the interpreter is not in the kernel at all.
- **Sorting.** The wavefront queues for `shade_surface` should be keyed by material id, or at least split into "has
  program" and "no program", as Cycles keys its sort by shader id. The interpreter's divergence is mostly
  between-material.

### 6.3 Feeding `Surface`

`read_material` and `read_colors` already separate the work that this needs:
- `read_material` handles direction-free, wavelength-free scalars.
- `read_colors` handles colors at the path's wavelengths.

The program runs once per vertex, before `read_material`. The two functions then read each parameter from the
override block when its bit is set, and from `material_at(m, row)` otherwise. The texture overrides in `read_material`
(`maps.x/y/z`) become the general case. The base color, roughness and metalness texture slots of record row 13 turn
into a compiled `IMAGE → param` program, so that row can be retired. `mapped_normal` becomes the Normal Map op, which
writes the shading normal before `surface_at`.

**Spectral handling.**
- Node math stays RGB, in ACEScg (our working space), as Cycles keeps RGB until `rgb_to_spectrum`.
- A color parameter that a program drives is upsampled in `read_colors` with the same path that textured base color
  uses today: `spec_of_texel` (`shaders/spectrum.glsl:67`), clamped to [0, 1] for reflectances.
- Constant colors keep their precomputed spectrum fits in the record.
- Emission driven by a program would use the same basis without the clamp.
- Dispersion and `hero_wavelength` stay in the fixed code after the program.

**Textures.**
- The Image op carries a slot index into the existing bindless table and calls `texture_at(slot, uv)`
  (`shaders/openpbr.glsl:59`). It returns linear Rec.709, which the op converts to ACEScg as `read_material` does now.
- A flag marks non-color data such as roughness and normal maps, which skip the color conversion.
- Sampling is `textureLod(..., 0)` today. Mip selection needs a footprint. Ray cones, or Cycles-style differentials
  that `mark_nodes_requiring_derivatives` limits to texture ancestors, are a later step. The bump trick needs the same
  footprint.

### 6.4 Consistency for light paths and MIS

luce-render evaluates the same point several ways: at the camera vertex (`shade_surface.comp`); at the light-path
vertex, read once as `lit` and re-read for the other side through `turned` and `read_material`
(`shaders/light_paths.glsl`); for the light sampler's pdf from the other side (`light_sampler_pdf`,
`shaders/connect.glsl:57`); and for shadow opacity (`intersect_shadow.comp`). The MIS weights between camera paths, light tracing and NEE assume that every one of these evaluations sees the same
parameters. Two rules follow:

1. **Programs must not depend on direction.** Leave out Fresnel, Layer Weight, Geometry Incoming, Texture Coordinate
   Reflection and Light Path in the first version. OpenPBR already models Fresnel inside the BSDF, and that is what
   artists usually build those nodes for. A "Backfacing" input is fine because it depends only on the side, and the
   side is known at every evaluation.
2. **Evaluate once per vertex and per side, and reuse.** `turned()` should keep the program's outputs and re-derive
   only what depends on the side (`inside`, `coat`, `eta`), instead of reading the record again. Otherwise the program
   would run twice at one light vertex.

Opacity and emission programs are slices of the same graph with the same inputs, so they agree with the full program.
The light tree needs a power estimate for emitters whose emission is textured. Like `Shader::estimate_emission`, use
the strength times a CPU-side average of the texture. That estimate only has to be positive wherever the emission can
be, so the result stays unbiased.

### 6.5 First node set

- **v1:** UV, Texture Coordinate (object, generated), Attribute and vertex color, Backfacing; Image Texture; Math,
  Vector Math, Mix (float, vector, color), Map Range, Clamp, Invert, Separate/Combine (XYZ, RGB), Value, RGB, Mapping;
  Color Ramp and Float Curve as LUTs; Normal Map; the OpenPBR Surface terminal.
- **v2:** Bump (Cycles' three-copy subgraph with shifted leaves, once ray footprints exist); Noise (fBM, octaves
  capped), Voronoi F1 (extra features gated like `VORONOI_EXTRA`), Gradient, Checker, White Noise; HSV, Gamma,
  Bright/Contrast, RGB Curves; Object Info random.
- **Not planned:** Mix Shader and Add Shader, since one OpenPBR has no closure sum. A "Mix Material" that interpolates
  parameters is possible, but it is a different operation from mixing BSDFs and should be named that way. Also not
  planned: AO, Bevel and Raycast (ray tracing inside shading), and the view-dependent nodes above.
