// What every integrator kernel shares: the push constants, the buffers at
// bindings 0..9, the path state and the queues. The host's side is
// src/render/integrator.lucb; keep the two in step.
#extension GL_GOOGLE_include_directive : require

#ifndef SPECTRAL
#define SPECTRAL 1
#endif
#ifndef RAY_QUERY
#define RAY_QUERY 0
#endif

#define PI 3.14159265358979323846
#define INFINITY 3.0e38
#define NONE 0xffffffffu

// A spectral value: four wavelengths' radiance, or ACEScg RGB in xyz (w unused),
// so both builds store and pass the same type.
#define Spec vec4

// The scene's features as specialization constants (src/render/integrator.lucb:
// scene_features): the compiler drops what a scene does not use. The defaults
// keep everything, so a kernel built without values is correct for any scene.
layout(constant_id = 0) const bool HAS_COAT = true;
layout(constant_id = 1) const bool HAS_TRANSMISSION = true;
layout(constant_id = 2) const bool HAS_METAL = true;
layout(constant_id = 3) const bool HAS_EMISSION = true;
layout(constant_id = 4) const bool CUTOUTS = true;
layout(constant_id = 5) const bool GUIDING = true;
// The number of lights; 0xffffffff leaves it to the constants buffer.
layout(constant_id = 6) const uint LIGHTS = 0xffffffffu;

layout(push_constant) uniform Params {
    vec4 position;      // camera; w: 1 orthographic
    vec4 corner;
    vec4 du;            // w: image width
    vec4 dv;            // w: image height
    uint band_start;    // first pixel of this pass's band
    uint band_count;    // paths in the band
    uint sample_index;
    uint bounce;
    uint current;       // the closest-hit queue being read (0 or 1)
    uint phase;         // schedule's step
    uint pool;          // path slots: the stride of every state field
    uint seed;
    uint64_t guide;     // the path-guiding grid's address (guide.glsl)
} params;

#include "scene.glsl"

// Lights: LIGHT_STRIDE vec4s each (src/render/lights.lucb).
layout(set = 0, binding = 3, std430) readonly buffer Lights { vec4 lights[]; };
// Path state, field-major: field f of path i at f * pool + i.
layout(set = 0, binding = 4, std430) buffer StateV { vec4 state_v[]; };
layout(set = 0, binding = 5, std430) buffer StateU { uint state_u[]; };
// Queue counters [0..15], indirect arguments [16..31], then each queue's paths.
layout(set = 0, binding = 6, std430) buffer Queues { uint queues[]; };
// The film, three planes of a vec4 a pixel (FILM_* below): the sum of samples
// (XYZ under E, or ACEScg; w their count), adaptive sampling's half buffer
// (twice the odd samples; w 1 once the pixel has converged), and a scratch
// plane for the convergence check.
layout(set = 0, binding = 7, std430) buffer Film { vec4 film[]; };
uint film_pixels() { return uint(params.du.w) * uint(params.dv.w); }
#define FILM_SUM 0u
#define FILM_HALF 1u
#define FILM_SCRATCH 2u
vec4 film_at(uint plane, uint pixel) { return film[plane * film_pixels() + pixel]; }
// Scene-wide constants (src/render/integrator.lucb: fill_constants).
layout(set = 0, binding = 9, std430) readonly buffer Constants { vec4 constants[]; };

#define K_INFO 0            // x lights, y max bounces, z indirect clamp (0 off), w 1 when a material cuts out
#define K_LIMITS 7          // the most diffuse, glossy and transmission bounces
#define K_BACKGROUND_FIT 1
#define K_BACKGROUND_RGB 2
#define K_FILM 3            // 3 rows: XYZ under E to ACEScg
#define K_CMF 6             // xyz: the color matching integrals
#define K_DISPLAY 9         // x exposure in stops, y view (0 standard, 1 neutral)
#define K_ADAPTIVE 11       // x noise threshold (0: off), y the samples before checks

// State fields.
#define SV_ORIGIN 0u
#define SV_DIRECTION 1u     // w: the BSDF pdf of the ray (for MIS at an emitter)
#define SV_THROUGHPUT 2u
#define SV_RADIANCE 3u
#define SV_LAMBDA 4u
#define SV_SHADOW_ORIGIN 5u // w: the shadow ray's length
#define SV_SHADOW_DIRECTION 6u
#define SV_SHADOW_RADIANCE 7u
#define SV_HIT 8u           // t, -, barycentrics u, v
#define SV_VERTEX 9u        // the last scattering point (origins move on through cut-outs; MIS pdfs are from here);
                            // w: unguided over guided throughput, which Russian roulette weighs (guiding must not change survival)
uint light_count() { return LIGHTS != 0xffffffffu ? LIGHTS : uint(constants[K_INFO].x); }

// Path guiding's ring of the first vertices (guide.glsl): pdf, throughput
// luminance after the vertex, luminance found before it, |cos| of the sampled
// direction; and cell, direction.
#define SV_GUIDE(k) (10u + (k))
#define SU_PIXEL 0u
#define SU_BOUNCE 1u        // bounces so far: total, diffuse, glossy, transmission (a byte each)
// What was hit: a triangle or LIGHT_HIT | light. Kept as an integer: bits stored
// in a float can be denormals, which Metal flushes to zero.
#define SU_HIT 2u
#define SU_NORMAL 3u        // the previous vertex's shading normal (octahedral), for the light tree's MIS pdf
#define SU_GUIDE_CELL(k) (4u + (k))
#define SU_GUIDE_DIRECTION(k) (7u + (k))

vec4 get_v(uint field, uint path) { return state_v[field * params.pool + path]; }
void set_v(uint field, uint path, vec4 value) { state_v[field * params.pool + path] = value; }
uint get_u(uint field, uint path) { return state_u[field * params.pool + path]; }
void set_u(uint field, uint path, uint value) { state_u[field * params.pool + path] = value; }

// Queues.
#define Q_CLOSEST 0u        // 0 and 1, alternating by bounce
#define Q_SURFACE 2u
#define Q_MISS 3u
#define Q_SHADOW 4u
#define QUEUE_BASE 32u

uint queue_count(uint queue) { return queues[queue]; }

// Scheduling without a scheduling kernel: the last workgroup of each kernel to
// finish (counted in queues[QUEUE_DONE]) writes the indirect arguments of the
// kernels that consume what it produced, and empties the queue it consumed.
// Every indirect dispatch launches at least one workgroup, so the chain never
// stops at an empty queue.
#define QUEUE_DONE 8u
uint arguments_slot(uint queue) { return 16u + 3u * queue; }

void arguments(uint queue) {
    uint count = atomicAdd(queues[queue], 0u);
    uint at = arguments_slot(queue);
    queues[at] = max(1u, (count + 63u) / 64u);
    queues[at + 1u] = 1u;
    queues[at + 2u] = 1u;
}

// Whether this thread is the first of the last of `groups` workgroups to
// finish; every thread of the kernel must call it. Each workgroup's writes
// are visible to the last one.
bool last_group(uint groups) {
    memoryBarrierBuffer();
    barrier();
    bool last = false;
    if (gl_LocalInvocationIndex == 0u) {
        uint done = atomicAdd(queues[QUEUE_DONE], 1u);
        if (done + 1u == groups) {
            last = true;
            queues[QUEUE_DONE] = 0u;
            memoryBarrierBuffer();
        }
    }
    return last;
}
uint queue_path(uint queue, uint at) { return queues[QUEUE_BASE + queue * params.pool + at]; }
void queue_push(uint queue, uint path) {
    uint at = atomicAdd(queues[queue], 1u);
    queues[QUEUE_BASE + queue * params.pool + at] = path;
}

// What a closest hit found: a triangle, a light (LIGHT_HIT | index) or nothing.
#define LIGHT_HIT 0x80000000u

uint bounce_total(uint bounces) { return bounces & 0xffu; }

// The bounces after one more of `lobe` (0 diffuse, 1 glossy, 2 transmission),
// or NONE past that kind's limit.
uint next_bounce(uint bounces, uint lobe) {
    uint shift = 8u * (lobe + 1u);
    uint kind = ((bounces >> shift) & 0xffu) + 1u;
    if (float(kind) > constants[K_LIMITS][lobe] || bounce_total(bounces) >= 255u) return NONE;
    return (bounces & ~(0xffu << shift)) + (kind << shift) + 1u;
}

float power_heuristic(float a, float b) {
    float a2 = a * a;
    return a2 / (a2 + b * b);
}

// An origin moved off a surface along its normal, scaled to the coordinates.
vec3 offset_origin(vec3 p, vec3 n) {
    float scale = max(1.0, max(abs(p.x), max(abs(p.y), abs(p.z))));
    return p + n * (1e-4 * scale);
}

// Unit vectors as two snorm16 octahedral coordinates.
uint octahedral_encode(vec3 n) {
    vec2 p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) p = (1.0 - abs(p.yx)) * vec2(p.x >= 0.0 ? 1.0 : -1.0, p.y >= 0.0 ? 1.0 : -1.0);
    return packSnorm2x16(p);
}

vec3 octahedral_decode(uint packed) {
    vec2 p = unpackSnorm2x16(packed);
    vec3 n = vec3(p, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

// An orthonormal basis around n (Duff et al. 2017).
void basis(vec3 n, out vec3 t, out vec3 b) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float c = n.x * n.y * a;
    t = vec3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    b = vec3(c, s + n.y * n.y * a, -n.y);
}
