// Path guiding (docs/research/REALTIME-STUDY.md §3.1): a world-space hash grid
// of cells keyed by quantised position and the octant of the normal, each a
// mixture of GUIDE_LOBES von Mises-Fisher lobes over the sphere of directions,
// fitted to the incident radiance paths have found there. Shading samples the
// mixture or the BSDF (one-sample MIS); `finish` credits each path's first
// GUIDE_RING vertices with the luminance that arrived along their sampled
// directions (statistics by float atomics, the nearest lobe taking each
// sample); guide_fit.comp turns statistics into lobes once a pass.
//
// The grid lives in one buffer reached by address (params.guide), in regions:
//   keys     u32 a cell (0: empty)
//   lobes    vec4 a lobe: mean direction, kappa; then vec4 a cell: the weights
//   stats    float × 5 a lobe: weight, weighted direction (x, y, z), count
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_atomic_float : require

#define GUIDE_CELLS 524288u
#define GUIDE_LOBES 4u
#define GUIDE_RING 3u       // the first vertices of a path that train, and that the guide samples at
#define GUIDE_TRAINING 1u   // one path in this many trains (the fit needs proportions, not every sample)
#define GUIDE_PROBES 8u
#define K_GUIDE 12          // x on, y cell width per unit of distance from the camera (8 pixels' footprint), z the guide's share, w samples before a cell guides
#define K_GUIDE_SIZE 13     // x the smallest cell width

layout(buffer_reference, std430, buffer_reference_align = 4) buffer GuideKeys { uint keys[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer GuideLobes { vec4 lobes[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer GuideStats { float stats[]; };

GuideKeys guide_keys() { return GuideKeys(params.guide); }
GuideLobes guide_lobes() { return GuideLobes(params.guide + uint64_t(GUIDE_CELLS) * 4ul); }
GuideStats guide_stats() { return GuideStats(params.guide + uint64_t(GUIDE_CELLS) * (4ul + 16ul * uint64_t(GUIDE_LOBES + 1u))); }

bool guiding() { return GUIDING && constants[K_GUIDE].x > 0.5; }

// The cell of point p with normal n, made on first use; NONE when the grid is
// full along the probe run. Cells are about 8 pixels of the camera's footprint
// wide where p is, in power-of-two levels (SHARC's scheme), so a cell gathers
// dozens of samples a pass at any resolution and distance.
uint guide_cell(vec3 p, vec3 n) {
    float minimum = constants[K_GUIDE_SIZE].x;
    float wanted = max(minimum, distance(p, params.position.xyz) * constants[K_GUIDE].y);
    uint level = uint(clamp(ceil(log2(wanted / minimum)), 0.0, 31.0));
    float size = minimum * exp2(float(level));
    ivec3 q = ivec3(floor(p / size));
    uint octant = (n.x >= 0.0 ? 1u : 0u) | (n.y >= 0.0 ? 2u : 0u) | (n.z >= 0.0 ? 4u : 0u);
    uint key = hash_u32(uint(q.x) * 73856093u ^ hash_u32(uint(q.y) * 19349663u ^ hash_u32(uint(q.z) * 83492791u ^ (octant | (level << 3u)))));
    key = max(key, 1u);
    uint slot = key % GUIDE_CELLS;
    GuideKeys keys = guide_keys();
    for (uint probe = 0u; probe < GUIDE_PROBES; probe++) {
        uint at = (slot + probe) % GUIDE_CELLS;
        // A plain read finds existing cells; only a new one takes an atomic.
        uint seen = keys.keys[at];
        if (seen == key) return at;
        if (seen != 0u) continue;
        uint found = atomicCompSwap(keys.keys[at], 0u, key);
        if (found == 0u || found == key) return at;
    }
    return NONE;
}

// How many samples have trained the cell (its lobes' counts).
float guide_count(uint cell) {
    GuideStats s = guide_stats();
    float count = 0.0;
    for (uint k = 0u; k < GUIDE_LOBES; k++) count += s.stats[(cell * GUIDE_LOBES + k) * 5u + 4u];
    return count;
}

// The lobes before any training: spread over the sphere (a tetrahedron), wide.
vec4 guide_lobe(uint cell, uint lobe) {
    vec4 v = guide_lobes().lobes[cell * (GUIDE_LOBES + 1u) + lobe];
    if (v.w > 0.0) return v;
    const vec3 spread[4] = vec3[4](vec3(1, 1, 1), vec3(1, -1, -1), vec3(-1, 1, -1), vec3(-1, -1, 1));
    return vec4(normalize(spread[lobe & 3u]), 1.0);
}

vec4 guide_weights(uint cell) {
    vec4 w = guide_lobes().lobes[cell * (GUIDE_LOBES + 1u) + GUIDE_LOBES];
    float total = w.x + w.y + w.z + w.w;
    return total > 0.0 ? w / total : vec4(0.25);
}

float vmf_pdf(vec4 lobe, vec3 d) {
    float kappa = lobe.w;
    if (kappa < 1e-3) return 1.0 / (4.0 * PI);
    return kappa / (2.0 * PI * (1.0 - exp(-2.0 * kappa))) * exp(kappa * (dot(lobe.xyz, d) - 1.0));
}

vec3 vmf_sample(vec4 lobe, vec2 u) {
    float kappa = max(lobe.w, 1e-3);
    float w = 1.0 + log(u.x + (1.0 - u.x) * exp(-2.0 * kappa)) / kappa;
    float s = sqrt(max(0.0, 1.0 - w * w));
    float phi = 2.0 * PI * u.y;
    vec3 t, b;
    basis(lobe.xyz, t, b);
    return normalize(t * (s * cos(phi)) + b * (s * sin(phi)) + lobe.xyz * w);
}

float guide_pdf(uint cell, vec3 d) {
    vec4 weights = guide_weights(cell);
    float pdf = 0.0;
    for (uint k = 0u; k < GUIDE_LOBES; k++) pdf += weights[k] * vmf_pdf(guide_lobe(cell, k), d);
    return pdf;
}

// A direction from the mixture: the lobe by its weight from u.z.
vec3 guide_sample(uint cell, vec3 u) {
    vec4 weights = guide_weights(cell);
    uint lobe = GUIDE_LOBES - 1u;
    float pick = u.z;
    for (uint k = 0u; k < GUIDE_LOBES; k++) {
        if (pick < weights[k]) { lobe = k; break; }
        pick -= weights[k];
    }
    return vmf_sample(guide_lobe(cell, lobe), u.xy);
}

// Credit cell `cell` with radiance `value` (luminance) arriving along d, sampled
// with pdf `pdf`: to the lobe that explains d best.
void guide_train(uint cell, vec3 d, float value, float pdf) {
    if (cell == NONE || !(value > 0.0) || !(pdf > 0.0) || isinf(value)) return;
    vec4 weights = guide_weights(cell);
    uint best = 0u;
    float best_score = -1.0;
    for (uint k = 0u; k < GUIDE_LOBES; k++) {
        // A floor on the weight, so an empty lobe can win samples again.
        float score = (weights[k] + 0.05) * vmf_pdf(guide_lobe(cell, k), d);
        if (score > best_score) { best_score = score; best = k; }
    }
    float w = min(value / pdf, 1e4);
    GuideStats s = guide_stats();
    uint at = (cell * GUIDE_LOBES + best) * 5u;
    atomicAdd(s.stats[at], w);
    atomicAdd(s.stats[at + 1u], w * d.x);
    atomicAdd(s.stats[at + 2u], w * d.y);
    atomicAdd(s.stats[at + 3u], w * d.z);
    atomicAdd(s.stats[at + 4u], 1.0);
}
