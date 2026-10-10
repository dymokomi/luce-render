// Gaussian splats along a ray, after 3D Gaussian Ray Tracing (Moenne-Loccoz et
// al. 2024). The host's side is src/render/splats.lucb; the records are
// reached by address (K_SPLATS) and traced through the TLAS's splat instance
// (mask 2, a Blas of boxes) or the splats' own software BVH.
//
// The model. A ray meets each Gaussian once, at its point of maximum response:
// there the splat covers α = min(0.99, opacity · exp(-½ d²)), d the ray's
// Mahalanobis distance from the center, and shows its color, the SH evaluated
// toward the ray's direction (the viewport's display_color). Splats under the
// cull (1/255) or past 3σ do not count. Along a segment of a path, the splats
// met in order of distance add radiance Σ Tᵢ αᵢ cᵢ (Tᵢ = Π_{j<i} (1 - αⱼ)) and
// pass on the fraction T = Π (1 - αᵢ) of whatever lies beyond.
//
// In the integrator splats are a non-scattering layer that emits and absorbs.
// Nothing samples them as lights, so the emission a camera path's segment
// gathers is the only technique for it (weight 1, no MIS term); light sampling
// and light tracing see them only as transmittance, which is deterministic and
// the same for every technique that carries light along a segment (BSDF rays,
// shadow rays, light paths, their camera connections), so the MIS weights
// between techniques are untouched and the estimate stays unbiased. Captured
// lighting is emitted as it was captured; relit splats (Relight GSplats) bake
// their new colors into Cd and the SH before rendering. Light paths carry no
// splat emission (they start at lights) and are only attenuated.
//
// Long segments end by Russian roulette once less than 1% gets through: the
// segment survives with probability T / 0.01 and keeps T = 0.01, so the
// expected transmittance and emission are exact.

#extension GL_EXT_control_flow_attributes : require

#define K_SPLATS 25         // xy the records' address, zw the harmonics' address
#define K_SPLAT_TREE 26     // xy the software BVH's address, z the splat count, w the cull
#define SPLAT_K 32          // hits gathered per round (the k-buffer)
#define SPLAT_ROULETTE 0.01

layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer SplatRecords { uvec4 records[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer SplatWords { uint words[]; };
struct SplatNode {
    vec3 lo;
    uint first;
    vec3 hi;
    uint count;
};
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer SplatNodes { SplatNode nodes[]; };

uint64_t splat_address(vec2 bits) { return packUint2x32(uvec2(floatBitsToUint(bits.x), floatBitsToUint(bits.y))); }
SplatRecords splat_records() { return SplatRecords(splat_address(constants[K_SPLATS].xy)); }
SplatWords splat_words() { return SplatWords(splat_address(constants[K_SPLATS].zw)); }
SplatNodes splat_tree() { return SplatNodes(splat_address(constants[K_SPLAT_TREE].xy)); }
bool splats_present() { return HAS_SPLATS && constants[K_SPLAT_TREE].z > 0.0; }
float splat_cull() { return constants[K_SPLAT_TREE].w; }

// `v` turned by unit quaternion `q` (xyzw).
vec3 splat_turn(vec4 q, vec3 v) {
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Splat `s`'s α where the ray from `o` along unit `d` passes nearest its center
// (in the splat's own metric), and the distance `t` there; 0 past 3σ. In the
// splat's frame scaled to unit σ the ray is o' + t d', and the squared distance
// of its nearest point is |o' × d'|² / |d'|², free of cancellation.
float splat_alpha(SplatRecords records, uint s, vec3 o, vec3 d, out float t) {
    uvec4 r0 = records.records[s * 4u];
    uvec4 r1 = records.records[s * 4u + 1u];
    uvec4 r2 = records.records[s * 4u + 2u];
    vec4 q = uintBitsToFloat(r1);
    vec3 inverse = uintBitsToFloat(r2.xyz);
    vec3 lo = splat_turn(q, o - uintBitsToFloat(r0.xyz)) * inverse;
    vec3 ld = splat_turn(q, d) * inverse;
    float dd = dot(ld, ld);
    t = -dot(lo, ld) / dd;
    vec3 c = cross(lo, ld);
    float m2 = dot(c, c) / dd;
    if (!(m2 <= 9.0)) return 0.0;
    float alpha = min(0.99, uintBitsToFloat(r0.w) * exp(-0.5 * m2));
    return alpha >= splat_cull() ? alpha : 0.0;
}

// The 3DGS SH constants (bands 1..3).
const float splat_c1 = 0.4886025119029199;
const float splat_c2[5] = float[5](1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396);
const float splat_c3[7] = float[7](-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
                                   1.445305721320277, -0.5900435899266435);

float splat_half(SplatWords words, uint first, uint index) {
    vec2 pair = unpackHalf2x16(words.words[first + (index >> 1u)]);
    return (index & 1u) == 0u ? pair.x : pair.y;
}

vec3 splat_item(SplatWords words, uint first, uint item) {
    return vec3(splat_half(words, first, item * 3u), splat_half(words, first, item * 3u + 1u), splat_half(words, first, item * 3u + 2u));
}

// The SH bands' color toward unit `d` (in the SH frame), over `used` items
// starting at word `first`; as luce-3d's splat_project.comp evaluates them.
vec3 splat_bands(SplatWords w, uint first, uint used, vec3 d) {
    float x = d.x, y = d.y, z = d.z;
    vec3 c = -splat_c1 * y * splat_item(w, first, 0u) + splat_c1 * z * splat_item(w, first, 1u) - splat_c1 * x * splat_item(w, first, 2u);
    if (used < 8u) return c;
    float xx = x * x, yy = y * y, zz = z * z;
    c += splat_c2[0] * x * y * splat_item(w, first, 3u) + splat_c2[1] * y * z * splat_item(w, first, 4u)
       + splat_c2[2] * (2.0 * zz - xx - yy) * splat_item(w, first, 5u) + splat_c2[3] * x * z * splat_item(w, first, 6u)
       + splat_c2[4] * (xx - yy) * splat_item(w, first, 7u);
    if (used < 15u) return c;
    c += splat_c3[0] * y * (3.0 * xx - yy) * splat_item(w, first, 8u) + splat_c3[1] * x * y * z * splat_item(w, first, 9u)
       + splat_c3[2] * y * (4.0 * zz - xx - yy) * splat_item(w, first, 10u) + splat_c3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * splat_item(w, first, 11u)
       + splat_c3[4] * x * (4.0 * zz - xx - yy) * splat_item(w, first, 12u) + splat_c3[5] * z * (xx - yy) * splat_item(w, first, 13u)
       + splat_c3[6] * x * (xx - 3.0 * yy) * splat_item(w, first, 14u);
    return c;
}

float splat_srgb_decode(float c) {
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

// Splat `s`'s radiance seen along unit `d`: the DC (in the cloud's encoding)
// plus the bands, clamped at 0, decoded, from linear Rec.709 to ACEScg, as a Spec.
Spec splat_radiance(SplatRecords records, uint s, vec3 d, vec4 lambda) {
    uvec4 r2 = records.records[s * 4u + 2u];
    uvec4 r3 = records.records[s * 4u + 3u];
    vec3 encoded = uintBitsToFloat(r3.xyz);
    uint items = r3.w & 31u;
    if (items >= 3u) {
        SplatWords words = splat_words();
        uint first = r2.w;
        vec4 frame = vec4(unpackHalf2x16(words.words[first]), unpackHalf2x16(words.words[first + 1u]));
        frame /= max(length(frame), 1e-20);
        encoded += splat_bands(words, first + 2u, items, splat_turn(frame, d));
    }
    encoded = max(encoded, vec3(0.0));
    if ((r3.w & 32u) != 0u)
        encoded = vec3(splat_srgb_decode(encoded.r), splat_srgb_decode(encoded.g), splat_srgb_decode(encoded.b));
    // Rec.709 to ACEScg (Bradford), as src/render/environment.lucb.
    vec3 aces = vec3(dot(vec3(0.613097, 0.339523, 0.047379), encoded), dot(vec3(0.070194, 0.916354, 0.013452), encoded),
                     dot(vec3(0.020616, 0.109570, 0.869815), encoded));
    return spec_of_texel(aces, lambda);
}

// A 32-bit mix (sampler.glsl's hash_u32): roulette needs no low-discrepancy numbers.
uint splat_hash(uint x) {
    x ^= x >> 16; x *= 0x21f0aaadu;
    x ^= x >> 15; x *= 0x735a2d97u;
    x ^= x >> 15;
    return x;
}

// A segment's seed: the path, the sample and the bounce, salted per use.
uint splat_seed(uint path, uint bounces, uint salt) {
    return splat_hash(path ^ splat_hash(params.sample_index * 0x9e3779b9u + params.seed) ^ (bounces * 0x85ebca6bu) ^ salt);
}

// A uniform number in [0, 1) for roulette `round` of a segment.
float splat_random(uint seed, uint round) {
    return float(splat_hash(seed ^ (round * 0x9e3779b9u)) >> 8) * (1.0 / 16777216.0);
}

// Russian roulette on a transmittance under SPLAT_ROULETTE: kept at
// SPLAT_ROULETTE with probability T / SPLAT_ROULETTE, else 0.
float splat_roulette(float kept, uint seed, uint round) {
    if (kept >= SPLAT_ROULETTE) return kept;
    return splat_random(seed, round) * SPLAT_ROULETTE < kept ? SPLAT_ROULETTE : 0.0;
}

// The k nearest splats of a round: distances ascending and the splats; empty
// slots hold t = INFINITY. Every index below is a constant once the loops are
// unrolled, so the arrays stay in registers (indexed at run time they went to
// thread memory, the kernel's largest cost), and α is not kept but evaluated
// again when the round is composited, which keeps the state a third smaller.
// 32 slots finish most rays in one round; a round costs a whole traversal.
struct SplatHits {
    float t[SPLAT_K];
    uint s[SPLAT_K];
};

void splat_hits_clear(out SplatHits hits) {
    [[unroll]] for (uint i = 0u; i < SPLAT_K; i++) {
        hits.t[i] = INFINITY;
        hits.s[i] = 0u;
    }
}

bool splat_hits_full(SplatHits hits) { return hits.t[SPLAT_K - 1u] < INFINITY; }

// Keep hit (t, s) among the nearest SPLAT_K by insertion: each slot keeps
// the nearer of itself and the hit carried down; the farthest falls off the end.
void splat_keep(inout SplatHits hits, float t, uint s) {
    [[unroll]] for (uint i = 0u; i < SPLAT_K; i++) {
        bool nearer = t < hits.t[i];
        float kt = hits.t[i];
        uint ks = hits.s[i];
        hits.t[i] = nearer ? t : kt;
        hits.s[i] = nearer ? s : ks;
        t = nearer ? kt : t;
        s = nearer ? ks : s;
    }
}

// The entry distance of the ray into a node's box within [t_lo, t_hi], or -1.
float splat_box_entry(vec3 lo, vec3 hi, vec3 origin, vec3 inverse, float t_lo, float t_hi) {
    vec3 a = (lo - origin) * inverse;
    vec3 b = (hi - origin) * inverse;
    vec3 near = min(a, b);
    vec3 far = max(a, b);
    float enter = max(max(near.x, near.y), max(near.z, t_lo));
    float leave = min(min(far.x, far.y), min(far.z, t_hi));
    return enter <= leave ? enter : -1.0;
}

#define SPLAT_STACK 48

// The SPLAT_K nearest splats whose maximum response lies in (t_lo, t_hi).
SplatHits splat_gather(vec3 o, vec3 d, float t_lo, float t_hi) {
    SplatHits hits;
    splat_hits_clear(hits);
    SplatRecords records = splat_records();
    float cutoff = t_hi;
#if RAY_QUERY
    rayQueryInitializeEXT(query, scene_tlas, gl_RayFlagsNoneEXT, 0x02u, o, t_lo, d, t_hi);
    while (rayQueryProceedEXT(query)) {
        if (rayQueryGetIntersectionTypeEXT(query, false) != gl_RayQueryCandidateIntersectionAABBEXT) continue;
        uint s = uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, false));
        float t;
        float alpha = splat_alpha(records, s, o, d, t);
        if (alpha <= 0.0 || !(t > t_lo && t < cutoff)) continue;
        splat_keep(hits, t, s);
        // Full: nothing past the farthest kept can enter, so the query culls beyond it.
        if (splat_hits_full(hits)) {
            cutoff = hits.t[SPLAT_K - 1u];
            rayQueryGenerateIntersectionEXT(query, cutoff);
        }
    }
#else
    SplatNodes tree = splat_tree();
    vec3 safe = mix(d, sign(d) * 1e-20 + vec3(equal(d, vec3(0.0))) * 1e-20, lessThan(abs(d), vec3(1e-20)));
    vec3 inverse = 1.0 / safe;
    uint stack[SPLAT_STACK];
    uint depth = 0u;
    uint node = 0u;
    if (tree.nodes[0].lo.x > tree.nodes[0].hi.x || splat_box_entry(tree.nodes[0].lo, tree.nodes[0].hi, o, inverse, t_lo, cutoff) < 0.0) return hits;
    while (true) {
        SplatNode current = tree.nodes[node];
        if (current.count > 0u) {
            for (uint i = 0u; i < current.count; i++) {
                uint s = current.first + i;
                float t;
                float alpha = splat_alpha(records, s, o, d, t);
                if (alpha <= 0.0 || !(t > t_lo && t < cutoff)) continue;
                splat_keep(hits, t, s);
                if (splat_hits_full(hits)) cutoff = hits.t[SPLAT_K - 1u];
            }
        } else {
            uint left = current.first;
            float a = splat_box_entry(tree.nodes[left].lo, tree.nodes[left].hi, o, inverse, t_lo, cutoff);
            float b = splat_box_entry(tree.nodes[left + 1u].lo, tree.nodes[left + 1u].hi, o, inverse, t_lo, cutoff);
            if (a >= 0.0 && b >= 0.0) {
                uint near_child = a <= b ? left : left + 1u;
                if (depth < SPLAT_STACK) stack[depth++] = a <= b ? left + 1u : left;
                node = near_child;
                continue;
            }
            if (a >= 0.0) { node = left; continue; }
            if (b >= 0.0) { node = left + 1u; continue; }
        }
        // Nodes waiting on the stack may have fallen past the cutoff since.
        bool found = false;
        while (depth > 0u && !found) {
            node = stack[--depth];
            found = splat_box_entry(tree.nodes[node].lo, tree.nodes[node].hi, o, inverse, t_lo, cutoff) >= 0.0;
        }
        if (!found) break;
    }
#endif
    return hits;
}

// The splats along the ray from `o` (unit `d`) up to `t_hi`, front to back:
// the radiance they add (`added`, to be weighed by the path's throughput) and
// the fraction they pass on (`kept`), in rounds of the SPLAT_K nearest.
void splat_composite(vec3 o, vec3 d, float t_hi, vec4 lambda, uint seed, out Spec added, out float kept) {
    added = Spec(0.0);
    kept = 1.0;
    SplatRecords records = splat_records();
    float t_lo = 0.0;
    for (uint round = 0u; round < 4096u; round++) {
        SplatHits hits = splat_gather(o, d, t_lo, t_hi);
        [[unroll]] for (uint i = 0u; i < SPLAT_K; i++) {
            if (hits.t[i] == INFINITY) break;
            float t;
            float alpha = splat_alpha(records, hits.s[i], o, d, t);
            added += splat_radiance(records, hits.s[i], d, lambda) * (kept * alpha);
            kept *= 1.0 - alpha;
        }
        if (!splat_hits_full(hits)) return;
        t_lo = hits.t[SPLAT_K - 1u];
        kept = splat_roulette(kept, seed, round);
        if (kept == 0.0) return;
    }
}

// The fraction of light the splats pass along the ray from `o` (unit `d`)
// over (t_lo, t_hi): the product of their (1 - α), in any order.
float splat_transmittance(vec3 o, vec3 d, float t_lo, float t_hi, uint seed) {
    float kept = 1.0;
    uint round = 0u;
    SplatRecords records = splat_records();
#if RAY_QUERY
    rayQueryInitializeEXT(query, scene_tlas, gl_RayFlagsNoneEXT, 0x02u, o, t_lo, d, t_hi);
    while (rayQueryProceedEXT(query)) {
        if (rayQueryGetIntersectionTypeEXT(query, false) != gl_RayQueryCandidateIntersectionAABBEXT) continue;
        float t;
        float alpha = splat_alpha(records, uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, false)), o, d, t);
        if (alpha <= 0.0 || !(t > t_lo && t < t_hi)) continue;
        kept *= 1.0 - alpha;
        if (kept < SPLAT_ROULETTE) {
            kept = splat_roulette(kept, seed, round++);
            if (kept == 0.0) {
                rayQueryTerminateEXT(query);
                return 0.0;
            }
        }
    }
#else
    SplatNodes tree = splat_tree();
    vec3 safe = mix(d, sign(d) * 1e-20 + vec3(equal(d, vec3(0.0))) * 1e-20, lessThan(abs(d), vec3(1e-20)));
    vec3 inverse = 1.0 / safe;
    uint stack[SPLAT_STACK];
    uint depth = 0u;
    uint node = 0u;
    if (tree.nodes[0].lo.x > tree.nodes[0].hi.x || splat_box_entry(tree.nodes[0].lo, tree.nodes[0].hi, o, inverse, t_lo, t_hi) < 0.0) return 1.0;
    while (true) {
        SplatNode current = tree.nodes[node];
        if (current.count > 0u) {
            for (uint i = 0u; i < current.count; i++) {
                float t;
                float alpha = splat_alpha(records, current.first + i, o, d, t);
                if (alpha <= 0.0 || !(t > t_lo && t < t_hi)) continue;
                kept *= 1.0 - alpha;
                if (kept < SPLAT_ROULETTE) {
                    kept = splat_roulette(kept, seed, round++);
                    if (kept == 0.0) return 0.0;
                }
            }
        } else {
            uint left = current.first;
            bool a = splat_box_entry(tree.nodes[left].lo, tree.nodes[left].hi, o, inverse, t_lo, t_hi) >= 0.0;
            bool b = splat_box_entry(tree.nodes[left + 1u].lo, tree.nodes[left + 1u].hi, o, inverse, t_lo, t_hi) >= 0.0;
            if (a && b) {
                if (depth < SPLAT_STACK) stack[depth++] = left + 1u;
                node = left;
                continue;
            }
            if (a) { node = left; continue; }
            if (b) { node = left + 1u; continue; }
        }
        if (depth == 0u) break;
        node = stack[--depth];
    }
#endif
    return kept;
}
