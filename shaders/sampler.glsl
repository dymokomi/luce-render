// Owen-scrambled Sobol points by Burley's hash-based scrambling ("Practical
// Hash-based Owen Scrambling", JCGT 2020). A path draws 4-D points: the index is
// the sample number, shuffled per pixel and dimension group so groups decorrelate,
// and each coordinate is nested-uniform scrambled. Paths keep no RNG state.

#include "sobol_table.glsl"

uint hash_u32(uint x) {
    x ^= x >> 16; x *= 0x21f0aaadu;
    x ^= x >> 15; x *= 0x735a2d97u;
    x ^= x >> 15;
    return x;
}

uint hash_combine(uint seed, uint value) {
    return seed ^ (hash_u32(value) + 0x9e3779b9u + (seed << 6) + (seed >> 2));
}

uint laine_karras(uint x, uint seed) {
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return x;
}

uint nested_uniform_scramble(uint x, uint seed) {
    return bitfieldReverse(laine_karras(bitfieldReverse(x), seed));
}

uint sobol(uint index, uint dimension) {
    uint x = 0u;
    for (uint bit = 0u; index != 0u; bit++, index >>= 1u)
        if ((index & 1u) != 0u) x ^= sobol_directions[dimension * 32u + bit];
    return x;
}

// Dimension groups: the camera's two, then three per bounce.
#define GROUP_CAMERA 0u
#define GROUP_WAVELENGTH 1u
#define GROUP_LIGHT(bounce) (2u + 3u * (bounce))
#define GROUP_BSDF(bounce) (3u + 3u * (bounce))
#define GROUP_OPACITY(bounce) (4u + 3u * (bounce))

// The 4-D point of `group` for this pixel's sample `index`.
vec4 sample4(uint pixel, uint index, uint group) {
    uint seed = hash_combine(hash_combine(params.seed, pixel), group);
    uint shuffled = nested_uniform_scramble(index, hash_u32(seed));
    uvec4 bits;
    for (uint d = 0u; d < 4u; d++)
        bits[d] = nested_uniform_scramble(sobol(shuffled, d), hash_combine(seed, d + 1u));
    // Top 24 bits, so values stay below 1 in f32.
    return vec4(bits >> 8u) * (1.0 / 16777216.0);
}
