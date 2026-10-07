// GGX microfacets in a local frame (normal +z): the anisotropic distribution,
// height-correlated Smith shadowing, visible-normal sampling by spherical caps
// (Dupuy and Benyoub 2023), and dielectric Fresnel. Shared by the shading
// kernels and the kernel that tabulates albedos (tables.comp), so the tables
// always match what is rendered.

// OpenPBR's roughness and anisotropy to GGX alphas along tangent and bitangent
// (alpha_t² + alpha_b² = 2 alpha²), clamped clear of a perfect mirror.
vec2 ggx_alpha(float roughness, float anisotropy) {
    float r2 = roughness * roughness;
    float t = r2 * sqrt(2.0 / (1.0 + (1.0 - anisotropy) * (1.0 - anisotropy)));
    return max(vec2(t, (1.0 - anisotropy) * t), vec2(1e-4));
}

float ggx_d(vec3 m, vec2 alpha) {
    if (m.z <= 0.0) return 0.0;
    vec3 s = vec3(m.x / alpha.x, m.y / alpha.y, m.z);
    float k = dot(s, s);
    return 1.0 / (PI * alpha.x * alpha.y * k * k);
}

float ggx_lambda(vec3 v, vec2 alpha) {
    float z2 = v.z * v.z;
    if (z2 <= 0.0) return INFINITY;
    float a2 = (alpha.x * alpha.x * v.x * v.x + alpha.y * alpha.y * v.y * v.y) / z2;
    return 0.5 * (-1.0 + sqrt(1.0 + a2));
}

float ggx_g1(vec3 v, vec2 alpha) { return 1.0 / (1.0 + ggx_lambda(v, alpha)); }
float ggx_g2(vec3 wi, vec3 wo, vec2 alpha) { return 1.0 / (1.0 + ggx_lambda(wi, alpha) + ggx_lambda(wo, alpha)); }

// A visible normal for wo (above the surface) from two uniform numbers.
vec3 ggx_sample(vec3 wo, vec2 alpha, vec2 u) {
    vec3 v = normalize(vec3(wo.xy * alpha, wo.z));
    float phi = 2.0 * PI * u.x;
    float z = (1.0 - u.y) * (1.0 + v.z) - v.z;
    float s = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
    vec3 h = vec3(s * cos(phi), s * sin(phi), z) + v;
    return normalize(vec3(h.xy * alpha, max(h.z, 1e-6)));
}

// The pdf of the reflected direction for wo, by visible normals.
float ggx_reflection_pdf(vec3 wo, vec3 h, vec2 alpha) {
    return ggx_g1(wo, alpha) * ggx_d(h, alpha) / (4.0 * wo.z);
}

// Unpolarized Fresnel reflectance of a dielectric interface; eta is the
// transmitted side's index over the incident side's.
float fresnel_dielectric(float cos_i, float eta) {
    float c = abs(cos_i);
    float g2 = eta * eta - 1.0 + c * c;
    if (g2 <= 0.0) return 1.0;
    float g = sqrt(g2);
    float a = (g - c) / (g + c);
    float b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return 0.5 * a * a * (1.0 + b * b);
}

// The hemispherical average of fresnel_dielectric (the closed fit OpenPBR's
// reference uses), for eta > 1.
float fresnel_dielectric_average(float eta) {
    return log((10893.0 * eta - 1438.2) / (-774.4 * eta * eta + 10212.0 * eta + 1.0));
}

// Table coordinates: z = sqrt(|eta - 1| / (eta + 1)) spreads precision where
// common indices are (Cycles' mapping), covering eta 1..4.6 over z 0..0.8.
#define TABLE_MU 32u
#define TABLE_ROUGHNESS 32u
#define TABLE_ETA 16u
#define TABLE_Z_MAX 0.8
#define TABLE_GGX 0u                                                   // E(mu, r): GGX with F = 1
#define TABLE_GGX_AVERAGE (TABLE_MU * TABLE_ROUGHNESS)                 // Eavg(r)
#define TABLE_DIELECTRIC (TABLE_GGX_AVERAGE + TABLE_ROUGHNESS)         // E(mu, r, z): GGX × Fresnel(eta)
#define TABLE_SIZE (TABLE_DIELECTRIC + TABLE_MU * TABLE_ROUGHNESS * TABLE_ETA)

float eta_of_z(float z) { return (1.0 + z * z) / (1.0 - z * z); }
float z_of_eta(float eta) { return sqrt(abs(eta - 1.0) / (eta + 1.0)); }

// The nodes along each axis: mu from just above grazing (where G1 vanishes) to 1,
// roughness 0..1, z 0..TABLE_Z_MAX.
#define TABLE_MU_MIN 0.02
float table_mu(uint i) { return mix(TABLE_MU_MIN, 1.0, float(i) / float(TABLE_MU - 1u)); }
float table_roughness(uint i) { return float(i) / float(TABLE_ROUGHNESS - 1u); }
float table_z(uint i) { return TABLE_Z_MAX * float(i) / float(TABLE_ETA - 1u); }
