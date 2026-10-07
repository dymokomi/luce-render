// OpenPBR Surface: material records (src/render/materials.lucb) and lobes.
// Records are MATERIAL_STRIDE vec4s; each color is its spectrum fit and its RGB:
//   0, 1   base color fit; base color RGB, base weight
//   2, 3   specular color fit; RGB, specular weight
//   4, 5   coat color fit; RGB, coat weight
//   6, 7   emission fit (× luminance); RGB (× luminance), luminance
//   8, 9   transmission color fit; RGB, transmission weight
//   10     metalness, diffuse roughness, specular roughness, specular anisotropy
//   11     specular IOR, coat roughness, coat IOR, coat darkening
//   12     opacity, thin walled, coat anisotropy, -
// Lobes so far, in a local frame with the normal along +z:
//   - EON diffuse (Portsmouth, Kutz and Hill, "EON: A practical energy-preserving
//     rough diffuse BRDF", 2025), sampled cosine-weighted;
//   - a GGX dielectric specular layer over it (exact Fresnel at specular_ior),
//     the diffuse scaled by what the layer does not reflect (albedo scaling);
//   - F82-tint metal (Kutz et al., Adobe Standard Material);
//   - emission.
// Microfacet lobes are compensated for multiple scattering by Kulla and Conty's
// 1 + F_avg (1 - E) / E, from albedo tables made on the GPU (tables.comp).

#define MATERIAL_STRIDE 16u

layout(set = 0, binding = 11, std430) readonly buffer Shading { uvec4 shading[]; };
layout(set = 0, binding = 12, std430) readonly buffer Materials { vec4 materials[]; };
layout(set = 0, binding = 13, std430) readonly buffer Tables { float tables[]; };

vec4 material_at(uint material, uint row) { return materials[material * MATERIAL_STRIDE + row]; }

// A material color as a Spec at the path's wavelengths.
Spec material_color(uint material, uint row, vec4 lambda) {
    return spec_of(material_at(material, row), material_at(material, row + 1u).xyz, lambda);
}

vec3 octahedral_decode(uint packed) {
    vec2 p = unpackSnorm2x16(packed);
    vec3 n = vec3(p, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

// -- EON diffuse.

const float FON_C1 = 0.5 - 2.0 / (3.0 * PI);
const float FON_C2 = 2.0 / 3.0 - 28.0 / (15.0 * PI);

// The FON lobe's directional albedo, by the paper's polynomial fit.
float fon_albedo(float mu, float r) {
    float m = 1.0 - mu;
    float g_over_pi = m * (0.0571085289 + m * (0.491881867 + m * (-0.332181442 + m * 0.0714429953)));
    return (1.0 + r * g_over_pi) / (1.0 + FON_C1 * r);
}

// EON at directions wi, wo (local, both above the surface) for albedo rho and
// roughness r: Fujii's Oren-Nayar plus the energy it loses, so a white surface
// reflects everything at any roughness.
Spec eon(Spec rho, float r, vec3 wi, vec3 wo) {
    float mu_i = wi.z;
    float mu_o = wo.z;
    float s = dot(wi, wo) - mu_i * mu_o;
    float s_over_t = s > 0.0 ? s / max(mu_i, mu_o) : s;
    float a = 1.0 / (1.0 + FON_C1 * r);
    Spec single = rho * (a * (1.0 + r * s_over_t) / PI);
    float e_o = fon_albedo(mu_o, r);
    float e_i = fon_albedo(mu_i, r);
    float average = a * (1.0 + FON_C2 * r);
    Spec rho_ms = rho * rho * average / (Spec(1.0) - rho * (1.0 - average));
    Spec multiple = rho_ms * (max(1e-7, 1.0 - e_o) * max(1e-7, 1.0 - e_i) / (PI * max(1e-7, 1.0 - average)));
    return single + multiple;
}

// EON's directional albedo at wo: what it reflects of uniform light.
Spec eon_albedo(Spec rho, float r, float mu_o) {
    float a = 1.0 / (1.0 + FON_C1 * r);
    float e = fon_albedo(mu_o, r);
    float average = a * (1.0 + FON_C2 * r);
    Spec rho_ms = rho * rho * average / (Spec(1.0) - rho * (1.0 - average));
    return rho * e + rho_ms * (1.0 - e);
}

// -- Albedo tables (nodes at both ends of each axis, linear interpolation).

float table_row(uint base, float r, float mu) {
    float x = clamp((mu - TABLE_MU_MIN) / (1.0 - TABLE_MU_MIN) * float(TABLE_MU - 1u), 0.0, float(TABLE_MU - 1u));
    float y = clamp(r * float(TABLE_ROUGHNESS - 1u), 0.0, float(TABLE_ROUGHNESS - 1u));
    uint x0 = uint(x), y0 = uint(y);
    uint x1 = min(x0 + 1u, TABLE_MU - 1u), y1 = min(y0 + 1u, TABLE_ROUGHNESS - 1u);
    float fx = x - float(x0), fy = y - float(y0);
    float a = mix(tables[base + y0 * TABLE_MU + x0], tables[base + y0 * TABLE_MU + x1], fx);
    float b = mix(tables[base + y1 * TABLE_MU + x0], tables[base + y1 * TABLE_MU + x1], fx);
    return mix(a, b, fy);
}

float table_ggx(float mu, float r) { return table_row(TABLE_GGX, r, mu); }

float table_ggx_average(float r) {
    float y = clamp(r * float(TABLE_ROUGHNESS - 1u), 0.0, float(TABLE_ROUGHNESS - 1u));
    uint y0 = uint(y), y1 = min(uint(y) + 1u, TABLE_ROUGHNESS - 1u);
    return mix(tables[TABLE_GGX_AVERAGE + y0], tables[TABLE_GGX_AVERAGE + y1], y - float(y0));
}

float table_dielectric(float mu, float r, float eta) {
    float z = clamp(z_of_eta(eta) / TABLE_Z_MAX * float(TABLE_ETA - 1u), 0.0, float(TABLE_ETA - 1u));
    uint z0 = uint(z), z1 = min(uint(z) + 1u, TABLE_ETA - 1u);
    uint plane = TABLE_MU * TABLE_ROUGHNESS;
    return mix(table_row(TABLE_DIELECTRIC + z0 * plane, r, mu), table_row(TABLE_DIELECTRIC + z1 * plane, r, mu), z - float(z0));
}

// -- F82-tint metal Fresnel, per lane.

Spec fresnel_f82(Spec f0, Spec tint, float mu) {
    const float bar = 1.0 / 7.0;
    float m = 1.0 - mu;
    float m5 = m * m * m * m * m;
    float b5 = pow(1.0 - bar, 5.0);
    Spec schlick = f0 + (1.0 - f0) * m5;
    Spec schlick_bar = f0 + (1.0 - f0) * b5;
    Spec a = schlick_bar * (1.0 - tint) / (bar * b5 * (1.0 - bar));
    return clamp(schlick - a * (mu * m5 * m), 0.0, 1.0);
}

// -- The surface at a hit.

float rgb_luminance(vec3 rgb) { return dot(rgb, vec3(0.2722287168, 0.6740817658, 0.0536895174)); }

struct Surface {
    vec3 normal;        // shading normal, facing the incoming ray's side
    vec3 tangent;
    vec3 bitangent;
    vec3 geometric;     // geometric normal on the same side
    uint material;
    Spec base;          // base color × base weight: the diffuse albedo and the metal's F0
    float diffuse_roughness;
    float metalness;
    Spec specular_tint; // specular color × specular weight
    Spec metal_tint;    // specular color (F82 tint)
    float specular_weight;
    vec2 alpha;
    float eta;
    float dielectric_compensation;  // Kulla-Conty factor of the dielectric layer
    float metal_missing;            // (1 - E) / E of the metal at wo
    float specular_albedo;          // what the dielectric layer reflects at wo
    float p_diffuse;    // lobe choice: wavelength-free, from the RGB parameters
    float p_specular;
    float p_metal;
};

vec3 to_local(Surface s, vec3 v) { return vec3(dot(v, s.tangent), dot(v, s.bitangent), dot(v, s.normal)); }
vec3 to_world(Surface s, vec3 v) { return s.tangent * v.x + s.bitangent * v.y + s.normal * v.z; }

// Read the material and prepare its lobes for the view direction `wo` (local).
void prepare_surface(inout Surface s, vec3 wo, vec4 lambda) {
    uint m = s.material;
    vec4 base_rgb = material_at(m, 1u);
    vec4 specular_rgb = material_at(m, 3u);
    vec4 scalars = material_at(m, 10u);
    s.base = material_color(m, 0u, lambda) * base_rgb.w;
    s.diffuse_roughness = scalars.y;
    s.metalness = clamp(scalars.x, 0.0, 1.0);
    s.specular_weight = max(specular_rgb.w, 0.0);
    s.metal_tint = material_color(m, 2u, lambda);
    s.specular_tint = s.metal_tint * s.specular_weight;
    float roughness = clamp(scalars.z, 0.0, 1.0);
    s.alpha = ggx_alpha(roughness, clamp(scalars.w, 0.0, 1.0));
    s.eta = max(material_at(m, 11u).x, 1.0001);
    float mu = clamp(wo.z, 0.0, 1.0);
    float e_ggx = max(table_ggx(mu, roughness), 1e-4);
    float missing = (1.0 - e_ggx) / e_ggx;
    s.metal_missing = missing;
    s.dielectric_compensation = 1.0 + fresnel_dielectric_average(s.eta) * missing;
    s.specular_albedo = clamp(s.specular_weight * s.dielectric_compensation * table_dielectric(mu, roughness, s.eta), 0.0, 1.0);
    float w_metal = s.metalness;
    float w_specular = (1.0 - s.metalness) * s.specular_albedo * max(rgb_luminance(specular_rgb.xyz), 0.0);
    float w_diffuse = (1.0 - s.metalness) * (1.0 - s.specular_albedo) * max(rgb_luminance(base_rgb.xyz * base_rgb.w), 0.0);
    float total = w_metal + w_specular + w_diffuse;
    s.p_metal = total > 0.0 ? w_metal / total : 0.0;
    s.p_specular = total > 0.0 ? w_specular / total : 0.0;
    s.p_diffuse = total > 0.0 ? w_diffuse / total : 0.0;
}

// The BSDF value at local directions wi (toward the light) and wo (toward the
// viewer), cosine not included, and the pdf `sample_surface` gives wi.
Spec evaluate_surface(Surface s, vec3 wi, vec3 wo, out float pdf) {
    pdf = 0.0;
    if (wi.z <= 0.0 || wo.z <= 0.0) return Spec(0.0);
    float dielectric = 1.0 - s.metalness;
    Spec f = eon(s.base, s.diffuse_roughness, wi, wo) * (dielectric * (1.0 - s.specular_albedo));
    pdf = s.p_diffuse * wi.z / PI;
    vec3 h = normalize(wi + wo);
    float microfacet = ggx_d(h, s.alpha) * ggx_g2(wi, wo, s.alpha) / (4.0 * wi.z * wo.z);
    float cos_h = dot(wo, h);
    if (dielectric > 0.0)
        f += s.specular_tint * (dielectric * fresnel_dielectric(cos_h, s.eta) * s.dielectric_compensation * microfacet);
    if (s.metalness > 0.0) {
        Spec f0 = s.base;
        Spec average = (20.0 * f0 + 1.0) / 21.0;
        Spec compensation = 1.0 + average * s.metal_missing;
        f += min(Spec(1.0), s.specular_weight * fresnel_f82(f0, s.metal_tint, cos_h)) * compensation * (s.metalness * microfacet);
    }
    pdf += (s.p_specular + s.p_metal) * ggx_reflection_pdf(wo, h, s.alpha);
    return f;
}

// A direction wi for wo from three uniform numbers (two for the direction, one
// to choose the lobe); `weight` is f · cos / pdf over all lobes.
bool sample_surface(Surface s, vec3 wo, vec3 u, out vec3 wi, out Spec weight, out float pdf) {
    if (wo.z <= 0.0) return false;
    if (u.z < s.p_diffuse) {
        vec2 disk = concentric(u.xy);
        wi = vec3(disk, sqrt(max(0.0, 1.0 - dot(disk, disk))));
    } else {
        if (s.p_specular + s.p_metal <= 0.0) return false;
        wi = reflect(-wo, ggx_sample(wo, s.alpha, u.xy));
    }
    if (wi.z <= 0.0) return false;
    Spec f = evaluate_surface(s, wi, wo, pdf);
    if (pdf <= 0.0) return false;
    weight = f * (wi.z / pdf);
    return true;
}
