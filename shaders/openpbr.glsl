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
// Lobes so far: EON diffuse (Portsmouth, Kutz and Hill, "EON: A practical
// energy-preserving rough diffuse BRDF", 2025), sampled cosine-weighted, and
// emission. All functions work in a local frame with the normal along +z.

#define MATERIAL_STRIDE 16u

layout(set = 0, binding = 11, std430) readonly buffer Shading { uvec4 shading[]; };
layout(set = 0, binding = 12, std430) readonly buffer Materials { vec4 materials[]; };

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

// -- The surface at a hit.

struct Surface {
    vec3 normal;       // shading normal, facing the incoming ray's side
    vec3 tangent;
    vec3 bitangent;
    vec3 geometric;    // geometric normal on the same side
    uint material;
    Spec base;         // base color × base weight
    float diffuse_roughness;
};

vec3 to_local(Surface s, vec3 v) { return vec3(dot(v, s.tangent), dot(v, s.bitangent), dot(v, s.normal)); }
vec3 to_world(Surface s, vec3 v) { return s.tangent * v.x + s.bitangent * v.y + s.normal * v.z; }

// The BSDF value at local directions wi (toward the light) and wo (toward the
// viewer), cosine not included, and the pdf `sample_surface` gives wi.
Spec evaluate_surface(Surface s, vec3 wi, vec3 wo, out float pdf) {
    if (wi.z <= 0.0 || wo.z <= 0.0) { pdf = 0.0; return Spec(0.0); }
    pdf = wi.z / PI;
    return eon(s.base, s.diffuse_roughness, wi, wo);
}

// A direction wi for wo from two uniform numbers; `weight` is f · cos / pdf.
bool sample_surface(Surface s, vec3 wo, vec3 u, out vec3 wi, out Spec weight, out float pdf) {
    vec2 disk = concentric(u.xy);
    wi = vec3(disk, sqrt(max(0.0, 1.0 - dot(disk, disk))));
    if (wi.z <= 0.0 || wo.z <= 0.0) return false;
    Spec f = evaluate_surface(s, wi, wo, pdf);
    if (pdf <= 0.0) return false;
    weight = f * (wi.z / pdf);
    return true;
}
