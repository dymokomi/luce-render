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
// OpenPBR v1, in a local frame with the normal along +z on the side the ray
// came from (the layering after OpenPBR's specification and its reference):
//   - coat: a GGX dielectric layer at coat_ior over everything below; the
//     substrate takes what it does not reflect, tinted by coat_color, darkened by
//     light trapped under it, and seen through a coat-roughened base;
//   - substrate: F82-tint metal (Kutz et al.) mixed by metalness with a dielectric
//     base at specular_ior (relative to the coat where coated);
//   - dielectric base: GGX specular reflection over EON diffuse (Portsmouth, Kutz
//     and Hill 2025; the diffuse scaled by what the specular does not reflect),
//     or, by transmission_weight, refraction into the object (Walter et al. 2007)
//     tinted by transmission_color. A ray leaving an object sees only the
//     interface: reflection and refraction at the inverse index;
//   - emission, and geometry_opacity (handled where the hit is shaded).
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

vec3 previous_normal(uint path) { return octahedral_decode(get_u(SU_NORMAL, path)); }

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

#define LOBE_DIFFUSE 0u
#define LOBE_GLOSSY 1u
#define LOBE_TRANSMISSION 2u

struct Surface {
    vec3 normal;        // shading normal, facing the incoming ray's side
    vec3 tangent;
    vec3 bitangent;
    vec3 geometric;     // geometric normal on the same side
    uint material;
    bool inside;        // the ray is leaving the object (it hit a back face)
    Spec base;          // base color × base weight: diffuse albedo, metal F0
    float diffuse_roughness;
    float metalness;
    Spec metal_tint;    // specular color (F82 tint)
    Spec specular_tint; // specular color × specular weight
    float specular_weight;
    float transmission;
    Spec transmission_tint;
    vec2 alpha;         // the base's (coat-roughened)
    float eta;          // the dielectric base's index: inside over outside
    float dielectric_compensation;
    float metal_missing;
    float specular_albedo;
    float coat;
    vec2 coat_alpha;
    float coat_eta;
    float coat_compensation;
    float coat_albedo;
    Spec substrate;     // what reaches the layers under the coat
    float p_coat;       // lobe choice: wavelength-free, from the RGB parameters
    float p_metal;
    float p_specular;
    float p_transmission;
    float p_diffuse;
};

vec3 to_local(Surface s, vec3 v) { return vec3(dot(v, s.tangent), dot(v, s.bitangent), dot(v, s.normal)); }
vec3 to_world(Surface s, vec3 v) { return s.tangent * v.x + s.bitangent * v.y + s.normal * v.z; }

// Read the material and prepare its lobes for the view direction `wo` (local).
void prepare_surface(inout Surface s, vec3 wo, vec4 lambda) {
    uint m = s.material;
    vec4 base_rgb = material_at(m, 1u);
    vec4 specular_rgb = material_at(m, 3u);
    vec4 coat_rgb = material_at(m, 5u);
    vec4 transmission_rgb = material_at(m, 9u);
    vec4 scalars = material_at(m, 10u);
    vec4 indices = material_at(m, 11u);
    vec4 geometry = material_at(m, 12u);
    float mu = clamp(wo.z, 0.0, 1.0);
    s.base = material_color(m, 0u, lambda) * base_rgb.w;
    s.diffuse_roughness = scalars.y;
    s.metalness = clamp(scalars.x, 0.0, 1.0);
    s.specular_weight = max(specular_rgb.w, 0.0);
    s.metal_tint = material_color(m, 2u, lambda);
    s.specular_tint = s.metal_tint * s.specular_weight;
    s.transmission = clamp(transmission_rgb.w, 0.0, 1.0);
    s.transmission_tint = material_color(m, 8u, lambda);
    // Only a transmissive material has an inside; any other back face shades
    // like its front (a closed room's walls, seen from within).
    s.inside = s.inside && s.transmission > 0.0;
    // The coat (none from inside an object: its layers face the outside).
    s.coat = s.inside ? 0.0 : clamp(coat_rgb.w, 0.0, 1.0);
    float coat_roughness = clamp(indices.y, 0.0, 1.0);
    s.coat_alpha = ggx_alpha(coat_roughness, clamp(geometry.z, 0.0, 1.0));
    s.coat_eta = max(indices.z, 1.0001);
    float e_coat_ggx = max(table_ggx(mu, coat_roughness), 1e-4);
    s.coat_compensation = 1.0 + fresnel_dielectric_average(s.coat_eta) * (1.0 - e_coat_ggx) / e_coat_ggx;
    s.coat_albedo = s.coat > 0.0 ? clamp(s.coat_compensation * table_dielectric(mu, coat_roughness, s.coat_eta), 0.0, 1.0) : 0.0;
    // The base, roughened under the coat; its index relative to the coat's.
    float roughness = clamp(scalars.z, 0.0, 1.0);
    float r4 = roughness * roughness * roughness * roughness;
    float c4 = coat_roughness * coat_roughness * coat_roughness * coat_roughness;
    roughness = mix(roughness, min(1.0, pow(r4 + 2.0 * c4, 0.25)), s.coat);
    s.alpha = ggx_alpha(roughness, clamp(scalars.w, 0.0, 1.0));
    float eta = max(indices.x, 1.0001);
    float relative = eta / s.coat_eta;
    if (relative < 1.0) relative = 1.0 / relative;
    eta = max(mix(eta, relative, s.coat), 1.0001);
    s.eta = s.inside ? 1.0 / eta : eta;
    float e_ggx = max(table_ggx(mu, roughness), 1e-4);
    float missing = (1.0 - e_ggx) / e_ggx;
    s.metal_missing = missing;
    s.dielectric_compensation = 1.0 + fresnel_dielectric_average(eta) * missing;
    s.specular_albedo = clamp(s.specular_weight * s.dielectric_compensation * table_dielectric(mu, roughness, eta), 0.0, 1.0);
    // Under the coat: what it lets through, tinted, and darkened by the light
    // the coat reflects back down (OpenPBR's Delta = (1 - K) / (1 - E_b K)).
    s.substrate = Spec(1.0);
    float substrate_rgb = 1.0;
    if (s.coat > 0.0) {
        float k_smooth = fresnel_dielectric(mu, s.coat_eta);
        float k_rough = 1.0 - (1.0 - fresnel_dielectric_average(s.coat_eta)) / (s.coat_eta * s.coat_eta);
        float base_roughness = mix(1.0, roughness, s.metalness);
        float k = mix(k_smooth, k_rough, base_roughness);
        float darkening = clamp(material_at(m, 11u).w, 0.0, 1.0);
        Spec delta = (1.0 - k) / (Spec(1.0) - s.base * k);
        Spec under = material_color(m, 4u, lambda) * (1.0 - s.coat_albedo) * mix(Spec(1.0), delta, darkening);
        s.substrate = mix(Spec(1.0), under, s.coat);
        float delta_rgb = (1.0 - k) / max(1e-4, 1.0 - rgb_luminance(base_rgb.xyz * base_rgb.w) * k);
        substrate_rgb = mix(1.0, rgb_luminance(coat_rgb.xyz) * (1.0 - s.coat_albedo) * mix(1.0, delta_rgb, darkening), s.coat);
    }
    // Lobe choice.
    float dielectric = 1.0 - s.metalness;
    float w_coat = s.coat * s.coat_albedo;
    float w_metal = substrate_rgb * s.metalness;
    float w_specular = substrate_rgb * dielectric * s.specular_albedo * max(rgb_luminance(specular_rgb.xyz), 0.0);
    float w_transmission = substrate_rgb * dielectric * s.transmission * (1.0 - s.specular_albedo) * max(rgb_luminance(transmission_rgb.xyz), 0.05);
    float w_diffuse = substrate_rgb * dielectric * (1.0 - s.transmission) * (1.0 - s.specular_albedo) * max(rgb_luminance(base_rgb.xyz * base_rgb.w), 0.0);
    if (s.inside) {
        // Leaving an object: only its interface.
        w_metal = 0.0;
        w_diffuse = 0.0;
        w_specular = max(w_specular, 0.05);
        w_transmission = max(w_transmission, s.transmission > 0.0 ? 0.05 : 0.0);
    }
    float total = w_coat + w_metal + w_specular + w_transmission + w_diffuse;
    float scale = total > 0.0 ? 1.0 / total : 0.0;
    s.p_coat = w_coat * scale;
    s.p_metal = w_metal * scale;
    s.p_specular = w_specular * scale;
    s.p_transmission = w_transmission * scale;
    s.p_diffuse = w_diffuse * scale;
}

// The refraction half-vector for wo and wi on opposite sides at index ratio eta
// (transmitted over incident), facing +z; false when the pair cannot refract.
bool refraction_half(vec3 wi, vec3 wo, float eta, out vec3 h) {
    h = normalize(wo + eta * wi);
    if (h.z < 0.0) h = -h;
    return dot(wo, h) > 0.0 && dot(wi, h) < 0.0;
}

// The BSDF value at local directions wi (toward the light) and wo (toward the
// viewer), cosine not included, and the pdf `sample_surface` gives wi.
Spec evaluate_surface(Surface s, vec3 wi, vec3 wo, out float pdf) {
    pdf = 0.0;
    if (wo.z <= 0.0 || wi.z == 0.0) return Spec(0.0);
    float dielectric = 1.0 - s.metalness;
    if (wi.z < 0.0) {
        // Refraction through the dielectric base.
        if (s.p_transmission <= 0.0) return Spec(0.0);
        vec3 h;
        if (!refraction_half(wi, wo, s.eta, h)) return Spec(0.0);
        float o_h = dot(wo, h), i_h = dot(wi, h);
        float denominator = o_h + s.eta * i_h;
        float jacobian = s.eta * s.eta * abs(i_h) / (denominator * denominator);
        float d = ggx_d(h, s.alpha);
        // Radiance transport: Walter et al.'s BTDF without its eta² (pbrt-v4's form).
        float f = (1.0 - fresnel_dielectric(o_h, s.eta)) * d * ggx_g2(-wi, wo, s.alpha) * abs(i_h) * o_h / (abs(wi.z) * wo.z * denominator * denominator);
        pdf = s.p_transmission * ggx_g1(wo, s.alpha) * d * o_h / wo.z * jacobian;
        return s.substrate * s.transmission_tint * (dielectric * s.transmission * f);
    }
    vec3 h = normalize(wi + wo);
    float cos_h = dot(wo, h);
    Spec f = Spec(0.0);
    if (s.p_coat > 0.0) {
        float coat = ggx_d(h, s.coat_alpha) * ggx_g2(wi, wo, s.coat_alpha) / (4.0 * wi.z * wo.z);
        f += Spec(s.coat * fresnel_dielectric(cos_h, s.coat_eta) * s.coat_compensation * coat);
        pdf += s.p_coat * ggx_reflection_pdf(wo, h, s.coat_alpha);
    }
    Spec below = Spec(0.0);
    if (!s.inside) {
        below += eon(s.base, s.diffuse_roughness, wi, wo) * (dielectric * (1.0 - s.transmission) * (1.0 - s.specular_albedo));
        pdf += s.p_diffuse * wi.z / PI;
    }
    float microfacet = ggx_d(h, s.alpha) * ggx_g2(wi, wo, s.alpha) / (4.0 * wi.z * wo.z);
    if (dielectric > 0.0)
        below += s.specular_tint * (dielectric * fresnel_dielectric(cos_h, s.eta) * s.dielectric_compensation * microfacet);
    if (s.metalness > 0.0 && !s.inside) {
        Spec average = (20.0 * s.base + 1.0) / 21.0;
        Spec compensation = 1.0 + average * s.metal_missing;
        below += min(Spec(1.0), s.specular_weight * fresnel_f82(s.base, s.metal_tint, cos_h)) * compensation * (s.metalness * microfacet);
    }
    pdf += (s.p_specular + s.p_metal) * ggx_reflection_pdf(wo, h, s.alpha);
    return f + s.substrate * below;
}

// A direction wi for wo from three uniform numbers (two for the direction, one
// to choose the lobe); `weight` is f · |cos| / pdf over all lobes, `lobe` what
// the sample was (LOBE_*) for the bounce limits.
bool sample_surface(Surface s, vec3 wo, vec3 u, out vec3 wi, out Spec weight, out float pdf, out uint lobe) {
    if (wo.z <= 0.0) return false;
    float pick = u.z;
    if (pick < s.p_diffuse) {
        vec2 disk = concentric(u.xy);
        wi = vec3(disk, sqrt(max(0.0, 1.0 - dot(disk, disk))));
        lobe = LOBE_DIFFUSE;
    } else if ((pick -= s.p_diffuse) < s.p_coat) {
        wi = reflect(-wo, ggx_sample(wo, s.coat_alpha, u.xy));
        lobe = LOBE_GLOSSY;
    } else if ((pick -= s.p_coat) < s.p_specular + s.p_metal) {
        wi = reflect(-wo, ggx_sample(wo, s.alpha, u.xy));
        lobe = LOBE_GLOSSY;
    } else if (s.p_transmission > 0.0) {
        vec3 h = ggx_sample(wo, s.alpha, u.xy);
        wi = refract(-wo, h, 1.0 / s.eta);
        if (dot(wi, wi) == 0.0) return false;   // total internal reflection
        lobe = LOBE_TRANSMISSION;
    } else {
        return false;
    }
    if (wi.z == 0.0) return false;
    Spec f = evaluate_surface(s, wi, wo, pdf);
    if (pdf <= 0.0) return false;
    weight = f * (abs(wi.z) / pdf);
    return true;
}
