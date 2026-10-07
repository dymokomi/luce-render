// OpenPBR Surface: material records (src/render/materials.lucb) and lobes.
// Records are MATERIAL_STRIDE vec4s; each color is its spectrum fit and its RGB:
//   0, 1   base color fit; base color RGB, base weight
//   2, 3   specular color fit; RGB, specular weight
//   4, 5   coat color fit; RGB, coat weight
//   6, 7   emission fit (× luminance); RGB (× luminance), luminance
//   8, 9   transmission color fit; RGB, transmission weight
//   10     metalness, diffuse roughness, specular roughness, specular anisotropy
//   11     specular IOR, coat roughness, coat IOR, coat darkening
//   12     opacity, thin walled, coat anisotropy, dispersion (Cauchy's B in nm², 0 none)
//   13     the node graph's program entry (-1 none; shader_program.glsl), -, -, -
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
// Buffers reached by address (K_ADDRESSES): the albedo tables, and each
// triangle's corners' texture coordinates (six floats).
#define K_ADDRESSES 23      // xy the albedo tables' address, zw the texture coordinates'
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer FloatData { float values[]; };
uint64_t address_of(vec2 bits) { return packUint2x32(uvec2(floatBitsToUint(bits.x), floatBitsToUint(bits.y))); }
float table_value(uint at) { return FloatData(address_of(constants[K_ADDRESSES].xy)).values[at]; }

vec4 material_at(uint material, uint row) { return materials[material * MATERIAL_STRIDE + row]; }

// The texture coordinates of a hit on `triangle` at barycentrics (b1, b2).
vec2 triangle_uv(uint triangle, vec2 b) {
    FloatData uvs = FloatData(address_of(constants[K_ADDRESSES].zw));
    uint at = triangle * 6u;
    vec2 a = vec2(uvs.values[at], uvs.values[at + 1u]);
    vec2 c1 = vec2(uvs.values[at + 2u], uvs.values[at + 3u]);
    vec2 c2 = vec2(uvs.values[at + 4u], uvs.values[at + 5u]);
    return a * (1.0 - b.x - b.y) + c1 * b.x + c2 * b.y;
}

// Where the surface being shaded is in its textures; set per hit.
vec2 surface_uv = vec2(0.0);

// The parameters the material's node graph drives at this hit
// (src/render/shader_graph.lucb's registers), and which: set per hit by
// run_program, read here in place of the record's.
uint program_mask = 0u;
float program_values[23];
bool driven(uint bit) { return (program_mask & (1u << bit)) != 0u; }
vec3 driven3(uint at) { return vec3(program_values[at], program_values[at + 1u], program_values[at + 2u]); }

#ifdef MATERIAL_TEXTURES
#extension GL_EXT_nonuniform_qualifier : require
layout(set = 0, binding = 13) uniform sampler texture_sampler;
layout(set = 1, binding = 0) uniform texture2D textures[];
vec4 texture_at(float slot, vec2 uv) {
    return textureLod(sampler2D(textures[nonuniformEXT(uint(slot))], texture_sampler), fract(uv), 0.0);
}
#else
vec4 texture_at(float slot, vec2 uv) { return vec4(1.0); }
#endif

// Linear Rec.709 (what textures sample as) to ACEScg.
const mat3 rec709_to_acescg = mat3(0.613097, 0.070194, 0.020616,
                                   0.339523, 0.916354, 0.109570,
                                   0.047379, 0.013452, 0.869815);

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
    float a = mix(table_value(base + y0 * TABLE_MU + x0), table_value(base + y0 * TABLE_MU + x1), fx);
    float b = mix(table_value(base + y1 * TABLE_MU + x0), table_value(base + y1 * TABLE_MU + x1), fx);
    return mix(a, b, fy);
}

float table_ggx(float mu, float r) { return table_row(TABLE_GGX, r, mu); }

float table_ggx_average(float r) {
    float y = clamp(r * float(TABLE_ROUGHNESS - 1u), 0.0, float(TABLE_ROUGHNESS - 1u));
    uint y0 = uint(y), y1 = min(uint(y) + 1u, TABLE_ROUGHNESS - 1u);
    return mix(table_value(TABLE_GGX_AVERAGE + y0), table_value(TABLE_GGX_AVERAGE + y1), y - float(y0));
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

// The wavelength dispersive indices are taken at: the path's hero (its first
// lane) once dispersive_hit has ended the others. Set per path by the kernels.
float hero_wavelength = 0.0;

// Cauchy's index at `lambda` nm through `ior` at 587.6 nm.
float dispersed_ior(float ior, float b, float lambda) {
    return ior + b * (1.0 / (lambda * lambda) - 1.0 / (587.6 * 587.6));
}

// Whether material `m` disperses (the spectral build only).
bool disperses(uint m) { return SPECTRAL != 0 && HAS_TRANSMISSION && material_at(m, 12u).w != 0.0; }

// A path meeting a dispersive material: its directions now depend on the
// wavelength, so the three secondary wavelengths end (their lanes zero, their
// wavelengths negative, which the film skips) and the hero carries the whole
// sample (pbrt-v4's TerminateSecondary). Once per path.
void dispersive_hit(inout Spec throughput, inout Spec radiance, inout vec4 lambda) {
    if (lambda.y < 0.0) return;
    throughput = Spec(throughput.x * 4.0, 0.0, 0.0, 0.0);
    radiance = Spec(radiance.x * 4.0, 0.0, 0.0, 0.0);
    lambda = vec4(lambda.x, -lambda.yzw);
}

struct Surface {
    vec3 normal;        // shading normal, facing the side the surface is seen from
    vec3 tangent;
    vec3 bitangent;
    vec3 geometric;     // geometric normal on the same side
    uint material;
    bool back;          // seen from behind its triangle's front face
    bool inside;        // seen from within a transmissive object (back, transmissive)
    // The material, as read (read_material): no direction, no wavelengths.
    float diffuse_roughness;
    float metalness;
    float specular_weight;
    float transmission;
    float roughness;    // the base's, coat-roughened
    float coat_roughness;
    float ior;          // the dielectric base's index (relative to the coat's where coated)
    float darkening;
    float lum_base;     // luminances of the RGB parameters the lobe choice weighs
    float lum_specular;
    float lum_transmission;
    float lum_coat;
    vec2 alpha;         // the base's
    float eta;          // the dielectric base's index: inside over outside
    float coat;
    vec2 coat_alpha;
    float coat_eta;
    // For a view direction (orient_lobes).
    float dielectric_compensation;
    float metal_missing;
    float specular_albedo;
    float coat_compensation;
    float coat_albedo;
    float coat_k;       // what the coat reflects back down (for its darkening)
    float p_coat;       // lobe choice: wavelength-free, from the RGB parameters
    float p_metal;
    float p_specular;
    float p_transmission;
    float p_diffuse;
    // At the path's wavelengths (read_colors, update_substrate).
    Spec base;          // base color × base weight: diffuse albedo, metal F0
    Spec metal_tint;    // specular color (F82 tint)
    Spec specular_tint; // specular color × specular weight
    Spec transmission_tint;
    Spec coat_tint;
    Spec substrate;     // what reaches the layers under the coat
};

vec3 to_local(Surface s, vec3 v) { return vec3(dot(v, s.tangent), dot(v, s.bitangent), dot(v, s.normal)); }
vec3 to_world(Surface s, vec3 v) { return s.tangent * v.x + s.bitangent * v.y + s.normal * v.z; }

// The surface at `normal` (shading) and `geometric`, both facing the side it is
// seen from; `back` when that is behind the triangle's front face.
Surface surface_at(vec3 normal, vec3 geometric, uint material, bool back) {
    Surface s;
    s.normal = normal;
    basis(normal, s.tangent, s.bitangent);
    s.geometric = geometric;
    s.material = material;
    s.back = back;
    return s;
}

// The same point seen from its other side (a refracted direction's viewer).
Surface turned(Surface s) {
    return surface_at(-s.normal, -s.geometric, s.material, !s.back);
}

// The material's parameters, for the side the surface is seen from.
void read_material(inout Surface s) {
    uint m = s.material;
    vec4 base_rgb = material_at(m, 1u);
    vec4 specular_rgb = material_at(m, 3u);
    vec4 coat_rgb = material_at(m, 5u);
    vec4 transmission_rgb = material_at(m, 9u);
    vec4 scalars = material_at(m, 10u);
    vec4 indices = material_at(m, 11u);
    vec4 geometry = material_at(m, 12u);
    // What the node graph drives replaces the record's.
    if (program_mask != 0u) {
        if (driven(0u)) base_rgb.xyz = driven3(0u);
        if (driven(1u)) base_rgb.w = program_values[3];
        if (driven(2u)) scalars.x = program_values[4];
        if (driven(3u)) scalars.y = program_values[5];
        if (driven(4u)) specular_rgb.w = program_values[6];
        if (driven(5u)) scalars.z = program_values[7];
        if (driven(6u)) coat_rgb.w = program_values[8];
        if (driven(7u)) indices.y = program_values[9];
        if (driven(8u)) transmission_rgb.w = program_values[10];
        if (driven(10u)) specular_rgb.xyz = driven3(14u);
        if (driven(11u)) transmission_rgb.xyz = driven3(17u);
        if (driven(12u)) coat_rgb.xyz = driven3(20u);
    }
    s.diffuse_roughness = scalars.y;
    s.metalness = HAS_METAL ? clamp(scalars.x, 0.0, 1.0) : 0.0;
    s.specular_weight = max(specular_rgb.w, 0.0);
    s.transmission = HAS_TRANSMISSION ? clamp(transmission_rgb.w, 0.0, 1.0) : 0.0;
    // Only a transmissive material has an inside; any other back face shades
    // like its front (a closed room's walls, seen from within).
    s.inside = s.back && s.transmission > 0.0;
    // The coat (none from inside an object: its layers face the outside).
    s.coat = (HAS_COAT && !s.inside) ? clamp(coat_rgb.w, 0.0, 1.0) : 0.0;
    s.coat_roughness = clamp(indices.y, 0.0, 1.0);
    s.coat_eta = max(indices.z, 1.0001);
    s.coat_alpha = vec2(1.0);
    if (HAS_COAT && s.coat > 0.0) s.coat_alpha = ggx_alpha(s.coat_roughness, clamp(geometry.z, 0.0, 1.0));
    s.darkening = clamp(indices.w, 0.0, 1.0);
    // The base, roughened under the coat; its index relative to the coat's.
    float roughness = clamp(scalars.z, 0.0, 1.0);
    float r4 = roughness * roughness * roughness * roughness;
    float c4 = s.coat_roughness * s.coat_roughness * s.coat_roughness * s.coat_roughness;
    s.roughness = mix(roughness, min(1.0, pow(r4 + 2.0 * c4, 0.25)), s.coat);
    s.alpha = ggx_alpha(s.roughness, clamp(scalars.w, 0.0, 1.0));
    float eta = max(indices.x, 1.0001);
    if (SPECTRAL != 0 && geometry.w != 0.0 && hero_wavelength > 0.0) eta = max(dispersed_ior(eta, geometry.w, hero_wavelength), 1.0001);
    float relative = eta / s.coat_eta;
    if (relative < 1.0) relative = 1.0 / relative;
    s.ior = max(mix(eta, relative, s.coat), 1.0001);
    s.eta = s.inside ? 1.0 / s.ior : s.ior;
    s.lum_base = max(rgb_luminance(base_rgb.xyz * base_rgb.w), 0.0);
    s.lum_specular = max(rgb_luminance(specular_rgb.xyz), 0.0);
    s.lum_transmission = max(rgb_luminance(transmission_rgb.xyz), 0.05);
    s.lum_coat = rgb_luminance(coat_rgb.xyz);
}

// The lobes for the view direction `wo` (local): the albedos f needs and the
// lobe choice a direction's pdf needs. Reads nothing: a read surface can be
// oriented again for another direction on the same side.
void orient_lobes(inout Surface s, vec3 wo) {
    float mu = clamp(wo.z, 0.0, 1.0);
    s.coat_compensation = 1.0;
    s.coat_albedo = 0.0;
    if (HAS_COAT && s.coat > 0.0) {
        float e_coat_ggx = max(table_ggx(mu, s.coat_roughness), 1e-4);
        s.coat_compensation = 1.0 + fresnel_dielectric_average(s.coat_eta) * (1.0 - e_coat_ggx) / e_coat_ggx;
        s.coat_albedo = clamp(s.coat_compensation * table_dielectric(mu, s.coat_roughness, s.coat_eta), 0.0, 1.0);
    }
    float e_ggx = max(table_ggx(mu, s.roughness), 1e-4);
    float missing = (1.0 - e_ggx) / e_ggx;
    s.metal_missing = missing;
    s.dielectric_compensation = 1.0 + fresnel_dielectric_average(s.ior) * missing;
    s.specular_albedo = clamp(s.specular_weight * s.dielectric_compensation * table_dielectric(mu, s.roughness, s.ior), 0.0, 1.0);
    // Under the coat: what it lets through, darkened by the light it reflects
    // back down (OpenPBR's Delta = (1 - K) / (1 - E_b K)).
    float substrate_rgb = 1.0;
    s.coat_k = 0.0;
    if (HAS_COAT && s.coat > 0.0) {
        float k_smooth = fresnel_dielectric(mu, s.coat_eta);
        float k_rough = 1.0 - (1.0 - fresnel_dielectric_average(s.coat_eta)) / (s.coat_eta * s.coat_eta);
        float base_roughness = mix(1.0, s.roughness, s.metalness);
        s.coat_k = mix(k_smooth, k_rough, base_roughness);
        float delta_rgb = (1.0 - s.coat_k) / max(1e-4, 1.0 - s.lum_base * s.coat_k);
        substrate_rgb = mix(1.0, s.lum_coat * (1.0 - s.coat_albedo) * mix(1.0, delta_rgb, s.darkening), s.coat);
    }
    // Lobe choice.
    float dielectric = 1.0 - s.metalness;
    float w_coat = s.coat * s.coat_albedo;
    float w_metal = substrate_rgb * s.metalness;
    float w_specular = substrate_rgb * dielectric * s.specular_albedo * s.lum_specular;
    float w_transmission = substrate_rgb * dielectric * s.transmission * (1.0 - s.specular_albedo) * s.lum_transmission;
    float w_diffuse = substrate_rgb * dielectric * (1.0 - s.transmission) * (1.0 - s.specular_albedo) * s.lum_base;
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

// The surface's lobes for the view direction `wo` (local); no wavelengths.
void prepare_lobes(inout Surface s, vec3 wo) {
    read_material(s);
    orient_lobes(s, wo);
}

// What reaches the layers under the coat, at the path's wavelengths (after
// the colors and the lobes for a direction).
void update_substrate(inout Surface s) {
    s.substrate = Spec(1.0);
    if (HAS_COAT && s.coat > 0.0) {
        Spec delta = (1.0 - s.coat_k) / (Spec(1.0) - s.base * s.coat_k);
        Spec under = s.coat_tint * (1.0 - s.coat_albedo) * mix(Spec(1.0), delta, s.darkening);
        s.substrate = mix(Spec(1.0), under, s.coat);
    }
}

// The surface's colors at the path's wavelengths (after read_material; the
// substrate follows with the lobes, update_substrate).
void read_colors(inout Surface s, vec4 lambda) {
    uint m = s.material;
    // Colors the node graph drives are reflectance spectra through the
    // environment basis (at most 1); the record's are fitted ones.
    float weight = driven(1u) ? program_values[3] : material_at(m, 1u).w;
    s.base = (driven(0u) ? clamp(spec_of_texel(driven3(0u), lambda), Spec(0.0), Spec(1.0)) : material_color(m, 0u, lambda)) * weight;
    s.metal_tint = driven(10u) ? clamp(spec_of_texel(driven3(14u), lambda), Spec(0.0), Spec(1.0)) : material_color(m, 2u, lambda);
    s.specular_tint = s.metal_tint * s.specular_weight;
    s.transmission_tint = driven(11u) ? clamp(spec_of_texel(driven3(17u), lambda), Spec(0.0), Spec(1.0)) : material_color(m, 8u, lambda);
    s.coat_tint = (HAS_COAT && s.coat > 0.0) ? (driven(12u) ? clamp(spec_of_texel(driven3(20u), lambda), Spec(0.0), Spec(1.0)) : material_color(m, 4u, lambda)) : Spec(1.0);
}

// Read the material and prepare its lobes and colors for the view direction
// `wo` (local).
void prepare_surface(inout Surface s, vec3 wo, vec4 lambda) {
    prepare_lobes(s, wo);
    read_colors(s, lambda);
    update_substrate(s);
}

// A prepared surface for another view direction `wo` on the same side.
void orient_surface(inout Surface s, vec3 wo) {
    orient_lobes(s, wo);
    update_substrate(s);
}

// The lobe choice light paths sample with (after read_material): the camera's
// weighing with Fresnel at normal incidence for the albedos, so it holds for
// every direction and the other side can recompute its pdfs without tables.
void light_lobes(inout Surface s) {
    float f0 = (s.ior - 1.0) / (s.ior + 1.0);
    float specular_albedo = clamp(s.specular_weight * f0 * f0, 0.0, 1.0);
    float coat_albedo = 0.0;
    if (HAS_COAT && s.coat > 0.0) {
        float c0 = (s.coat_eta - 1.0) / (s.coat_eta + 1.0);
        coat_albedo = c0 * c0;
    }
    float substrate_rgb = mix(1.0, s.lum_coat * (1.0 - coat_albedo), s.coat);
    float dielectric = 1.0 - s.metalness;
    float w_coat = s.coat * coat_albedo;
    float w_metal = substrate_rgb * s.metalness;
    float w_specular = substrate_rgb * dielectric * specular_albedo * s.lum_specular;
    float w_transmission = substrate_rgb * dielectric * s.transmission * (1.0 - specular_albedo) * s.lum_transmission;
    float w_diffuse = substrate_rgb * dielectric * (1.0 - s.transmission) * (1.0 - specular_albedo) * s.lum_base;
    if (s.inside) {
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

// The pdf `sample_direction` gives wi for wo (local), over all lobes; needs
// only prepare_lobes.
float surface_pdf(Surface s, vec3 wi, vec3 wo) {
    if (wo.z <= 0.0 || wi.z == 0.0) return 0.0;
    if (wi.z < 0.0) {
        if (!HAS_TRANSMISSION || s.p_transmission <= 0.0) return 0.0;
        vec3 h;
        if (!refraction_half(wi, wo, s.eta, h)) return 0.0;
        float o_h = dot(wo, h), i_h = dot(wi, h);
        float denominator = o_h + s.eta * i_h;
        float jacobian = s.eta * s.eta * abs(i_h) / (denominator * denominator);
        return s.p_transmission * ggx_g1(wo, s.alpha) * ggx_d(h, s.alpha) * o_h / wo.z * jacobian;
    }
    vec3 h = normalize(wi + wo);
    float pdf = 0.0;
    if (HAS_COAT && s.p_coat > 0.0) pdf += s.p_coat * ggx_reflection_pdf(wo, h, s.coat_alpha);
    if (!s.inside) pdf += s.p_diffuse * wi.z / PI;
    pdf += (s.p_specular + s.p_metal) * ggx_reflection_pdf(wo, h, s.alpha);
    return pdf;
}

// The BSDF value at local directions wi (toward the light) and wo (toward the
// viewer), cosine not included, and the pdf `sample_surface` gives wi.
Spec evaluate_surface(Surface s, vec3 wi, vec3 wo, out float pdf) {
    pdf = 0.0;
    if (wo.z <= 0.0 || wi.z == 0.0) return Spec(0.0);
    pdf = surface_pdf(s, wi, wo);
    float dielectric = 1.0 - s.metalness;
    if (wi.z < 0.0) {
        // Refraction through the dielectric base.
        if (!HAS_TRANSMISSION || s.p_transmission <= 0.0) return Spec(0.0);
        vec3 h;
        if (!refraction_half(wi, wo, s.eta, h)) return Spec(0.0);
        float o_h = dot(wo, h), i_h = dot(wi, h);
        float denominator = o_h + s.eta * i_h;
        float d = ggx_d(h, s.alpha);
        // Radiance transport: Walter et al.'s BTDF without its eta² (pbrt-v4's form).
        float f = (1.0 - fresnel_dielectric(o_h, s.eta)) * d * ggx_g2(-wi, wo, s.alpha) * abs(i_h) * o_h / (abs(wi.z) * wo.z * denominator * denominator);
        return s.substrate * s.transmission_tint * (dielectric * s.transmission * f);
    }
    vec3 h = normalize(wi + wo);
    float cos_h = dot(wo, h);
    Spec f = Spec(0.0);
    if (HAS_COAT && s.p_coat > 0.0) {
        float coat = ggx_d(h, s.coat_alpha) * ggx_g2(wi, wo, s.coat_alpha) / (4.0 * wi.z * wo.z);
        f += Spec(s.coat * fresnel_dielectric(cos_h, s.coat_eta) * s.coat_compensation * coat);
    }
    Spec below = Spec(0.0);
    if (!s.inside)
        below += eon(s.base, s.diffuse_roughness, wi, wo) * (dielectric * (1.0 - s.transmission) * (1.0 - s.specular_albedo));
    float microfacet = ggx_d(h, s.alpha) * ggx_g2(wi, wo, s.alpha) / (4.0 * wi.z * wo.z);
    if (dielectric > 0.0)
        below += s.specular_tint * (dielectric * fresnel_dielectric(cos_h, s.eta) * s.dielectric_compensation * microfacet);
    if (HAS_METAL && s.metalness > 0.0 && !s.inside) {
        Spec average = (20.0 * s.base + 1.0) / 21.0;
        Spec compensation = 1.0 + average * s.metal_missing;
        below += min(Spec(1.0), s.specular_weight * fresnel_f82(s.base, s.metal_tint, cos_h)) * compensation * (s.metalness * microfacet);
    }
    return f + s.substrate * below;
}

// A direction wi for wo from three uniform numbers (two for the direction, one
// to choose the lobe), its pdf over all lobes, and `lobe`, what the sample was
// (LOBE_*) for the bounce limits. Needs only prepare_lobes.
bool sample_direction(Surface s, vec3 wo, vec3 u, out vec3 wi, out float pdf, out uint lobe) {
    pdf = 0.0;
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
    } else if (HAS_TRANSMISSION && s.p_transmission > 0.0) {
        vec3 h = ggx_sample(wo, s.alpha, u.xy);
        wi = refract(-wo, h, 1.0 / s.eta);
        if (dot(wi, wi) == 0.0) return false;   // total internal reflection
        lobe = LOBE_TRANSMISSION;
    } else {
        return false;
    }
    if (wi.z == 0.0) return false;
    pdf = surface_pdf(s, wi, wo);
    return pdf > 0.0;
}

// sample_direction, and `weight`: f · |cos| / pdf.
bool sample_surface(Surface s, vec3 wo, vec3 u, out vec3 wi, out Spec weight, out float pdf, out uint lobe) {
    if (!sample_direction(s, wo, u, wi, pdf, lobe)) return false;
    float pdf_again;
    Spec f = evaluate_surface(s, wi, wo, pdf_again);
    weight = f * (abs(wi.z) / pdf);
    return true;
}
